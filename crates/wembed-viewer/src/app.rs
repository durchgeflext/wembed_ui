//! winit ApplicationHandler. Owns the Window, the (lazily-constructed) Renderer, the
//! EmbedderHandle, the Camera, an egui context, and an FPS counter. Each frame:
//!  - poll the embedder → maybe update positions
//!  - refit the camera → push new view uniforms
//!  - build the egui overlay (FPS + iteration stats + pause + mode toggle + hover tooltip)
//!  - render (scene pass + egui pass)
//!
//! Running vs. inspection: while running, the camera auto-refits every frame to the
//! incoming positions and drag/zoom/hover are gated off. While in an interactive mode
//! (Paused or Finished), `refit_camera` is a no-op and mouse drag / scroll drive the
//! camera directly (pan / zoom-at-cursor). Space toggles pause. The embedder auto-pauses
//! when it finishes so the viewer transitions cleanly into the "explore the final layout"
//! mode.
//!
//! All of that state lives in [`crate::inspection::InspectionMode`] — a three-state
//! machine (`Live` / `Paused` / `Finished`) synced from the embedder's flags each tick.
//! The picker, hover vertex, and "may I drag/zoom?" predicate are its responsibility.
//! App just relays events into it and reads out `hover()` / `is_interactive()` at the
//! call sites that used to sprinkle `is_paused()` checks.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use egui::ViewportId;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

use crate::attrs::Attributes;
use crate::camera::Camera;
use crate::cli::{EdgeColorSource, ScalarSource, VertexColorSource};
use crate::config::CameraConfig;
use crate::convergence::Convergence;
use crate::density::DensityAnchors;
use crate::embedder::{format_progress_with_iteration, TopologyProducer, Update};
use crate::exporter::ExportConfig;
use crate::framing::{
    best_rotation_deg_for_aspect, point_radius, scaled_sizing, SizingConfig, View, Q_BUFFER, Q_HI,
    Q_LO, REFERENCE_HEIGHT,
};
use crate::highlight::Highlight;
use crate::inspection::InspectionMode;
use crate::projection::{ProjectionMode, Projector};
use crate::render::{Renderer, UiFrame};
use crate::scene::{Scene, Topology};
use crate::scene_renderer::{Channels, TextQuad};
use crate::settings::{DensityParams, LegendParams, LegendSwatch, SessionInfo, TextParams};
use crate::theme::{wgpu_clear_color, ThemeRegistry};

/// Debounce before auto-exposure recomputes the density anchors after camera motion stops.
/// Readbacks stall the pipeline and tonal pumping while panning is unwanted, so anchors only
/// refresh once the view has been at rest this long.
const DENSITY_EXPOSURE_DEBOUNCE: Duration = Duration::from_millis(200);

/// How long the search field must be idle before a query is applied. Below this,
/// keystrokes accumulate without re-running the label scan — a 15-20ms scan per
/// keystroke on 322k labels would hog the frame otherwise.
const SEARCH_DEBOUNCE: Duration = Duration::from_millis(150);

/// EMA weight for FPS smoothing. Lower = smoother but slower to react.
const FPS_EMA_ALPHA: f32 = 0.05;

/// Zoom sensitivity per wheel line. Chosen so a single notch is a perceptible but not jarring
/// step; sign of `dy` decides direction (positive scroll → zoom in → factor < 1).
const ZOOM_PER_LINE: f32 = 0.1;

/// Directory that interactive exports (`save png` / `save embedding`) land in, relative
/// to CWD. Created on demand; gitignored. Headless mode ignores this — it writes to the
/// explicit `--output` path.
const EXPORT_DIR: &str = "exports";

/// Per-channel source selections for the current presentation strategy (currently just
/// "points"; the density strategy will land alongside as a nested submode). This struct
/// is pure config — no wgpu, no locks — so egui can mutate it freely inside a UI closure.
/// The tick loop reads it *after* the UI runs and pushes any deltas into Renderer.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct VisualState {
    /// Vertex color source. Falls back to the theme color while its file isn't loaded or
    /// doesn't match the vertex count.
    pub vertex_color: VertexColorSource,
    /// Size channel source — `Constant` = the density-clamped base radius, `Weight` =
    /// times the per-vertex `sqrt(w/mean_w)`. Defaults to `Weight`, the usual intent.
    pub vertex_size: ScalarSource,
    /// Global point-size multiplier from the UI slider (log-scale in the UI).
    pub vertex_size_scale: f32,
    /// Global point-alpha multiplier from the UI slider. Linear scale [0, 1]. `0.0` =
    /// vertices fully transparent (replaces the historical on/off toggle).
    pub vertex_alpha_scale: f32,
    /// Alpha multiplier applied ONLY to uncategorized (`-1`) vertices when a
    /// `--categories` file drives colors. Linear [0, 1]; `1.0` = no dimming. Lets the
    /// categorized islands pop on graphs where most vertices carry no category.
    pub uncategorized_alpha: f32,
    /// Edge alpha channel source — `Constant`, or `Weight` = times the per-edge
    /// `1/(w_a*w_b)` (hub↔hub edges recede). Defaults to `Weight`.
    pub edge_alpha: ScalarSource,
    /// Global edge-alpha multiplier from the UI slider. `0.0` = edges invisible.
    pub edge_alpha_scale: f32,
    /// Global edge-width multiplier from the UI slider.
    pub edge_width_scale: f32,
    /// Edge color source — theme (default) / vertex / file / length. Independent of the
    /// alpha channel above; alpha always rides the `edge_alpha_scale × weight` path.
    pub edge_color: EdgeColorSource,
    /// Stress-ratio clamp `[lo, hi]` for the `length` source's diverging colormap. `r = 1`
    /// (ideal length) is the neutral midpoint. Only consulted when `edge_length_auto` is
    /// off. Default `[0.25, 4.0]`.
    pub edge_length_range: [f32; 2],
    /// Derive the clamp window automatically from the stress distribution (q05/q95
    /// quantiles of `ln r`, recomputed with the ratios). Default on; a TOML
    /// `edge_length_range` switches to that fixed manual window.
    pub edge_length_auto: bool,
    /// How dimension > 2 embeddings project into the 2D view. Irrelevant at dimension 2.
    pub projection: ProjectionMode,
    /// Depth-shading strength [0, 1] — vertices fade by their out-of-projection-plane
    /// residual. `0` = off; only meaningful at dimension > 2.
    pub depth_fade: f32,
    /// Camera rotation in DEGREES (converted to radians before pushing to Camera). Kept
    /// in degrees for the UI (`DragValue` reads more naturally). Positive = CCW.
    pub rotation_deg: f32,
    /// Mirror the rendered image left↔right. Clip-space, so it composes with rotation
    /// as "rotate, then mirror the picture" (see [`Camera::set_flip`]).
    pub flip_x: bool,
    /// Mirror the rendered image top↔bottom.
    pub flip_y: bool,
}

/// Common wallpaper / display resolutions offered by the export panel. The `Custom`
/// variant defers to the width/height fields in [`ExportSettings`], which the UI edits
/// via `DragValue`. Adding a new preset = one enum variant + one match arm.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ResolutionPreset {
    Window,
    P1080,
    P1440,
    K4,
    K5,
    K8,
    Ultrawide,
    UltrawideK5,
    PhonePortrait,
    Custom,
}

impl ResolutionPreset {
    /// `None` for `Window` and `Custom` — the caller resolves those from surface size or
    /// the manual width/height fields. Every other variant returns its fixed dimensions.
    fn dims(self) -> Option<(u32, u32)> {
        Some(match self {
            ResolutionPreset::Window | ResolutionPreset::Custom => return None,
            ResolutionPreset::P1080 => (1920, 1080),
            ResolutionPreset::P1440 => (2560, 1440),
            ResolutionPreset::K4 => (3840, 2160),
            ResolutionPreset::K5 => (5120, 2880),
            ResolutionPreset::K8 => (7680, 4320),
            ResolutionPreset::Ultrawide => (3440, 1440),
            ResolutionPreset::UltrawideK5 => (5120, 2160),
            ResolutionPreset::PhonePortrait => (1170, 2532),
        })
    }

    fn label(self) -> &'static str {
        match self {
            ResolutionPreset::Window => "window",
            ResolutionPreset::P1080 => "1080p (1920×1080)",
            ResolutionPreset::P1440 => "1440p (2560×1440)",
            ResolutionPreset::K4 => "4K (3840×2160)",
            ResolutionPreset::K5 => "5K (5120×2880)",
            ResolutionPreset::K8 => "8K (7680×4320)",
            ResolutionPreset::Ultrawide => "ultrawide (3440×1440)",
            ResolutionPreset::UltrawideK5 => "ultrawide 5K (5120×2160)",
            ResolutionPreset::PhonePortrait => "phone portrait (1170×2532)",
            ResolutionPreset::Custom => "custom",
        }
    }
}

/// Persistent state of the export panel — resolution / manual dims / preset. MSAA is
/// always on for exports (4× — cheap, meaningful only for wallpaper renders); no reason
/// to expose a toggle.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ExportSettings {
    pub preset: ResolutionPreset,
    /// Manual width, used when `preset == Custom`. Kept around even when a fixed preset is
    /// selected so switching to Custom preserves the last edit.
    pub custom_width: u32,
    pub custom_height: u32,
}

impl ExportSettings {
    fn default_settings() -> Self {
        Self {
            preset: ResolutionPreset::K4,
            custom_width: 3840,
            custom_height: 2160,
        }
    }

    /// Resolve dimensions to (width, height), given the current window size for the
    /// `Window` preset. Falls back to `(1920, 1080)` if the window isn't sized yet — the
    /// user is asking for an export before the surface is ready, which shouldn't happen
    /// in practice but we don't want to crash.
    fn dimensions(&self, window_size: (u32, u32)) -> (u32, u32) {
        match self.preset {
            ResolutionPreset::Window => window_size,
            ResolutionPreset::Custom => (self.custom_width.max(1), self.custom_height.max(1)),
            other => other.dims().expect("preset has fixed dims"),
        }
    }
}

/// Editable + display state for the density UI section, marshaled in/out of the egui
/// closure (like the visual/export scratch). σ + freeze are edited; `anchors` is read-only.
#[derive(Copy, Clone, Debug)]
struct DensityUi {
    sigma: f32,
    freeze: bool,
    anchors: Option<DensityAnchors>,
}

impl VisualState {
    /// The UI defaults for a given vertex color source. `settings::resolve` starts from this and
    /// overlays any `[visual]` TOML fields, so these literals stay the single source of
    /// truth for the slider defaults.
    pub(crate) fn new(vertex_color: VertexColorSource) -> Self {
        Self {
            vertex_color,
            vertex_size: ScalarSource::Weight,
            vertex_size_scale: 1.0,
            vertex_alpha_scale: 1.0,
            uncategorized_alpha: 1.0,
            edge_alpha: ScalarSource::Weight,
            edge_alpha_scale: 0.35,
            edge_width_scale: 1.0,
            edge_color: EdgeColorSource::Theme,
            edge_length_range: [0.25, 4.0],
            edge_length_auto: true,
            projection: ProjectionMode::Pca,
            depth_fade: 0.0,
            rotation_deg: 0.0,
            flip_x: false,
            flip_y: false,
        }
    }

    /// Everything [`Renderer::set_channels`] needs. `edge_colors_available` downgrades a
    /// `file` edge-color selection (see [`Self::edge_color_source_value`]); `auto_range` is
    /// the last automatic `length` window.
    pub(crate) fn channels(
        &self,
        edge_colors_available: bool,
        auto_range: Option<(f32, f32)>,
    ) -> Channels {
        let (edge_len_log_min, edge_len_log_max) = self.edge_len_log_range_effective(auto_range);
        Channels {
            vertex_size_source: self.vertex_size.shader_value(),
            edge_alpha_source: self.edge_alpha.shader_value(),
            edge_color_source: self.edge_color_source_value(edge_colors_available),
            vertex_size_scale: self.vertex_size_scale,
            vertex_alpha_scale: self.vertex_alpha_scale,
            edge_alpha_scale: self.edge_alpha_scale,
            edge_width_scale: self.edge_width_scale,
            edge_len_log_min,
            edge_len_log_max,
        }
    }

    /// `edge_color_source` value for the shader uniform. `edge_colors_available` lets the
    /// caller downgrade a `file` selection to `theme` when no matching `--edge-colors` file
    /// is loaded (the buffer would be the white dummy fill), keeping the shader's file
    /// branch off real data only.
    pub(crate) fn edge_color_source_value(&self, edge_colors_available: bool) -> u32 {
        match self.edge_color {
            EdgeColorSource::File if !edge_colors_available => {
                EdgeColorSource::Theme.shader_value()
            }
            other => other.shader_value(),
        }
    }

    /// `(ln(lo), ln(hi))` for the `length` source's clamp window. Guards against
    /// non-positive / inverted ranges so the shader always sees `log_min < 0 < log_max`.
    pub(crate) fn edge_len_log_range(&self) -> (f32, f32) {
        let lo = self.edge_length_range[0].max(1e-6);
        let hi = self.edge_length_range[1].max(lo * 1.0001);
        (lo.ln().min(-1e-6), hi.ln().max(1e-6))
    }

    /// The window actually pushed to the shader: the auto-computed quantile window when
    /// `edge_length_auto` (falling back to the manual values until the first stress
    /// compute delivers one), else the manual range.
    pub(crate) fn edge_len_log_range_effective(&self, auto: Option<(f32, f32)>) -> (f32, f32) {
        if self.edge_length_auto {
            if let Some(window) = auto {
                return window;
            }
        }
        self.edge_len_log_range()
    }
}

/// Startup bundle for [`App::new`] — everything `main` resolves from CLI + TOML that the
/// interactive session needs beyond the producer/attrs/themes/sizing it already took.
pub struct AppInit {
    /// Whether the GUI is in debugger-like interactive mode with single-step controls.
    pub interactive: bool,
    /// Initial visual state (color mode + channel toggles + slider scales).
    pub visual: VisualState,
    /// `[visual].search` — pre-fills the search box; applied on the first topology
    /// publish through the existing re-apply path.
    pub search: Option<String>,
    /// `[camera]` section. `main` guarantees `Some` only when the producer replays an
    /// `.emb`; applied once on the first camera fit, then never again.
    pub camera: Option<CameraConfig>,
    /// `[export]` dims — seeds the export panel with a Custom preset.
    pub export_dims: Option<(u32, u32)>,
    /// Resolved `[text]` section (+ `--title`) — title string seeds the UI field.
    pub text: TextParams,
    /// Resolved `[legend]` section — rows drawn window + export, TOML-only for now.
    pub legend: Option<LegendParams>,
    /// Resolved `[density]` params — `Some` enables the heatmap render strategy.
    pub density: Option<DensityParams>,
    /// Title font, loaded (and validated) once by `main` — shared with headless.
    pub font: ab_glyph::FontArc,
    /// Input paths + embedder provenance for the TOML export button.
    pub session: SessionInfo,
}

pub struct App {
    /// Source of topology + positions. Boxed trait object so App is agnostic to whether
    /// the source is a live [`crate::embedder::EmbedderHandle`] or (future) a load-once
    /// `LoadedEmbedding`.
    embedder: Box<dyn TopologyProducer>,
    interactive: bool,
    sizing: SizingConfig,
    /// Raw d-dim positions, reused across frames — embedder writes into it on poll.
    /// NEVER read by the render/camera/pick stack (that's `view_positions`); kept for
    /// re-projection, depth values, and full-fidelity `.emb` saves.
    frame_positions: Vec<f32>,
    /// The 2D display positions — `frame_positions` through the projector (verbatim copy
    /// at dimension 2). Everything downstream (scene, camera, framing, picking) reads
    /// this; refreshed whenever `frame_positions` changes or the projection changes.
    view_positions: Vec<f32>,
    /// Per-vertex depth-shade scratch, recomputed on projection refits.
    depth_values: Vec<f32>,
    /// d-dim → 2D projection state. Rebuilt on every topology publish (dim + vertex
    /// count can change across layered-embedder layer transitions).
    projector: Projector,
    /// Per-vertex raw weights from the last topology publish. Kept on App so the "save
    /// embedding" path has them to hand — Topology's `weights` field otherwise gets
    /// consumed into Scene and lost.
    frame_weights: Vec<f32>,
    /// Edges + per-edge alphas from the last topology publish (moved out of the
    /// producer's Topology), retained for the `length` edge-color source's per-update
    /// stress recompute.
    frame_edges: Vec<[u32; 2]>,
    frame_edge_alphas: Vec<f32>,
    /// Per-topology cache of `(w_a·w_b)^(-1/dim)` for the stress loop; empty = not yet
    /// computed (filled lazily when the `length` source is first active).
    edge_inv_norms: Vec<f32>,
    /// Last auto-computed `ln(r)` colormap window (q05/q95 of the stress sample). `None`
    /// until the first compute; also shown in the UI and used to seed the manual range
    /// when the user unchecks auto.
    edge_len_auto_range: Option<(f32, f32)>,
    /// The UI flipped something that needs a stress recompute (source switched to
    /// `length`, auto toggled) while the renderer borrow was live — handled at the top
    /// of the next tick.
    edge_stress_dirty: bool,
    /// Per-edge stress-ratio scratch, uploaded via `Scene::write_edge_lengths`.
    edge_stress: Vec<f32>,
    camera: Option<Camera>,
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    /// Per-topology GPU buffers (positions, edges, colors, mask + bind group).
    /// Created on the first `Update::Topology`, replaced in-place on each
    /// subsequent layered-embedder layer transition via `Scene::reload`.
    scene: Option<Scene>,

    egui_ctx: egui::Context,
    /// Constructed in `resumed` once we have a Window.
    egui_winit_state: Option<egui_winit::State>,
    fps: FpsCounter,

    /// Latest cursor position in physical pixels. Populated on every CursorMoved; used both
    /// for drag deltas and as the anchor for zoom-at-cursor.
    cursor: Option<(f32, f32)>,
    /// Left mouse pressed AND drag started outside egui (i.e. was a scene drag, not a UI
    /// panel drag). Cleared on release regardless of pause state so we never get stuck.
    dragging: bool,

    // ---- Per-vertex attributes ----
    attributes: Attributes,
    /// Per-channel source selections. Mutated by egui inside `build_ui`; the tick loop
    /// reads it after the UI closure and pushes deltas into Renderer's uniforms.
    visual: VisualState,
    /// Loaded themes + which one is currently active. Mutated by the theme picker.
    themes: ThemeRegistry,

    // ---- Inspection state machine (running vs. paused vs. finished) ----
    /// Owns the picker, hover vertex, and the interactive-vs-live distinction.
    /// Synced from `EmbedderHandle::{is_paused, is_finished}` once per tick.
    /// See [`crate::inspection`] for the transition rules.
    inspection: InspectionMode,

    // ---- Highlight state ----
    /// Union of hover neighborhood + search matches, uploaded to the mask buffer.
    highlight: Highlight,
    /// Live-editable text field bound directly by egui. Applied to `highlight` after
    /// `SEARCH_DEBOUNCE` idle time so a burst of typing doesn't re-scan every keystroke.
    search_query: String,
    /// When `search_query` last changed. `None` between applies. The apply drains this.
    search_dirty_since: Option<Instant>,

    // ---- Export panel state ----
    /// Persisted export settings — resolution preset, custom dims, MSAA toggle. Mutated
    /// by egui inside the export panel; App reads it on the save trigger.
    export: ExportSettings,
    /// Status line shown under the save button — "saved to foo.png" or an error string.
    /// Set on save attempt; cleared on next mutation of `export`.
    export_status: Option<String>,
    /// Set by the 'S' keyboard shortcut in `window_event`; consumed by the next
    /// `tick_and_render` (which has the renderer borrow needed to actually run the
    /// export). Same trigger path as the "save png" button in the UI panel.
    export_pending: bool,
    /// Set by the 'E' keyboard shortcut or the "save embedding" button. Consumed by the
    /// next `tick_and_render`. Independent of `export_pending` because saving an
    /// embedding doesn't need the renderer at all — just `frame_positions` +
    /// `frame_weights`.
    save_embedding_pending: bool,

    // ---- TOML config session state ----
    /// Input paths + embedder provenance, captured at startup for the TOML export.
    session: SessionInfo,
    /// One-shot `[camera]` restore. `Some` only when replaying an `.emb`; taken (and
    /// applied over the auto-fit) on the first camera creation in `refit_camera`.
    initial_camera: Option<CameraConfig>,
    /// Name field next to the "save toml" button. Trimmed + sanitized on save; empty →
    /// timestamp fallback, matching the PNG naming.
    toml_name: String,

    // ---- Title text ----
    /// Styling knobs from `[text]` (size/anchor/color/font). The title *string* lives in
    /// `title_text` (editable in the UI); these are config-only.
    text_params: TextParams,
    /// Live-editable title, bound by egui like `search_query`. Empty = no title drawn.
    title_text: String,
    /// The loaded title font (bundled or `[text].font`). Loaded once at startup.
    title_font: ab_glyph::FontArc,
    /// Cached GPU quad for the current (title, px, color) triple, plus the bitmap's
    /// pixel dims for anchor math. Rebuilt when the key changes (edit, theme switch,
    /// zoom changing the preview scale).
    title_quad: Option<(TextQuad, u32, u32)>,
    /// The (text, rounded px, color) the cached quad was built for.
    title_quad_key: (String, u32, [u8; 4]),

    // ---- Legend ----
    /// Resolved `[legend]` rows + styling. Static over a session (TOML-only); swatch
    /// colors resolve against the ACTIVE theme at rasterize time.
    legend_params: Option<LegendParams>,
    /// Cached legend quad + bitmap dims, same lifecycle as `title_quad`.
    legend_quad: Option<(TextQuad, u32, u32)>,
    /// The (rounded px, text color, theme index) the cached legend quad was built for —
    /// entries can't change at runtime, so these three cover every rebuild trigger.
    legend_quad_key: (u32, [u8; 4], usize),

    // ---- Convergence metrics ----
    /// Sample history + plot rendering. Drained from the producer each tick; empty on
    /// `.emb` replay (the section renders nothing then).
    convergence: Convergence,

    // ---- Density (heatmap) render strategy ----
    /// Resolved `[density]` params — `Some` switches the frame to the heatmap pipeline.
    /// The UI's σ slider mutates `sigma`; the ramp resolves from the active theme when empty.
    /// The GPU-side [`DensityRenderer`] lives inside [`Renderer`] (lazily built) to keep
    /// borrows simple — App only holds the params + cached exposure state.
    density: Option<DensityParams>,
    /// Last computed exposure anchors (knee/white). Displayed in the UI so the user can copy
    /// frozen values into a config. `None` until the first readback.
    density_anchors: Option<DensityAnchors>,
    /// Freeze auto-exposure: hold the current anchors instead of recomputing on camera rest.
    density_freeze: bool,
    /// When the view last changed (camera/positions). Auto-exposure recomputes anchors once
    /// this has been idle for [`DENSITY_EXPOSURE_DEBOUNCE`]. `None` between recomputes.
    density_view_dirty_since: Option<Instant>,
}

impl App {
    pub fn new(
        embedder: Box<dyn TopologyProducer>,
        sizing: SizingConfig,
        attributes: Attributes,
        themes: ThemeRegistry,
        init: AppInit,
    ) -> Self {
        let export = match init.export_dims {
            // `[export]` in the TOML seeds the panel as a Custom preset so the first 'S'
            // press renders at exactly the configured size.
            Some((w, h)) => ExportSettings {
                preset: ResolutionPreset::Custom,
                custom_width: w,
                custom_height: h,
            },
            None => ExportSettings::default_settings(),
        };
        // Copy, read before `init.session` is moved into the struct below
        let embedder_params = init.session.embedder;
        Self {
            embedder,
            interactive: init.interactive,
            sizing,
            frame_positions: Vec::new(),
            view_positions: Vec::new(),
            depth_values: Vec::new(),
            projector: Projector::new(2, init.visual.projection),
            frame_weights: Vec::new(),
            frame_edges: Vec::new(),
            frame_edge_alphas: Vec::new(),
            edge_inv_norms: Vec::new(),
            edge_len_auto_range: None,
            edge_stress_dirty: false,
            edge_stress: Vec::new(),
            camera: None,
            window: None,
            renderer: None,
            scene: None,
            egui_ctx: egui::Context::default(),
            egui_winit_state: None,
            fps: FpsCounter::new(),
            cursor: None,
            dragging: false,
            attributes,
            visual: init.visual,
            themes,
            inspection: InspectionMode::new(),
            highlight: Highlight::new(),
            // Pre-filled query is applied by the first topology publish's re-apply path,
            // no debounce round-trip needed.
            search_query: init.search.unwrap_or_default(),
            search_dirty_since: None,
            export,
            export_status: None,
            export_pending: false,
            save_embedding_pending: false,
            session: init.session,
            initial_camera: init.camera,
            toml_name: String::new(),
            title_text: init.text.title.clone().unwrap_or_default(),
            text_params: init.text,
            title_font: init.font,
            title_quad: None,
            title_quad_key: (String::new(), 0, [0; 4]),
            legend_params: init.legend,
            legend_quad: None,
            legend_quad_key: (0, [0; 4], usize::MAX),
            convergence: Convergence::new(embedder_params),
            density: init.density,
            density_anchors: None,
            density_freeze: false,
            density_view_dirty_since: Some(Instant::now()),
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("wembed-viewer")
            .with_inner_size(winit::dpi::LogicalSize::new(900.0, 700.0));
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));

        let egui_state = egui_winit::State::new(
            self.egui_ctx.clone(),
            ViewportId::ROOT,
            &*window,
            Some(window.scale_factor() as f32),
            None, // system theme; egui picks
            None, // max_texture_side; use wgpu default
        );
        self.egui_winit_state = Some(egui_state);
        self.window = Some(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        // Give egui first crack at the event. We do NOT early-return on `consumed` because we
        // still need to handle a few things unconditionally: window resize / redraw / close,
        // cursor tracking (so zoom-at-cursor has a valid anchor even after the cursor crosses
        // the panel), and mouse-button *release* (so we don't get stuck mid-drag if the
        // release lands on the panel).
        let consumed = if let (Some(window), Some(state)) =
            (self.window.as_ref(), self.egui_winit_state.as_mut())
        {
            state.on_window_event(window, &event).consumed
        } else {
            false
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(r) = self.renderer.as_mut() {
                    r.resize(size.width, size.height);
                    self.push_view();
                    if let Some(w) = &self.window {
                        w.request_redraw();
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                self.tick_and_render(event_loop);
            }
            WindowEvent::CursorMoved { position, .. } => {
                let x = position.x as f32;
                let y = position.y as f32;
                // Pan only while the drag is genuinely on the scene. If egui starts using the
                // pointer mid-drag (e.g. a resize on the convergence window whose press slipped
                // past the `!consumed` gate below), `consumed` flips true and we stop panning —
                // otherwise resizing that window would drag the graph with it.
                if self.dragging && !consumed {
                    if let Some((px, py)) = self.cursor {
                        self.pan(x - px, y - py);
                    }
                }
                self.cursor = Some((x, y));
                // `inspection.hit_test` no-ops in Live, so we don't gate on interactivity
                // — just on whether egui claimed the cursor move (in which case the tooltip
                // should vanish, so we clear hover explicitly).
                if !consumed {
                    self.update_hover();
                } else {
                    self.inspection.clear_hover();
                }
            }
            WindowEvent::CursorLeft { .. } => {
                self.inspection.clear_hover();
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => match state {
                // Drag only in an interactive mode (Paused or Finished) so the camera isn't
                // fighting the auto-refit during a live run.
                ElementState::Pressed => {
                    self.dragging = !consumed && self.inspection.is_interactive();
                }
                // Always clear on release — no way to get stuck mid-drag if the release event
                // hits egui or fires while running.
                ElementState::Released => self.dragging = false,
            },
            WindowEvent::MouseWheel { delta, .. }
                if !consumed && self.inspection.is_interactive() =>
            {
                self.zoom(scroll_delta_y(delta));
            }
            WindowEvent::KeyboardInput {
                event: key_event, ..
            } => {
                if consumed || key_event.state != ElementState::Pressed {
                    return;
                }
                match &key_event.logical_key {
                    Key::Named(NamedKey::Space) => self.embedder.toggle_paused(),
                    // 'S' triggers an export at the currently-selected settings. Sets a
                    // flag consumed by the next tick_and_render — we can't do the export
                    // inline because it needs `&mut renderer` and we're deep in an event
                    // handler here.
                    Key::Character(c) if c.eq_ignore_ascii_case("s") => {
                        self.export_pending = true;
                    }
                    Key::Character(c) if c.eq_ignore_ascii_case("e") => {
                        self.save_embedding_pending = true;
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }
}

impl App {
    fn tick_and_render(&mut self, event_loop: &ActiveEventLoop) {
        self.fps.tick();

        // Deferred from the previous frame's UI (the renderer borrow was live there):
        // stress recompute for a freshly-activated `length` source / auto-range toggle.
        if self.edge_stress_dirty {
            self.edge_stress_dirty = false;
            self.update_edge_stress();
        }

        // Convergence samples accumulate independently of position/topology updates.
        self.embedder.drain_metrics(self.convergence.samples_mut());

        // ---- embedder poll → maybe update GPU positions + refit camera ----
        let mut positions_changed = false;
        match self.embedder.poll(&mut self.frame_positions) {
            Update::Nothing => {}
            Update::Positions => {
                positions_changed = true;
                // project BEFORE the camera refit — everything downstream reads the view
                self.projector
                    .update_view(&self.frame_positions, &mut self.view_positions);
                self.refit_camera();
                if let (Some(r), Some(scene)) = (self.renderer.as_ref(), self.scene.as_ref()) {
                    scene.write_positions(r.queue(), &self.view_positions);
                    if !self.projector.is_identity() {
                        self.projector
                            .depth_values(&self.frame_positions, &mut self.depth_values);
                        scene.write_depth(r.queue(), &self.depth_values);
                    }
                }
                self.update_edge_stress();
                self.push_view();
            }
            Update::Topology(topology) => {
                positions_changed = true;
                let Some(window) = self.window.clone() else {
                    return;
                };
                // fresh projector: dim + vertex count can change across layer transitions
                self.projector = Projector::new(topology.dim, self.visual.projection);
                self.projector
                    .update_view(&self.frame_positions, &mut self.view_positions);
                self.refit_camera();

                // Renderer is scene-independent, so we can lazily build it on the first
                // topology publish and keep the same instance across every subsequent
                // layered layer transition. Egui's font atlas survives, since the
                // egui_wgpu::Renderer lives inside Renderer.
                if self.renderer.is_none() {
                    let mut renderer =
                        pollster::block_on(Renderer::new(window)).expect("renderer init");
                    // Push the initial theme's colors so the first-ever frame matches it.
                    // The edge/size/alpha *channels* are pushed after the scene builds below
                    // (the effective edge-color source depends on the edge count).
                    renderer.set_clear_color(wgpu_clear_color(self.themes.current().background()));
                    renderer.set_theme_colors(self.themes.current());
                    self.renderer = Some(renderer);
                }
                let renderer = self.renderer.as_ref().expect("just created above");

                // Snapshot raw weights for the save-embedding path — Scene consumes the
                // derived size_multipliers below but we want the raw values on hand.
                self.frame_weights.clear();
                self.frame_weights.extend_from_slice(&topology.weights);

                let n = self.view_positions.len() / 2;
                // Preflight the buffer sizes against the granted limits — a graph past
                // the cap dies here with a readable line instead of a wgpu validation
                // panic inside Scene::new.
                if let Err(e) =
                    crate::scene::check_device_limits(renderer.device(), n, topology.edges.len())
                {
                    eprintln!("{e}");
                    event_loop.exit();
                    return;
                }
                let colors = self.attributes.colors_for(
                    self.visual.vertex_color,
                    n,
                    Some(self.themes.current()),
                    self.visual.uncategorized_alpha,
                );
                let vertex_color = self.visual.vertex_color.effective(colors.is_some());
                // Per-edge colors for the `file` source, applied only on an exact edge-count
                // match (a layered coarser layer falls back — see `edge_color_source_value`).
                // `edge_colors` borrows `self.attributes`, so snapshot the availability bool
                // now — it must outlive the borrow (the `&mut self` in `push_view` below).
                let edge_colors = self.attributes.edge_colors_for(topology.edges.len());
                let edge_colors_available = edge_colors.is_some();
                let topo = Topology {
                    positions: &self.view_positions,
                    edges: &topology.edges,
                    colors: colors.as_deref(),
                    size_multipliers: Some(&topology.size_multipliers),
                    edge_alphas: Some(&topology.edge_alphas),
                    edge_colors,
                };
                if let Some(scene) = self.scene.as_mut() {
                    // Layer transition: rebuild all scene buffers in place. Highlight's mask
                    // and Scene's mask reset in parallel (Scene::reload fills with bright,
                    // Highlight::set_topology below clears its tracking sets).
                    scene.reload(renderer.device(), renderer.scene_bind_group_layout(), &topo);
                } else {
                    self.scene = Some(Scene::new(
                        renderer.device(),
                        renderer.scene_bind_group_layout(),
                        &topo,
                    ));
                }
                self.push_view();
                // Push the full channel set now the edge count is known — the effective
                // edge-color source (file→theme downgrade) depends on it, and a layer
                // transition can flip edge-colors availability. set_channels no-ops when
                // unchanged, so this is free on the steady-state layer.
                if let Some(r) = self.renderer.as_mut() {
                    r.set_vertex_color_source(vertex_color.shader_value());
                    r.set_channels(
                        &self
                            .visual
                            .channels(edge_colors_available, self.edge_len_auto_range),
                    );
                    r.set_depth_fade(self.visual.depth_fade);
                }
                // depth values for the fresh scene (reload zero-fills the buffer)
                if !self.projector.is_identity() {
                    self.projector
                        .depth_values(&self.frame_positions, &mut self.depth_values);
                    if let (Some(r), Some(scene)) = (self.renderer.as_ref(), self.scene.as_ref()) {
                        scene.write_depth(r.queue(), &self.depth_values);
                    }
                }

                // Rebuild highlight state for the new topology. Vertex ids in `search_set`
                // and `hover_set` don't survive a layered-embedder layer change (N changes,
                // adjacency is different), so `set_topology` resets both. We then re-apply
                // the current query so a mid-embed search stays live across transitions.
                let labels = self.attributes.labels_for(n).map(<[String]>::to_vec);
                self.highlight
                    .set_topology(n, &topology.edges, labels.as_deref());
                if !self.search_query.is_empty() {
                    self.highlight.set_query(&self.search_query);
                }
                // Any hover from before the layer transition refers to old vertex ids —
                // sync will rebuild the picker over the new positions on the next tick.
                self.inspection.clear_hover();

                // Retain edges + alphas for the stress recompute (moves, no copies —
                // this is the last use of `topology`). Inv-norms and the auto window are
                // topology-derived, so both caches die with the old topology.
                self.edge_len_auto_range = None;
                self.frame_edges = topology.edges;
                self.frame_edge_alphas = topology.edge_alphas;
                self.edge_inv_norms.clear();
                self.update_edge_stress();
            }
        }

        // ---- reconcile inspection state ----
        // `sync` handles all Live↔Paused↔Finished transitions internally: building the
        // picker on entering inspection, dropping it on leaving, transferring the session
        // between Paused and Finished without rebuilding.
        if positions_changed {
            self.inspection.update_positions(&self.view_positions);
        }
        self.inspection.sync(
            self.embedder.is_paused(),
            self.embedder.is_finished(),
            &self.view_positions,
        );
        if positions_changed {
            self.update_hover();
        }

        // Read everything we need off `self` BEFORE grabbing the mutable borrows of
        // `egui_winit_state` and `renderer` — hover_tooltip() wants `&self` which the
        // borrow checker won't accept once the mut refs are live.
        let fps = self.fps.value();
        let iter = self.embedder.iteration();
        let progress = self.embedder.progress();
        let embedder_error = self.embedder.error();
        let available = Available {
            categories: self.attributes.has_categories(),
            vertex_colors: self.attributes.has_colors(),
            edge_colors: self.attributes.has_edge_colors(),
        };
        let paused = self.inspection.is_paused();
        let finished = self.inspection.is_finished();
        // Pass current N so `label_for` can bail on a layered coarser-layer mismatch —
        // vertex ids there index supernodes, not products, so the labels file is stale.
        let n_for_labels = self.view_positions.len() / 2;
        // Projection UI state — the section only shows at dimension > 2.
        let projection_dim = self.projector.dim();
        let explained_variance = self.projector.explained_variance();
        // Auto stress window in r-space for the UI readout.
        let edge_auto_range = self
            .edge_len_auto_range
            .map(|(lo, hi)| [lo.exp(), hi.exp()]);
        // Graph size for the status strip. Vertices come straight off the position scratch;
        // edges from the live Scene (0 before the first topology publish).
        let n_vertices = n_for_labels;
        let n_edges = self
            .scene
            .as_ref()
            .map(|s| s.num_edges() as usize)
            .unwrap_or(0);
        let hover_text = self
            .inspection
            .hover()
            .map(|id| self.attributes.label_for(id, n_for_labels));
        let cursor = self.cursor;
        let visual_before = self.visual;
        let export_before = self.export;
        let export_status_msg = self.export_status.clone();
        let theme_index_before = self.themes.current_index();
        // Snapshot theme names for the picker — cheap Vec<String> alloc, avoids passing
        // &ThemeRegistry (which conflicts with `&mut self.themes` in the outer borrow).
        let theme_options: Vec<(usize, String)> = self
            .themes
            .names()
            .map(|(i, n)| (i, n.to_string()))
            .collect();

        // Nothing to render yet (embedder still loading the graph).
        let (Some(window), Some(state), Some(r)) = (
            self.window.as_ref(),
            self.egui_winit_state.as_mut(),
            self.renderer.as_mut(),
        ) else {
            return;
        };

        // Snapshot everything the closure needs so egui's closure captures a small set of
        // scalars + one &mut String, not `&mut self`. `search_query` is mutated in place
        // via TextEdit — we compare against `prev_search_query` after the run to detect
        // typing and reset the debounce clock.
        let prev_search_query = self.search_query.clone();
        let search_matches = self.highlight.search_match_count();
        let search_has_labels = self.highlight.has_labels();

        // ---- build egui frame ----
        let raw_input = state.take_egui_input(window);
        // Captured by the closures below and consulted after `run`. Egui can't call back
        // into `self` directly (borrow-checker), so button/radio state rides through scalars.
        let mut toggle_pause = false;
        let mut step_once_requested = false;
        let mut new_visual = visual_before;
        let mut new_export = export_before;
        let mut export_requested = false;
        let mut save_embedding_requested = false;
        let mut autorotate_requested = false;
        let mut reset_view_requested = false;
        let mut save_toml_requested = false;
        let mut new_theme_index = theme_index_before;
        let search_query = &mut self.search_query;
        let toml_name = &mut self.toml_name;
        let title_text = &mut self.title_text;
        let convergence = &self.convergence;
        // Density UI scratch: the section only draws when density mode is active. σ + freeze
        // are edited in place; the applied-anchor readouts are display-only.
        let mut density_ui = self.density.as_ref().map(|p| DensityUi {
            sigma: p.sigma,
            freeze: self.density_freeze,
            anchors: self.density_anchors,
        });
        // Where the export target's viewport lands on the current preview canvas, in
        // *physical* pixels. `None` before the first topology publish (no camera yet) or
        // when preview and export exactly match (nothing informative to draw). build_ui
        // converts to logical points via `ctx.pixels_per_point()`.
        let export_preview_dims = self.export.dimensions(r.size());
        let export_frame_rect = self.camera.as_ref().and_then(|cam| {
            if r.size() == export_preview_dims {
                return None;
            }
            export_frame_pixel_rect(cam, r.size(), export_preview_dims)
        });
        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            build_ui(
                ctx,
                fps,
                iter,
                progress,
                embedder_error.as_deref(),
                n_vertices,
                n_edges,
                self.interactive,
                paused,
                finished,
                &mut toggle_pause,
                &mut step_once_requested,
                &mut new_visual,
                &mut new_export,
                &mut export_requested,
                &mut save_embedding_requested,
                &mut autorotate_requested,
                &mut reset_view_requested,
                &mut save_toml_requested,
                export_status_msg.as_deref(),
                &theme_options,
                &mut new_theme_index,
                hover_text.as_deref(),
                cursor,
                search_query,
                toml_name,
                title_text,
                search_matches,
                search_has_labels,
                export_frame_rect,
                convergence,
                projection_dim,
                explained_variance,
                edge_auto_range,
                density_ui.as_mut(),
                available,
            );
        });
        // Apply density UI edits: σ change re-arms exposure (the blurred field shifts);
        // freeze toggle just latches. Both are cheap Option writes.
        if let (Some(ui_state), Some(params)) = (density_ui, self.density.as_mut()) {
            if ui_state.sigma != params.sigma {
                params.sigma = ui_state.sigma.max(0.0);
                self.density_view_dirty_since = Some(Instant::now());
            }
            self.density_freeze = ui_state.freeze;
        }
        // Unchecking auto range seeds the manual values from the last computed window so
        // the picture doesn't jump at the moment of the switch.
        if !new_visual.edge_length_auto && visual_before.edge_length_auto {
            if let Some((lo, hi)) = self.edge_len_auto_range {
                new_visual.edge_length_range = [lo.exp(), hi.exp()];
            }
        }
        if toggle_pause {
            self.embedder.toggle_paused();
        }
        if step_once_requested {
            self.embedder.step_once();
        }
        if new_visual != visual_before {
            self.visual = new_visual;
            // Push each channel to Renderer. Setters early-return on no-op so calling
            // both per state change is cheap.
            // Effective edge-color source needs the current edge count for the file→theme
            // downgrade rule (0 before the first topology publish → file downgrades, fine).
            let n_edges = self
                .scene
                .as_ref()
                .map(|s| s.num_edges() as usize)
                .unwrap_or(0);
            let edge_colors_available = self.attributes.edge_colors_for(n_edges).is_some();
            r.set_channels(&new_visual.channels(edge_colors_available, self.edge_len_auto_range));
            r.set_depth_fade(new_visual.depth_fade);
            // Switching TO the length source (or flipping its auto-range) needs an
            // immediate stress recompute — positions may be frozen (paused), so no
            // positions update will trigger one. Deferred to the top of the next tick
            // because update_edge_stress would re-borrow self while `r` is live.
            if new_visual.edge_color == EdgeColorSource::Length
                && (visual_before.edge_color != EdgeColorSource::Length
                    || new_visual.edge_length_auto != visual_before.edge_length_auto)
            {
                self.edge_stress_dirty = true;
            }
            if new_visual.vertex_color != visual_before.vertex_color
                || new_visual.uncategorized_alpha != visual_before.uncategorized_alpha
            {
                if let Some(scene) = self.scene.as_ref() {
                    push_vertex_colors(
                        r,
                        scene,
                        &self.attributes,
                        self.themes.current(),
                        &new_visual,
                        self.view_positions.len() / 2,
                    );
                }
            }
            // Rotation and flips are camera concerns, not shader uniforms. Push to camera
            // and re-upload view_proj so the change takes effect this frame. Inlined
            // push_view logic — self.push_view() would re-borrow self while `r` is live.
            if new_visual.rotation_deg != visual_before.rotation_deg
                || new_visual.flip_x != visual_before.flip_x
                || new_visual.flip_y != visual_before.flip_y
            {
                if let Some(cam) = self.camera.as_mut() {
                    cam.set_rotation(new_visual.rotation_deg.to_radians());
                    cam.set_flip(new_visual.flip_x, new_visual.flip_y);
                    let (w, h) = r.size();
                    let n = self.view_positions.len() / 2;
                    r.set_view(&View::new(cam, n, w, h, &self.sizing));
                }
            }
        }
        // Projection switch: re-project the whole cloud, refresh depth, and refit the
        // camera — the display coordinates changed wholesale, so the old viewport is
        // meaningless (same rationale as reset view; rotation/flip are framing choices
        // and survive).
        if new_visual.projection != visual_before.projection && !self.projector.is_identity() {
            self.projector.set_mode(new_visual.projection);
            self.projector
                .update_view(&self.frame_positions, &mut self.view_positions);
            self.projector
                .depth_values(&self.frame_positions, &mut self.depth_values);
            if let Some(scene) = self.scene.as_ref() {
                scene.write_positions(r.queue(), &self.view_positions);
                scene.write_depth(r.queue(), &self.depth_values);
            }
            let mut cam = Camera::fit_to_quantile_flat(&self.view_positions, Q_LO, Q_HI, Q_BUFFER);
            cam.set_rotation(self.visual.rotation_deg.to_radians());
            cam.set_flip(self.visual.flip_x, self.visual.flip_y);
            let (w, h) = r.size();
            let n = self.view_positions.len() / 2;
            r.set_view(&View::new(&cam, n, w, h, &self.sizing));
            self.camera = Some(cam);
        }
        if new_export != export_before {
            self.export = new_export;
            // User changed a knob → clear stale "saved X" status so it doesn't confuse.
            self.export_status = None;
        }
        if new_theme_index != theme_index_before {
            self.themes.set_current(new_theme_index);
            self.attributes.invalidate_theme_cache();
            let theme = self.themes.current();
            r.set_clear_color(wgpu_clear_color(theme.background()));
            r.set_theme_colors(theme);
            if let Some(scene) = self.scene.as_ref() {
                push_vertex_colors(
                    r,
                    scene,
                    &self.attributes,
                    theme,
                    &new_visual,
                    self.view_positions.len() / 2,
                );
            }
        }
        // Autorotate: brute-force the best rotation for the current export target aspect,
        // set visual.rotation_deg, and let the visual-delta block above pick it up next
        // tick. We do it BEFORE the export block so a save-and-autorotate sequence
        // captures the new rotation.
        if autorotate_requested && !self.view_positions.is_empty() {
            let window_size = r.size();
            let (w, h) = self.export.dimensions(window_size);
            let target_aspect = w as f32 / h.max(1) as f32;
            let deg = best_rotation_deg_for_aspect(&self.view_positions, target_aspect);
            self.visual.rotation_deg = deg;
            if let Some(cam) = self.camera.as_mut() {
                cam.set_rotation(deg.to_radians());
                let (ww, wh) = r.size();
                let n = self.view_positions.len() / 2;
                r.set_view(&View::new(cam, n, ww, wh, &self.sizing));
            }
        }
        // Reset view: re-fit the camera to the current position cloud on demand — the
        // escape hatch after panning/zooming way off while paused (auto-refit no-ops in
        // interactive modes). Preserves the current rotation: that's a framing choice,
        // not a navigation accident.
        if reset_view_requested && !self.view_positions.is_empty() {
            let mut cam = Camera::fit_to_quantile_flat(&self.view_positions, Q_LO, Q_HI, Q_BUFFER);
            cam.set_rotation(self.visual.rotation_deg.to_radians());
            cam.set_flip(self.visual.flip_x, self.visual.flip_y);
            let (w, h) = r.size();
            let n = self.view_positions.len() / 2;
            r.set_view(&View::new(&cam, n, w, h, &self.sizing));
            self.camera = Some(cam);
        }
        // Save embedding — button click OR keyboard 'E'. Purely CPU-side (no renderer
        // access needed), so it can borrow `&self` fields disjointly from `r`.
        if save_embedding_requested || self.save_embedding_pending {
            self.save_embedding_pending = false;
            // Raw d-dim positions — the save is full-fidelity; projection is display-only.
            self.export_status = Some(run_save_embedding(
                &self.frame_positions,
                self.projector.dim(),
                &self.frame_weights,
            ));
        }
        // Save TOML — snapshot the whole session as a curated, named config bundle.
        // CPU-side like the embedding save; `r` is only consulted for the window size
        // (to resolve the export panel's `Window` preset into concrete dims).
        if save_toml_requested {
            self.export_status = Some(match self.camera.as_ref() {
                Some(cam) => run_save_toml(
                    &self.toml_name,
                    cam,
                    &self.visual,
                    &self.search_query,
                    &self.title_text,
                    &self.text_params,
                    self.legend_params.as_ref(),
                    &self.themes.current().name,
                    &self.sizing,
                    &self.session,
                    self.export.dimensions(r.size()),
                    &self.frame_positions,
                    self.projector.dim(),
                    &self.frame_weights,
                    self.density.as_ref(),
                    self.density_anchors,
                ),
                None => "save toml: nothing loaded yet".to_string(),
            });
        }
        // Button click sets `export_requested`; keyboard 'S' set `self.export_pending`.
        // Either triggers the same export path.
        if export_requested || self.export_pending {
            self.export_pending = false;
            // Free function so we don't have to release `r` (which would break the render
            // call below). `self.camera` and `self.scene` are field-disjoint from
            // `self.renderer` so the borrow checker lets us pass all three.
            let status = match (self.camera.as_ref(), self.scene.as_ref()) {
                (Some(cam), Some(scene)) => run_export(
                    r,
                    scene,
                    cam,
                    &self.sizing,
                    self.view_positions.len() / 2,
                    &self.export,
                    r.size(),
                    wgpu_clear_color(self.themes.current().background()),
                    &self.title_text,
                    &self.text_params,
                    self.legend_params.as_ref(),
                    &self.title_font,
                    self.themes.current(),
                ),
                _ => "export: scene not ready yet".to_string(),
            };
            self.export_status = Some(status);
        }

        // ---- highlight state: debounced search + hover sync + mask upload ----
        if self.search_query != prev_search_query {
            self.search_dirty_since = Some(Instant::now());
        }
        if let Some(t) = self.search_dirty_since {
            if t.elapsed() >= SEARCH_DEBOUNCE {
                self.highlight.set_query(&self.search_query);
                self.search_dirty_since = None;
            }
        }
        // Hover assignment is idempotent inside Highlight, so calling every frame is fine —
        // when hover_vertex hasn't changed the call is a single Option comparison.
        self.highlight.set_hover(self.inspection.hover());
        if let Some(mask) = self.highlight.drain_dirty_mask() {
            if let Some(scene) = self.scene.as_ref() {
                scene.write_mask(r.queue(), mask);
            }
        }

        state.handle_platform_output(window, full_output.platform_output);
        let pixels_per_point = full_output.pixels_per_point;
        let paint_jobs = self
            .egui_ctx
            .tessellate(full_output.shapes, pixels_per_point);

        // ---- render (scene + egui) ----
        let ui = UiFrame {
            paint_jobs: &paint_jobs,
            textures_delta: &full_output.textures_delta,
            pixels_per_point,
        };

        // ---- title quad upkeep ----
        // The title anchors to the export-frame overlay when active (so the preview shows
        // where it lands in the saved image), else to the window. Preview px follows the
        // anchor rect's height against the 1080p reference — the same fraction of the
        // frame the export will use. Re-rasterize only when (text, rounded px, color)
        // changes; zoom moves the export frame, so the key naturally tracks it.
        let title_ndc = {
            let (w, h) = r.size();
            let rect = export_frame_rect
                .map(|rr| (rr.min.x, rr.min.y, rr.width(), rr.height()))
                .unwrap_or((0.0, 0.0, w as f32, h as f32));
            let px = self.text_params.size * (rect.3 / 1080.0);
            let title = self.title_text.trim();
            if title.is_empty() || px < 1.0 {
                self.title_quad = None;
                self.title_quad_key = (String::new(), 0, [0; 4]);
                None
            } else {
                let color = self
                    .text_params
                    .color
                    .unwrap_or_else(|| self.themes.current().foreground());
                let key = (title.to_string(), px.round().max(1.0) as u32, color);
                if self.title_quad.is_none() || self.title_quad_key != key {
                    self.title_quad =
                        crate::text::rasterize(&self.title_font, title, key.1 as f32, color)
                            .map(|bmp| (r.make_text_quad(&bmp), bmp.width, bmp.height));
                    self.title_quad_key = key;
                }
                self.title_quad.as_ref().map(|(_, bw, bh)| {
                    let k = px / self.text_params.size;
                    let mx = self.text_params.margin_x.map_or(0.5 * px, |m| m * k);
                    let my = self.text_params.margin_y.map_or(0.5 * px, |m| m * k);
                    let (x, y) = crate::text::anchor_position(
                        self.text_params.anchor,
                        rect,
                        *bw as f32,
                        *bh as f32,
                        mx,
                        my,
                    );
                    crate::text::pixel_rect_to_ndc(x, y, *bw as f32, *bh as f32, w as f32, h as f32)
                })
            }
        };
        // ---- legend quad upkeep ----
        // Same anchoring + scaling rules as the title. Entries are session-static, so
        // the cache key is only (px, text color, theme index) — a theme switch recolors
        // both the swatches and the default text color.
        let legend_ndc = {
            let (w, h) = r.size();
            let rect = export_frame_rect
                .map(|rr| (rr.min.x, rr.min.y, rr.width(), rr.height()))
                .unwrap_or((0.0, 0.0, w as f32, h as f32));
            match self.legend_params.as_ref() {
                None => None,
                Some(params) => {
                    let px = params.size * (rect.3 / 1080.0);
                    if px < 1.0 {
                        self.legend_quad = None;
                        None
                    } else {
                        let theme = self.themes.current();
                        let color = params.color.unwrap_or_else(|| theme.foreground());
                        let key = (
                            px.round().max(1.0) as u32,
                            color,
                            self.themes.current_index(),
                        );
                        if self.legend_quad.is_none() || self.legend_quad_key != key {
                            let rows = params.rows(theme);
                            self.legend_quad = crate::text::rasterize_legend(
                                &self.title_font,
                                &rows,
                                key.0 as f32,
                                color,
                            )
                            .map(|bmp| (r.make_text_quad(&bmp), bmp.width, bmp.height));
                            self.legend_quad_key = key;
                        }
                        self.legend_quad.as_ref().map(|(_, bw, bh)| {
                            let k = px / params.size;
                            let mx = params.margin_x.map_or(0.5 * px, |m| m * k);
                            let my = params.margin_y.map_or(0.5 * px, |m| m * k);
                            let (x, y) = crate::text::anchor_position(
                                params.anchor,
                                rect,
                                *bw as f32,
                                *bh as f32,
                                mx,
                                my,
                            );
                            crate::text::pixel_rect_to_ndc(
                                x, y, *bw as f32, *bh as f32, w as f32, h as f32,
                            )
                        })
                    }
                }
            }
        };

        let mut text_draws: Vec<(&TextQuad, [f32; 4])> = Vec::new();
        if let (Some(ndc), Some((q, _, _))) = (title_ndc, self.title_quad.as_ref()) {
            text_draws.push((q, ndc));
        }
        if let (Some(ndc), Some((q, _, _))) = (legend_ndc, self.legend_quad.as_ref()) {
            text_draws.push((q, ndc));
        }

        // Scene is required to render; if it's not up yet (unusual — would mean we passed
        // the outer `Some(r)` guard without a topology publish), skip this frame silently.
        let Some(scene) = self.scene.as_ref() else {
            return;
        };
        // Density mode: auto-exposure recomputes anchors once the view rests (debounced —
        // readbacks stall the pipe and tonal pumping while panning is unwanted). Frozen
        // holds the current anchors. Then render the heatmap frame instead of the scene.
        let render_result = if let Some(params) = self.density.as_ref() {
            if !self.density_freeze {
                let due = self
                    .density_view_dirty_since
                    .map(|t| t.elapsed() >= DENSITY_EXPOSURE_DEBOUNCE)
                    .unwrap_or(false);
                if due || self.density_anchors.is_none() {
                    if let Some(a) = r.recompute_density_anchors(scene, params) {
                        self.density_anchors = Some(a);
                    }
                    self.density_view_dirty_since = None;
                }
            }
            // Fallback anchors until the first readback lands (a bland but valid first frame).
            let anchors = self.density_anchors.unwrap_or(DensityAnchors {
                knee: 0.0,
                white: 1.0,
            });
            let theme = self.themes.current();
            let ramp = crate::density::effective_ramp(
                params,
                theme.background(),
                crate::density::pick_ramp_accent(&theme.accents()),
                theme.foreground(),
            );
            r.render_density(scene, &ui, &text_draws, params, anchors, &ramp)
        } else {
            r.render(scene, &ui, &text_draws)
        };
        match render_result {
            Ok(()) => {}
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                let (w, h) = r.size();
                r.resize(w, h);
                self.push_view();
            }
            Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
            Err(e) => eprintln!("surface error: {e:?}"),
        }
    }

    /// Recompute + upload the per-edge stress ratios for the `length` edge-color source
    /// from the RAW d-dim positions (projection-independent), then refresh the auto
    /// colormap window from their distribution. No-op unless the source is active and a
    /// scene exists. Called per positions update, on topology publish, and via the
    /// `edge_stress_dirty` deferral when the UI activates the source while paused.
    fn update_edge_stress(&mut self) {
        if self.visual.edge_color != EdgeColorSource::Length || self.frame_edges.is_empty() {
            return;
        }
        if self.renderer.is_none() || self.scene.is_none() {
            return;
        }
        let dim = self.projector.dim();
        if self.edge_inv_norms.is_empty() {
            self.edge_inv_norms = crate::embedder::edge_inv_norms(&self.frame_edge_alphas, dim);
        }
        crate::embedder::edge_length_ratios(
            &self.frame_positions,
            dim,
            &self.frame_edges,
            &self.edge_inv_norms,
            &mut self.edge_stress,
        );
        if self.visual.edge_length_auto {
            self.edge_len_auto_range = crate::embedder::edge_length_log_range(&self.edge_stress);
        }
        let edge_colors_available = self
            .attributes
            .edge_colors_for(self.frame_edges.len())
            .is_some();
        let r = self.renderer.as_mut().expect("checked above");
        r.set_channels(
            &self
                .visual
                .channels(edge_colors_available, self.edge_len_auto_range),
        );
        let scene = self.scene.as_ref().expect("checked above");
        scene.write_edge_lengths(r.queue(), &self.edge_stress);
    }

    /// Auto-fit the camera to the current position cloud. No-op once we already have a
    /// camera AND we're in an interactive inspection mode (Paused or Finished) — so the
    /// user's pan/zoom sticks. The "already have a camera" caveat ensures the initial
    /// topology publish still lands even if the embedder starts out finished
    /// (e.g. `max_iterations=0`).
    ///
    /// The current `visual.rotation_deg` is re-applied to every fresh fit so a rotation
    /// set during a live run survives the per-frame refits instead of snapping back to 0.
    fn refit_camera(&mut self) {
        if self.camera.is_some() && self.inspection.is_interactive() {
            return;
        }
        let mut cam = Camera::fit_to_quantile_flat(&self.view_positions, Q_LO, Q_HI, Q_BUFFER);
        // One-shot `[camera]` restore from the TOML — present only when replaying an
        // `.emb` (main strips it otherwise). Overrides the fit and seeds the rotation
        // UI state so the DragValue shows the restored angle.
        if let Some(cfg) = self.initial_camera.take() {
            if let (Some(center), Some(half)) = (cfg.center, cfg.half) {
                cam = cam.with_view(center, half);
            }
            if let Some(deg) = cfg.rotation_deg {
                self.visual.rotation_deg = deg;
            }
        }
        cam.set_rotation(self.visual.rotation_deg.to_radians());
        cam.set_flip(self.visual.flip_x, self.visual.flip_y);
        self.camera = Some(cam);
    }

    fn pan(&mut self, dx: f32, dy: f32) {
        let (Some(cam), Some(r)) = (self.camera.as_mut(), self.renderer.as_ref()) else {
            return;
        };
        let (w, h) = r.size();
        cam.pan_pixels(dx, dy, w, h);
        self.push_view();
    }

    fn zoom(&mut self, wheel_dy: f32) {
        let (Some(cam), Some(r), Some((mx, my))) =
            (self.camera.as_mut(), self.renderer.as_ref(), self.cursor)
        else {
            return;
        };
        let (w, h) = r.size();
        // Positive dy (scroll up / away from user) → zoom in → factor < 1. Exponential
        // mapping keeps zoom smooth across the range.
        let factor = (-wheel_dy * ZOOM_PER_LINE).exp();
        cam.zoom_at_pixel(factor, mx, my, w, h);
        self.push_view();
    }

    /// Push view_proj + hybrid-clamped point radius to the renderer. No-op if either the
    /// camera or the renderer isn't set up yet.
    fn push_view(&mut self) {
        let Some(cam) = self.camera.as_ref() else {
            return;
        };
        let n = self.view_positions.len() / 2;
        let sizing = &self.sizing;
        // The view moved — density auto-exposure re-arms its debounce (recomputes anchors
        // once the view rests). Cheap Option write; the density path gates on it.
        if self.density.is_some() && self.density_view_dirty_since.is_none() {
            self.density_view_dirty_since = Some(Instant::now());
        }
        let Some(r) = self.renderer.as_mut() else {
            return;
        };
        let (w, h) = r.size();
        r.set_view(&View::new(cam, n, w, h, sizing));
    }

    /// Delegate a cursor position to the inspection state's picker. `radius` is recomputed
    /// from live camera + sizing state — cheaper than caching and can't go stale.
    /// [`InspectionMode::hit_test`] no-ops in Live, so callers don't need to gate.
    fn update_hover(&mut self) {
        let (Some(cam), Some(r), Some((mx, my))) =
            (self.camera.as_ref(), self.renderer.as_ref(), self.cursor)
        else {
            self.inspection.clear_hover();
            return;
        };
        let (w, h) = r.size();
        let world = cam.pixel_to_world(mx, my, w, h);
        let n = self.view_positions.len() / 2;
        let radius = point_radius(
            cam,
            n,
            w,
            h,
            &scaled_sizing(&self.sizing, h, REFERENCE_HEIGHT),
        );
        self.inspection.hit_test(world, radius);
    }
}

/// Save the current positions + weights to a timestamped `.emb` file in [`EXPORT_DIR`].
/// Returns a status string suitable for the UI (same shape as the PNG-export status).
fn run_save_embedding(positions: &[f32], dim: usize, weights: &[f32]) -> String {
    if positions.is_empty() || weights.is_empty() {
        return "save embedding: no topology yet (still loading?)".to_string();
    }
    let path = timestamped_export_path("emb");
    match crate::loaded_embedding::save_embedding_file(&path, positions, dim, weights) {
        Ok(()) => format!("saved {} ({} vertices)", path.display(), weights.len()),
        Err(e) => format!("save embedding failed: {e}"),
    }
}

/// The exports directory, created on demand. Falls back to CWD (`.`) when creation fails
/// (read-only checkout) — the save still lands somewhere and the status line shows the
/// actual path.
fn ensure_export_dir() -> PathBuf {
    let dir = PathBuf::from(EXPORT_DIR);
    match std::fs::create_dir_all(&dir) {
        Ok(()) => dir,
        Err(_) => PathBuf::from("."),
    }
}

/// `wembed_<unix_seconds>` — the fallback filename stem. Not human-readable but avoids a
/// `chrono` dependency and always sorts chronologically.
fn timestamp_stem() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("wembed_{secs}")
}

/// Timestamped path in [`EXPORT_DIR`] for interactive exports:
/// `exports/wembed_<unix_seconds>.<ext>`.
fn timestamped_export_path(extension: &str) -> PathBuf {
    ensure_export_dir().join(format!("{}.{extension}", timestamp_stem()))
}

/// Filename stem for the TOML export: the user's name field, trimmed, with path
/// separators and whitespace flattened to `-` (a name is a name, not a directory
/// traversal). Empty after trimming → timestamp fallback, matching the PNG naming.
fn sanitize_toml_stem(name: &str) -> String {
    let cleaned: String = name
        .trim()
        .chars()
        .map(|c| {
            if matches!(c, '/' | '\\') || c.is_whitespace() {
                '-'
            } else {
                c
            }
        })
        .collect();
    if cleaned.is_empty() {
        timestamp_stem()
    } else {
        cleaned
    }
}

/// `#rrggbb` (or `#rrggbbaa` when alpha < 255) for TOML round-trips.
fn hex_color(c: [u8; 4]) -> String {
    if c[3] == 0xff {
        format!("#{:02x}{:02x}{:02x}", c[0], c[1], c[2])
    } else {
        format!("#{:02x}{:02x}{:02x}{:02x}", c[0], c[1], c[2], c[3])
    }
}

/// Best-effort absolute path. Exported TOMLs must work from any CWD, so CWD-relative
/// input paths get anchored; if the CWD is unavailable the original path is kept (the
/// export still round-trips on the machine it was made on). Manual CWD-join instead of
/// `std::path::absolute` — that needs Rust 1.79, above the workspace's 1.75 MSRV.
fn absolutize(p: &std::path::Path) -> PathBuf {
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(p))
            .unwrap_or_else(|_| p.to_path_buf())
    }
}

/// Serialize the current session into a curated `FileConfig` bundle in [`EXPORT_DIR`]:
/// `<stem>.toml`, plus `<stem>.emb` when the session runs a live embedder (a TOML can
/// only reproduce a picture whose positions exist on disk). Returns a UI status string.
///
/// Path policy: inputs (graph, attrs, themes dir, pre-existing `.emb`) are absolutized so
/// the TOML loads from anywhere; a freshly bundled `.emb` is referenced by bare filename
/// instead — relative paths resolve against the TOML's own directory, so the pair can be
/// moved (or committed) together.
#[allow(clippy::too_many_arguments)]
fn run_save_toml(
    name: &str,
    cam: &Camera,
    visual: &VisualState,
    search: &str,
    title: &str,
    text_params: &TextParams,
    legend: Option<&LegendParams>,
    theme_name: &str,
    sizing: &SizingConfig,
    session: &SessionInfo,
    export_dims: (u32, u32),
    positions: &[f32],
    dim: usize,
    weights: &[f32],
    density: Option<&DensityParams>,
    density_anchors: Option<DensityAnchors>,
) -> String {
    use crate::config;

    let stem = sanitize_toml_stem(name);
    let dir = ensure_export_dir();
    let toml_path = dir.join(format!("{stem}.toml"));

    let embedding = match &session.input_embedding {
        Some(p) => absolutize(p),
        None => {
            if positions.is_empty() || weights.is_empty() {
                return "save toml: no topology yet (still loading?)".to_string();
            }
            let emb_name = format!("{stem}.emb");
            if let Err(e) = crate::loaded_embedding::save_embedding_file(
                &dir.join(&emb_name),
                positions,
                dim,
                weights,
            ) {
                return format!("save toml: bundling .emb failed: {e}");
            }
            PathBuf::from(emb_name)
        }
    };

    let has_attrs = session.colors.is_some()
        || session.labels.is_some()
        || session.categories.is_some()
        || session.edge_colors.is_some();
    let cfg = config::FileConfig {
        graph: Some(absolutize(&session.graph_path)),
        input_embedding: Some(embedding),
        output_embedding: None,
        // Provenance: with `input_embedding` set these are ignored on load, but they
        // record how the bundled positions were produced.
        embedder: Some(config::EmbedderConfig {
            iterations: Some(session.embedder.iterations),
            seed: Some(session.embedder.seed),
            dimension: Some(session.embedder.dimension),
            dimension_hint: Some(session.embedder.dimension_hint),
            learning_rate: Some(session.embedder.learning_rate),
            lr_cooling_factor: Some(session.embedder.lr_cooling_factor),
            layered: Some(session.embedder.layered),
            centre: Some(session.embedder.centre),
            dyn_buffer: Some(session.embedder.dyn_buffer),
            lr_schedule: Some(session.embedder.lr_schedule),
            warmup_steps: Some(session.embedder.warmup_steps),
            lr_decay_factor: Some(session.embedder.lr_decay_factor),
            lr_decay_threshold: Some(session.embedder.lr_decay_threshold),
            lr_adapt_patience: Some(session.embedder.lr_adapt_patience),
            lr_growth_factor: Some(session.embedder.lr_growth_factor),
            lr_growth_threshold: Some(session.embedder.lr_growth_threshold),
            stop_criterion: Some(session.embedder.stop_criterion),
            stop_displacement_tol: Some(session.embedder.stop_displacement_tol),
            stop_displacement_patience: Some(session.embedder.stop_displacement_patience),
            stop_loss_tol: Some(session.embedder.stop_loss_tol),
            stop_loss_patience: Some(session.embedder.stop_loss_patience),
        }),
        attrs: has_attrs.then(|| config::AttrsConfig {
            colors: session.colors.as_deref().map(absolutize),
            labels: session.labels.as_deref().map(absolutize),
            categories: session.categories.as_deref().map(absolutize),
            edge_colors: session.edge_colors.as_deref().map(absolutize),
        }),
        theme: Some(config::ThemeConfig {
            dir: Some(absolutize(&session.themes_dir)),
            name: Some(theme_name.to_string()),
        }),
        visual: Some(config::VisualConfig {
            vertex_color_source: Some(visual.vertex_color),
            vertex_size_source: Some(visual.vertex_size),
            vertex_size_scale: Some(visual.vertex_size_scale),
            vertex_alpha_scale: Some(visual.vertex_alpha_scale),
            uncategorized_alpha: Some(visual.uncategorized_alpha),
            edge_color_source: Some(visual.edge_color),
            edge_alpha_source: Some(visual.edge_alpha),
            edge_alpha_scale: Some(visual.edge_alpha_scale),
            edge_width_scale: Some(visual.edge_width_scale),
            // Absence = auto on reload, so only a manual window is persisted.
            edge_length_range: (!visual.edge_length_auto).then_some(visual.edge_length_range),
            projection: Some(visual.projection),
            depth_fade: Some(visual.depth_fade),
            flip_x: Some(visual.flip_x),
            flip_y: Some(visual.flip_y),
            search: (!search.is_empty()).then(|| search.to_string()),
        }),
        sizing: Some(config::SizingConfig {
            density_factor: Some(sizing.density_factor),
            point_min_pixels: Some(sizing.min_pixels),
            point_max_pixels: Some(sizing.max_pixels),
        }),
        camera: Some(config::CameraConfig {
            center: Some(cam.center()),
            half: Some(cam.half()),
            rotation_deg: Some(visual.rotation_deg),
        }),
        export: Some(config::ExportConfig {
            width: Some(export_dims.0),
            height: Some(export_dims.1),
        }),
        text: {
            let title = title.trim();
            let any =
                !title.is_empty() || text_params.color.is_some() || text_params.font.is_some();
            any.then(|| config::TextConfig {
                title: (!title.is_empty()).then(|| title.to_string()),
                size: Some(text_params.size),
                anchor: Some(text_params.anchor),
                margin_x: text_params.margin_x,
                margin_y: text_params.margin_y,
                // Only an explicit color is written — a theme-driven title stays
                // theme-driven on reload.
                color: text_params.color.map(hex_color),
                font: text_params.font.as_deref().map(absolutize),
            })
        },
        legend: legend.map(|p| config::LegendConfig {
            entries: p
                .entries
                .iter()
                .map(|(label, swatch)| config::LegendEntryConfig {
                    label: label.clone(),
                    category: match swatch {
                        LegendSwatch::Category(c) => Some(*c),
                        LegendSwatch::Fixed(_) => None,
                    },
                    color: match swatch {
                        LegendSwatch::Fixed(c) => Some(hex_color(*c)),
                        LegendSwatch::Category(_) => None,
                    },
                })
                .collect(),
            size: Some(p.size),
            anchor: Some(p.anchor),
            margin_x: p.margin_x,
            margin_y: p.margin_y,
            // Same rule as [text]: theme-driven text color stays theme-driven on reload.
            color: p.color.map(hex_color),
            // A names / entries_from file was already expanded into `entries` at startup —
            // the export snapshots the resolved rows rather than referencing the file.
            names: None,
            entries_from: None,
        }),
        density: density.map(|d| {
            // Effective anchors export as explicit knee/white so the picture reproduces
            // exactly (this is the "currently-frozen/effective values" the emitter promises);
            // before the first exposure lands, fall back to the config's percentile/explicit
            // encoding. The ramp is only written when explicitly set (a theme-fallback ramp
            // stays theme-driven on reload).
            let (knee, white, knee_pct, white_pct) = match density_anchors {
                Some(a) => (Some(a.knee), Some(a.white), None, None),
                None => (d.knee.ok(), d.white.ok(), d.knee.err(), d.white.err()),
            };
            config::DensityConfig {
                sigma: Some(d.sigma),
                curve: Some("log".to_string()),
                knee_percentile: knee_pct,
                white_percentile: white_pct,
                knee,
                white,
                ramp: (!d.ramp.is_empty()).then(|| d.ramp.iter().map(|c| hex_color(*c)).collect()),
            }
        }),
    };

    match cfg.save(&toml_path) {
        Ok(()) => format!("saved {}", toml_path.display()),
        Err(e) => format!("save toml failed: {e}"),
    }
}

/// Kick off a PNG export. Free function (not a method on App) so it can be called from
/// inside the render borrow scope without violating field-disjoint borrow rules — App
/// hands us a `&mut Renderer` and immutable refs to the sibling fields it needs.
///
/// Returns a status string suitable for the UI. Mutates the renderer's uniforms during
/// the export; caller is expected to push the interactive view uniforms on the next
/// frame (App's `refit_camera` + `push_view` already do this every tick).
#[allow(clippy::too_many_arguments)]
fn run_export(
    r: &mut Renderer,
    scene: &Scene,
    cam: &Camera,
    sizing: &SizingConfig,
    num_vertices: usize,
    settings: &ExportSettings,
    window_size: (u32, u32),
    clear_color: wgpu::Color,
    title: &str,
    text_params: &TextParams,
    legend: Option<&LegendParams>,
    title_font: &ab_glyph::FontArc,
    theme: &crate::theme::Theme,
) -> String {
    let (w, h) = settings.dimensions(window_size);
    let view = View::new(cam, num_vertices, w, h, sizing);
    let path = timestamped_export_path("png");
    let config = ExportConfig {
        width: w,
        height: h,
        // Always MSAA — vertex disks are fragment-shader antialiased already but line
        // edges alias badly at wallpaper resolutions without it. Cheap 4× is enough
        // (see `MSAA_SAMPLE_COUNT` note in scene_renderer.rs).
        msaa: true,
        clear_color,
        path: path.clone(),
    };
    // Title at export scale: re-rasterize at the target's px size (same 1080p-reference
    // rule as the pixel clamps) instead of stretching the preview texture.
    let export_text = {
        let title = title.trim();
        let px = text_params.size * (h as f32 / 1080.0);
        if title.is_empty() || px < 1.0 {
            None
        } else {
            let color = text_params.color.unwrap_or_else(|| theme.foreground());
            crate::text::rasterize(title_font, title, px, color).map(|bmp| {
                let quad = r.make_text_quad(&bmp);
                let k = px / text_params.size;
                let (x, y) = crate::text::anchor_position(
                    text_params.anchor,
                    (0.0, 0.0, w as f32, h as f32),
                    bmp.width as f32,
                    bmp.height as f32,
                    text_params.margin_x.map_or(0.5 * px, |m| m * k),
                    text_params.margin_y.map_or(0.5 * px, |m| m * k),
                );
                let ndc = crate::text::pixel_rect_to_ndc(
                    x,
                    y,
                    bmp.width as f32,
                    bmp.height as f32,
                    w as f32,
                    h as f32,
                );
                (quad, ndc)
            })
        }
    };
    // Legend at export scale — same re-rasterize rule.
    let export_legend = legend.and_then(|params| {
        let px = params.size * (h as f32 / 1080.0);
        if px < 1.0 {
            return None;
        }
        let color = params.color.unwrap_or_else(|| theme.foreground());
        crate::text::rasterize_legend(title_font, &params.rows(theme), px, color).map(|bmp| {
            let quad = r.make_text_quad(&bmp);
            let k = px / params.size;
            let (x, y) = crate::text::anchor_position(
                params.anchor,
                (0.0, 0.0, w as f32, h as f32),
                bmp.width as f32,
                bmp.height as f32,
                params.margin_x.map_or(0.5 * px, |m| m * k),
                params.margin_y.map_or(0.5 * px, |m| m * k),
            );
            let ndc = crate::text::pixel_rect_to_ndc(
                x,
                y,
                bmp.width as f32,
                bmp.height as f32,
                w as f32,
                h as f32,
            );
            (quad, ndc)
        })
    });
    let mut text_draws: Vec<(&TextQuad, [f32; 4])> = Vec::new();
    if let Some((q, ndc)) = export_text.as_ref() {
        text_draws.push((q, *ndc));
    }
    if let Some((q, ndc)) = export_legend.as_ref() {
        text_draws.push((q, *ndc));
    }
    let status = match r.export_png(scene, &view, &config, &text_draws) {
        Ok(()) => format!("saved {} ({w}×{h})", path.display()),
        Err(e) => format!("export failed: {e}"),
    };
    // The export mutated view uniforms to target dimensions. Restore them for the window
    // so the next interactive render pass draws with the right framing — cheaper than
    // relying on refit_camera firing on the next tick (which needs a positions update to
    // fire, and might not while paused).
    let (ww, wh) = window_size;
    r.set_view(&View::new(cam, num_vertices, ww, wh, sizing));
    status
}

/// Upload the per-vertex colors of `visual`'s source and point the shader at them (or at the
/// theme color). Needed whenever the source, the uncategorized alpha or the theme changes.
fn push_vertex_colors(
    r: &mut Renderer,
    scene: &Scene,
    attributes: &Attributes,
    theme: &crate::theme::Theme,
    visual: &VisualState,
    n: usize,
) {
    let colors = attributes.colors_for(
        visual.vertex_color,
        n,
        Some(theme),
        visual.uncategorized_alpha,
    );
    if let Some(c) = &colors {
        scene.write_colors(r.queue(), c);
    }
    r.set_vertex_color_source(
        visual
            .vertex_color
            .effective(colors.is_some())
            .shader_value(),
    );
}

const ERROR_RED: egui::Color32 = egui::Color32::from_rgb(0xf7, 0x76, 0x8e);

#[allow(clippy::too_many_arguments)]
fn build_ui(
    ctx: &egui::Context,
    fps: f32,
    iteration: u64,
    progress: Option<wembed_sys::Progress>,
    error: Option<&str>,
    n_vertices: usize,
    n_edges: usize,
    interactive: bool,
    paused: bool,
    finished: bool,
    toggle_pause: &mut bool,
    step_once_requested: &mut bool,
    visual: &mut VisualState,
    export: &mut ExportSettings,
    export_requested: &mut bool,
    save_embedding_requested: &mut bool,
    autorotate_requested: &mut bool,
    reset_view_requested: &mut bool,
    save_toml_requested: &mut bool,
    export_status: Option<&str>,
    theme_options: &[(usize, String)],
    theme_index: &mut usize,
    hover_text: Option<&str>,
    cursor: Option<(f32, f32)>,
    search_query: &mut String,
    toml_name: &mut String,
    title_text: &mut String,
    search_matches: usize,
    search_has_labels: bool,
    export_frame_rect: Option<egui::Rect>,
    convergence: &Convergence,
    projection_dim: usize,
    explained_variance: Option<f32>,
    edge_auto_range: Option<[f32; 2]>,
    density_ui: Option<&mut DensityUi>,
    available: Available,
) {
    // Convergence lives in its own pinned top-right window; it self-gates on samples and
    // starts collapsed (title-bar triangle to expand).
    convergence.window(ctx);

    // Left control column: an always-visible status/playback strip, then collapsible
    // sections (open by default) of label/control grids in a scroll area. Chromeless +
    // pinned top-left.
    egui::Window::new("controls")
        .title_bar(false)
        .resizable(false)
        .movable(false)
        .anchor(egui::Align2::LEFT_TOP, egui::vec2(10.0, 10.0))
        // egui's default window height (420) would cap the scroll area below
        .default_height(ctx.screen_rect().height())
        .show(ctx, |ui| {
            // ---- Status + playback (always visible) ----
            ui.horizontal(|ui| {
                let (color, word) = if error.is_some() {
                    (ERROR_RED, "error")
                } else if finished {
                    (egui::Color32::from_rgb(0x7a, 0xa2, 0xf7), "finished")
                } else if paused {
                    (egui::Color32::from_rgb(0xe0, 0xaf, 0x68), "paused")
                } else {
                    (egui::Color32::from_rgb(0x9e, 0xce, 0x6a), "live")
                };
                // Painted disk, not a glyph — the ● codepoint is missing from egui's default
                // font and renders as a tofu box.
                let d = ui.text_style_height(&egui::TextStyle::Body);
                let (rect, _) = ui.allocate_exact_size(egui::vec2(d, d), egui::Sense::hover());
                ui.painter().circle_filled(rect.center(), d * 0.3, color);
                ui.strong(word);
                match progress {
                    Some(p) => ui.label(format!(
                        "· {}",
                        format_progress_with_iteration(&p, iteration)
                    )),
                    None => ui.label(format!("· iter {iteration}")),
                };
            });
            if let Some(e) = error {
                ui.colored_label(ERROR_RED, e);
            }
            ui.label(format!(
                "{fps:.0} fps · {} verts · {} edges",
                fmt_compact(n_vertices),
                fmt_compact(n_edges)
            ));
            ui.horizontal(|ui| {
                if !finished && error.is_none() {
                    let label = if paused {
                        "resume (Space)"
                    } else {
                        "pause (Space)"
                    };
                    if ui.button(label).clicked() {
                        *toggle_pause = true;
                    }
                }
                if interactive && paused && !finished && error.is_none() {
                    if ui.button("step").clicked() {
                        *step_once_requested = true;
                    }
                }
                // Re-fit the camera to the current graph — undoes pan/zoom drift while
                // paused. Rotation is preserved (App reapplies `visual.rotation_deg`).
                if ui.button("reset view").clicked() {
                    *reset_view_requested = true;
                }
            });

            let max_height = ui.ctx().screen_rect().bottom() - ui.cursor().top() - 20.0;
            egui::ScrollArea::vertical()
                .max_height(max_height)
                .show(ui, |ui| {
                    section(ui, "Scene", |ui| {
                        grid(ui, "scene", |ui| {
                            // App applies a pick after the closure (invalidate cache + push colors
                            // + push bg).
                            ui.label("theme");
                            let current_name = theme_options
                                .iter()
                                .find(|(i, _)| *i == *theme_index)
                                .map(|(_, n)| n.as_str())
                                .unwrap_or("");
                            egui::ComboBox::from_id_salt("theme_picker")
                                .selected_text(current_name)
                                .width(CONTROL_W)
                                .show_ui(ui, |ui| {
                                    for (i, name) in theme_options {
                                        ui.selectable_value(theme_index, *i, name);
                                    }
                                });
                            ui.end_row();

                            // positive = CCW; autorotate picks the best angle for the export aspect
                            ui.label("rotation");
                            ui.horizontal(|ui| {
                                ui.add(
                                    egui::DragValue::new(&mut visual.rotation_deg)
                                        .speed(1.0)
                                        .suffix("°"),
                                );
                                if ui.button("autorotate").clicked() {
                                    *autorotate_requested = true;
                                }
                            });
                            ui.end_row();

                            // clip-space: "x" mirrors the picture left-right whatever the rotation
                            ui.label("flip");
                            ui.horizontal(|ui| {
                                ui.checkbox(&mut visual.flip_x, "x");
                                ui.checkbox(&mut visual.flip_y, "y");
                            });
                            ui.end_row();

                            // drawn into window and export alike; styling from the TOML [text]
                            ui.label("title");
                            ui.add(
                                egui::TextEdit::singleline(title_text)
                                    .hint_text("(none)")
                                    .desired_width(CONTROL_W),
                            );
                            ui.end_row();
                        });
                    });

                    section(ui, "Vertices", |ui| {
                        grid(ui, "vertices", |ui| {
                            ui.label("color");
                            source_combo(
                                ui,
                                "vertex_color",
                                &mut visual.vertex_color,
                                &[
                                    (VertexColorSource::Theme, "theme", None),
                                    (
                                        VertexColorSource::Categories,
                                        "categories",
                                        (!available.categories).then_some("load with --categories"),
                                    ),
                                    (
                                        VertexColorSource::File,
                                        "from file",
                                        (!available.vertex_colors).then_some("load with --colors"),
                                    ),
                                ],
                            );
                            ui.end_row();
                            ui.label("size");
                            source_combo(
                                ui,
                                "vertex_size",
                                &mut visual.vertex_size,
                                SCALAR_SOURCES,
                            );
                            ui.end_row();
                            ui.label("size ×");
                            ui.add(scale_slider(&mut visual.vertex_size_scale, 0.01..=10.0));
                            ui.end_row();
                            ui.label("alpha ×");
                            ui.add(alpha_slider(&mut visual.vertex_alpha_scale));
                            ui.end_row();
                            if visual.vertex_color == VertexColorSource::Categories {
                                ui.label("uncat. alpha ×");
                                ui.add(alpha_slider(&mut visual.uncategorized_alpha));
                                ui.end_row();
                            }
                        });
                    });

                    section(ui, "Edges", |ui| {
                        grid(ui, "edges", |ui| {
                            ui.label("color");
                            source_combo(
                                ui,
                                "edge_color",
                                &mut visual.edge_color,
                                &[
                                    (EdgeColorSource::Theme, "theme", None),
                                    (EdgeColorSource::Vertex, "by endpoints", None),
                                    (
                                        EdgeColorSource::File,
                                        "from file",
                                        (!available.edge_colors)
                                            .then_some("load with --edge-colors"),
                                    ),
                                    (EdgeColorSource::Length, "length (stress)", None),
                                ],
                            );
                            ui.end_row();
                            // Stress-ratio clamp of the length colormap. Auto derives it from the
                            // stress quantiles; unchecking seeds the manual values from the last
                            // auto window (handled App-side).
                            if visual.edge_color == EdgeColorSource::Length {
                                ui.label("range");
                                ui.horizontal(|ui| {
                                    ui.checkbox(&mut visual.edge_length_auto, "auto");
                                    if !visual.edge_length_auto {
                                        ui.add(
                                            egui::DragValue::new(&mut visual.edge_length_range[0])
                                                .speed(0.01)
                                                .range(0.01..=1.0)
                                                .prefix("≥"),
                                        );
                                        ui.add(
                                            egui::DragValue::new(&mut visual.edge_length_range[1])
                                                .speed(0.05)
                                                .range(1.0..=100.0)
                                                .prefix("≤"),
                                        );
                                    } else if let Some([lo, hi]) = edge_auto_range {
                                        ui.label(format!("{lo:.2} – {hi:.2}"));
                                    }
                                });
                                ui.end_row();
                            }
                            ui.label("alpha");
                            source_combo(ui, "edge_alpha", &mut visual.edge_alpha, SCALAR_SOURCES);
                            ui.end_row();
                            ui.label("alpha ×");
                            ui.add(alpha_slider(&mut visual.edge_alpha_scale));
                            ui.end_row();
                            ui.label("width ×");
                            ui.add(scale_slider(&mut visual.edge_width_scale, 0.01..=10.0));
                            ui.end_row();
                        });
                    });

                    if let Some(d) = density_ui {
                        section(ui, "Density", |ui| {
                            grid(ui, "density", |ui| {
                                ui.label("blur σ");
                                ui.add(egui::Slider::new(&mut d.sigma, 0.0..=8.0).suffix(" px"));
                                ui.end_row();
                                // freeze holds the exposure so its anchors can be copied into a
                                // config; otherwise they recompute once the camera rests
                                ui.label("exposure");
                                ui.checkbox(&mut d.freeze, "freeze");
                                ui.end_row();
                                ui.label("");
                                match d.anchors {
                                    Some(a) => ui.label(format!(
                                        "knee {:.4} · white {:.4}",
                                        a.knee, a.white
                                    )),
                                    None => ui.label("computing…"),
                                };
                                ui.end_row();
                            });
                        });
                    }

                    if projection_dim > 2 {
                        section(ui, "Projection", |ui| {
                            grid(ui, "projection", |ui| {
                                ui.label("method");
                                source_combo(
                                    ui,
                                    "projection",
                                    &mut visual.projection,
                                    &[
                                        (
                                            ProjectionMode::Pca,
                                            projection_label(ProjectionMode::Pca),
                                            None,
                                        ),
                                        (
                                            ProjectionMode::FirstTwo,
                                            projection_label(ProjectionMode::FirstTwo),
                                            None,
                                        ),
                                    ],
                                );
                                ui.end_row();
                                // fades vertices by how far they sit outside the display plane,
                                // i.e. where the 2D view misrepresents them
                                ui.label("depth fade");
                                ui.add(egui::Slider::new(&mut visual.depth_fade, 0.0..=1.0));
                                ui.end_row();
                            });
                            let variance = explained_variance
                                .map(|ev| format!(", view captures {:.1}% of variance", ev * 100.0))
                                .unwrap_or_default();
                            ui.label(format!("{projection_dim}D → 2D{variance}"));
                        });
                    }

                    section(ui, "Export", |ui| {
                        grid(ui, "export", |ui| {
                            ui.label("resolution");
                            egui::ComboBox::from_id_salt("export_preset")
                                .selected_text(export.preset.label())
                                .width(CONTROL_W)
                                .show_ui(ui, |ui| {
                                    for preset in [
                                        ResolutionPreset::Window,
                                        ResolutionPreset::P1080,
                                        ResolutionPreset::P1440,
                                        ResolutionPreset::K4,
                                        ResolutionPreset::K5,
                                        ResolutionPreset::K8,
                                        ResolutionPreset::Ultrawide,
                                        ResolutionPreset::UltrawideK5,
                                        ResolutionPreset::PhonePortrait,
                                        ResolutionPreset::Custom,
                                    ] {
                                        ui.selectable_value(
                                            &mut export.preset,
                                            preset,
                                            preset.label(),
                                        );
                                    }
                                });
                            ui.end_row();
                            if export.preset == ResolutionPreset::Custom {
                                ui.label("");
                                ui.horizontal(|ui| {
                                    ui.add(
                                        egui::DragValue::new(&mut export.custom_width)
                                            .range(1..=16384)
                                            .prefix("W "),
                                    );
                                    ui.add(
                                        egui::DragValue::new(&mut export.custom_height)
                                            .range(1..=16384)
                                            .prefix("H "),
                                    );
                                });
                                ui.end_row();
                            }
                        });
                        ui.horizontal(|ui| {
                            if ui.button("save png (S)").clicked() {
                                *export_requested = true;
                            }
                            if ui.button("save embedding (E)").clicked() {
                                *save_embedding_requested = true;
                            }
                        });
                        // A named snapshot of the whole session (`exports/<name>.toml`, bundling
                        // an `.emb` for live sessions). Empty name falls back to a timestamp.
                        ui.horizontal(|ui| {
                            ui.add(
                                egui::TextEdit::singleline(toml_name)
                                    .hint_text("config name…")
                                    .desired_width(112.0),
                            );
                            if ui.button("save toml").clicked() {
                                *save_toml_requested = true;
                            }
                        });
                        if let Some(msg) = export_status {
                            ui.label(msg);
                        }
                    });

                    section(ui, "Find", |ui| {
                        grid(ui, "find", |ui| {
                            ui.label("search");
                            let hint = if search_has_labels {
                                "label substring…"
                            } else {
                                "(no labels loaded)"
                            };
                            ui.add_enabled(
                                search_has_labels,
                                egui::TextEdit::singleline(search_query)
                                    .hint_text(hint)
                                    .desired_width(CONTROL_W),
                            );
                            ui.end_row();
                            if search_has_labels && !search_query.is_empty() {
                                ui.label("");
                                ui.label(format!("{search_matches} match(es)"));
                                ui.end_row();
                            }
                        });
                    });
                });
        });

    // Export-frame overlay on the main canvas. Drawn in the Background layer with
    // interactable=false so it doesn't eat pointer events, and stroked (not filled) so it
    // never obscures the scene. Rect comes in as *physical pixels*; egui works in logical
    // points, so convert via pixels_per_point.
    if let Some(rect_px) = export_frame_rect {
        let ppp = ctx.pixels_per_point();
        let rect = egui::Rect::from_min_max(
            egui::pos2(rect_px.min.x / ppp, rect_px.min.y / ppp),
            egui::pos2(rect_px.max.x / ppp, rect_px.max.y / ppp),
        );
        let painter = ctx.layer_painter(egui::LayerId::new(
            egui::Order::Foreground,
            egui::Id::new("export_frame_overlay"),
        ));
        // Two-tone stroke so the frame reads over both light and dark scene content: a
        // wider dim underlay + a thinner bright line on top. Selection color matches the
        // theme picker's active-item highlight.
        let visuals = ctx.style().visuals.clone();
        painter.rect_stroke(
            rect,
            0.0,
            egui::Stroke::new(3.0, egui::Color32::from_black_alpha(120)),
            egui::StrokeKind::Middle,
        );
        painter.rect_stroke(
            rect,
            0.0,
            egui::Stroke::new(1.5, visuals.selection.bg_fill),
            egui::StrokeKind::Middle,
        );
    }

    // Floating tooltip anchored near the cursor (upper-left offset). Only shown when we
    // have both a hovered vertex AND a valid cursor position — cursor may be `None` right
    // after the window gains focus, before the first mouse-move event.
    if let (Some(text), Some((mx, my))) = (hover_text, cursor) {
        let pixels_per_point = ctx.pixels_per_point();
        // Cursor is in physical pixels; egui works in logical points → divide.
        let anchor = egui::pos2(mx / pixels_per_point + 12.0, my / pixels_per_point + 12.0);
        egui::Area::new(egui::Id::new("hover_tooltip"))
            .fixed_pos(anchor)
            .order(egui::Order::Tooltip)
            .interactable(false)
            .show(ctx, |ui| {
                egui::Frame::popup(ui.style()).show(ui, |ui| {
                    // Cap width so multi-KB "fat" labels (e.g. Amazon product
                    // descriptions) wrap instead of stretching off-screen.
                    ui.set_max_width(420.0);
                    ui.label(text);
                });
            });
    }
}

/// Where on the preview canvas the export target's viewport lands, in *physical* pixels
/// (matches the values Renderer knows about). Since the preview and the export share the
/// same camera center and rotation, only the aspect-corrected half-extents differ — so the
/// overlay is axis-aligned in preview pixel space regardless of rotation.
///
/// The rectangle may extend beyond the preview canvas when the export's viewport shows
/// *more* of the world than the current window does (e.g. portrait export from a landscape
/// window). The overlay painter handles that by drawing the rect verbatim; nothing clips
/// it, so the user sees "the export includes area beyond what's currently visible."
pub fn export_frame_pixel_rect(
    cam: &Camera,
    window_size: (u32, u32),
    export_size: (u32, u32),
) -> Option<egui::Rect> {
    let (ww, wh) = window_size;
    let (ew, eh) = export_size;
    if ww == 0 || wh == 0 || ew == 0 || eh == 0 {
        return None;
    }
    let (hx_p, hy_p) = cam.half_extents(ww, wh);
    let (hx_e, hy_e) = cam.half_extents(ew, eh);
    if hx_p <= 0.0 || hy_p <= 0.0 {
        return None;
    }
    let rx = hx_e / hx_p;
    let ry = hy_e / hy_p;
    let (ww_f, wh_f) = (ww as f32, wh as f32);
    let x_left = (1.0 - rx) * 0.5 * ww_f;
    let x_right = (1.0 + rx) * 0.5 * ww_f;
    let y_top = (1.0 - ry) * 0.5 * wh_f;
    let y_bottom = (1.0 + ry) * 0.5 * wh_f;
    Some(egui::Rect::from_min_max(
        egui::pos2(x_left, y_top),
        egui::pos2(x_right, y_bottom),
    ))
}

/// Human label for the projection dropdown.
/// Width of the controls-grid dropdowns and text fields.
const CONTROL_W: f32 = 150.0;
/// Fixed label-column width so the controls line up across sections.
const LABEL_W: f32 = 92.0;

const SCALAR_SOURCES: &[(ScalarSource, &str, Option<&str>)] = &[
    (ScalarSource::Constant, "constant", None),
    (ScalarSource::Weight, "by weight", None),
];

/// Which optional attribute files are loaded — decides which color sources the UI offers.
#[derive(Copy, Clone)]
struct Available {
    categories: bool,
    vertex_colors: bool,
    edge_colors: bool,
}

fn section(ui: &mut egui::Ui, title: &str, add: impl FnOnce(&mut egui::Ui)) {
    egui::CollapsingHeader::new(title)
        .default_open(true)
        .show(ui, add);
}

/// Two-column label/control grid; the closure ends each row with `ui.end_row()`.
fn grid(ui: &mut egui::Ui, id: &str, add: impl FnOnce(&mut egui::Ui)) {
    egui::Grid::new(id)
        .num_columns(2)
        .min_col_width(LABEL_W)
        .spacing([10.0, 4.0])
        .show(ui, add);
}

/// Dropdown over `(value, label, reason it's unavailable)`. Unavailable options stay listed
/// but disabled, with the reason on hover.
fn source_combo<T: PartialEq + Copy>(
    ui: &mut egui::Ui,
    id: &str,
    value: &mut T,
    options: &[(T, &str, Option<&str>)],
) {
    let selected = options
        .iter()
        .find(|(v, _, _)| v == value)
        .map_or("", |(_, label, _)| *label);
    egui::ComboBox::from_id_salt(id)
        .selected_text(selected)
        .width(CONTROL_W)
        .show_ui(ui, |ui| {
            for &(option, label, unavailable) in options {
                let response = ui.add_enabled(
                    unavailable.is_none(),
                    egui::Button::selectable(*value == option, label),
                );
                if response.clicked() {
                    *value = option;
                }
                if let Some(reason) = unavailable {
                    response.on_disabled_hover_text(reason);
                }
            }
        });
}

/// Log-scale multiplier slider. A value outside `range` (e.g. from a config) is kept; only
/// edits clamp.
fn scale_slider(value: &mut f32, range: std::ops::RangeInclusive<f32>) -> egui::Slider<'_> {
    egui::Slider::new(value, range)
        .logarithmic(true)
        .clamping(egui::SliderClamping::Edits)
        .custom_formatter(|v, _| fmt_significant(v))
}

/// Alpha multiplier: log-scale because on big graphs the useful values sit around 1e-3;
/// below 1e-4 snaps to exactly 0 (hidden).
fn alpha_slider(value: &mut f32) -> egui::Slider<'_> {
    scale_slider(value, 0.0..=1.0).smallest_positive(1e-4)
}

/// Three significant digits (`1.00`, `0.350`, `0.00123`), so tiny alphas stay readable.
fn fmt_significant(v: f64) -> String {
    if v == 0.0 {
        return "0".to_string();
    }
    let decimals = (2 - v.abs().log10().floor() as i32).max(0) as usize;
    format!("{v:.decimals$}")
}

fn projection_label(mode: ProjectionMode) -> &'static str {
    match mode {
        ProjectionMode::Pca => "PCA (top-2 plane)",
        ProjectionMode::FirstTwo => "first two coords",
    }
}

/// Human label for the edge-color dropdown. `length` gets a hint at its diagnostic role.
/// Compact integer for the status strip — `12_345 → "12.3k"`, `4_500_000 → "4.5M"`. Keeps the
/// panel narrow on large graphs where the exact vertex/edge counts are just noise.
fn fmt_compact(n: usize) -> String {
    match n {
        n if n >= 1_000_000 => format!("{:.1}M", n as f64 / 1e6),
        n if n >= 1_000 => format!("{:.1}k", n as f64 / 1e3),
        n => n.to_string(),
    }
}

/// Extract the vertical component of a scroll delta, normalizing pixel deltas (touchpads /
/// smooth wheels) into "line" units so `ZOOM_PER_LINE` behaves consistently across devices.
fn scroll_delta_y(delta: MouseScrollDelta) -> f32 {
    match delta {
        MouseScrollDelta::LineDelta(_, y) => y,
        // 100px/line matches winit's own heuristic for touchpad → line conversion.
        MouseScrollDelta::PixelDelta(pos) => (pos.y as f32) / 100.0,
    }
}

// ----------------------------------------------------------------------------

/// EMA-smoothed frame-rate estimate. Cheap: one `Instant::now` per frame, no allocations.
struct FpsCounter {
    ema: f32,
    last_frame: Option<Instant>,
}

impl FpsCounter {
    fn new() -> Self {
        Self {
            ema: 0.0,
            last_frame: None,
        }
    }

    fn tick(&mut self) {
        let now = Instant::now();
        if let Some(prev) = self.last_frame {
            let dt = (now - prev).as_secs_f32().max(1e-9);
            let instant_fps = 1.0 / dt;
            self.ema = if self.ema == 0.0 {
                instant_fps
            } else {
                FPS_EMA_ALPHA * instant_fps + (1.0 - FPS_EMA_ALPHA) * self.ema
            };
        }
        self.last_frame = Some(now);
    }

    fn value(&self) -> f32 {
        self.ema
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fmt_significant_keeps_small_alphas_readable() {
        // a fixed 2-3 decimals would show the 1e-3 alphas big graphs need as 0.00
        assert_eq!(fmt_significant(1.0), "1.00");
        assert_eq!(fmt_significant(0.35), "0.350");
        assert_eq!(fmt_significant(0.00123), "0.00123");
        assert_eq!(fmt_significant(0.0), "0");
        assert_eq!(fmt_significant(250.0), "250");
    }
    use crate::framing::{Q_BUFFER, Q_HI, Q_LO};

    const EPS: f32 = 1e-3;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= EPS
    }

    /// Camera framing a symmetric cloud → predictable center (0,0).
    fn camera() -> Camera {
        let pts = [
            -10.0_f32, -10.0, 10.0, 10.0, -10.0, 10.0, 10.0, -10.0, 0.0, 0.0,
        ];
        Camera::fit_to_quantile_flat(&pts, Q_LO, Q_HI, Q_BUFFER)
    }

    // ---- ExportSettings::dimensions ----

    #[test]
    fn dimensions_custom_floored_at_one() {
        let s = ExportSettings {
            preset: ResolutionPreset::Custom,
            custom_width: 800,
            custom_height: 600,
        };
        assert_eq!(s.dimensions((1920, 1080)), (800, 600));

        let zeroed = ExportSettings {
            preset: ResolutionPreset::Custom,
            custom_width: 0,
            custom_height: 0,
        };
        assert_eq!(zeroed.dimensions((1920, 1080)), (1, 1));
    }

    // ---- export_frame_pixel_rect ----

    #[test]
    fn export_rect_preview_equals_export_is_full_window() {
        // Pinned: when preview == export, the function itself returns a Some rect equal to
        // the whole window (rx = ry = 1). The App call site short-circuits to None *before*
        // calling in that case — this documents the function's own behavior.
        let cam = camera();
        let rect = export_frame_pixel_rect(&cam, (800, 600), (800, 600)).expect("some rect");
        assert!(close(rect.min.x, 0.0) && close(rect.min.y, 0.0), "{rect:?}");
        assert!(
            close(rect.max.x, 800.0) && close(rect.max.y, 600.0),
            "{rect:?}"
        );
    }

    #[test]
    fn export_rect_zero_sized_is_none() {
        let cam = camera();
        assert!(export_frame_pixel_rect(&cam, (0, 600), (800, 600)).is_none());
        assert!(export_frame_pixel_rect(&cam, (800, 0), (800, 600)).is_none());
        assert!(export_frame_pixel_rect(&cam, (800, 600), (0, 600)).is_none());
        assert!(export_frame_pixel_rect(&cam, (800, 600), (800, 0)).is_none());
    }

    #[test]
    fn export_rect_portrait_export_from_landscape_extends_vertically() {
        // Landscape window (wide), portrait export (tall) shows MORE world vertically →
        // the rect's top edge goes above the canvas (min.y < 0).
        let cam = camera();
        let rect = export_frame_pixel_rect(&cam, (1600, 900), (1080, 1920)).expect("some rect");
        assert!(rect.min.y < 0.0, "expected min.y < 0, got {}", rect.min.y);
        assert!(
            rect.max.y > 900.0,
            "expected max.y > canvas, got {}",
            rect.max.y
        );
    }

    #[test]
    fn export_rect_wider_export_from_square_is_wider_and_centered() {
        // Square window, wide export shows MORE world horizontally → rect wider than
        // canvas, symmetric about the center.
        let cam = camera();
        let rect = export_frame_pixel_rect(&cam, (1000, 1000), (2000, 1000)).expect("some rect");
        assert!(rect.min.x < 0.0, "min.x {}", rect.min.x);
        assert!(rect.max.x > 1000.0, "max.x {}", rect.max.x);
        // Centered horizontally: left overshoot mirrors right overshoot.
        assert!(
            close(rect.min.x, 1000.0 - rect.max.x),
            "not centered: {rect:?}"
        );
        // Vertical extent unchanged (same aspect on that axis).
        assert!(
            close(rect.min.y, 0.0) && close(rect.max.y, 1000.0),
            "y changed: {rect:?}"
        );
    }

    // ---- timestamped_export_path ----

    #[test]
    fn timestamped_path_uses_exports_dir_and_extension() {
        // Uses the process CWD (the repo root under test), so it lands under `exports/`
        // (gitignored) when the dir is creatable. We assert the file-name shape and
        // extension rather than an absolute location.
        let p = timestamped_export_path("png");
        assert_eq!(p.extension().and_then(|s| s.to_str()), Some("png"));
        let name = p.file_name().and_then(|s| s.to_str()).unwrap();
        assert!(name.starts_with("wembed_"), "unexpected file name: {name}");
        // When create_dir_all succeeds the path is `exports/<name>`; the fallback path is
        // the bare name in CWD. Either way the parent is `exports` or empty.
        match p.parent().and_then(|s| s.to_str()) {
            Some("") | None => {} // fallback: bare filename in CWD
            Some(parent) => assert_eq!(parent, EXPORT_DIR, "parent {parent}"),
        }
    }
    // NOTE: the "creates the directory if missing (run inside a temp CWD)" sub-item is
    // skipped — `timestamped_export_path` reads the *process* CWD, and changing it in a
    // multi-threaded test runner races other tests. Covered indirectly by the assertion
    // above that the parent is `EXPORT_DIR` when creation succeeds.

    // ---- sanitize_toml_stem ----

    #[test]
    fn sanitize_toml_stem_cleans_name() {
        // Normal names pass through untouched.
        assert_eq!(sanitize_toml_stem("my_session"), "my_session");
        // Interior `/`, `\`, and whitespace flatten to `-` (a name, not a path).
        assert_eq!(sanitize_toml_stem("a/b\\c d"), "a-b-c-d");
        assert_eq!(sanitize_toml_stem("tab\tname"), "tab-name");
        // Trim happens *before* flattening, so edge whitespace vanishes instead of → `-`.
        assert_eq!(sanitize_toml_stem("  name  "), "name");
        assert_eq!(sanitize_toml_stem("\t hi \n"), "hi");
    }

    #[test]
    fn sanitize_toml_stem_empty_falls_back_to_timestamp() {
        for input in ["", "   ", "\t\n "] {
            let stem = sanitize_toml_stem(input);
            assert!(
                stem.starts_with("wembed_"),
                "expected timestamp fallback, got {stem:?}"
            );
            let digits = &stem["wembed_".len()..];
            assert!(!digits.is_empty(), "no digits in fallback stem {stem:?}");
            assert!(
                digits.chars().all(|c| c.is_ascii_digit()),
                "fallback tail not all digits: {stem:?}"
            );
        }
    }

    // ---- absolutize ----

    #[test]
    fn absolutize_leaves_absolute_paths_unchanged() {
        let abs = std::path::Path::new("/etc/hosts");
        assert_eq!(absolutize(abs), PathBuf::from("/etc/hosts"));
    }

    #[test]
    fn absolutize_anchors_relative_paths_to_cwd() {
        let rel = std::path::Path::new("sub/graph.edg");
        let out = absolutize(rel);
        assert!(out.is_absolute(), "expected absolute path, got {out:?}");
        assert!(
            out.ends_with("sub/graph.edg"),
            "expected to end with the relative path, got {out:?}"
        );
    }
}
