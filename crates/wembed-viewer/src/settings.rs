//! Precedence merge: built-in defaults → `--config` TOML → explicitly-passed CLI flags.
//!
//! [`Cli`](crate::cli::Cli) keeps every tunable as `Option<T>` precisely so this module
//! can tell "user passed it" from "user said nothing"; the TOML layer
//! ([`FileConfig`](crate::config::FileConfig)) is all-`Option` for the same reason. The
//! output is [`Settings`] — fully resolved, no `Option`s except things that are genuinely
//! optional (attribute files, theme name, `[camera]`).
//!
//! The defaults in [`defaults`] are the single source of truth; the `[default: ...]`
//! notes in cli.rs doc comments mirror them — keep both in sync.

use std::path::PathBuf;

use anyhow::{bail, Context, Result};

use crate::app::VisualState;
use crate::cli::{Cli, Command, VertexColorSource};
use crate::config::{CameraConfig, FileConfig};
use crate::framing::SizingConfig;

/// Built-in defaults, applied when neither the TOML nor the CLI says otherwise.
///
/// The embedder knobs here mirror `wembed::Options`'s C++ defaults. The real source of
/// truth is the C++ header, reachable at runtime via `wembed_sys::Options::default()`
/// (which reads the member initializers over FFI) — but these are kept as compile-time
/// `const`s because they're the values the clap `[default: N]` help strings in cli.rs
/// quote, and a `///` doc-comment can't interpolate a runtime value. The
/// `defaults_track_wembed` test asserts each equals the C++ default, so drift fails loudly
/// instead of silently changing behavior (as `warmup_steps` once did: UI 0 vs wembed 20).
mod defaults {
    use crate::cli::{LrSchedule, StopCriterion};

    pub const ITERATIONS: i32 = 10_000;
    pub const SEED: i32 = 42;
    /// Deliberately diverges from wembed's C++ default (4): the viewer is a 2D tool
    /// first, so dimension > 2 (and the projection machinery it activates) is opt-in.
    /// Exempt from `defaults_track_wembed`.
    pub const DIMENSION: i32 = 2;
    /// Non-positive = off (degree weights used as-is).
    pub const DIMENSION_HINT: f64 = -1.0;
    pub const LEARNING_RATE: f64 = 10.0;
    pub const CENTRE: f64 = 0.0;
    /// Negative = auto (3 / dimension); 0 disables dynamic queries.
    pub const DYN_BUFFER: f64 = -1.0;
    // Learning-rate schedule knobs — values mirror wembed's `Options` defaults so an
    // un-tuned run matches the C++ CLI's out-of-the-box behavior.
    pub const LR_SCHEDULE: LrSchedule = LrSchedule::Exponential;
    pub const WARMUP_STEPS: i32 = 20;
    pub const LR_COOLING_FACTOR: f64 = 0.995;
    pub const LR_DECAY_FACTOR: f64 = 0.5;
    pub const LR_DECAY_THRESHOLD: f64 = 1e-2;
    pub const LR_ADAPT_PATIENCE: i32 = 20;
    // Growth is opt-in and off by default (factor 1.0 → pure plateau decay).
    pub const LR_GROWTH_FACTOR: f64 = 1.0;
    pub const LR_GROWTH_THRESHOLD: f64 = 1e-1;
    // Stopping-criterion knobs — values mirror wembed's `Options` defaults.
    pub const STOP_CRITERION: StopCriterion = StopCriterion::Loss;
    pub const STOP_DISPLACEMENT_TOL: f64 = 3e-4;
    pub const STOP_DISPLACEMENT_PATIENCE: i32 = 5;
    pub const STOP_LOSS_TOL: f64 = 1e-3;
    pub const STOP_LOSS_PATIENCE: i32 = 50;
    pub const DENSITY_FACTOR: f32 = 0.5;
    pub const POINT_MIN_PIXELS: f32 = 3.0;
    pub const POINT_MAX_PIXELS: f32 = 40.0;
    /// The bundled themes, found from any CWD. Packagers that don't ship the source tree
    /// relocate them at build time via `WEMBED_VIEWER_THEMES_DIR` (the Nix package does).
    pub const THEMES_DIR: &str = match option_env!("WEMBED_VIEWER_THEMES_DIR") {
        Some(dir) => dir,
        None => concat!(env!("CARGO_MANIFEST_DIR"), "/themes"),
    };
    pub const RESOLUTION: (u32, u32) = (3840, 2160);
    /// Title text height in px at the 1080p reference height.
    pub const TITLE_SIZE: f32 = 28.0;
    /// Legend row text height in px at the 1080p reference height.
    pub const LEGEND_SIZE: f32 = 24.0;
    /// Density blur sigma in pixels.
    pub const DENSITY_SIGMA: f32 = 1.2;
    /// Density knee anchor, ABSOLUTE. 0 = curve rises from the first density quantum — a
    /// percentile knee crushes everything below it to one flat floor color.
    pub const DENSITY_KNEE: f32 = 0.0;
    /// Density white anchor percentile over lit pixels.
    pub const DENSITY_WHITE_PCT: f32 = 99.5;
}

/// Resolved `[text]` section. `title` is the only field with a CLI counterpart
/// (`--title`); the styling knobs are TOML-only.
#[derive(Clone, Debug)]
pub struct TextParams {
    pub title: Option<String>,
    /// Px at the 1080p reference height (scaled per target, like the pixel clamps).
    pub size: f32,
    pub anchor: crate::text::Anchor,
    /// Margins from the anchored corner in 1080p-reference px (scaled like `size`).
    /// `None` → half the scaled text size (the historical default).
    pub margin_x: Option<f32>,
    pub margin_y: Option<f32>,
    /// `None` → the active theme's foreground (base05), resolved at draw time so theme
    /// switches recolor the title.
    pub color: Option<[u8; 4]>,
    /// Font file override; `None` → bundled font.
    pub font: Option<PathBuf>,
}

/// A legend row's swatch color source. `Category` resolves against the ACTIVE theme at
/// draw time (so runtime theme switches recolor the legend along with the vertices);
/// `Fixed` is an explicit `color = "#..."`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum LegendSwatch {
    Category(i64),
    Fixed([u8; 4]),
}

/// Resolved `[legend]` section. Entries are TOML-order rows; `color` `None` → theme
/// foreground, like the title.
#[derive(Clone, Debug)]
pub struct LegendParams {
    pub entries: Vec<(String, LegendSwatch)>,
    /// Px at the 1080p reference height (scaled per target, like `[text].size`).
    pub size: f32,
    pub anchor: crate::text::Anchor,
    /// Margins from the anchored corner in 1080p-reference px (scaled like `size`).
    /// `None` → half the scaled text size (the historical default).
    pub margin_x: Option<f32>,
    pub margin_y: Option<f32>,
    pub color: Option<[u8; 4]>,
}

/// Expand a category-names file's lines into legend entries: line i → `(name,
/// Category(i))`. Blank lines skip their category WITHOUT shifting the ids below —
/// positional alignment with the `--categories` id space is the whole point of the file.
pub fn legend_entries_from_names(names: &[String]) -> Vec<(String, LegendSwatch)> {
    names
        .iter()
        .enumerate()
        .filter(|(_, n)| !n.trim().is_empty())
        .map(|(i, n)| (n.trim().to_string(), LegendSwatch::Category(i as i64)))
        .collect()
}

/// Validate `[[legend.entries]]`-shaped rows (inline or from an external file) into
/// swatch rows: exactly one of `category` / `color` per row, hex parsed.
pub fn legend_entries_from_configs(
    entries: &[crate::config::LegendEntryConfig],
) -> Result<Vec<(String, LegendSwatch)>> {
    entries
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let swatch = match (e.category, e.color.as_deref()) {
                (Some(c), None) => LegendSwatch::Category(c),
                (None, Some(hex)) => LegendSwatch::Fixed(
                    crate::text::parse_color(hex)
                        .with_context(|| format!("legend entry #{}: color", i + 1))?,
                ),
                _ => bail!(
                    "legend entry #{} ({:?}): set exactly one of `category` or `color`",
                    i + 1,
                    e.label
                ),
            };
            Ok((e.label.clone(), swatch))
        })
        .collect()
}

impl LegendParams {
    /// Resolve every row's swatch color against `theme` — the same mapping
    /// `Attributes::colors_for` applies to the `--categories` file: accents indexed
    /// modulo palette length, negative ids → the neutral base04.
    pub fn rows(&self, theme: &crate::theme::Theme) -> Vec<(String, [u8; 4])> {
        let accents = theme.accents();
        self.entries
            .iter()
            .map(|(label, swatch)| {
                let color = match swatch {
                    LegendSwatch::Fixed(c) => *c,
                    LegendSwatch::Category(c) if *c < 0 => theme.neutral(),
                    LegendSwatch::Category(c) => accents[(*c as usize) % accents.len()],
                };
                (label.clone(), color)
            })
            .collect()
    }
}

/// Resolved density (heatmap) mode parameters. `Some` on [`Settings`] iff density mode is
/// enabled (a `[density]` section OR any `--density*` flag). The ramp is resolved to RGBA
/// stops here; anchors stay in "percentile OR explicit value" form so the tone-mapping pass
/// can compute the percentile ones from the actual density histogram at render time.
#[derive(Clone, Debug, PartialEq)]
pub struct DensityParams {
    /// Blur sigma in pixels. `0` = off.
    pub sigma: f32,
    /// Knee anchor: `Ok(density)` explicit, `Err(percentile)` derived over lit pixels.
    pub knee: Result<f32, f32>,
    /// White anchor: same encoding as `knee`.
    pub white: Result<f32, f32>,
    /// Ramp stops (dark → bright), >= 2, sRGB RGBA bytes. Empty = fall back to theme
    /// bg → fg at LUT-build time (the theme isn't known here).
    pub ramp: Vec<[u8; 4]>,
}

/// Resolved embedder knobs — the numbers that go into `SpawnConfig`, kept as a separate
/// struct because the TOML export also records them as provenance for a live session.
#[derive(Copy, Clone, Debug)]
pub struct EmbedderParams {
    pub iterations: i32,
    pub seed: i32,
    pub dimension: i32,
    pub dimension_hint: f64,
    pub learning_rate: f64,
    pub lr_cooling_factor: f64,
    pub layered: bool,
    pub centre: f64,
    pub dyn_buffer: f64,
    // ---- Learning-rate schedule ----
    pub lr_schedule: crate::cli::LrSchedule,
    pub warmup_steps: i32,
    pub lr_decay_factor: f64,
    pub lr_decay_threshold: f64,
    pub lr_adapt_patience: i32,
    pub lr_growth_factor: f64,
    pub lr_growth_threshold: f64,
    // ---- Stopping criterion ----
    pub stop_criterion: crate::cli::StopCriterion,
    pub stop_displacement_tol: f64,
    pub stop_displacement_patience: i32,
    // ---- Loss stop ----
    pub stop_loss_tol: f64,
    pub stop_loss_patience: i32,
}

/// Everything the interactive TOML export needs to reconstruct a `FileConfig`: the input
/// paths and embedder provenance as resolved at startup. Snapshot taken once in `main`
/// and handed to `App` — the values never change over a session.
#[derive(Clone, Debug)]
pub struct SessionInfo {
    pub graph_path: PathBuf,
    /// `Some` when this session loads positions from an `.emb` (the export references it
    /// verbatim); `None` for a live embedder (the export bundles a fresh `.emb` alongside
    /// the TOML).
    pub input_embedding: Option<PathBuf>,
    pub colors: Option<PathBuf>,
    pub labels: Option<PathBuf>,
    pub categories: Option<PathBuf>,
    pub edge_colors: Option<PathBuf>,
    pub themes_dir: PathBuf,
    pub embedder: EmbedderParams,
}

/// Fully-resolved runtime configuration. Consumed by `main` (producer spawn, attrs,
/// themes), `App` (visual/search/camera/export panel seeds), and `headless::run`.
pub struct Settings {
    pub graph_path: PathBuf,
    pub input_embedding: Option<PathBuf>,
    pub embedder: EmbedderParams,
    pub colors: Option<PathBuf>,
    pub labels: Option<PathBuf>,
    pub categories: Option<PathBuf>,
    pub edge_colors: Option<PathBuf>,
    pub themes_dir: PathBuf,
    pub theme: Option<String>,
    pub sizing: SizingConfig,
    /// Initial UI visual state (color mode + channel toggles + slider scales).
    /// `rotation_deg` is always 0 here — rotation lives in `[camera]`, which only
    /// applies on `.emb` replay.
    pub visual: VisualState,
    /// Label-search query from `[visual].search`, re-applied once labels are loaded.
    pub search: Option<String>,
    /// The TOML `[camera]` section, verbatim. `main` strips this to `None` (with a
    /// warning) unless the producer is an `.emb` replay — downstream code may assume
    /// `Some` implies replay.
    pub camera: Option<CameraConfig>,
    /// `[export]` width/height — seeds the interactive export panel's Custom preset.
    pub export_dims: Option<(u32, u32)>,
    /// Resolved `[text]` section (+ `--title`).
    pub text: TextParams,
    /// Resolved `[legend]` section. `None` when the TOML has no entries — nothing drawn.
    /// When `legend_names`/`legend_entries_from` is set this is `Some` with EMPTY entries;
    /// `main` loads the file and fills them (same eager-parse stage as the attribute files).
    pub legend: Option<LegendParams>,
    /// `[legend].names` — category-names file (line i = name of category id i).
    pub legend_names: Option<PathBuf>,
    /// `[legend].entries_from` or `--legend-entries` — external `[[entries]]` TOML.
    pub legend_entries_from: Option<PathBuf>,
    /// Resolved `[density]` params. `Some` iff density mode is enabled (section present OR
    /// a `--density*` flag passed). `None` = normal points/edges rendering.
    pub density: Option<DensityParams>,
    /// GUI-only mode: wait for user input before starting embedding iterations.
    pub interactive: bool,

    // ---- headless ----
    pub headless: bool,
    /// PNG destination (`-o`). `None` on an embed-only `headless` run.
    pub output: Option<PathBuf>,
    /// Headless render resolution: CLI `--resolution` > TOML `[export]` > 4K.
    pub resolution: (u32, u32),
    /// Explicit CLI `--rotation-deg` only — headless resolves the full priority chain
    /// (autorotate > this > `[camera].rotation_deg` > 0) itself, since only it knows
    /// whether autorotate is in play.
    pub rotation_deg_cli: Option<f32>,
    pub autorotate: bool,
    /// `.emb` destination (`--output-embedding`, or TOML `output_embedding =`).
    pub output_embedding: Option<PathBuf>,
}

/// CLI over TOML over built-in default — the precedence every knob follows.
fn pick<T>(cli: Option<T>, file: Option<T>, default: T) -> T {
    cli.or(file).unwrap_or(default)
}

fn set_if_some<T>(dst: &mut T, src: Option<T>) {
    if let Some(x) = src {
        *dst = x;
    }
}

impl Settings {
    /// Merge CLI over TOML over defaults. Errors on: no graph from either source, a
    /// half-specified `[export]` section, or a `headless` run with neither `-o` nor
    /// `--output-embedding` (nothing to do).
    pub fn resolve(cli: Cli, file: FileConfig) -> Result<Settings> {
        let mut interactive = false;
        // Split the subcommand: headless-only knobs:
        let mut headless = false;
        let mut output = None;
        let mut resolution_cli = None;
        let mut rotation_deg_cli = None;
        let mut autorotate = false;
        let mut output_embedding = None;
        let common = match cli.command {
            Command::Gui(g) => {
                interactive = g.interactive;
                g.common
            }
            Command::Headless(r) => {
                headless = true;
                output = r.output;
                resolution_cli = r.resolution;
                rotation_deg_cli = r.rotation_deg;
                autorotate = r.autorotate;
                output_embedding = r.output_embedding;
                r.common
            }
        };

        let e = file.embedder.unwrap_or_default();
        let embedder = EmbedderParams {
            iterations: pick(common.iterations, e.iterations, defaults::ITERATIONS),
            seed: pick(common.seed, e.seed, defaults::SEED),
            dimension: pick(common.dimension, e.dimension, defaults::DIMENSION),
            dimension_hint: pick(common.dim_hint, e.dimension_hint, defaults::DIMENSION_HINT),
            learning_rate: pick(
                common.learning_rate,
                e.learning_rate,
                defaults::LEARNING_RATE,
            ),
            lr_cooling_factor: pick(
                common.lr_cooling_factor,
                e.lr_cooling_factor,
                defaults::LR_COOLING_FACTOR,
            ),
            // A bool flag can only be switched ON from the CLI, so or-merge is exact.
            layered: common.layered || e.layered.unwrap_or(false),
            centre: pick(common.centre, e.centre, defaults::CENTRE),
            dyn_buffer: pick(common.dyn_buffer, e.dyn_buffer, defaults::DYN_BUFFER),
            lr_schedule: pick(common.lr_schedule, e.lr_schedule, defaults::LR_SCHEDULE),
            warmup_steps: pick(common.warmup_steps, e.warmup_steps, defaults::WARMUP_STEPS),
            lr_decay_factor: pick(
                common.lr_decay_factor,
                e.lr_decay_factor,
                defaults::LR_DECAY_FACTOR,
            ),
            lr_decay_threshold: pick(
                common.lr_decay_threshold,
                e.lr_decay_threshold,
                defaults::LR_DECAY_THRESHOLD,
            ),
            lr_adapt_patience: pick(
                common.lr_adapt_patience,
                e.lr_adapt_patience,
                defaults::LR_ADAPT_PATIENCE,
            ),
            lr_growth_factor: pick(
                common.lr_growth_factor,
                e.lr_growth_factor,
                defaults::LR_GROWTH_FACTOR,
            ),
            lr_growth_threshold: pick(
                common.lr_growth_threshold,
                e.lr_growth_threshold,
                defaults::LR_GROWTH_THRESHOLD,
            ),
            stop_criterion: pick(
                common.stop_criterion,
                e.stop_criterion,
                defaults::STOP_CRITERION,
            ),
            stop_displacement_tol: pick(
                common.stop_displacement_tol,
                e.stop_displacement_tol,
                defaults::STOP_DISPLACEMENT_TOL,
            ),
            stop_displacement_patience: pick(
                common.stop_displacement_patience,
                e.stop_displacement_patience,
                defaults::STOP_DISPLACEMENT_PATIENCE,
            ),
            stop_loss_tol: pick(
                common.stop_loss_tol,
                e.stop_loss_tol,
                defaults::STOP_LOSS_TOL,
            ),
            stop_loss_patience: pick(
                common.stop_loss_patience,
                e.stop_loss_patience,
                defaults::STOP_LOSS_PATIENCE,
            ),
        };
        if embedder.dimension < 2 {
            bail!(
                "--dimension must be >= 2 (got {}); the viewer displays a 2D projection",
                embedder.dimension
            );
        }

        let a = file.attrs.unwrap_or_default();
        let t = file.theme.unwrap_or_default();
        let v = file.visual.unwrap_or_default();
        let s = file.sizing.unwrap_or_default();

        let vertex_color = pick(
            common.vertex_color_source,
            v.vertex_color_source,
            VertexColorSource::Theme,
        );
        let mut visual = VisualState::new(vertex_color);
        set_if_some(&mut visual.vertex_size, v.vertex_size_source);
        set_if_some(&mut visual.vertex_size_scale, v.vertex_size_scale);
        set_if_some(&mut visual.vertex_alpha_scale, v.vertex_alpha_scale);
        set_if_some(&mut visual.edge_alpha, v.edge_alpha_source);
        set_if_some(&mut visual.edge_alpha_scale, v.edge_alpha_scale);
        set_if_some(&mut visual.edge_width_scale, v.edge_width_scale);
        set_if_some(&mut visual.edge_color, v.edge_color_source);
        if let Some(x) = v.edge_length_range {
            // An explicit range opts out of the automatic quantile window.
            visual.edge_length_range = x;
            visual.edge_length_auto = false;
        }
        set_if_some(&mut visual.projection, v.projection);
        if let Some(x) = v.depth_fade {
            visual.depth_fade = x.clamp(0.0, 1.0);
        }
        set_if_some(&mut visual.uncategorized_alpha, v.uncategorized_alpha);
        set_if_some(&mut visual.flip_x, v.flip_x);
        set_if_some(&mut visual.flip_y, v.flip_y);

        let ex = file.export.unwrap_or_default();
        let export_dims = match (ex.width, ex.height) {
            (Some(w), Some(h)) => Some((w, h)),
            (None, None) => None,
            _ => bail!("[export] needs both width and height"),
        };

        let tx = file.text.unwrap_or_default();
        let text = TextParams {
            title: common.title.or(tx.title).filter(|t| !t.trim().is_empty()),
            size: tx.size.unwrap_or(defaults::TITLE_SIZE),
            anchor: tx.anchor.unwrap_or(crate::text::Anchor::TopLeft),
            margin_x: tx.margin_x,
            margin_y: tx.margin_y,
            color: tx
                .color
                .as_deref()
                .map(crate::text::parse_color)
                .transpose()
                .context("[text].color")?,
            font: tx.font,
        };

        let lg = file.legend.unwrap_or_default();
        // At most one entry source in the config. `--legend-entries` overrides the lot.
        let source_count = lg.names.is_some() as u8
            + lg.entries_from.is_some() as u8
            + !lg.entries.is_empty() as u8;
        if source_count > 1 {
            bail!("[legend]: set at most one of `names`, `entries_from`, or `[[legend.entries]]`");
        }

        // `--legend-entries` supersedes every config source; otherwise the config's
        // single source (if any) wins. `names`/`entries_from` produce a legend with EMPTY
        // entries here — `main` fills them once the file is parsed (eager, alongside the
        // attribute files); inline `entries` are validated immediately.
        let cli_entries = common.legend_entries.clone();
        let (legend_names, legend_entries_from, inline_entries) = if let Some(p) = cli_entries {
            (None, Some(p), Vec::new())
        } else {
            (
                lg.names,
                lg.entries_from,
                legend_entries_from_configs(&lg.entries)?,
            )
        };
        let deferred_source = legend_names.is_some() || legend_entries_from.is_some();
        let legend = (deferred_source || !inline_entries.is_empty()).then(|| LegendParams {
            entries: inline_entries,
            size: lg.size.unwrap_or(defaults::LEGEND_SIZE),
            anchor: lg.anchor.unwrap_or(crate::text::Anchor::BottomLeft),
            margin_x: lg.margin_x,
            margin_y: lg.margin_y,
            color: None,
        });
        // Text color parsed outside the closure so the error propagates.
        let legend = match (legend, lg.color.as_deref()) {
            (Some(mut params), Some(hex)) => {
                params.color = Some(crate::text::parse_color(hex).context("[legend].color")?);
                Some(params)
            }
            (legend, _) => legend,
        };

        // Density mode: enabled by a `[density]` section OR any `--density*` flag. CLI
        // overrides layer over the (possibly absent) TOML section, defaults fill the rest.
        let density_enabled =
            common.density || common.density_sigma.is_some() || common.density_ramp.is_some();
        let density = if density_enabled || file.density.is_some() {
            let d = file.density.unwrap_or_default();
            if let Some(curve) = d.curve.as_deref() {
                if curve != "log" {
                    bail!("[density].curve: only \"log\" is supported (got {curve:?})");
                }
            }
            let sigma = pick(common.density_sigma, d.sigma, defaults::DENSITY_SIGMA).max(0.0);
            // Explicit value beats the percentile; the default is absolute 0, NOT a
            // percentile — knee_percentile is opt-in floor-crushing.
            let knee = match (d.knee, d.knee_percentile) {
                (Some(v), _) => Ok(v),
                (None, Some(p)) => Err(p),
                (None, None) => Ok(defaults::DENSITY_KNEE),
            };
            let white = match d.white {
                Some(v) => Ok(v),
                None => Err(d.white_percentile.unwrap_or(defaults::DENSITY_WHITE_PCT)),
            };
            // CLI ramp beats TOML ramp; empty Vec → theme bg→fg at LUT-build time. The CLI
            // form is one comma-separated string (>= 2 stops); TOML is already a list.
            let cli_ramp = match common.density_ramp.as_deref() {
                Some(s) => Some(crate::cli::parse_hex_list(s).map_err(|e| anyhow::anyhow!(e))?),
                None => None,
            };
            let ramp_hex = cli_ramp.or(d.ramp);
            let ramp = match ramp_hex {
                Some(stops) => {
                    if stops.len() < 2 {
                        bail!("[density].ramp: need at least 2 hex stops");
                    }
                    stops
                        .iter()
                        .enumerate()
                        .map(|(i, hex)| {
                            crate::text::parse_color(hex)
                                .with_context(|| format!("[density].ramp stop #{}", i + 1))
                        })
                        .collect::<Result<Vec<_>>>()?
                }
                None => Vec::new(),
            };
            Some(DensityParams {
                sigma,
                knee,
                white,
                ramp,
            })
        } else {
            None
        };

        let Some(graph_path) = common.graph_path.or(file.graph) else {
            bail!("no graph given: pass a graph path on the CLI or set `graph = ...` in the --config TOML");
        };

        // CLI beats TOML, like every other knob.
        let output_embedding = output_embedding.or(file.output_embedding);
        if headless && output.is_none() && output_embedding.is_none() {
            bail!(
                "render has nothing to do: pass -o/--output for a PNG \
                 and/or --output-embedding for an .emb"
            );
        }

        Ok(Settings {
            graph_path,
            input_embedding: common.input_embedding.or(file.input_embedding),
            embedder,
            colors: common.colors.or(a.colors),
            labels: common.labels.or(a.labels),
            categories: common.categories.or(a.categories),
            edge_colors: common.edge_colors.or(a.edge_colors),
            themes_dir: common
                .themes_dir
                .or(t.dir)
                .unwrap_or_else(|| PathBuf::from(defaults::THEMES_DIR)),
            theme: common.theme.or(t.name),
            sizing: SizingConfig {
                density_factor: pick(
                    common.density_factor,
                    s.density_factor,
                    defaults::DENSITY_FACTOR,
                ),
                min_pixels: pick(
                    common.point_min_pixels,
                    s.point_min_pixels,
                    defaults::POINT_MIN_PIXELS,
                ),
                max_pixels: pick(
                    common.point_max_pixels,
                    s.point_max_pixels,
                    defaults::POINT_MAX_PIXELS,
                ),
            },
            visual,
            text,
            legend,
            legend_names,
            legend_entries_from,
            density,
            search: v.search.filter(|q| !q.is_empty()),
            camera: file.camera,
            export_dims,
            headless,
            interactive,
            output,
            resolution: pick(resolution_cli, export_dims, defaults::RESOLUTION),
            rotation_deg_cli,
            autorotate,
            output_embedding,
        })
    }

    /// Snapshot for the interactive TOML export. Clones — taken once at startup.
    pub fn session_info(&self) -> SessionInfo {
        SessionInfo {
            graph_path: self.graph_path.clone(),
            input_embedding: self.input_embedding.clone(),
            colors: self.colors.clone(),
            labels: self.labels.clone(),
            categories: self.categories.clone(),
            edge_colors: self.edge_colors.clone(),
            themes_dir: self.themes_dir.clone(),
            embedder: self.embedder,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        CameraConfig, EmbedderConfig, ExportConfig, SizingConfig as FileSizing, VisualConfig,
    };
    use clap::Parser;

    /// Build a `gui`-subcommand `Cli` the same way `main` does — via clap's arg parser.
    /// `argv` is the tail after `wembed-viewer gui`; every test that needs a graph
    /// passes `"g.edg"` (or omits it deliberately to exercise the TOML-`graph` path).
    fn cli(argv: &[&str]) -> Cli {
        let mut full = vec!["wembed-viewer", "gui"];
        full.extend_from_slice(argv);
        Cli::try_parse_from(full).expect("cli parses")
    }

    /// Same, for the `headless` subcommand (headless-only knobs live there).
    fn cli_headless(argv: &[&str]) -> Cli {
        let mut full = vec!["wembed-viewer", "headless"];
        full.extend_from_slice(argv);
        Cli::try_parse_from(full).expect("cli parses")
    }

    /// The error message from a `resolve` that is expected to fail. `Settings` has no
    /// `Debug` impl, so `Result::unwrap_err` is unavailable — match the `Err` out by hand.
    fn resolve_err(cli: Cli, file: FileConfig) -> String {
        match Settings::resolve(cli, file) {
            Ok(_) => panic!("expected resolve to fail"),
            Err(e) => e.to_string(),
        }
    }

    /// The compiled-in `VisualState::new(Theme)` — the overlay baseline.
    fn default_visual() -> VisualState {
        VisualState::new(VertexColorSource::Theme)
    }

    // ---- defaults only ----

    #[test]
    fn defaults_only_wires_through_to_documented_defaults() {
        // Wiring canary: with an empty TOML and only a graph path, a representative field
        // from each resolve path lands on its documented default. Catches a field wired to
        // the wrong `defaults::` constant or the wrong `unwrap_or`. Not a per-field audit —
        // the override tests below cover the merge behavior.
        let s = Settings::resolve(cli(&["g.edg"]), FileConfig::empty()).expect("resolve ok");

        assert_eq!(s.graph_path, PathBuf::from("g.edg"));
        assert_eq!(s.embedder.iterations, 10_000, "embedder default");
        assert_eq!(s.embedder.dimension, 2, "viewer defaults to 2D");
        assert_eq!(s.sizing.max_pixels, 40.0, "sizing default");
        assert_eq!(s.resolution, (3840, 2160), "resolution default");
        assert_eq!(s.visual, default_visual(), "visual baseline");
        assert!(!s.headless, "gui subcommand is not headless");
        assert!(!s.interactive, "GUI starts immediately by default");
        // LR-schedule defaults mirror wembed's Options — exponential + the numeric knobs.
        assert_eq!(s.embedder.lr_schedule, crate::cli::LrSchedule::Exponential);
        assert_eq!(
            s.embedder.stop_loss_patience, 50,
            "stop_loss_patience default"
        );
        assert_eq!(s.embedder.lr_decay_factor, 0.5, "lr_decay_factor default");
        assert_eq!(s.embedder.lr_growth_factor, 1.0, "growth off by default");
        assert_eq!(
            s.embedder.dimension_hint, -1.0,
            "dimension hint off by default"
        );
        // Stopping-criterion defaults mirror wembed's Options (loss by default).
        assert_eq!(s.embedder.stop_criterion, crate::cli::StopCriterion::Loss);
        assert_eq!(s.embedder.stop_displacement_tol, 3e-4);
    }

    #[test]
    fn gui_interactive_flag_pauses_before_embedding() {
        let s = Settings::resolve(cli(&["g.edg", "--interactive"]), FileConfig::empty())
            .expect("resolve ok");
        assert!(s.interactive);
    }

    // ---- defaults track wembed ----

    #[test]
    fn defaults_track_wembed() {
        // The embedder `defaults::` consts exist only so the clap `[default: N]` help
        // strings in cli.rs have a compile-time value to quote — the C++ header is the real
        // source of truth. This test keeps the hand-written copy honest: each const must
        // equal wembed's C++ default (read over FFI). A field that intentionally diverges
        // from wembed gets an explicit exception line with a reason, never a silent edit.
        // DIMENSION is exempt: it deliberately diverges from wembed's default (4) — see
        // the const's doc comment.
        let w = wembed_sys::Options::default(); // C++ member initializers, via FFI
        assert_eq!(defaults::ITERATIONS, w.max_iterations);
        assert_eq!(defaults::LEARNING_RATE, w.learning_rate);
        assert_eq!(defaults::CENTRE, w.centre_scale);
        assert_eq!(defaults::DYN_BUFFER, w.dynamic_query_buffer);
        assert_eq!(defaults::WARMUP_STEPS, w.warmup_steps);
        assert_eq!(defaults::LR_COOLING_FACTOR, w.lr_cooling_factor);
        assert_eq!(defaults::LR_DECAY_FACTOR, w.lr_decay_factor);
        assert_eq!(defaults::LR_DECAY_THRESHOLD, w.lr_decay_threshold);
        assert_eq!(defaults::LR_ADAPT_PATIENCE, w.lr_adapt_patience);
        assert_eq!(defaults::LR_GROWTH_FACTOR, w.lr_growth_factor);
        assert_eq!(defaults::LR_GROWTH_THRESHOLD, w.lr_growth_threshold);
        assert_eq!(defaults::STOP_DISPLACEMENT_TOL, w.stop_displacement_tol);
        assert_eq!(
            defaults::STOP_DISPLACEMENT_PATIENCE,
            w.stop_displacement_patience
        );
        assert_eq!(defaults::DIMENSION_HINT, w.dimension_hint);
        assert_eq!(defaults::STOP_LOSS_TOL, w.stop_loss_tol);
        assert_eq!(defaults::STOP_LOSS_PATIENCE, w.stop_loss_patience);
        // Enum defaults: map the viewer const through the same cli→FFI conversion production
        // uses, then compare to wembed's default. Catches both a stale const AND a mis-wired
        // mapping. (SEED has no Options counterpart — it's applied via `set_seed`.)
        assert_eq!(
            crate::embedder::lr_schedule_to_ffi(defaults::LR_SCHEDULE),
            w.lr_schedule
        );
        assert_eq!(
            crate::embedder::stop_criterion_to_ffi(defaults::STOP_CRITERION),
            w.stop_criterion
        );
        // the worker hard-codes IndexSprk, wembed's default: a stale discriminant in the Rust
        // mirror shows up here instead of silently selecting another backend
        assert_eq!(w.index_type, wembed_sys::SpatialIndex::IndexSprk);
    }

    // ---- LR-schedule flags ----

    #[test]
    fn lr_schedule_flags_resolve_and_speed_alias() {
        // The lr-prefixed flags plus a schedule and its knobs all land on `EmbedderParams`.
        let s = Settings::resolve(
            cli(&[
                "g.edg",
                "--lr-schedule",
                "loss-adaptive",
                "--speed",
                "3.5",
                "--lr-cooling",
                "0.5",
                "--lr-growth-factor",
                "1.2",
                "--dim-hint",
                "3",
            ]),
            FileConfig::empty(),
        )
        .expect("resolve ok");
        assert_eq!(s.embedder.lr_schedule, crate::cli::LrSchedule::LossAdaptive);
        assert_eq!(s.embedder.learning_rate, 3.5, "--speed → learning_rate");
        assert_eq!(
            s.embedder.lr_cooling_factor, 0.5,
            "--lr-cooling → lr_cooling_factor"
        );
        assert_eq!(s.embedder.lr_growth_factor, 1.2);
        assert_eq!(
            s.embedder.dimension_hint, 3.0,
            "--dim-hint → dimension_hint"
        );

        // `--learning-rate` is still a back-compat alias for `--speed`.
        let s = Settings::resolve(
            cli(&["g.edg", "--learning-rate", "7.0"]),
            FileConfig::empty(),
        )
        .expect("resolve ok");
        assert_eq!(s.embedder.learning_rate, 7.0);
    }

    #[test]
    fn cli_lr_schedule_beats_toml() {
        let file = FileConfig {
            embedder: Some(EmbedderConfig {
                lr_schedule: Some(crate::cli::LrSchedule::Exponential),
                lr_adapt_patience: Some(3),
                ..Default::default()
            }),
            ..FileConfig::empty()
        };
        let s = Settings::resolve(cli(&["g.edg", "--lr-schedule", "loss-adaptive"]), file)
            .expect("resolve ok");
        assert_eq!(
            s.embedder.lr_schedule,
            crate::cli::LrSchedule::LossAdaptive,
            "CLI wins"
        );
        // A TOML-only schedule knob with no CLI counterpart survives.
        assert_eq!(s.embedder.lr_adapt_patience, 3, "TOML-only field survives");
    }

    // ---- TOML overrides defaults ----

    #[test]
    fn toml_overrides_defaults_untouched_fields_keep_defaults() {
        let file = FileConfig {
            embedder: Some(EmbedderConfig {
                seed: Some(7),
                ..Default::default()
            }),
            sizing: Some(FileSizing {
                density_factor: Some(0.9),
                ..Default::default()
            }),
            visual: Some(VisualConfig {
                edge_alpha_scale: Some(0.7),
                ..Default::default()
            }),
            ..FileConfig::empty()
        };
        let s = Settings::resolve(cli(&["g.edg"]), file).expect("resolve ok");

        // Overridden fields win.
        assert_eq!(s.embedder.seed, 7);
        assert_eq!(s.sizing.density_factor, 0.9);
        assert_eq!(s.visual.edge_alpha_scale, 0.7);

        // Untouched siblings keep the compiled-in defaults.
        assert_eq!(s.embedder.iterations, 10_000);
        assert_eq!(s.embedder.learning_rate, 10.0);
        assert_eq!(s.sizing.min_pixels, 3.0);
        assert_eq!(s.sizing.max_pixels, 40.0);
        assert_eq!(
            s.visual.vertex_size_scale,
            default_visual().vertex_size_scale
        );
    }

    // ---- CLI overrides TOML ----

    #[test]
    fn cli_overrides_toml_but_toml_only_field_survives() {
        let file = FileConfig {
            embedder: Some(EmbedderConfig {
                seed: Some(7),
                // TOML-only field: no CLI counterpart passed, so it survives.
                learning_rate: Some(3.5),
                ..Default::default()
            }),
            sizing: Some(FileSizing {
                density_factor: Some(0.1),
                ..Default::default()
            }),
            ..FileConfig::empty()
        };
        let s = Settings::resolve(cli(&["g.edg", "-s", "7", "--density-factor", "0.9"]), file)
            .expect("resolve ok");

        // CLI wins the contested fields.
        assert_eq!(s.embedder.seed, 7);
        assert_eq!(s.sizing.density_factor, 0.9);
        // TOML-only field still comes from the TOML.
        assert_eq!(s.embedder.learning_rate, 3.5);
    }

    #[test]
    fn manual_edge_length_range_disables_auto() {
        let file = FileConfig {
            visual: Some(VisualConfig {
                edge_length_range: Some([0.5, 2.0]),
                ..Default::default()
            }),
            ..FileConfig::empty()
        };
        let s = Settings::resolve(cli(&["g.edg"]), file).expect("resolve ok");
        assert!(!s.visual.edge_length_auto, "explicit range → manual");
        assert_eq!(s.visual.edge_length_range, [0.5, 2.0]);

        let s = Settings::resolve(cli(&["g.edg"]), FileConfig::empty()).expect("resolve ok");
        assert!(s.visual.edge_length_auto, "absent range → auto");
    }

    #[test]
    fn dimension_below_two_errors() {
        let err = resolve_err(cli(&["g.edg", "-d", "1"]), FileConfig::empty());
        assert!(err.contains(">= 2"), "err: {err}");
    }

    // ---- graph resolution ----

    #[test]
    fn graph_from_toml_when_no_positional() {
        let file = FileConfig {
            graph: Some(PathBuf::from("from_toml.edg")),
            ..FileConfig::empty()
        };
        let s = Settings::resolve(cli(&[]), file).expect("resolve ok");
        assert_eq!(s.graph_path, PathBuf::from("from_toml.edg"));
    }

    #[test]
    fn no_graph_from_either_source_errors_mentioning_both() {
        let err = resolve_err(cli(&[]), FileConfig::empty());
        assert!(err.contains("graph path on the CLI"), "err: {err}");
        assert!(err.contains("--config TOML"), "err: {err}");
    }

    #[test]
    fn cli_positional_graph_beats_toml_graph() {
        let file = FileConfig {
            graph: Some(PathBuf::from("from_toml.edg")),
            ..FileConfig::empty()
        };
        let s = Settings::resolve(cli(&["from_cli.edg"]), file).expect("resolve ok");
        assert_eq!(s.graph_path, PathBuf::from("from_cli.edg"));
    }

    // ---- layered or-merge ----

    #[test]
    fn layered_or_merge_cases() {
        // CLI off + TOML true → true.
        let file = FileConfig {
            embedder: Some(EmbedderConfig {
                layered: Some(true),
                ..Default::default()
            }),
            ..FileConfig::empty()
        };
        assert!(
            Settings::resolve(cli(&["g.edg"]), file)
                .unwrap()
                .embedder
                .layered
        );

        // CLI --layered + TOML absent → true.
        assert!(
            Settings::resolve(cli(&["g.edg", "--layered"]), FileConfig::empty())
                .unwrap()
                .embedder
                .layered
        );

        // Both absent → false.
        assert!(
            !Settings::resolve(cli(&["g.edg"]), FileConfig::empty())
                .unwrap()
                .embedder
                .layered
        );
    }

    // ---- vertex color source chain ----

    #[test]
    fn vertex_color_source_chain_cli_beats_toml_beats_builtin() {
        // The only [visual] field that also has a CLI flag, so the only 3-level chain there.
        let toml = |src| FileConfig {
            visual: Some(VisualConfig {
                vertex_color_source: Some(src),
                ..Default::default()
            }),
            ..FileConfig::empty()
        };
        let s = Settings::resolve(
            cli(&["g.edg", "--vertex-color-source", "file"]),
            toml(VertexColorSource::Categories),
        )
        .expect("resolve");
        assert_eq!(
            s.visual.vertex_color,
            VertexColorSource::File,
            "CLI beats TOML"
        );
        let s = Settings::resolve(cli(&["g.edg"]), toml(VertexColorSource::Categories))
            .expect("resolve");
        assert_eq!(
            s.visual.vertex_color,
            VertexColorSource::Categories,
            "TOML beats built-in"
        );
    }

    // ---- visual overlay ----

    #[test]
    fn visual_overlay_only_set_fields_change_rest_from_baseline() {
        let file = FileConfig {
            visual: Some(VisualConfig {
                vertex_size_scale: Some(2.5),
                vertex_alpha_scale: Some(0.25),
                ..Default::default()
            }),
            ..FileConfig::empty()
        };
        let s = Settings::resolve(cli(&["g.edg"]), file).expect("resolve ok");

        let base = default_visual();
        // The two set fields are applied.
        assert_eq!(s.visual.vertex_size_scale, 2.5);
        assert_eq!(s.visual.vertex_alpha_scale, 0.25);
        // Everything else matches the baseline.
        assert_eq!(s.visual.vertex_size, base.vertex_size);
        assert_eq!(s.visual.edge_alpha, base.edge_alpha);
        assert_eq!(s.visual.edge_alpha_scale, base.edge_alpha_scale);
        assert_eq!(s.visual.rotation_deg, base.rotation_deg);
    }

    // ---- [export] pairing + resolution ----

    #[test]
    fn export_width_without_height_errors() {
        let file = FileConfig {
            export: Some(ExportConfig {
                width: Some(1920),
                height: None,
            }),
            ..FileConfig::empty()
        };
        let err = resolve_err(cli(&["g.edg"]), file);
        assert!(
            err.contains("[export] needs both width and height"),
            "err: {err}"
        );
    }

    #[test]
    fn export_both_dims_set_export_dims_and_resolution_picks_them_up() {
        let file = FileConfig {
            export: Some(ExportConfig {
                width: Some(1280),
                height: Some(720),
            }),
            ..FileConfig::empty()
        };
        let s = Settings::resolve(cli(&["g.edg"]), file).expect("resolve ok");
        assert_eq!(s.export_dims, Some((1280, 720)));
        // With no CLI --resolution, the export dims become the render resolution.
        assert_eq!(s.resolution, (1280, 720));
    }

    #[test]
    fn cli_resolution_beats_export_dims() {
        let file = FileConfig {
            export: Some(ExportConfig {
                width: Some(1280),
                height: Some(720),
            }),
            ..FileConfig::empty()
        };
        // Headless-only flags live on the `headless` subcommand.
        let s = Settings::resolve(
            cli_headless(&["g.edg", "-o", "out.png", "--resolution", "800x600"]),
            file,
        )
        .expect("resolve ok");
        assert_eq!(s.resolution, (800, 600));
        // The [export] dims are still surfaced for the interactive panel seed.
        assert_eq!(s.export_dims, Some((1280, 720)));
    }

    // ---- search ----

    #[test]
    fn search_empty_string_is_none_nonempty_is_some() {
        let empty = FileConfig {
            visual: Some(VisualConfig {
                search: Some(String::new()),
                ..Default::default()
            }),
            ..FileConfig::empty()
        };
        assert!(Settings::resolve(cli(&["g.edg"]), empty)
            .unwrap()
            .search
            .is_none());

        let set = FileConfig {
            visual: Some(VisualConfig {
                search: Some("hub".to_string()),
                ..Default::default()
            }),
            ..FileConfig::empty()
        };
        assert_eq!(
            Settings::resolve(cli(&["g.edg"]), set).unwrap().search,
            Some("hub".to_string())
        );
    }

    // ---- render output validation ----

    #[test]
    fn render_with_no_outputs_errors_embed_only_passes() {
        // `-o` went optional so embed-only runs exist; the failure mode to guard is a
        // render that produces NOTHING (silent no-op burning minutes of embedding).
        let err = resolve_err(cli_headless(&["g.edg"]), FileConfig::empty());
        assert!(err.contains("nothing to do"), "err: {err}");

        let s = Settings::resolve(
            cli_headless(&["g.edg", "--output-embedding", "x.emb"]),
            FileConfig::empty(),
        )
        .expect("embed-only resolve ok");
        assert!(s.output.is_none());
        assert_eq!(s.output_embedding, Some(PathBuf::from("x.emb")));
    }

    // ---- legend ----

    #[test]
    fn legend_entry_needs_exactly_one_of_category_or_color() {
        // Both directions of the exactly-one rule: hand-written TOMLs get a named entry
        // in the error instead of a silently-arbitrary swatch.
        for entry in [
            crate::config::LegendEntryConfig {
                label: "both".into(),
                category: Some(1),
                color: Some("#ff0000".into()),
            },
            crate::config::LegendEntryConfig {
                label: "neither".into(),
                category: None,
                color: None,
            },
        ] {
            let file = FileConfig {
                legend: Some(crate::config::LegendConfig {
                    entries: vec![entry],
                    ..Default::default()
                }),
                ..FileConfig::empty()
            };
            let err = resolve_err(cli(&["g.edg"]), file);
            assert!(
                err.contains("exactly one of `category` or `color`"),
                "err: {err}"
            );
        }
    }

    #[test]
    fn legend_resolves_with_documented_defaults() {
        let file = FileConfig {
            legend: Some(crate::config::LegendConfig {
                entries: vec![crate::config::LegendEntryConfig {
                    label: "hubs".into(),
                    category: Some(3),
                    color: None,
                }],
                ..Default::default()
            }),
            ..FileConfig::empty()
        };
        let s = Settings::resolve(cli(&["g.edg"]), file).expect("resolve ok");
        let legend = s.legend.expect("legend present");
        assert_eq!(legend.entries.len(), 1);
        assert_eq!(legend.entries[0].1, LegendSwatch::Category(3));
        assert_eq!(legend.size, 24.0, "documented default");
        assert_eq!(legend.anchor, crate::text::Anchor::BottomLeft);
        assert!(legend.color.is_none(), "text color stays theme-driven");

        // No entries → no legend at all.
        let s = Settings::resolve(cli(&["g.edg"]), FileConfig::empty()).expect("resolve ok");
        assert!(s.legend.is_none());
    }

    #[test]
    fn legend_names_and_entries_are_mutually_exclusive() {
        let file = FileConfig {
            legend: Some(crate::config::LegendConfig {
                names: Some(PathBuf::from("cats.txt")),
                entries: vec![crate::config::LegendEntryConfig {
                    label: "x".into(),
                    category: Some(0),
                    color: None,
                }],
                ..Default::default()
            }),
            ..FileConfig::empty()
        };
        let err = resolve_err(cli(&["g.edg"]), file);
        assert!(err.contains("at most one"), "err: {err}");
    }

    #[test]
    fn legend_entries_from_conflicts_with_inline_entries_and_names() {
        for other in [
            crate::config::LegendConfig {
                entries_from: Some(PathBuf::from("legend.toml")),
                entries: vec![crate::config::LegendEntryConfig {
                    label: "x".into(),
                    category: Some(0),
                    color: None,
                }],
                ..Default::default()
            },
            crate::config::LegendConfig {
                entries_from: Some(PathBuf::from("legend.toml")),
                names: Some(PathBuf::from("cats.txt")),
                ..Default::default()
            },
        ] {
            let file = FileConfig {
                legend: Some(other),
                ..FileConfig::empty()
            };
            let err = resolve_err(cli(&["g.edg"]), file);
            assert!(err.contains("at most one"), "err: {err}");
        }
    }

    #[test]
    fn legend_entries_from_defers_to_main_with_empty_entries() {
        let file = FileConfig {
            legend: Some(crate::config::LegendConfig {
                entries_from: Some(PathBuf::from("legend.toml")),
                ..Default::default()
            }),
            ..FileConfig::empty()
        };
        let s = Settings::resolve(cli(&["g.edg"]), file).expect("resolve ok");
        assert_eq!(s.legend_entries_from, Some(PathBuf::from("legend.toml")));
        let legend = s.legend.expect("deferred legend present");
        assert!(legend.entries.is_empty(), "main fills these later");
    }

    #[test]
    fn cli_legend_entries_overrides_config_source() {
        let file = FileConfig {
            legend: Some(crate::config::LegendConfig {
                names: Some(PathBuf::from("cats.txt")),
                ..Default::default()
            }),
            ..FileConfig::empty()
        };
        let s = Settings::resolve(cli(&["g.edg", "--legend-entries", "cli.toml"]), file)
            .expect("resolve ok");
        assert!(s.legend_names.is_none(), "CLI supersedes config names");
        assert_eq!(s.legend_entries_from, Some(PathBuf::from("cli.toml")));
        assert!(s.legend.is_some());
    }

    #[test]
    fn legend_entries_from_names_skips_blanks_without_shifting_ids() {
        // Line index IS the category id — a naive filter-before-enumerate would silently
        // recolor every category after a blank line.
        let names: Vec<String> = ["Physics", "", "  ", "Biology"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let entries = legend_entries_from_names(&names);
        assert_eq!(
            entries,
            vec![
                ("Physics".to_string(), LegendSwatch::Category(0)),
                ("Biology".to_string(), LegendSwatch::Category(3)),
            ]
        );
    }

    // ---- density ----

    #[test]
    fn density_off_by_default_on_by_flag_or_section() {
        // No section, no flag → None.
        let s = Settings::resolve(cli(&["g.edg"]), FileConfig::empty()).expect("resolve ok");
        assert!(s.density.is_none());

        // Bare --density flag → defaults.
        let s = Settings::resolve(cli(&["g.edg", "--density"]), FileConfig::empty()).expect("ok");
        let d = s.density.expect("density on");
        assert_eq!(d.sigma, 1.2);
        // absolute 0, not a percentile — a percentile knee floor-crushes by default
        assert_eq!(d.knee, Ok(0.0));
        assert_eq!(d.white, Err(99.5));
        assert!(d.ramp.is_empty(), "no ramp → theme fallback later");

        // Bare [density] section → also on.
        let file = FileConfig {
            density: Some(crate::config::DensityConfig::default()),
            ..FileConfig::empty()
        };
        assert!(Settings::resolve(cli(&["g.edg"]), file)
            .unwrap()
            .density
            .is_some());
    }

    #[test]
    fn density_cli_overrides_section_and_explicit_anchors() {
        let file = FileConfig {
            density: Some(crate::config::DensityConfig {
                sigma: Some(2.0),
                knee: Some(3.0),
                white_percentile: Some(98.0),
                ramp: Some(vec!["#000000".into(), "#ffffff".into()]),
                ..Default::default()
            }),
            ..FileConfig::empty()
        };
        let s = Settings::resolve(
            cli(&[
                "g.edg",
                "--density-sigma",
                "0.5",
                "--density-ramp",
                "#111111,#eeeeee",
            ]),
            file,
        )
        .expect("resolve ok");
        let d = s.density.expect("density on");
        assert_eq!(d.sigma, 0.5, "CLI sigma wins");
        assert_eq!(d.knee, Ok(3.0), "explicit knee value beats percentile");
        assert_eq!(d.white, Err(98.0));
        assert_eq!(
            d.ramp,
            vec![[0x11, 0x11, 0x11, 0xff], [0xee, 0xee, 0xee, 0xff]]
        );
    }

    #[test]
    fn density_bad_curve_and_short_ramp_error() {
        let file = FileConfig {
            density: Some(crate::config::DensityConfig {
                curve: Some("sqrt".into()),
                ..Default::default()
            }),
            ..FileConfig::empty()
        };
        assert!(resolve_err(cli(&["g.edg"]), file).contains("only \"log\""));

        let file = FileConfig {
            density: Some(crate::config::DensityConfig {
                ramp: Some(vec!["#000000".into()]),
                ..Default::default()
            }),
            ..FileConfig::empty()
        };
        assert!(resolve_err(cli(&["g.edg"]), file).contains("at least 2"));
    }

    // ---- camera passthrough ----

    #[test]
    fn camera_passed_through_verbatim() {
        let cam = CameraConfig {
            center: Some([1.5, -2.5]),
            half: Some(42.0),
            rotation_deg: Some(15.0),
        };
        let file = FileConfig {
            camera: Some(cam.clone()),
            ..FileConfig::empty()
        };
        let s = Settings::resolve(cli(&["g.edg"]), file).expect("resolve ok");
        // resolve does NOT strip on non-replay — that's main's job.
        assert_eq!(s.camera, Some(cam));
    }
}
