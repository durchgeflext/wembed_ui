//! Headless batch runner: run the embedder to completion (or load an `--input-embedding`),
//! then write an `.emb` and/or render one PNG at a fixed resolution, and exit. No window,
//! no egui, no event loop. An embed-only invocation (no `-o`) returns before any wgpu
//! init, so it works on GPU-less machines and parallelizes freely.
//!
//! Shares [`SceneRenderer`] + [`crate::exporter`] with the interactive path so pipelines
//! and colorspace handling stay in one place. The only headless-specific bits are:
//!   - wgpu init with `compatible_surface: None`
//!   - a fixed target format (`Rgba8UnormSrgb` — sRGB target so the clear color and shader
//!     outputs match interactive-mode behavior, no BGRA byte-swap needed)
//!   - a drain loop over the [`TopologyProducer`] that blocks until `is_finished()` and
//!     tracks the latest published topology + positions
//!   - a reference-height rule for the pixel-clamp scaling — see
//!     [`crate::framing::REFERENCE_HEIGHT`].
//!
//! Failure modes are surfaced as `anyhow::Error` back to `main`, which exits non-zero. No
//! partial writes: PNG encoding + save happens after the render fully readbacks.

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result};

use crate::app::VisualState;
use crate::attrs::Attributes;
use crate::camera::Camera;
use crate::config::CameraConfig;
use crate::embedder::{preprocess_topology, TopologyProducer, Update};
use crate::exporter::{export_png, ExportConfig};
use crate::framing::{
    best_rotation_deg_for_aspect, SizingConfig, View, Q_BUFFER, Q_HI, Q_LO, REFERENCE_HEIGHT,
};
use crate::highlight::Highlight;
use crate::projection::Projector;
use crate::scene::{Scene, Topology};
use crate::scene_renderer::SceneRenderer;
use crate::theme::{wgpu_clear_color, ThemeRegistry};

pub struct HeadlessConfig {
    /// PNG destination. `None` = embed-only run: save the `.emb`, never touch the GPU.
    pub output: Option<PathBuf>,
    pub resolution: (u32, u32),
    /// Explicit CLI `--rotation-deg`, if passed. Full priority chain resolved here:
    /// `--autorotate` > this > `[camera].rotation_deg` > 0.
    pub rotation_deg_cli: Option<f32>,
    pub autorotate: bool,
    /// Optional path to save the final `.emb` file (`--output-embedding`); combined
    /// with `output` we embed once, save, and render in the same run.
    pub output_embedding: Option<PathBuf>,
    /// Resolved visual state (color mode + channel toggles + slider scales) — same values
    /// an interactive session would start with, so the two paths produce identical output
    /// at identical settings.
    pub visual: VisualState,
    /// TOML `[camera]` section. `main` guarantees `Some` only when the producer replays
    /// an `.emb`.
    pub camera: Option<CameraConfig>,
    /// `[visual].search` — reproduces search-highlight fading in the render.
    pub search: Option<String>,
    /// Resolved `[text]` section (+ `--title`).
    pub text: crate::settings::TextParams,
    /// Resolved `[legend]` section — rows drawn into the export like the title.
    pub legend: Option<crate::settings::LegendParams>,
    /// Resolved `[density]` params. `Some` → render via the density heatmap pipeline
    /// (accumulate → blur → tonemap) instead of the normal points/edges pass.
    pub density: Option<crate::settings::DensityParams>,
    /// Title font, loaded and validated by `main`.
    pub font: ab_glyph::FontArc,
}

pub fn run(
    mut producer: Box<dyn TopologyProducer>,
    attributes: Attributes,
    themes: ThemeRegistry,
    sizing: SizingConfig,
    config: HeadlessConfig,
) -> Result<()> {
    // ---- 1. Drain the producer to completion. ----
    println!("headless: waiting for producer to finish…");
    let (raw_positions, topology) = drain_producer(&mut *producer)?;
    let n = raw_positions.len() / topology.dim;
    println!(
        "headless: producer finished — {n} vertices (dim {}), {} edges, {} iterations",
        topology.dim,
        topology.edges.len(),
        producer.iteration()
    );

    // Optional output-embedding: do it before rendering so a render failure doesn't lose
    // the (potentially expensive) embedding result. Full d-dim — projection is display-only.
    if let Some(path) = &config.output_embedding {
        crate::loaded_embedding::save_embedding_file(
            path,
            &raw_positions,
            topology.dim,
            &topology.weights,
        )
        .with_context(|| format!("save embedding to {}", path.display()))?;
        println!("headless: saved embedding to {}", path.display());
    }

    // Embed-only run: no PNG requested → done, without ever touching wgpu.
    let Some(png_path) = config.output.clone() else {
        return Ok(());
    };

    // ---- 1b. Project to the 2D view — same Projector the interactive path runs. ----
    let mut projector = Projector::new(topology.dim, config.visual.projection);
    let mut positions = Vec::new();
    projector.update_view(&raw_positions, &mut positions);
    if let Some(ev) = projector.explained_variance() {
        println!(
            "headless: {}D → 2D via {:?}, view captures {:.1}% of variance",
            topology.dim,
            config.visual.projection,
            ev * 100.0
        );
    }

    // ---- 2. Boot wgpu without a surface. ----
    let (device, queue) = pollster::block_on(init_wgpu_headless())?;
    // sRGB target: exporter path already handles this correctly (no BGRA byte-swap needed
    // for Rgba8UnormSrgb — see `is_bgra_format` in exporter.rs).
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut scene_renderer = SceneRenderer::new(device, queue, format);

    // Apply the resolved visual state — same values an interactive session starts with.
    let visual = config.visual;
    let theme = themes.current();
    // Per-edge colors for the `file` source (exact edge-count match only); `is_some()`
    // drives the file→theme downgrade so the shader's file branch only reads real data.
    let edge_colors = attributes.edge_colors_for(topology.edges.len());
    scene_renderer.set_channels(&visual.channels(edge_colors.is_some(), None));
    scene_renderer.set_theme_colors(theme);
    scene_renderer.set_depth_fade(visual.depth_fade);

    // ---- 3. Build the Scene from the drained topology. ----
    crate::scene::check_device_limits(scene_renderer.device(), n, topology.edges.len())?;
    let colors = attributes.colors_for(
        visual.vertex_color,
        n,
        Some(theme),
        visual.uncategorized_alpha,
    );
    scene_renderer.set_vertex_color_source(
        visual
            .vertex_color
            .effective(colors.is_some())
            .shader_value(),
    );
    let scene = Scene::new(
        scene_renderer.device(),
        scene_renderer.scene_bind_group_layout(),
        &Topology {
            positions: &positions,
            edges: &topology.edges,
            colors: colors.as_deref(),
            size_multipliers: Some(&topology.size_multipliers),
            edge_alphas: Some(&topology.edge_alphas),
            edge_colors,
        },
    );
    if !projector.is_identity() {
        let mut depth = Vec::new();
        projector.depth_values(&raw_positions, &mut depth);
        scene.write_depth(scene_renderer.queue(), &depth);
    }
    // Stress ratios for the `length` edge-color source — full d-dim, same math as App —
    // plus the auto colormap window (re-pushed via set_channels; the setter dedups).
    if visual.edge_color == crate::cli::EdgeColorSource::Length && !topology.edges.is_empty() {
        let inv_norms = crate::embedder::edge_inv_norms(&topology.edge_alphas, topology.dim);
        let mut stress = Vec::new();
        crate::embedder::edge_length_ratios(
            &raw_positions,
            topology.dim,
            &topology.edges,
            &inv_norms,
            &mut stress,
        );
        let auto = visual
            .edge_length_auto
            .then(|| crate::embedder::edge_length_log_range(&stress))
            .flatten();
        scene_renderer.set_channels(&visual.channels(edge_colors.is_some(), auto));
        if let Some((lo, hi)) = auto {
            println!(
                "headless: auto stress window r ∈ [{:.3}, {:.3}]",
                lo.exp(),
                hi.exp()
            );
        }
        scene.write_edge_lengths(scene_renderer.queue(), &stress);
    }

    // Search-highlight mask: a `[visual].search` query fades non-matching vertices in the
    // interactive view, so a reproducing render must apply the same mask. Mirrors App's
    // Highlight wiring minus hover.
    if let Some(query) = config.search.as_deref().filter(|q| !q.is_empty()) {
        let labels = attributes.labels_for(n).map(<[String]>::to_vec);
        let mut highlight = Highlight::new();
        highlight.set_topology(n, &topology.edges, labels.as_deref());
        highlight.set_query(query);
        if let Some(mask) = highlight.drain_dirty_mask() {
            scene.write_mask(scene_renderer.queue(), mask);
        }
        let matches = highlight.search_match_count();
        println!("headless: search {query:?} highlighted {matches} vertices");
    }

    // ---- 4. Camera + rotation. ----
    // `[camera]` center/zoom (replay-only; enforced by main) beats auto-fit. Rotation
    // priority: --autorotate > --rotation-deg > [camera].rotation_deg > 0.
    let fit = Camera::fit_to_quantile_flat(&positions, Q_LO, Q_HI, Q_BUFFER);
    let mut camera = match config.camera.as_ref().and_then(|c| c.center.zip(c.half)) {
        Some((center, half)) => fit.with_view(center, half),
        None => fit,
    };
    let rot_deg = if config.autorotate {
        best_rotation_deg_for_aspect(
            &positions,
            config.resolution.0 as f32 / config.resolution.1.max(1) as f32,
        )
    } else {
        config
            .rotation_deg_cli
            .or(config.camera.as_ref().and_then(|c| c.rotation_deg))
            .unwrap_or(0.0)
    };
    camera.set_rotation(rot_deg.to_radians());
    camera.set_flip(visual.flip_x, visual.flip_y);
    if config.autorotate {
        println!(
            "headless: autorotated to {rot_deg:.0}° for aspect {}:{}",
            config.resolution.0, config.resolution.1
        );
    }

    // ---- 5. Render + save. ----
    let (w, h) = config.resolution;
    let view = View::new(&camera, n, w, h, &sizing);
    let export_config = ExportConfig {
        width: w,
        height: h,
        msaa: true,
        clear_color: wgpu_clear_color(theme.background()),
        path: png_path.clone(),
    };
    // Title at export scale — identical logic to the interactive export path.
    let export_text = {
        let title = config.text.title.as_deref().unwrap_or("").trim();
        let px = config.text.size * (h as f32 / (REFERENCE_HEIGHT as f32));
        if title.is_empty() || px < 1.0 {
            None
        } else {
            let color = config.text.color.unwrap_or_else(|| theme.foreground());
            crate::text::rasterize(&config.font, title, px, color).map(|bmp| {
                let quad = scene_renderer.make_text_quad(&bmp);
                let k = px / config.text.size;
                let (x, y) = crate::text::anchor_position(
                    config.text.anchor,
                    (0.0, 0.0, w as f32, h as f32),
                    bmp.width as f32,
                    bmp.height as f32,
                    config.text.margin_x.map_or(0.5 * px, |m| m * k),
                    config.text.margin_y.map_or(0.5 * px, |m| m * k),
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
    // Legend at export scale — same 1080p-reference rule as the title.
    let export_legend = config.legend.as_ref().and_then(|params| {
        let px = params.size * (h as f32 / (REFERENCE_HEIGHT as f32));
        if px < 1.0 {
            return None;
        }
        let color = params.color.unwrap_or_else(|| theme.foreground());
        crate::text::rasterize_legend(&config.font, &params.rows(theme), px, color).map(|bmp| {
            let quad = scene_renderer.make_text_quad(&bmp);
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
    let mut export_texts: Vec<(&crate::scene_renderer::TextQuad, [f32; 4])> = Vec::new();
    if let Some((q, ndc)) = export_text.as_ref() {
        export_texts.push((q, *ndc));
    }
    if let Some((q, ndc)) = export_legend.as_ref() {
        export_texts.push((q, *ndc));
    }
    let render_start = std::time::Instant::now();
    if let Some(params) = config.density.as_ref() {
        // Density heatmap path: accumulate → blur → tonemap. The scene + channel uniforms are
        // already set above; the density pipelines read the same geometry inputs.
        let mut density = crate::density::DensityRenderer::new(
            scene_renderer.device().clone(),
            scene_renderer.queue().clone(),
            scene_renderer.uniform_bind_group_layout(),
            scene_renderer.scene_bind_group_layout(),
            format,
        );
        let ramp = crate::density::effective_ramp(
            params,
            theme.background(),
            crate::density::pick_ramp_accent(&theme.accents()),
            theme.foreground(),
        );
        crate::exporter::export_png_density(
            &mut scene_renderer,
            &mut density,
            &scene,
            &view,
            params,
            &ramp,
            &export_config,
            &export_texts,
        )
        .with_context(|| format!("density export PNG to {}", png_path.display()))?;
    } else {
        export_png(
            &mut scene_renderer,
            &scene,
            &view,
            &export_config,
            &export_texts,
        )
        .with_context(|| format!("export PNG to {}", png_path.display()))?;
    }
    println!(
        "headless: wrote {} ({w}×{h}) in {:.2}s",
        png_path.display(),
        render_start.elapsed().as_secs_f32()
    );

    Ok(())
}

/// Block on producer polling until `is_finished()` returns true. Tracks the latest published
/// topology (there may be several across a layered embedder's coarse→fine transitions) and
/// the latest positions.
///
/// Errors with the producer's message if it fails (e.g. an unreadable graph), or if it
/// somehow finishes without ever publishing a topology.
fn drain_producer(
    producer: &mut dyn TopologyProducer,
) -> Result<(Vec<f32>, crate::embedder::Topology)> {
    let mut positions: Vec<f32> = Vec::new();
    let mut topology: Option<crate::embedder::Topology> = None;
    let mut last_report = std::time::Instant::now();
    loop {
        if let Some(e) = producer.error() {
            anyhow::bail!(e);
        }
        match producer.poll(&mut positions) {
            Update::Nothing => {}
            Update::Positions => {}
            Update::Topology(t) => {
                topology = Some(t);
            }
        }
        if producer.is_finished() {
            // One more poll after the finished flag flips — the worker may have written a
            // final positions_gen bump between our last poll and its finish signal.
            match producer.poll(&mut positions) {
                Update::Nothing | Update::Positions => {}
                Update::Topology(t) => {
                    topology = Some(t);
                }
            }
            break;
        }
        // wembed's CLI cadence
        if last_report.elapsed() >= Duration::from_secs(30) {
            if let Some(p) = producer.progress() {
                println!("headless:   … {}", crate::embedder::format_progress(&p));
            }
            last_report = std::time::Instant::now();
        }
        // 10ms is small relative to a single embedder step (typically 20–200ms on a large
        // graph) so we don't miss much freshness by sleeping.
        std::thread::sleep(Duration::from_millis(10));
    }
    let topology = topology.context("producer finished without publishing a topology")?;
    // Preprocess_topology is what the worker does before publishing; verify shape here so
    // the exporter's Scene::new assertions can't trip on a corrupted producer output.
    debug_assert_eq!(topology.size_multipliers.len(), topology.weights.len());
    debug_assert_eq!(topology.edge_alphas.len(), topology.edges.len());
    // Silence unused warning if debug_assertions are off.
    let _ = preprocess_topology as fn(&[f32], &[[u32; 2]]) -> (Vec<f32>, Vec<f32>);
    Ok((positions, topology))
}

/// Async wgpu init with no surface. Mirrors `Renderer::new`'s adapter+device request but
/// omits `compatible_surface` (we're not presenting) and drops the surface-format probe.
async fn init_wgpu_headless() -> Result<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..Default::default()
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .context("no suitable GPU adapter (headless)")?;
    // Same limits as the interactive path (`render::request_limits`) — 8 vertex-stage
    // SSBOs, adapter-level storage/buffer size caps, texture dims for 8K exports.
    let limits = crate::render::request_limits(&adapter);
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("wembed-viewer headless device"),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        })
        .await
        .context("request device (headless)")?;
    crate::render::print_granted_limits(&device);
    Ok((device, queue))
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    /// A vertex's hex must export as exactly that hex. Catches double sRGB encoding (colours
    /// uploaded raw into the *Srgb target — `#70b6fd` exported as `#b1dcfe`). Needs a GPU
    /// adapter; passes vacuously without one (e.g. in the Nix build sandbox).
    #[test]
    fn exported_vertex_pixel_matches_its_hex() {
        if pollster::block_on(init_wgpu_headless()).is_err() {
            eprintln!("skipped: no GPU adapter");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let png = dir.path().join("out.png");
        let cli = crate::cli::Cli::parse_from([
            "wembed-viewer",
            "headless",
            "unused.edg",
            "--vertex-color-source",
            "file",
            "--resolution",
            "64x64",
            // floor far above the 64px frame so the disks cover the centre pixel
            "--point-min-pixels",
            "400",
            "--point-max-pixels",
            "800",
            "-o",
            png.to_str().unwrap(),
        ]);
        let settings =
            crate::settings::Settings::resolve(cli, crate::config::FileConfig::empty()).unwrap();
        let rgba = [0x70, 0xb6, 0xfd, 0xff];
        // three collinear vertices so quantile framing has extent; the middle one is centred
        let producer = Box::new(crate::loaded_embedding::LoadedEmbedding::new(
            vec![-1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            2,
            vec![1.0; 3],
            vec![],
        ));
        let attributes = Attributes::new(
            Some(vec![crate::theme::pack_rgba(rgba); 3]),
            None,
            None,
            None,
        );
        let config = HeadlessConfig {
            output: Some(png.clone()),
            resolution: settings.resolution,
            rotation_deg_cli: None,
            autorotate: false,
            output_embedding: None,
            visual: settings.visual,
            camera: None,
            search: None,
            text: settings.text.clone(),
            legend: None,
            density: None,
            font: crate::text::load_font(None).unwrap(),
        };
        // empty dir → the compiled-in fallback theme
        let themes = ThemeRegistry::from_dir(dir.path(), None).unwrap();
        run(
            producer,
            attributes,
            themes,
            settings.sizing.clone(),
            config,
        )
        .unwrap();

        let img = image::open(&png).unwrap().to_rgba8();
        let px = img.get_pixel(32, 32).0;
        for c in 0..3 {
            assert!(
                px[c].abs_diff(rgba[c]) <= 1,
                "exported {px:02x?}, authored {rgba:02x?}"
            );
        }
    }

    /// A missing graph must come back as an error carrying wembed's message. Catches a
    /// dropped `Result` on the FFI (a C++ throw would terminate the process) and a worker
    /// failure that never reaches the consumer (the drain would spin forever).
    #[test]
    fn missing_graph_is_an_error_not_an_abort() {
        let cli =
            crate::cli::Cli::parse_from(["wembed-viewer", "headless", "unused.edg", "-o", "x.png"]);
        let settings =
            crate::settings::Settings::resolve(cli, crate::config::FileConfig::empty()).unwrap();
        let mut producer = crate::embedder::spawn(crate::embedder::SpawnConfig {
            graph_path: "does/not/exist.edg".into(),
            embedder: settings.embedder,
            start_paused: false, //headless has no reason to pause and no possibility to resume
        })
        .unwrap();
        let Err(err) = drain_producer(&mut producer) else {
            panic!("a missing graph loaded");
        };
        let err = err.to_string();
        assert!(err.contains("does/not/exist.edg"), "{err}");
    }
}
