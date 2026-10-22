//! wembed-viewer: interactive viewer for the wembed graph embedder.

mod app;
mod attrs;
mod camera;
mod cli;
mod config;
mod convergence;
mod density;
mod embedder;
mod exporter;
mod framing;
mod headless;
mod highlight;
mod inspection;
mod loaded_embedding;
mod pick;
mod projection;
mod render;
mod scene;
mod scene_renderer;
mod settings;
mod text;
mod theme;

use anyhow::Result;
use clap::Parser;
use winit::event_loop::{ControlFlow, EventLoop};

use crate::app::AppInit;
use crate::attrs::Attributes;
use crate::embedder::SpawnConfig;
use crate::headless::HeadlessConfig;
use crate::loaded_embedding::{parse_edges_file, parse_embedding_file, LoadedEmbedding};
use crate::settings::Settings;
use crate::theme::ThemeRegistry;

fn main() -> Result<()> {
    let args = cli::Cli::parse();

    // Layer the config: built-in defaults → TOML (if any) → explicit CLI flags.
    let file = match &args.common().config {
        Some(path) => config::FileConfig::load(path)?,
        None => config::FileConfig::empty(),
    };
    let mut settings = Settings::resolve(args, file)?;

    // `[camera]` only applies when replaying an `.emb` — a live embedder's positions
    // drift (and the layered embedder even changes vertex counts), so a fixed framing is
    // meaningless there. Strip it here with a warning so downstream never re-checks.
    if settings.camera.is_some() && settings.input_embedding.is_none() {
        eprintln!(
            "warning: [camera] section ignored — it only applies when replaying an --input-embedding"
        );
        settings.camera = None;
    }

    println!(
        "loading {} (iters={} seed={} lr={} cool={} centre={})",
        settings.graph_path.display(),
        settings.embedder.iterations,
        settings.embedder.seed,
        settings.embedder.learning_rate,
        settings.embedder.lr_cooling_factor,
        settings.embedder.centre,
    );

    // Parse attribute files eagerly, before the (potentially slow) graph load. A typo in
    // a hex color or JSON label should fail immediately rather than after a 30-second wait.
    let colors = settings
        .colors
        .as_deref()
        .map(attrs::parse_colors_file)
        .transpose()?;
    let labels = settings
        .labels
        .as_deref()
        .map(attrs::parse_labels_file)
        .transpose()?;
    let categories = settings
        .categories
        .as_deref()
        .map(attrs::parse_categories_file)
        .transpose()?;
    // Per-edge colors reuse the vertex-colors hex parser (line i = edge i).
    let edge_colors = settings
        .edge_colors
        .as_deref()
        .map(attrs::parse_colors_file)
        .transpose()?;
    let attributes = Attributes::new(colors, categories, labels, edge_colors);

    // Deferred legend sources (`[legend].names`, `[legend].entries_from` /
    // `--legend-entries`) — parse the referenced file here, same eager fail-fast stage as
    // the attribute files above. resolve() guarantees at most one is set. Empty → no legend.
    if let Some(path) = &settings.legend_names {
        let names = attrs::parse_category_names_file(path)?;
        let entries = settings::legend_entries_from_names(&names);
        if entries.is_empty() {
            settings.legend = None;
        } else if let Some(legend) = settings.legend.as_mut() {
            legend.entries = entries;
        }
    } else if let Some(path) = &settings.legend_entries_from {
        let entries =
            settings::legend_entries_from_configs(&config::LegendEntriesFile::load(path)?)?;
        if entries.is_empty() {
            settings.legend = None;
        } else if let Some(legend) = settings.legend.as_mut() {
            legend.entries = entries;
        }
    }

    // Title font: load + validate once, shared by interactive and headless. Fail fast on
    // a bad `[text].font` before the (potentially slow) graph load.
    let title_font = text::load_font(settings.text.font.as_deref())?;

    // Themes: load once at startup. Failures inside `from_dir` are warnings (skip file);
    // the compiled-in fallback keeps the picker non-empty.
    let themes = ThemeRegistry::from_dir(&settings.themes_dir, settings.theme.as_deref())?;

    // Producer: either load a precomputed embedding (skips the embedder entirely) or
    // spawn a live embedder worker.
    let producer: Box<dyn embedder::TopologyProducer> =
        if let Some(emb_path) = &settings.input_embedding {
            let (positions, dim, weights) = parse_embedding_file(emb_path)?;
            let n = weights.len();
            let edges = parse_edges_file(&settings.graph_path, n)?;
            println!(
                "loaded embedding: {n} vertices (dim {dim}), {} edges (embedder skipped)",
                edges.len()
            );
            Box::new(LoadedEmbedding::new(positions, dim, weights, edges))
        } else {
            let handle = embedder::spawn(SpawnConfig {
                graph_path: settings.graph_path.clone(),
                embedder: settings.embedder,
                start_paused: settings.interactive,
            })?;
            Box::new(handle)
        };

    if settings.headless {
        let config = HeadlessConfig {
            // `None` = embed-only run; resolve() guarantees output_embedding is Some then.
            output: settings.output.clone(),
            resolution: settings.resolution,
            rotation_deg_cli: settings.rotation_deg_cli,
            autorotate: settings.autorotate,
            output_embedding: settings.output_embedding.clone(),
            visual: settings.visual,
            camera: settings.camera.clone(),
            search: settings.search.clone(),
            text: settings.text.clone(),
            legend: settings.legend.clone(),
            density: settings.density.clone(),
            font: title_font,
        };
        return headless::run(
            producer,
            attributes,
            themes,
            settings.sizing.clone(),
            config,
        );
    }

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let init = AppInit {
        visual: settings.visual,
        search: settings.search.clone(),
        camera: settings.camera.clone(),
        export_dims: settings.export_dims,
        text: settings.text.clone(),
        legend: settings.legend.clone(),
        density: settings.density.clone(),
        font: title_font,
        session: settings.session_info(),
    };
    let mut app = app::App::new(producer, settings.sizing.clone(), attributes, themes, init);
    event_loop.run_app(&mut app)?;
    Ok(())
}
