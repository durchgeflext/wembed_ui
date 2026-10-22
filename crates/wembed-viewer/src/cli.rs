//! Command-line arguments. Single source of truth for the *flag surface*; the actual
//! defaults live in [`crate::settings`], because precedence is `defaults → --config TOML
//! → explicitly-passed CLI flags` and clap can't express the middle layer. That's why
//! every tunable here is `Option<T>` with no `default_value`: `None` means "the user
//! didn't say", which the resolve step needs to distinguish from "the user typed the
//! default value on purpose". Defaults quoted in the doc comments mirror
//! `settings::defaults` — keep them in sync.
//!
//! Two subcommands share [`CommonArgs`]:
//! - `gui` — interactive viewer (window, egui, live embedding).
//! - `headless` — embed and/or render, driven by which outputs are requested (`-o` PNG,
//!   `--output-embedding` .emb, at least one). Headless-only knobs (`--output`,
//!   `--resolution`, `--rotation-deg`, `--autorotate`, `--output-embedding`) live only
//!   here, so clap itself rejects them on `gui`.
//!
//! Bool flags (`--layered`, `--autorotate`) stay plain `bool`: they can only be switched
//! ON from the CLI, so `false` is never "explicitly passed" and or-merging with the TOML
//! is exactly equivalent to an `Option<bool>`.

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};

/// Source of per-vertex color (`--vertex-color-source`, the `[visual] vertex_color_source`
/// TOML key and the vertices' color dropdown). Falls back to `theme` wherever the selected
/// file isn't loaded or its line count doesn't match the current vertex count (e.g. a
/// layered run's coarser layers). Converts to the shader uniform via
/// [`VertexColorSource::shader_value`].
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum VertexColorSource {
    /// The theme's vertex color (`base09`) for every vertex.
    #[default]
    Theme,
    /// Theme accents by category id, from `--categories`.
    Categories,
    /// Per-vertex colors from `--colors`.
    File,
}

impl VertexColorSource {
    /// Value written into the shader `vertex_color_source` uniform — whether the per-vertex
    /// colors buffer is read. Keep in lockstep with `points.wgsl`'s `VERTEX_COLOR_*`.
    pub fn shader_value(self) -> u32 {
        match self {
            VertexColorSource::Theme => 0,
            VertexColorSource::Categories | VertexColorSource::File => 1,
        }
    }

    /// The source the shader actually draws: `self`, or `Theme` when it has no colors for
    /// the current vertex count.
    pub fn effective(self, has_colors: bool) -> Self {
        if has_colors {
            self
        } else {
            VertexColorSource::Theme
        }
    }
}

/// Source of a per-element scalar channel: vertex size (`[visual] vertex_size_source`) and
/// edge alpha (`[visual] edge_alpha_source`). `weight` multiplies in `sqrt(w/mean_w)` for
/// vertex size and `1/(w_a·w_b)` for edge alpha.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ScalarSource {
    Constant,
    Weight,
}

impl ScalarSource {
    /// Keep in lockstep with `SIZE_*` in points.wgsl and `EDGE_ALPHA_*` in lines.wgsl.
    pub fn shader_value(self) -> u32 {
        match self {
            ScalarSource::Constant => 0,
            ScalarSource::Weight => 1,
        }
    }
}

/// Source of per-edge color (the `[visual] edge_color_source` TOML key + the "edge color"
/// UI dropdown). No CLI flag — it's chosen in the TOML or the UI; the `file` variant is fed
/// by the separate `--edge-colors` / `[attrs] edge_colors` file. Derives serde (kebab-case)
/// so the TOML strings and the runtime [`crate::app::VisualState`] share one type; converts
/// to the shader uniform via [`EdgeColorSource::shader_value`].
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum EdgeColorSource {
    /// Fixed theme edge color (the theme's `edge` key, else its `base03`). The default when a
    /// config omits the field.
    #[default]
    Theme,
    /// Each edge inherits its endpoint vertex colors' RGB, interpolated along the segment.
    /// Alpha stays on the `edge_alpha_scale × weight` path (endpoint alpha is ignored, so
    /// alpha-`00` junction vertices don't erase their edges).
    Vertex,
    /// Per-edge baked color from `--edge-colors` (line i = edge i). An 8-digit hex alpha
    /// multiplies into the edge alpha.
    File,
    /// Computed stress: `|pos_a − pos_b| / (w_a·w_b)^(1/dim)` over the FULL d-dim
    /// positions (matches `WeightedGeometric::getSimilarity`), on a diverging colormap
    /// centered at 1 (compressed = cool, stretched = warm). Live while the embedder runs.
    Length,
}

impl EdgeColorSource {
    /// Value written into the shader `edge_color_source` uniform. Keep in lockstep with
    /// `lines.wgsl`'s `EDGE_COLOR_*` constants.
    pub fn shader_value(self) -> u32 {
        match self {
            EdgeColorSource::Theme => 0,
            EdgeColorSource::Vertex => 1,
            EdgeColorSource::File => 2,
            EdgeColorSource::Length => 3,
        }
    }
}

/// Learning-rate schedule for the embedder (`--lr-schedule`). One type deriving both
/// clap's `ValueEnum` and serde so the CLI flag values and the TOML `[embedder]
/// lr_schedule` string stay identical; `settings` maps it to `wembed_sys::LRSchedule`.
/// Variants are numbered to match wembed's `LRSchedule` enum (0/1). Default =
/// `exponential` (wembed's `LRExponentialCooling`). Each schedule reads a different
/// subset of the LR knobs; the irrelevant ones are ignored.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum LrSchedule {
    /// (0) Exponential cooling — multiply the LR by `--lr-cooling` each step. wembed's default.
    Exponential,
    /// (1) Loss-adaptive — three-zone controller on the loss-decrease `rate(t)`: grow the LR
    /// by `--lr-growth-factor` above `--lr-growth-threshold`, cut it by `--lr-decay-factor`
    /// below `--lr-decay-threshold`, hold in between — each after `--lr-adapt-patience`
    /// in-zone steps.
    LossAdaptive,
}

/// Signal that stops the embedding run (`--stop-criterion`; `--iterations` always caps on
/// top). One type deriving both clap's `ValueEnum` and serde so the CLI flag values and the
/// TOML `[embedder] stop_criterion` string stay identical; `settings` maps it to
/// `wembed_sys::StopCriterion`. Variants are numbered to match wembed's `StopCriterion`
/// enum (0/1). Default = `loss` (wembed's `StopLoss`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum StopCriterion {
    /// (0) Displacement — stop once the relative per-step node movement settles
    /// (`--stop-displacement-tol` / `--stop-displacement-patience`).
    Displacement,
    /// (1) Loss — stop once the loss-decrease `rate(t)` stays below `--stop-loss-tol` for
    /// `--stop-loss-patience` steps (ftol; `--loss-smoothing`/`--loss-rate-window` shape it).
    /// wembed's default.
    Loss,
}

#[derive(Parser, Debug)]
#[command(
    name = "wembed-viewer",
    about = "Live visualization of the wembed graph embedder"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

impl Cli {
    /// The shared argument block, regardless of subcommand. `main` needs `--config`
    /// before the precedence resolve can run.
    pub fn common(&self) -> &CommonArgs {
        match &self.command {
            Command::Gui(g) => &g.common,
            Command::Headless(h) => &h.common,
        }
    }
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Interactive viewer: watch the embedding live, inspect while paused, export from
    /// the UI panel.
    Gui(GuiArgs),
    /// Headless: run the embedder to completion (or replay an `--input-embedding`),
    /// then write a PNG (`-o`) and/or the finished `.emb` (`--output-embedding`), exit.
    /// No window, no UI; embed-only invocations don't even need a GPU.
    Headless(HeadlessArgs),
}

#[derive(Args, Debug)]
pub struct GuiArgs {
    #[command(flatten)]
    pub common: CommonArgs,

    /// Pause after loading the initial embedding, before the first iteration.
    #[arg(long)]
    pub interactive: bool,
}

#[derive(Args, Debug)]
pub struct HeadlessArgs {
    #[command(flatten)]
    pub common: CommonArgs,

    /// Where to write the rendered PNG. Optional: an embed-only run passes just
    /// `--output-embedding` (at least one of the two is required).
    #[arg(short = 'o', long, help_heading = "Input/Output")]
    pub output: Option<PathBuf>,

    /// Export resolution as `WxH` (e.g. `3840x2160`). Vertex sizing is scaled so a
    /// `--point-max-pixels 40` behaves the same at 4K as it does on a 1080p window in
    /// interactive mode (see `REFERENCE_HEIGHT` in [`crate::framing`]).
    /// [default: 3840x2160, or the TOML's [export] section]
    #[arg(long, value_parser = parse_resolution, help_heading = "Rendering")]
    pub resolution: Option<(u32, u32)>,

    /// Camera rotation in degrees (CCW). Overrides a TOML `[camera].rotation_deg`;
    /// ignored if `--autorotate` is set. [default: 0.0]
    #[arg(long, help_heading = "Rendering")]
    pub rotation_deg: Option<f32>,

    /// Pick the rotation that best matches `--resolution`'s aspect (brute-force over
    /// whole-degree rotations, minimizes `|log(bbox_aspect / target_aspect)|`).
    #[arg(long, help_heading = "Rendering")]
    pub autorotate: bool,

    /// Write the finished positions to this `.emb` path — the output counterpart of
    /// `--input-embedding`. With no `-o` this is an embed-only run (no GPU touched);
    /// reuse the result later via `--input-embedding` / `input_embedding =`.
    #[arg(long, help_heading = "Input/Output")]
    pub output_embedding: Option<PathBuf>,
}

/// Arguments shared by `gui` and `headless`.
#[derive(Args, Debug)]
pub struct CommonArgs {
    /// Path to a wembed edge-list file (one undirected edge per line, 0-indexed).
    /// Optional when a `--config` TOML supplies `graph`; passing it here overrides the TOML.
    pub graph_path: Option<PathBuf>,

    /// TOML config file supplying defaults for every knob below (and a `[camera]`
    /// section that has no CLI equivalent). Any flag passed explicitly on the command
    /// line overrides its TOML counterpart. Relative paths inside the TOML resolve
    /// against the TOML's own directory.
    #[arg(long, help_heading = "Input/Output")]
    pub config: Option<PathBuf>,

    // ---------- Embedder knobs (forwarded to wembed::Options) → "Embedding" heading ----------
    /// Max embedder iterations. Loop halts sooner if the embedder converges.
    /// [default: 10000]
    #[arg(short = 'i', long, help_heading = "Embedding")]
    pub iterations: Option<i32>,

    /// RNG seed for random initial positions. [default: 42]
    #[arg(short = 's', long, help_heading = "Embedding")]
    pub seed: Option<i32>,

    /// Embedding dimension (>= 2). The display is always a 2D projection — pick it via
    /// `[visual] projection` (`first-two` or `pca`) or the UI dropdown. [default: 2]
    #[arg(short = 'd', long, help_heading = "Embedding")]
    pub dimension: Option<i32>,

    /// Dimension hint for the degree weights: weights become degree^(dimension/dim-hint),
    /// so a vertex's radius grows as degree^(1/dim-hint) instead of degree^(1/dimension).
    /// Non-positive = off. Matches wembed's `--dim-hint`. [default: -1]
    #[arg(long, help_heading = "Embedding")]
    pub dim_hint: Option<f64>,

    /// Initial learning rate (both schedules). Higher = faster but noisier convergence.
    /// Matches wembed's `--speed`; `--learning-rate` is a back-compat alias. [default: 10.0]
    #[arg(
        short = 'l',
        long = "speed",
        alias = "learning-rate",
        help_heading = "Embedding"
    )]
    pub learning_rate: Option<f64>,

    /// Per-step multiplicative LR decay (exponential schedule only). Lower = faster
    /// cooldown. Matches wembed's `--lr-cooling`. [default: 0.995]
    #[arg(short = 'c', long = "lr-cooling", help_heading = "Embedding")]
    pub lr_cooling_factor: Option<f64>,

    /// Use the multilevel LayeredEmbedder — embed a coarsened graph first, then refine.
    /// Slower per step, often better final quality on large graphs.
    #[arg(long, help_heading = "Embedding")]
    pub layered: bool,

    /// Pull each vertex toward the origin with this strength every step. `0.0` disables it
    /// (default). Useful for graphs with disconnected components — repulsion alone would
    /// blow the islands out to infinity.
    #[arg(long, alias = "center", help_heading = "Embedding")]
    pub centre: Option<f64>,

    /// Slack added to repulsion query radii; the spatial index is only rebuilt once
    /// accumulated node movement exceeds it. Negative picks `3 / dimension`
    /// automatically; `0.0` rebuilds every iteration. [default: -1]
    #[arg(long, help_heading = "Embedding")]
    pub dyn_buffer: Option<f64>,

    // ---------- Learning-rate schedule ----------
    /// Learning-rate schedule. [default: exponential]
    #[arg(long, value_enum, help_heading = "Learning-rate schedule")]
    pub lr_schedule: Option<crate::cli::LrSchedule>,

    /// Linear LR ramp-up over the first N steps (both schedules). [default: 20]
    #[arg(long, help_heading = "Learning-rate schedule")]
    pub warmup_steps: Option<i32>,

    /// Multiplicative LR drop on a decay event (loss-adaptive schedule only). [default: 0.5]
    #[arg(long, help_heading = "Learning-rate schedule")]
    pub lr_decay_factor: Option<f64>,

    /// Decay the LR when the loss-decrease `rate(t)` stays below this (loss-adaptive schedule
    /// only). [default: 0.01]
    #[arg(long, help_heading = "Learning-rate schedule")]
    pub lr_decay_threshold: Option<f64>,

    /// Consecutive in-zone steps before a decay OR growth event (loss-adaptive schedule
    /// only — gates both zones). [default: 20]
    #[arg(long, help_heading = "Learning-rate schedule")]
    pub lr_adapt_patience: Option<i32>,

    /// Multiplicative LR growth while the loss keeps decreasing fast (loss-adaptive schedule
    /// only; `1.0` disables growth → pure plateau decay). [default: 1.0]
    #[arg(long, help_heading = "Learning-rate schedule")]
    pub lr_growth_factor: Option<f64>,

    /// Grow the LR when the loss-decrease `rate(t)` stays above this (loss-adaptive schedule
    /// only). [default: 0.1]
    #[arg(long, help_heading = "Learning-rate schedule")]
    pub lr_growth_threshold: Option<f64>,

    // ---------- Stopping ----------
    /// Which signal terminates the run (`--iterations` always caps on top).
    /// [default: loss]
    #[arg(long, value_enum, help_heading = "Stopping")]
    pub stop_criterion: Option<crate::cli::StopCriterion>,

    /// Relative per-step node movement (mean displacement / radius of gyration) below which
    /// the layout counts as settled (displacement criterion). [default: 0.0003]
    #[arg(long, help_heading = "Stopping")]
    pub stop_displacement_tol: Option<f64>,

    /// Settled steps in a row before stopping (displacement criterion). [default: 5]
    #[arg(long, help_heading = "Stopping")]
    pub stop_displacement_patience: Option<i32>,

    /// ftol: converged once the loss-decrease `rate(t)` stays below this relative decrease
    /// (loss criterion). [default: 0.001]
    #[arg(long, help_heading = "Stopping")]
    pub stop_loss_tol: Option<f64>,

    /// Sub-tolerance steps in a row before stopping, loss criterion (`--iterations`
    /// always caps). [default: 50]
    #[arg(long, help_heading = "Stopping")]
    pub stop_loss_patience: Option<i32>,

    // ---------- Rendering knobs ----------
    /// Point radius as a fraction of estimated nearest-neighbor spacing (world units, before
    /// pixel clamping). Lower = more breathing room between dots in dense areas. The default
    /// dominates at medium zoom levels (before `max_pixels` clamps kick in), so a value that
    /// looks reasonable across the graph sizes we ship matters more than being conservative.
    /// [default: 0.5]
    #[arg(long, help_heading = "Rendering")]
    pub density_factor: Option<f32>,

    /// Minimum point radius in pixels at a 1080p reference height (scaled by
    /// `actual_height / 1080` in both the live window and exports, so it's resolution-
    /// independent). Dots below this become invisible; the floor ensures large graphs still
    /// show density at fit-view zoom levels. [default: 3.0]
    #[arg(long, help_heading = "Rendering")]
    pub point_min_pixels: Option<f32>,

    /// Maximum point radius in pixels at a 1080p reference height (see `--point-min-pixels`
    /// for the scaling rule). Dots above this dominate the frame; the
    /// ceiling stops small graphs from becoming a wall of blobs. Generous because size-by-
    /// weight can multiply the base radius by `sqrt(w_i / mean_w)` — a hub vertex with 10x
    /// mean weight is already ~3.2x base, so the cap needs headroom before hub dots start
    /// getting clipped to a screen-pixel plateau. [default: 40.0]
    #[arg(long, help_heading = "Rendering")]
    pub point_max_pixels: Option<f32>,

    /// Initial vertex color source. Runtime-switchable in the UI. [default: theme]
    #[arg(long, value_enum, help_heading = "Rendering")]
    pub vertex_color_source: Option<VertexColorSource>,

    // ---------- Per-vertex attributes ----------
    /// Optional per-vertex colors file. One hex color per line (line i → vertex i). `#`
    /// prefix and alpha channel are optional; alpha defaults to `ff`. Shown with
    /// `--vertex-color-source file`, and only while the line count matches the current vertex
    /// count (so `--layered` runs fall back to the theme color on coarser layers).
    #[arg(long, help_heading = "Per-vertex attributes")]
    pub colors: Option<PathBuf>,

    /// Optional per-vertex labels file, JSON Lines (one JSON string per line, line i →
    /// vertex i). Shown as hover tooltip while paused. Absent → hover shows `#<vertex_id>`.
    #[arg(long, help_heading = "Per-vertex attributes")]
    pub labels: Option<PathBuf>,

    /// Optional per-vertex categories file — one integer per line, line i = vertex i's
    /// category id. `-1` marks the vertex as uncategorized (rendered in the theme's
    /// neutral base04). When a theme is active AND this file matches the vertex count,
    /// the theme's accent palette is applied by `color = accents[category % accents.len()]`.
    /// Shown with `--vertex-color-source categories`.
    #[arg(long, help_heading = "Per-vertex attributes")]
    pub categories: Option<PathBuf>,

    /// Optional per-edge colors file — one hex color per line, line i → edge i of the loaded
    /// `.edg` (same format as `--colors`; `#` and alpha optional). Feeds the `file`
    /// edge-color source (select it via `[visual] edge_color_source = "file"` or the UI
    /// dropdown); an 8-digit alpha multiplies into the edge alpha. Applied only when the line
    /// count equals the edge count (layered coarser layers fall back to the theme edge color).
    #[arg(long, help_heading = "Per-vertex attributes")]
    pub edge_colors: Option<PathBuf>,

    // ---------- Theming ----------
    /// Directory of tinted-theming YAML files. Every `*.yaml` is loaded at startup and
    /// shown in the theme picker. [default: the bundled themes]
    #[arg(long, help_heading = "Theming")]
    pub themes_dir: Option<PathBuf>,

    /// Initial theme by substring match (case- and separator-insensitive, against
    /// display name or file stem) — e.g. `--theme tokyo` or `--theme tokyo-night-storm`.
    /// Falls back to the first theme (alphabetical) when omitted or no match (warns).
    #[arg(long, help_heading = "Theming")]
    pub theme: Option<String>,

    // ---------- Text ----------
    /// Title drawn into the render (window preview + export, identical pixels).
    /// Overrides the TOML's `[text].title`; size/anchor/color/font are TOML-only.
    #[arg(long, help_heading = "Text")]
    pub title: Option<String>,

    /// External legend-entries TOML (bare `[[entries]]` rows, same schema as
    /// `[[legend.entries]]`). Overrides the config's `[legend]` entry source
    /// (`names`/`entries`/`entries_from`); path resolves against CWD. Styling stays TOML-only.
    #[arg(long, help_heading = "Text")]
    pub legend_entries: Option<PathBuf>,

    // ---------- Density render mode ----------
    /// Enable density (heatmap) render mode with defaults. Equivalent to a bare `[density]`
    /// TOML section; edges + points splat their weight into an accumulation buffer that's
    /// blurred and tone-mapped through a color ramp. Orthogonal to `--vertex-color-source`.
    #[arg(long, help_heading = "Density")]
    pub density: bool,

    /// Density blur sigma in pixels (`0` = off). Overrides `[density].sigma`. Implies
    /// `--density`. [default: 1.2]
    #[arg(long, help_heading = "Density")]
    pub density_sigma: Option<f32>,

    /// Density colormap ramp as comma-separated hex stops, dark→bright (e.g.
    /// `"#0b0e14,#ff9940,#fff2e0"`). Overrides `[density].ramp`. Implies `--density`.
    /// Lets the pipeline fan per-theme ramps over one config. Validated (>= 2 stops) at
    /// resolve time.
    #[arg(long, help_heading = "Density")]
    pub density_ramp: Option<String>,

    // ---------- Precomputed embedding ----------
    /// Optional `.emb` CSV to load positions from instead of running the embedder.
    /// Format: `id,x1,...,xd,w` per line (dim inferred from the column count, >= 2).
    /// Requires a graph (positional or TOML) for
    /// the edges — the `.emb` only stores per-vertex data. Skips the embedder entirely
    /// (embedder knobs are ignored); a missing file is a hard error. This is also the
    /// only producer where a TOML `[camera]` section applies.
    #[arg(long, help_heading = "Input/Output")]
    pub input_embedding: Option<PathBuf>,
}

/// Parse `WxH` (e.g. `3840x2160`) into `(u32, u32)`. Zero dimensions are rejected — a
/// zero-sized render target crashes wgpu allocation. Also accepts `WXH` (capital X).
fn parse_resolution(s: &str) -> Result<(u32, u32), String> {
    let sep = s
        .find(['x', 'X'])
        .ok_or_else(|| format!("resolution {s:?}: expected `WxH`, e.g. `3840x2160`"))?;
    let (w, h) = s.split_at(sep);
    let h = &h[1..];
    let w: u32 = w
        .parse()
        .map_err(|e| format!("resolution {s:?}: width {w:?} not an integer: {e}"))?;
    let h: u32 = h
        .parse()
        .map_err(|e| format!("resolution {s:?}: height {h:?} not an integer: {e}"))?;
    if w == 0 || h == 0 {
        return Err(format!("resolution {s:?}: dimensions must be > 0"));
    }
    Ok((w, h))
}

/// Split a comma-separated hex list (`--density-ramp`) into trimmed non-empty stops. The
/// hex strings themselves are validated later (in `settings`, via the shared color parser),
/// so this only enforces the >= 2 stops rule the ramp needs to interpolate.
pub(crate) fn parse_hex_list(s: &str) -> Result<Vec<String>, String> {
    let stops: Vec<String> = s
        .split(',')
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .map(str::to_string)
        .collect();
    if stops.len() < 2 {
        return Err(format!(
            "density ramp {s:?}: need at least 2 hex stops, e.g. `#0b0e14,#ff9940`"
        ));
    }
    Ok(stops)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_resolution_accepts_lower_and_upper_x() {
        assert_eq!(parse_resolution("3840x2160").unwrap(), (3840, 2160));
        assert_eq!(parse_resolution("3840X2160").unwrap(), (3840, 2160));
    }

    #[test]
    fn parse_resolution_missing_separator_errors() {
        let err = parse_resolution("38402160").unwrap_err();
        assert!(err.contains("expected `WxH`"), "unexpected error: {err}");
    }

    #[test]
    fn parse_resolution_zero_dimension_errors() {
        assert!(parse_resolution("0x2160").unwrap_err().contains("> 0"));
        assert!(parse_resolution("3840x0").unwrap_err().contains("> 0"));
    }

    #[test]
    fn parse_resolution_non_integer_parts_error() {
        assert!(parse_resolution("axb").is_err());
        assert!(parse_resolution("3840x21.6").is_err());
        assert!(parse_resolution("-5x100").is_err());
    }

    #[test]
    fn vertex_color_source_shader_values_are_stable() {
        // Guards the lockstep with `VERTEX_COLOR_*` in points.wgsl.
        assert_eq!(VertexColorSource::Theme.shader_value(), 0);
        assert_eq!(VertexColorSource::Categories.shader_value(), 1);
        assert_eq!(VertexColorSource::File.shader_value(), 1);
    }

    #[test]
    fn edge_color_source_shader_values_are_stable() {
        // Guards the lockstep with `EDGE_COLOR_*` in lines.wgsl.
        assert_eq!(EdgeColorSource::Theme.shader_value(), 0);
        assert_eq!(EdgeColorSource::Vertex.shader_value(), 1);
        assert_eq!(EdgeColorSource::File.shader_value(), 2);
        assert_eq!(EdgeColorSource::Length.shader_value(), 3);
        // Default (missing-config fallback) is Theme.
        assert_eq!(EdgeColorSource::default(), EdgeColorSource::Theme);
    }

    #[test]
    fn parse_hex_list_splits_and_trims() {
        assert_eq!(
            parse_hex_list("#0b0e14, #ff9940 ,#fff2e0").unwrap(),
            vec!["#0b0e14", "#ff9940", "#fff2e0"]
        );
    }

    #[test]
    fn parse_hex_list_needs_two_stops() {
        assert!(parse_hex_list("#0b0e14")
            .unwrap_err()
            .contains("at least 2"));
        assert!(parse_hex_list("").is_err());
    }

    #[test]
    fn headless_only_flags_rejected_on_gui() {
        // The whole point of the subcommand split: clap statically rejects headless-only
        // knobs on `gui` instead of us needing `requires`-style runtime gates.
        assert!(Cli::try_parse_from(["wembed-viewer", "gui", "g.edg", "-o", "x.png"]).is_err());
        assert!(Cli::try_parse_from(["wembed-viewer", "gui", "g.edg", "--autorotate"]).is_err());
    }
}
