//! Threaded embedder driver.
//!
//! The wembed FFI types (`Graph`, `Embedder`) are held via `cxx::UniquePtr` which is not `Send`.
//! We work around that by constructing them *inside* the worker thread — nothing FFI-related
//! ever crosses the thread boundary. What crosses is: an `Arc<Mutex<Shared>>` for the
//! streaming state and an `Arc<AtomicBool>` for shutdown.
//!
//! Design goals:
//! - `spawn` returns instantly. Graph load (which can take seconds on large graphs) happens
//!   entirely on the worker thread. Consumer sees `Update::Nothing` until the worker publishes.
//! - The consumer polls once per frame via `poll`. It gets a small enum telling it what
//!   changed — nothing / just positions / topology (which requires a renderer rebuild).
//! - "Graph changed at runtime" is representable in the state machine: the worker sets a new
//!   `topology` and bumps both generation counters; the consumer's next poll returns
//!   `Update::Topology(...)` and rebuilds its GPU state. Implementing the actual graph swap
//!   is a future feature — the state machine already allows it.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use anyhow::{Context, Result};
use wembed_sys::{ffi, Options, Progress, SpatialIndex};

use crate::loaded_embedding::parse_edges_file;

#[derive(Clone)]
pub struct Topology {
    /// Embedding dimension of the accompanying positions (coords per vertex, >= 2).
    /// `num_vertices` is derivable as `positions.len() / dim`.
    pub dim: usize,
    /// Undirected edges as `[src, dst]` vertex-index pairs.
    pub edges: Vec<[u32; 2]>,
    /// Raw per-vertex weights from the embedder. Kept alongside `size_multipliers` because
    /// the embedding save-path needs them verbatim — we can reconstruct size_multipliers
    /// from raw weights but not the other way around (mean_w is destroyed by normalization).
    /// Length == N.
    pub weights: Vec<f32>,
    /// Per-vertex size multipliers, precomputed as `sqrt(w_i / mean_w)`. Baked here rather
    /// than shipping raw weights + a normalization uniform so the shader is one multiply and
    /// the "typical vertex = base radius" invariant holds by construction. Length == N.
    /// When the embedder was built with `use_unit_weights=true`, every entry is `1.0`.
    pub size_multipliers: Vec<f32>,
    /// Per-edge alpha multiplier, precomputed as `1 / (w_a * w_b)` from raw weights. Length
    /// == `edges.len()`. Consumed by the lines shader with no interpolation (the whole
    /// segment gets one constant alpha), so the "hub↔hub edges recede" behavior is exact
    /// rather than an interpolated approximation. `use_unit_weights=true` collapses every
    /// entry to `1.0`.
    pub edge_alphas: Vec<f32>,
}

/// Preprocess raw weights + edges into the derived per-vertex `size_multipliers` and
/// per-edge `edge_alphas` arrays used by the renderer. Extracted here so `LoadedEmbedding`
/// (loading from a `.emb` file) can produce the same-shape `Topology` as a live embedder.
pub fn preprocess_topology(weights: &[f32], edges: &[[u32; 2]]) -> (Vec<f32>, Vec<f32>) {
    let mean_w: f64 = if weights.is_empty() {
        0.0
    } else {
        weights.iter().map(|w| *w as f64).sum::<f64>() / weights.len() as f64
    };
    let size_multipliers: Vec<f32> = if mean_w <= 0.0 {
        vec![1.0; weights.len()]
    } else {
        weights
            .iter()
            .map(|w| ((*w as f64 / mean_w).max(0.0).sqrt()) as f32)
            .collect()
    };
    let edge_alphas: Vec<f32> = edges
        .iter()
        .map(|&[a, b]| {
            let wa = weights.get(a as usize).copied().unwrap_or(0.0);
            let wb = weights.get(b as usize).copied().unwrap_or(0.0);
            let prod = (wa as f64) * (wb as f64);
            if prod > 0.0 {
                (1.0 / prod) as f32
            } else {
                1.0
            }
        })
        .collect();
    (size_multipliers, edge_alphas)
}

/// Per-edge weight normalizer `(w_a·w_b)^(-1/dim)` — i.e. `edge_alphas^(1/dim)`.
/// Precomputed once per topology so the per-step ratio loop below avoids a `powf` per
/// edge (which would dominate the distance math ~10×).
pub fn edge_inv_norms(edge_alphas: &[f32], dim: usize) -> Vec<f32> {
    let inv_d = 1.0 / dim as f32;
    edge_alphas.iter().map(|a| a.powf(inv_d)).collect()
}

/// Per-edge stress ratio `r_i = |pos_a − pos_b| / (w_a·w_b)^(1/dim)` over the FULL d-dim
/// positions — the exact quantity `WeightedGeometric::getSimilarity` computes (r < 1 =
/// compressed, r > 1 = stretched, r = 1 = ideal). Feeds the `length` edge-color source;
/// recomputed per positions update while that source is active.
pub fn edge_length_ratios(
    positions: &[f32],
    dim: usize,
    edges: &[[u32; 2]],
    inv_norms: &[f32],
    out: &mut Vec<f32>,
) {
    debug_assert_eq!(edges.len(), inv_norms.len());
    out.clear();
    out.reserve(edges.len());
    for (e, inv_norm) in edges.iter().zip(inv_norms) {
        let a = &positions[e[0] as usize * dim..(e[0] as usize + 1) * dim];
        let b = &positions[e[1] as usize * dim..(e[1] as usize + 1) * dim];
        let mut acc = 0.0_f32;
        for k in 0..dim {
            let c = a[k] - b[k];
            acc += c * c;
        }
        out.push(acc.sqrt() * inv_norm);
    }
}

/// Automatic colormap window for the `length` edge-color source: q05/q95 quantiles of
/// `ln(r)` over a deterministic <=10k stride sample, each side pushed past 0 so the
/// r = 1 neutral anchor stays strictly inside the window (a side with no tail collapses
/// to the anchor — every edge beyond it saturates, which is the intent: "short enough"
/// is cool, "too long" is warm, and the quantiles decide where full saturation begins).
/// `None` when no positive finite ratios exist.
pub fn edge_length_log_range(ratios: &[f32]) -> Option<(f32, f32)> {
    const MAX_SAMPLE: usize = 10_000;
    if ratios.is_empty() {
        return None;
    }
    let stride = ratios.len().div_ceil(MAX_SAMPLE);
    let mut logs: Vec<f32> = ratios
        .iter()
        .step_by(stride)
        .copied()
        .filter(|r| r.is_finite() && *r > 0.0)
        .map(|r| r.ln())
        .collect();
    if logs.is_empty() {
        return None;
    }
    logs.sort_by(f32::total_cmp);
    let q = |frac: f64| logs[((logs.len() - 1) as f64 * frac).round() as usize];
    Some((q(0.05).min(-1e-6), q(0.95).max(1e-6)))
}

/// One per-step convergence sample, published by the worker after every embedder step.
/// Consumed by the convergence plot in the stats panel.
#[derive(Copy, Clone, Debug)]
pub struct MetricSample {
    /// Worker-side step count (1 = after the first `calculate_step`). Monotonic across
    /// layered-embedder layer transitions.
    pub iteration: u64,
    /// Learning rate the optimizer used at this step (whatever the active schedule set it
    /// to — read verbatim from `getCurrentLearningRate`).
    pub learning_rate: f64,
    /// Total loss from this step's force computation (attractive + repulsive).
    pub loss: f64,
    /// Relative node displacement of this step (mean per-node movement / radius of
    /// gyration) — the signal the displacement stop criterion watches.
    pub rel_displacement: f64,
    /// Windowed relative loss-decrease rate `rate(t)` of this step — the signal the loss stop
    /// criterion watches (stagnant while `|rate| < stop_loss_tol`). `+inf` until the window
    /// has filled; negative while the loss rises (e.g. right after a layer expansion).
    pub rel_loss_improvement: f64,
    /// Vertex count of the layer this step ran on.
    pub num_vertices: u32,
    /// wembed's layer (counts down to 0) and layer count; the convergence panel starts a new
    /// column where `layer` changes.
    pub layer: i32,
    pub num_layers: i32,
}

/// What the last `poll` observed.
pub enum Update {
    /// Producer hasn't advanced since last poll. Cheap — one uncontended lock acquire.
    Nothing,
    /// Positions changed; topology is unchanged. `positions_out` was rewritten.
    Positions,
    /// Topology changed — this is either "the graph finished loading" or "we swapped to a
    /// different graph." The renderer must rebuild its GPU buffers because sizes changed.
    /// `positions_out` was rewritten to match the new topology.
    Topology(Topology),
}

/// Source of `Update`s for the App. Implemented by [`EmbedderHandle`] for live embedding
/// runs and by [`LoadedEmbedding`](crate::loaded_embedding::LoadedEmbedding) for `.emb`
/// replay (publishes exactly once).
///
/// Playback controls (`is_paused`, `toggle_paused`, `is_finished`, `iteration`) are on the
/// trait even though a loaded embedding has no meaningful playback — sensible defaults
/// (`paused = true`, `finished = true`, `iteration = 0`, `toggle` = no-op) let the App
/// treat both sources uniformly without special-casing.
pub trait TopologyProducer {
    /// Check for new state. See [`Update`] variants for what the return value means and
    /// what `positions_out` looks like afterward.
    fn poll(&mut self, positions_out: &mut Vec<f32>) -> Update;

    /// Whether the producer is in a paused state. For a live embedder, this is the user's
    /// pause toggle; for a loaded embedding, always `true` (nothing to run).
    fn is_paused(&self) -> bool {
        true
    }

    /// Whether the producer has finished producing new state. For a live embedder, set
    /// when the embedder converges; for a loaded embedding, always `true`.
    fn is_finished(&self) -> bool {
        true
    }

    /// Flip pause state. No-op for producers that don't have a run loop.
    fn toggle_paused(&self) {}

    /// Advance one iteration while paused. No-op for producers without a step loop.
    fn step_once(&self) {}

    /// Number of iterations completed. `0` for producers without a step loop.
    fn iteration(&self) -> u64 {
        0
    }

    /// Move any convergence samples published since the last drain into `out`. Default
    /// no-op — only a live embedder produces metrics; `.emb` replay has none.
    fn drain_metrics(&mut self, _out: &mut Vec<MetricSample>) {}

    /// wembed's progress estimate after the latest step. `None` without a step loop.
    fn progress(&self) -> Option<Progress> {
        None
    }

    /// Why the producer stopped without a result (e.g. an unreadable graph). Terminal:
    /// nothing is published afterwards and `is_finished` never flips.
    fn error(&self) -> Option<String> {
        None
    }
}

/// One-line progress summary shared by the status strip and the headless heartbeat, e.g.
/// `layer 3/7 · iter 640/~1200 · ETA 1m42s`. Parts wembed can't estimate yet are omitted.
pub fn format_progress(p: &Progress) -> String {
    format_progress_at_iteration(p, p.iteration)
}

/// Format progress using the consumer's monotonic published-step count rather than the
/// embedder's per-layer iteration field. This is useful in the GUI, where a single overall
/// iteration counter should continue across layered transitions.
pub fn format_progress_with_iteration(p: &Progress, iteration: u64) -> String {
    format_progress_at_iteration(p, iteration)
}

fn format_progress_at_iteration(p: &Progress, iteration: impl std::fmt::Display) -> String {
    let mut out = String::new();
    if p.num_layers > 1 {
        out += &layer_label(p.layer, p.num_layers);
        out += " · ";
    }
    out += &format!("iter {iteration}");
    if p.expected_iterations >= 0 {
        out += &format!("/~{}", p.expected_iterations);
    }
    let finished = p.layer == 0 && p.layer_finished;
    if p.eta_seconds >= 0.0 && !finished {
        out += &format!(" · ETA {}", format_seconds(p.eta_seconds));
    }
    out
}

/// `layer 3/7`, shared by the status strip and the convergence panel. wembed counts layers
/// down to 0; they're shown counting up so the last one reads 7/7.
pub fn layer_label(layer: i32, num_layers: i32) -> String {
    format!("layer {}/{}", num_layers - layer, num_layers)
}

/// Same format as wembed's CLI: `42s`, `1m05s`, `2h03m`.
fn format_seconds(seconds: f64) -> String {
    let s = seconds.round() as u64;
    if s >= 3600 {
        format!("{}h{:02}m", s / 3600, (s % 3600) / 60)
    } else if s >= 60 {
        format!("{}m{:02}s", s / 60, s % 60)
    } else {
        format!("{s}s")
    }
}

struct Shared {
    /// `None` until the worker has finished reading the graph and set up the embedder.
    topology: Option<Topology>,
    /// Bumps whenever `topology` is replaced. Consumer compares to detect graph swaps.
    topology_gen: u64,
    /// Flat row-major coords, `topology.dim` per vertex. Sized to match the current topology.
    positions: Vec<f32>,
    /// Bumps every step. Consumer compares to detect stale-frame skips.
    positions_gen: u64,
    /// Per-step convergence samples appended by the worker, moved out wholesale by
    /// `drain_metrics`. Stays small in interactive runs (drained every frame); an
    /// undrained headless run tops out at max_iterations × 24 B.
    metrics: Vec<MetricSample>,
    progress: Option<Progress>,
    /// Set once if loading fails; the worker exits right after.
    error: Option<String>,
}

pub struct EmbedderHandle {
    shared: Arc<Mutex<Shared>>,
    stop: Arc<AtomicBool>,
    /// Consumer toggles to pause/resume the step loop. Worker sets it to true when the
    /// embedder finishes so the App transitions cleanly into the paused view.
    paused: Arc<AtomicBool>,
    /// Queued one-step requests, consumed by the worker only while paused.
    step_requests: Arc<AtomicU64>,
    /// Set once by the worker when `Embedder::is_finished()` returns true. Sticky — never
    /// clears. Consumer uses it to disable the Resume button (resuming a finished embedder
    /// would just re-pause on the next worker tick).
    finished: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
    /// Consumer-side view of what we've already seen. Kept on the handle so `poll` is one
    /// method call — the App doesn't need its own bookkeeping.
    last_topology_gen: u64,
    last_positions_gen: u64,
}

impl TopologyProducer for EmbedderHandle {
    /// Check for new state. If the topology advanced, returns a fresh `Topology` and
    /// overwrites `positions_out` to match. If only positions advanced, just overwrites
    /// `positions_out`. Otherwise `Nothing`.
    fn poll(&mut self, positions_out: &mut Vec<f32>) -> Update {
        let g = self.shared.lock().unwrap();

        let topology_advanced = g.topology_gen != self.last_topology_gen;
        let positions_advanced = g.positions_gen != self.last_positions_gen;
        if !topology_advanced && !positions_advanced {
            return Update::Nothing;
        }

        // Both cases below require topology to be present. If we saw an advance but topology
        // is still None (shouldn't happen with a well-behaved worker) treat as Nothing.
        let Some(topology) = &g.topology else {
            return Update::Nothing;
        };

        positions_out.clear();
        positions_out.extend_from_slice(&g.positions);
        self.last_positions_gen = g.positions_gen;

        if topology_advanced {
            self.last_topology_gen = g.topology_gen;
            Update::Topology(topology.clone())
        } else {
            Update::Positions
        }
    }

    fn is_paused(&self) -> bool {
        self.paused.load(Ordering::Acquire)
    }

    fn is_finished(&self) -> bool {
        self.finished.load(Ordering::Acquire)
    }

    /// Flip pause state. No-op when finished — worker would re-pause anyway.
    fn toggle_paused(&self) {
        if self.finished.load(Ordering::Acquire) {
            return;
        }
        if self.paused.load(Ordering::Acquire) {
            self.step_requests.store(0, Ordering::Release);
        }
        self.paused.fetch_xor(true, Ordering::AcqRel);
    }

    fn step_once(&self) {
        if self.paused.load(Ordering::Acquire) && !self.finished.load(Ordering::Acquire) {
            loop {
                let pending = self.step_requests.load(Ordering::Acquire);
                if pending == u64::MAX
                    || self
                        .step_requests
                        .compare_exchange_weak(
                            pending,
                            pending + 1,
                            Ordering::AcqRel,
                            Ordering::Acquire,
                        )
                        .is_ok()
                {
                    break;
                }
            }
        }
    }

    /// Number of embedder steps completed at the time of the last `poll`. The first publish
    /// is the pre-step initial state (iteration 0); every `calculate_step` increments this.
    fn iteration(&self) -> u64 {
        self.last_positions_gen.saturating_sub(1)
    }

    fn drain_metrics(&mut self, out: &mut Vec<MetricSample>) {
        let mut g = self.shared.lock().unwrap();
        out.append(&mut g.metrics);
    }

    fn progress(&self) -> Option<Progress> {
        self.shared.lock().unwrap().progress
    }

    fn error(&self) -> Option<String> {
        self.shared.lock().unwrap().error.clone()
    }
}

impl Drop for EmbedderHandle {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

pub struct SpawnConfig {
    pub graph_path: PathBuf,
    /// The fully-resolved embedder knobs (iterations, seed, LR schedule, …). Carried
    /// whole rather than field-by-field so adding an `Options` knob is one struct edit.
    pub embedder: crate::settings::EmbedderParams,
    /// Start the worker paused after it has loaded and published the initial embedding.
    pub start_paused: bool,
}

/// Worker-side view of `SpawnConfig`: `graph_path` is a `String` because the FFI wants
/// `&str` and we validate UTF-8 once in `spawn` rather than every use site; the embedder
/// knobs ride along verbatim.
struct WorkerConfig {
    graph_path: String,
    embedder: crate::settings::EmbedderParams,
}

/// Cross-thread channels shared between `spawn` (which builds them) and the worker (which
/// writes into them). Consumer-side aliases live on `EmbedderHandle`. Bundled so the worker
/// signature is one parameter instead of four Arcs.
struct WorkerChannels {
    shared: Arc<Mutex<Shared>>,
    stop: Arc<AtomicBool>,
    paused: Arc<AtomicBool>,
    step_requests: Arc<AtomicU64>,
    finished: Arc<AtomicBool>,
}

/// Spawn the embedder worker. Returns immediately — no blocking. The worker takes over
/// graph loading and embedding; the consumer polls via `EmbedderHandle::poll` to observe
/// progress.
pub fn spawn(cfg: SpawnConfig) -> Result<EmbedderHandle> {
    let worker_cfg = WorkerConfig {
        graph_path: cfg
            .graph_path
            .to_str()
            .context("graph path is not UTF-8")?
            .to_owned(),
        embedder: cfg.embedder,
    };

    let shared = Arc::new(Mutex::new(Shared {
        topology: None,
        topology_gen: 0,
        positions: Vec::new(),
        positions_gen: 0,
        metrics: Vec::new(),
        progress: None,
        error: None,
    }));
    let stop = Arc::new(AtomicBool::new(false));
    let paused = Arc::new(AtomicBool::new(cfg.start_paused));
    let step_requests = Arc::new(AtomicU64::new(0));
    let finished = Arc::new(AtomicBool::new(false));

    let channels = WorkerChannels {
        shared: Arc::clone(&shared),
        stop: Arc::clone(&stop),
        paused: Arc::clone(&paused),
        step_requests: Arc::clone(&step_requests),
        finished: Arc::clone(&finished),
    };

    let handle = std::thread::Builder::new()
        .name("wembed-embedder".into())
        .spawn(move || run_worker(worker_cfg, channels))
        .context("spawn embedder thread")?;

    Ok(EmbedderHandle {
        shared,
        stop,
        paused,
        step_requests,
        finished,
        handle: Some(handle),
        last_topology_gen: 0,
        last_positions_gen: 0,
    })
}

/// Map the CLI/TOML `LrSchedule` to the FFI enum. Kept here (not on `cli::LrSchedule`) so
/// `cli.rs` stays free of a `wembed_sys` dependency. Discriminants match by construction —
/// both enums are numbered against wembed's `LRSchedule` — but the explicit match makes a
/// future variant a compile error rather than a silent misroute.
pub(crate) fn lr_schedule_to_ffi(s: crate::cli::LrSchedule) -> ffi::LRSchedule {
    use crate::cli::LrSchedule as C;
    use ffi::LRSchedule as F;
    match s {
        C::Exponential => F::LRExponentialCooling,
        C::LossAdaptive => F::LRLossAdaptive,
    }
}

/// Map the CLI/TOML `StopCriterion` to the FFI enum. Same rationale as
/// [`lr_schedule_to_ffi`]: kept off `cli::StopCriterion` so `cli.rs` stays free of a
/// `wembed_sys` dependency, and the explicit match turns a future variant into a compile
/// error rather than a silent misroute.
pub(crate) fn stop_criterion_to_ffi(c: crate::cli::StopCriterion) -> ffi::StopCriterion {
    use crate::cli::StopCriterion as C;
    use ffi::StopCriterion as F;
    match c {
        C::Displacement => F::StopDisplacement,
        C::Loss => F::StopLoss,
    }
}

/// The wembed FFI surface the worker drives — the step loop and the publish functions only
/// see plain Rust data. Built on the worker thread (`UniquePtr` is `!Send`).
struct Wembed {
    embedder: wembed_sys::UniquePtr<ffi::Embedder>,
    graph_path: String,
    /// wembed's f64 copy destination, reused across steps (resized on layer transitions)
    coords: Vec<f64>,
}

impl Wembed {
    /// Graph load + embedder setup. Can take seconds on 1M vertices — this is exactly what we
    /// don't want on the main thread.
    fn load(graph_path: &str, e: &crate::settings::EmbedderParams) -> Result<Self> {
        ffi::set_seed(e.seed);
        let graph = ffi::graph_from_file(graph_path, "#", " ")?;
        // Optimizer + force scales keep their C++ defaults via `..Options::default()`.
        // Everything the CLI/TOML exposes flows from the resolved `EmbedderParams`.
        let opts = Options {
            embedding_dimension: e.dimension,
            dimension_hint: e.dimension_hint,
            index_type: SpatialIndex::IndexSprk,
            max_iterations: e.iterations,
            learning_rate: e.learning_rate,
            lr_cooling_factor: e.lr_cooling_factor,
            layered_embedding: e.layered,
            centre_scale: e.centre,
            dynamic_query_buffer: e.dyn_buffer,
            lr_schedule: lr_schedule_to_ffi(e.lr_schedule),
            warmup_steps: e.warmup_steps,
            lr_decay_factor: e.lr_decay_factor,
            lr_decay_threshold: e.lr_decay_threshold,
            lr_adapt_patience: e.lr_adapt_patience,
            lr_growth_factor: e.lr_growth_factor,
            lr_growth_threshold: e.lr_growth_threshold,
            stop_criterion: stop_criterion_to_ffi(e.stop_criterion),
            stop_displacement_tol: e.stop_displacement_tol,
            stop_displacement_patience: e.stop_displacement_patience,
            stop_loss_tol: e.stop_loss_tol,
            stop_loss_patience: e.stop_loss_patience,
            ..Options::default()
        };
        Ok(Self {
            embedder: ffi::create_embedder(&graph, &opts)?,
            graph_path: graph_path.to_owned(),
            coords: Vec::new(),
        })
    }

    fn step(&mut self) {
        self.embedder.pin_mut().calculate_step();
    }

    fn is_finished(&self) -> bool {
        self.embedder.is_finished()
    }

    fn num_vertices(&self) -> usize {
        self.embedder.num_vertices() as usize
    }

    fn progress(&self) -> Progress {
        ffi::embedder_progress(&self.embedder)
    }

    fn dim(&self) -> usize {
        self.embedder.embedding_dimension() as usize
    }

    /// Convergence sample of the last step. Every getter reads state tracked during the step,
    /// so these are cheap FFI calls, not a recomputation.
    fn sample(&self, iteration: u64, progress: &Progress) -> MetricSample {
        MetricSample {
            iteration,
            learning_rate: self.embedder.current_learning_rate(),
            loss: ffi::embedder_loss(&self.embedder).total,
            rel_displacement: self.embedder.last_rel_displacement(),
            rel_loss_improvement: self.embedder.last_rel_loss_improvement(),
            num_vertices: progress.num_vertices as u32,
            layer: progress.layer,
            num_layers: progress.num_layers,
        }
    }

    /// Edges + raw weights of the current graph: the coarsest layer for `LayeredEmbedder`,
    /// else the whole graph. `getCurrentGraph` allocates a fresh Graph for the layered
    /// embedder — keep this outside any lock.
    ///
    /// INVARIANT: the input graph (layer 0) takes its edges from the file in line order, like
    /// `.emb` replay does — wembed's own list is sorted, and `--edge-colors` line i must name
    /// the same edge live and on replay.
    fn current_graph(&self) -> Result<(Vec<[u32; 2]>, Vec<f32>)> {
        let weights: Vec<f32> = ffi::embedder_weights(&self.embedder)
            .iter()
            .map(|w| *w as f32)
            .collect();
        let edges = if self.progress().layer == 0 {
            parse_edges_file(Path::new(&self.graph_path), weights.len())?
        } else {
            let graph = ffi::embedder_current_graph(&self.embedder);
            ffi::graph_edge_list(&graph)
                .iter()
                .map(|e| [e.src as u32, e.dst as u32])
                .collect()
        };
        Ok((edges, weights))
    }

    /// Row-major coordinates of the current layer (`num_vertices * dim`).
    fn coords(&mut self) -> &[f64] {
        self.coords.resize(self.num_vertices() * self.dim(), 0.0);
        ffi::embedder_copy_coords(&self.embedder, &mut self.coords);
        &self.coords
    }
}

fn run_worker(cfg: WorkerConfig, ch: WorkerChannels) {
    if let Err(e) = worker_loop(&cfg, &ch) {
        ch.shared.lock().unwrap().error = Some(format!("{e:#}"));
    }
}

fn worker_loop(cfg: &WorkerConfig, ch: &WorkerChannels) -> Result<()> {
    let mut wembed = Wembed::load(&cfg.graph_path, &cfg.embedder)?;

    // Initial publish uses the same topology path as layer transitions.
    publish_topology(&ch.shared, &mut wembed)?;
    let mut progress = wembed.progress();
    ch.shared.lock().unwrap().progress = Some(progress);

    // Step loop. A layer change means the layered embedder expanded: republish the topology
    // (edges + resize positions), and the App tears down / rebuilds the Renderer for the new
    // sizes.
    let mut step_count: u64 = 0;
    while !ch.stop.load(Ordering::Acquire) {
        if wembed.is_finished() {
            // Auto-pause on finish so the App transitions into "paused view" (frozen
            // positions, user-driven camera) without extra plumbing. `finished` is sticky
            // so the UI can disable Resume — otherwise a user resume would just re-enter
            // this branch on the next tick.
            if !ch.finished.load(Ordering::Acquire) {
                ch.finished.store(true, Ordering::Release);
                ch.paused.store(true, Ordering::Release);
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
            continue;
        }
        if ch.paused.load(Ordering::Acquire) {
            if !take_step_request(&ch.step_requests) {
                std::thread::sleep(std::time::Duration::from_millis(20));
                continue;
            }
        }
        wembed.step();
        step_count += 1;

        let prev_layer = progress.layer;
        progress = wembed.progress();
        if progress.layer != prev_layer {
            publish_topology(&ch.shared, &mut wembed)?;
        } else {
            publish_positions(&ch.shared, &mut wembed);
        }
        let sample = wembed.sample(step_count, &progress);
        let mut g = ch.shared.lock().unwrap();
        g.metrics.push(sample);
        g.progress = Some(progress);
    }
    Ok(())
}

fn take_step_request(requests: &AtomicU64) -> bool {
    loop {
        let pending = requests.load(Ordering::Acquire);
        if pending == 0 {
            return false;
        }
        if requests
            .compare_exchange_weak(
                pending,
                pending - 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
        {
            return true;
        }
    }
}

/// Publish the current graph's edges + coordinates as a new topology. Called on worker
/// startup and again on every layered-embedder layer transition. Does the heavy FFI work
/// outside the lock; only the state swap happens under it.
fn publish_topology(shared: &Mutex<Shared>, wembed: &mut Wembed) -> Result<()> {
    let dim = wembed.dim();
    let (edges, weights) = wembed.current_graph()?;
    let (size_multipliers, edge_alphas) = preprocess_topology(&weights, &edges);
    let coords = wembed.coords();

    let mut g = shared.lock().unwrap();
    g.topology = Some(Topology {
        dim,
        edges,
        weights,
        size_multipliers,
        edge_alphas,
    });
    g.topology_gen += 1;
    g.positions.resize(coords.len(), 0.0);
    for (dst, &src) in g.positions.iter_mut().zip(coords) {
        *dst = src as f32;
    }
    g.positions_gen += 1;
    Ok(())
}

fn publish_positions(shared: &Mutex<Shared>, wembed: &mut Wembed) {
    let coords = wembed.coords();
    let mut g = shared.lock().unwrap();
    for (dst, &src) in g.positions.iter_mut().zip(coords) {
        *dst = src as f32;
    }
    g.positions_gen += 1;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn progress_format_can_use_monotonic_iteration_count() {
        let progress = Progress {
            layer: 0,
            num_layers: 1,
            num_vertices: 10,
            iteration: 0,
            expected_iterations: 100,
            layer_seconds: 0.0,
            eta_seconds: -1.0,
            layer_finished: false,
        };
        assert_eq!(
            format_progress_with_iteration(&progress, 3),
            "iter 3/~100"
        );
    }

    #[test]
    fn step_requests_are_consumed_one_at_a_time() {
        let requests = AtomicU64::new(2);
        assert!(take_step_request(&requests));
        assert!(take_step_request(&requests));
        assert!(!take_step_request(&requests));
    }

    /// wembed hands out its edges sorted; the input graph must keep the file's line order so
    /// `--edge-colors` line i colors the same edge live as on `.emb` replay.
    #[test]
    fn live_input_graph_keeps_file_edge_order() {
        use clap::Parser;
        use std::io::Write;
        let mut f = tempfile::NamedTempFile::new().unwrap();
        writeln!(f, "2 1\n0 2\n1 0").unwrap();
        let cli = crate::cli::Cli::parse_from(["wembed-viewer", "gui", "g.edg"]);
        let settings =
            crate::settings::Settings::resolve(cli, crate::config::FileConfig::empty()).unwrap();
        let wembed = Wembed::load(f.path().to_str().unwrap(), &settings.embedder).unwrap();
        let (edges, _) = wembed.current_graph().unwrap();
        assert_eq!(edges, vec![[2, 1], [0, 2], [1, 0]]);
    }

    /// The CLI/TOML schedule enum must map onto the FFI enum with matching discriminants —
    /// a mismatch would silently run the wrong schedule. Checks every variant by its i32
    /// value (the shim `static_cast`s, so the numbers are what actually cross the boundary).
    #[test]
    fn lr_schedule_maps_to_ffi_by_discriminant() {
        use crate::cli::LrSchedule as C;
        let cases = [(C::Exponential, 0), (C::LossAdaptive, 1)];
        // cxx bridge enums are newtypes over an i32 exposed as `.repr` (not `as`-castable).
        for (variant, want) in cases {
            assert_eq!(lr_schedule_to_ffi(variant).repr, want, "{variant:?}");
        }
    }

    const EPS: f32 = 1e-5;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= EPS
    }

    #[test]
    fn uniform_weights_give_unit_multipliers_and_alphas() {
        let weights = vec![2.0_f32; 4];
        let edges = vec![[0, 1], [2, 3]];
        let (mults, alphas) = preprocess_topology(&weights, &edges);
        for m in &mults {
            assert!(close(*m, 1.0), "size multiplier {m} != 1.0");
        }
        // Uniform weights: w_a*w_b = 4, alpha = 1/4 (not 1.0). Pin actual math: alpha
        // depends on raw weight, so uniform-but-not-1 weights give a uniform non-1 alpha.
        for a in &alphas {
            assert!(close(*a, 0.25), "edge alpha {a} != 0.25");
        }
    }

    #[test]
    fn size_multiplier_is_sqrt_of_normalized_weight() {
        // weights [1, 4] → mean 2.5 → sqrt(1/2.5)=0.6324..., sqrt(4/2.5)=1.2649...
        let weights = vec![1.0_f32, 4.0];
        let (mults, _) = preprocess_topology(&weights, &[]);
        assert!(close(mults[0], (1.0_f32 / 2.5).sqrt()), "got {}", mults[0]);
        assert!(close(mults[1], (4.0_f32 / 2.5).sqrt()), "got {}", mults[1]);
    }

    #[test]
    fn edge_alpha_is_reciprocal_of_raw_weight_product() {
        // Alpha uses RAW (unnormalized) weights: 1/(w_a*w_b).
        let weights = vec![2.0_f32, 5.0, 4.0];
        let edges = vec![[0, 1], [1, 2]];
        let (_, alphas) = preprocess_topology(&weights, &edges);
        assert!(close(alphas[0], 1.0 / (2.0 * 5.0)), "got {}", alphas[0]);
        assert!(close(alphas[1], 1.0 / (5.0 * 4.0)), "got {}", alphas[1]);
    }

    #[test]
    fn nonpositive_mean_weight_falls_back_to_units() {
        // All-zero weights → mean 0 → size multipliers all 1.0, no NaN.
        let weights = vec![0.0_f32; 3];
        let edges = vec![[0, 1]];
        let (mults, alphas) = preprocess_topology(&weights, &edges);
        for m in &mults {
            assert!(close(*m, 1.0) && m.is_finite(), "size multiplier {m}");
        }
        // Product is 0 → alpha falls back to 1.0.
        for a in &alphas {
            assert!(close(*a, 1.0) && a.is_finite(), "edge alpha {a}");
        }

        // Negative mean: weights sum to a negative mean, still all-1.0 fallback, no NaN.
        let neg = vec![-1.0_f32, -2.0];
        let (mults, _) = preprocess_topology(&neg, &[]);
        for m in &mults {
            assert!(close(*m, 1.0) && m.is_finite(), "neg-mean multiplier {m}");
        }
    }

    #[test]
    fn edge_length_ratio_uses_dimension_exponent() {
        // w_a·w_b = 16 → alpha 1/16; at d=4 the normalizer is 16^(1/4) = 2, at d=2 it's
        // 16^(1/2) = 4. Distance 3 along one axis → r = 1.5 vs 0.75. Pins the ^(1/d)
        // generalization of the old hardcoded sqrt.
        let alphas = vec![1.0_f32 / 16.0];
        let edges = vec![[0_u32, 1]];
        let positions4 = vec![0.0_f32, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0];
        let mut out = Vec::new();
        edge_length_ratios(
            &positions4,
            4,
            &edges,
            &edge_inv_norms(&alphas, 4),
            &mut out,
        );
        assert!(close(out[0], 1.5), "d=4 ratio {}", out[0]);

        let positions2 = vec![0.0_f32, 0.0, 3.0, 0.0];
        edge_length_ratios(
            &positions2,
            2,
            &edges,
            &edge_inv_norms(&alphas, 2),
            &mut out,
        );
        assert!(close(out[0], 0.75), "d=2 ratio {}", out[0]);
    }

    #[test]
    fn auto_edge_length_range_tracks_quantiles_and_keeps_anchor() {
        // 90 ideal edges + 10 stretched 8× — the warm side must reach the tail while the
        // tail-less cool side collapses to the r=1 anchor (stays strictly negative).
        let mut ratios = vec![1.0_f32; 90];
        ratios.extend(vec![8.0_f32; 10]);
        let (lo, hi) = edge_length_log_range(&ratios).expect("range");
        assert!((-1e-5..0.0).contains(&lo), "cool side anchored: {lo}");
        assert!((hi - 8.0_f32.ln()).abs() < 1e-3, "warm side at tail: {hi}");

        // zero/non-finite ratios are ignored, not propagated into the window
        let degenerate = vec![0.0_f32, f32::NAN, 0.5];
        let (lo, _) = edge_length_log_range(&degenerate).expect("range");
        assert!(
            (lo - 0.5_f32.ln()).abs() < 1e-6,
            "lo from the finite ratio: {lo}"
        );
    }

    #[test]
    fn out_of_range_edge_vertex_falls_back_to_unit_alpha() {
        // Edge references vertex 5 which is beyond weights.len() — no panic, alpha 1.0.
        let weights = vec![2.0_f32, 3.0];
        let edges = vec![[0, 5]];
        let (_, alphas) = preprocess_topology(&weights, &edges);
        assert_eq!(alphas.len(), 1);
        assert!(close(alphas[0], 1.0), "out-of-range alpha {}", alphas[0]);
    }
}
