//! Inspection-mode state machine — the "am I looking at the graph or watching
//! it embed?" concept, made structural.
//!
//! Three states:
//! - **Live**: embedder is running. No picker, no hover, camera auto-fits to
//!   incoming positions.
//! - **Paused**: user hit space (or the pause button). Picker built over the
//!   frozen point cloud; hover / drag / zoom-at-cursor active.
//! - **Finished**: embedder converged and auto-paused. Same interactive
//!   affordances as Paused, distinguished only for UI purposes (only Paused
//!   shows a "resume" button; Finished shows the disabled "finished" state).
//!
//! Paused and Finished share the same interactive surface via an inner
//! [`Session`] — the tag distinguishes intent rather than shape. That's why
//! Q2 landed on the inner-struct model: when Paused vs Finished behavior
//! diverges later (e.g. Finished shows a "reset" button, or disables camera
//! pan) the divergent fields move onto the specific variant.
//!
//! Frame-flag drive: App calls [`InspectionMode::sync`] once per tick with
//! the embedder's current `paused` and `finished` flags plus the position
//! cloud. Transitions live in `sync`. No embedder reference is held — the
//! entire module is testable without a Window or a wgpu device.

use crate::pick::Picker;

/// Interactive state shared by [`InspectionMode::Paused`] and
/// [`InspectionMode::Finished`]. Owns the picker (a k-d tree over the frozen
/// point cloud) and the currently hovered vertex.
///
/// Private — external callers query state via [`InspectionMode`]'s methods
/// (`hover`, `is_interactive`, ...) rather than destructuring the variant.
pub struct Session {
    picker: Picker,
    hover: Option<u32>,
}

impl Session {
    /// Build a session over the current position cloud. Returns `None` when
    /// there's nothing to pick against (positions empty) — [`InspectionMode`]
    /// stays in Live in that case rather than materialize a useless session.
    fn new(positions: &[f32]) -> Option<Self> {
        Some(Self {
            picker: Picker::for_positions(positions)?,
            hover: None,
        })
    }
}

#[derive(Default)]
pub enum InspectionMode {
    #[default]
    Live,
    Paused(Session),
    Finished(Session),
}

/// Which state the current flags say we should be in. Computed once at the
/// top of `sync` and matched against the current state to decide the transition.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum TargetKind {
    Live,
    Paused,
    Finished,
}

impl TargetKind {
    fn from_flags(paused: bool, finished: bool) -> Self {
        // Finished dominates: the embedder worker sets paused=true when it
        // flips finished=true, but we don't want to depend on that ordering.
        // Finished ⇒ Finished; paused-but-not-finished ⇒ Paused; else Live.
        if finished {
            TargetKind::Finished
        } else if paused {
            TargetKind::Paused
        } else {
            TargetKind::Live
        }
    }
}

impl InspectionMode {
    pub fn new() -> Self {
        InspectionMode::Live
    }

    /// Reconcile with the embedder's flags. Idempotent when the target state
    /// matches the current state. Transitions:
    ///
    /// - Live → Paused/Finished: build a fresh [`Session`] from `positions`.
    ///   If positions is empty, stays Live.
    /// - Paused/Finished → Live: drop the session (picker + hover).
    /// - Paused ↔ Finished: reuse the same session (picker positions haven't
    ///   moved since the pause), preserving hover.
    pub fn sync(&mut self, paused: bool, finished: bool, positions: &[f32]) {
        let target = TargetKind::from_flags(paused, finished);
        // std::mem::replace lets us match `current` by value and rebuild `self`
        // from what we take. `Live` is the cheap default while we're computing.
        let current = std::mem::replace(self, InspectionMode::Live);
        *self = match (current, target) {
            // Target is Live — regardless of where we came from, drop the session.
            (_, TargetKind::Live) => InspectionMode::Live,
            // Entering inspection — build a session, or stay Live if we can't.
            (InspectionMode::Live, TargetKind::Paused) => match Session::new(positions) {
                Some(s) => InspectionMode::Paused(s),
                None => InspectionMode::Live,
            },
            (InspectionMode::Live, TargetKind::Finished) => match Session::new(positions) {
                Some(s) => InspectionMode::Finished(s),
                None => InspectionMode::Live,
            },
            // Already in the target state — preserve everything (picker + hover).
            (existing @ InspectionMode::Paused(_), TargetKind::Paused) => existing,
            (existing @ InspectionMode::Finished(_), TargetKind::Finished) => existing,
            // Transfer the session between the two inspection variants.
            (InspectionMode::Paused(s), TargetKind::Finished) => InspectionMode::Finished(s),
            (InspectionMode::Finished(s), TargetKind::Paused) => InspectionMode::Paused(s),
        };
    }

    /// Rebuild the picker after a single-step or topology update while inspection remains
    /// active. Position changes invalidate both the spatial index and the hovered vertex.
    pub fn update_positions(&mut self, positions: &[f32]) {
        let current = std::mem::replace(self, InspectionMode::Live);
        *self = match current {
            InspectionMode::Live => InspectionMode::Live,
            InspectionMode::Paused(_) => match Session::new(positions) {
                Some(session) => InspectionMode::Paused(session),
                None => InspectionMode::Live,
            },
            InspectionMode::Finished(_) => match Session::new(positions) {
                Some(session) => InspectionMode::Finished(session),
                None => InspectionMode::Live,
            },
        };
    }

    /// Update hover from a cursor world-position. No-op in Live. When the
    /// cursor doesn't hit any vertex, hover becomes `None`.
    pub fn hit_test(&mut self, cursor_world: [f32; 2], rendered_radius: f32) {
        if let Some(session) = self.session_mut() {
            session.hover = session.picker.hit_test(cursor_world, rendered_radius);
        }
    }

    /// Clear the current hover — cursor left the window, moved over the UI,
    /// etc. No-op in Live.
    pub fn clear_hover(&mut self) {
        if let Some(session) = self.session_mut() {
            session.hover = None;
        }
    }

    /// The vertex currently under the cursor, or `None`.
    pub fn hover(&self) -> Option<u32> {
        self.session().and_then(|s| s.hover)
    }

    pub fn is_paused(&self) -> bool {
        matches!(self, InspectionMode::Paused(_))
    }

    pub fn is_finished(&self) -> bool {
        matches!(self, InspectionMode::Finished(_))
    }

    /// True when the user should be able to drag/zoom the camera and when
    /// hover picks should fire — Paused or Finished, either way.
    pub fn is_interactive(&self) -> bool {
        !matches!(self, InspectionMode::Live)
    }

    fn session(&self) -> Option<&Session> {
        match self {
            InspectionMode::Live => None,
            InspectionMode::Paused(s) | InspectionMode::Finished(s) => Some(s),
        }
    }

    fn session_mut(&mut self) -> Option<&mut Session> {
        match self {
            InspectionMode::Live => None,
            InspectionMode::Paused(s) | InspectionMode::Finished(s) => Some(s),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn some_positions() -> Vec<f32> {
        vec![0.0, 0.0, 1.0, 1.0, 2.0, 0.5]
    }

    #[test]
    fn starts_live() {
        let mode = InspectionMode::new();
        assert!(!mode.is_paused());
        assert!(!mode.is_finished());
        assert!(!mode.is_interactive());
        assert_eq!(mode.hover(), None);
    }

    #[test]
    fn live_to_paused_and_back() {
        let mut mode = InspectionMode::new();
        let positions = some_positions();

        mode.sync(true, false, &positions);
        assert!(mode.is_paused());
        assert!(mode.is_interactive());

        mode.sync(false, false, &positions);
        assert!(!mode.is_paused());
        assert!(!mode.is_interactive());
    }

    #[test]
    fn paused_to_finished_preserves_session() {
        let mut mode = InspectionMode::new();
        let positions = some_positions();

        mode.sync(true, false, &positions);
        mode.hit_test([0.0, 0.0], 0.5);
        let hover_before = mode.hover();
        assert!(hover_before.is_some(), "test setup expected a hit");

        mode.sync(true, true, &positions);
        assert!(mode.is_finished());
        assert_eq!(mode.hover(), hover_before);
    }

    #[test]
    fn paused_position_update_rebuilds_picker_and_clears_hover() {
        let mut mode = InspectionMode::new();
        let initial = [0.0_f32, 0.0, 10.0, 10.0];
        mode.sync(true, false, &initial);
        mode.hit_test([0.0, 0.0], 1.0);
        assert_eq!(mode.hover(), Some(0));

        let stepped = [10.0_f32, 10.0, 20.0, 20.0];
        mode.update_positions(&stepped);
        assert_eq!(mode.hover(), None);
        mode.hit_test([0.0, 0.0], 1.0);
        assert_eq!(mode.hover(), None, "old positions must not remain pickable");
        mode.hit_test([10.0, 10.0], 1.0);
        assert_eq!(mode.hover(), Some(0), "new positions must be pickable");
    }

    #[test]
    fn live_stays_live_with_empty_positions() {
        let mut mode = InspectionMode::new();
        mode.sync(true, false, &[]);
        assert!(matches!(mode, InspectionMode::Live));
    }

    #[test]
    fn hit_test_noop_in_live() {
        let mut mode = InspectionMode::new();
        mode.hit_test([0.0, 0.0], 100.0);
        assert_eq!(mode.hover(), None);
    }

    #[test]
    fn clear_hover_noop_in_live() {
        let mut mode = InspectionMode::new();
        mode.clear_hover();
        assert!(!mode.is_interactive());
    }
}
