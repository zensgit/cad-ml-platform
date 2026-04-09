# PR Stack Finalization 2026-03-06

## Scope

This document records the final state of the four parallel pull requests that
were prepared in isolated worktrees, validated, merged into `main`, and then
cleaned up locally.

## Merged Pull Requests

1. `#85` `feat: reconcile history sequence tooling and validation`
   - Merged at: `2026-03-06T14:05:53Z`
   - Merge commit: `b0df3b5981a28f4b3607eb25848c10f76351a333`
2. `#84` `feat: emit graph2d training metrics artifacts`
   - Merged at: `2026-03-06T14:06:25Z`
   - Merge commit: `c3ada8a99a5e713e123467760d6b0e5a3b08e224`
3. `#82` `fix: tighten assistant security event time filters`
   - Merged at: `2026-03-06T14:06:44Z`
   - Merge commit: `8eca676842fe075337cb1d4540cd96c68346552e`
4. `#83` `feat: extend graph2d evaluation diagnostics`
   - Base branch was updated from `feat/history-sequence-reconcile` to `main`
     after `#85` merged.
   - Merged at: `2026-03-06T14:07:05Z`
   - Merge commit: `48e300e62afdb9b6602f6cbd2cd882b7e46712f7`

## Merge Order

The merge order was intentionally fixed to preserve branch dependencies:

1. `#85`
2. `#84`
3. `#82`
4. `#83`

This order ensured that the history-sequence toolchain landed before the
Graph2D evaluation diagnostics branch was rebased to `main`.

## Validation Basis

Each PR had already passed its targeted local validation before push:

- `#85`: history-sequence tooling, datasets, scripts, and validation tests
- `#84`: Graph2D training metrics artifact emission
- `#82`: assistant security time-boundary filtering
- `#83`: Graph2D evaluation diagnostics and history-aware eval helpers

During merge, GitHub required checks were satisfied. Some non-blocking checks
continued to complete after merge due to repository workflow configuration.

## Local Cleanup Completed

After merge, the temporary worktrees used for isolated development were removed:

- `/private/tmp/cad-ml-platform-main-clean`
- `/private/tmp/cad-ml-platform-graph2d-train`
- `/private/tmp/cad-ml-platform-assistant-security`
- `/private/tmp/cad-ml-platform-graph2d-eval`

The corresponding local branches were also deleted:

- `feat/history-sequence-reconcile`
- `feat/graph2d-train-metrics`
- `feat/assistant-security-boundary`
- `feat/graph2d-eval-history`
- `work/main-clean-20260306`

## Current Safe Development Baseline

- Remote `main` head after the final merge: `48e300e62afdb9b6602f6cbd2cd882b7e46712f7`
- New clean local worktree created from `origin/main`:
  - `/private/tmp/cad-ml-platform-main-post-merge`
  - branch: `work/main-post-merge-20260306`

This new worktree is the correct place for subsequent development. The original
main worktree remains intentionally untouched because it still contains user
local changes.
