# PR Stack Merge And Rollback Plan

## PR Stack

1. `#85` `feat: reconcile history sequence tooling and validation`
2. `#84` `feat: emit graph2d training metrics artifacts`
3. `#82` `fix: tighten assistant security event time filters`
4. `#83` `feat: extend graph2d evaluation diagnostics`

## Merge Order

Recommended order:

1. `#85`
2. `#84`
3. `#82`
4. `#83`

Reasoning:

- `#83` depends on `#85` because `scripts/eval_with_history.sh` invokes the history-sequence tooling introduced there.
- `#84` and `#82` are independent of the history-sequence stack and can merge after `#85`.
- `#82` is isolated and low-risk, so it can merge before or after `#84`; keeping it third preserves the main feature-first order.

## Rollback Order

Rollback in reverse dependency order:

1. `#83`
2. `#82`
3. `#84`
4. `#85`

Reasoning:

- Revert the dependent eval branch before reverting the history tooling it consumes.
- Revert isolated branches independently.

## Validation Gates Before Merge

For each PR:

- GitHub Actions required checks green
- branch clean after latest docs commit
- branch-specific validation doc reviewed

Additional for `#83`:

- verify base branch remains `feat/history-sequence-reconcile` until `#85` merges
- rebase or retarget to `main` after `#85` merge if needed

## Post-Merge Checks

After merging `#85` and `#83`:

- verify `scripts/eval_with_history.sh` still references existing history tooling paths
- rerun targeted history-sequence tests on `main`

After merging `#84`:

- verify `train_2d_graph.py --metrics-out` still emits expected JSON fields

After merging `#82`:

- verify boundary-time filtering callers do not rely on inclusive semantics
