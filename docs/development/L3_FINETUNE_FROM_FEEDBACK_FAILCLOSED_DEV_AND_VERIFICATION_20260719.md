# L3 — seal offline `finetune_from_feedback` train-and-reload bypass — Dev & Verification (2026-07-19)

> **Honest scope.** This slice seals **only** the offline CLI
> `scripts/finetune_from_feedback.py` behind the unconditional
> `scripts/eval_integrity_gate.check()` contract already used by
> `scripts/auto_retrain.sh` (#509). It does **not** claim Track E, Phase B,
> production activation/enablement, or that all offline research training
> scripts are sealed.

## Why

`auto_retrain.sh` is fail-closed at Step 0 (#509). Independently,
`finetune_from_feedback.py` trained from active-learning feedback and called
`reload_model(..., force=True)` with **no** evaluation-integrity gate — a
train-and-reload promotion sink around that stop.

Historical reference only: closed PR #514 explored a broader Phase-A seal
(including API reload). This work is the **narrow offline-finetune slice** and
does not cherry-pick that PR or touch C1 model-activation / unrelated loaders.

## What changed

| File | Change |
|---|---|
| `scripts/finetune_from_feedback.py` | Bootstrap is **stdlib + logging + `_enforce_evaluation_integrity_gate` only**. CLI `if __name__ == "__main__"` runs the gate **before** `numpy` / `src` imports and before `sys.path` mutation. **Deterministic gate import:** if `__package__` set → `scripts.eval_integrity_gate`; if empty (direct CLI) → sibling `eval_integrity_gate` from script dir (`sys.path[0]`) — no repo-root mutation, no broad fallback. `main()` still starts with the same gate. Import/`check()` catch **`BaseException`**; every refusal branch raises explicit `SystemExit(1)`. **`GateBlocked` logs fixed text only**. |
| `tests/unit/test_finetune_from_feedback_failclosed.py` | Fail-closed modes for both import names (incl. import-time `SystemExit(0)` / `KeyboardInterrupt`), `--help`/malformed argv, path-secret log proof, AST lock for numpy/src **and** `sys.path` mutation (with observed-RED mutation fixture), subprocess without repo `PYTHONPATH` proving **canonical GateBlocked log**, poisoned numpy/src with **fake-only** `PYTHONPATH`. |
| `.github/workflows/ci.yml`, `ci-tiered-tests.yml` | New tests on the curated L3 evaluation-integrity fail-closed route. |
| `Makefile` `test-training-governance` | Includes the fail-closed file next to existing finetune tests. |

Existing label/summary unit tests in `tests/unit/test_finetune_from_feedback.py` are
unchanged (helpers only; they do not exercise `main()`).

## Verification

```bash
python3.11 -m pytest -q \
  tests/unit/test_finetune_from_feedback.py \
  tests/unit/test_finetune_from_feedback_failclosed.py \
  tests/unit/test_eval_integrity_gate.py \
  tests/unit/test_auto_retrain_failclosed.py
# → 68 passed

python3.11 -m py_compile scripts/finetune_from_feedback.py \
  tests/unit/test_finetune_from_feedback_failclosed.py
python3.11 -m flake8 scripts/finetune_from_feedback.py \
  tests/unit/test_finetune_from_feedback_failclosed.py \
  --max-line-length=100
python3.11 -m black --check scripts/finetune_from_feedback.py \
  tests/unit/test_finetune_from_feedback_failclosed.py
python3.11 -m isort --check-only --profile black \
  scripts/finetune_from_feedback.py \
  tests/unit/test_finetune_from_feedback_failclosed.py
python3.11 -m mypy scripts/finetune_from_feedback.py

# Direct CLI repro (no PYTHONPATH): must log fixed GateBlocked text, not gate-unavailable
PYTHONPATH= python3.11 scripts/finetune_from_feedback.py --help
# → exit 1, stderr contains: retraining blocked by the evaluation-integrity gate
```

| Proof | Test |
|---|---|
| Gate raises → exit 1; no learner/export/train/write/reload | `test_gate_raises_blocks_before_any_side_effect` |
| Missing gate module → exit 1; zero side effects | `test_missing_gate_module_blocks_with_zero_side_effects` |
| Each import name (`scripts.*` / sibling) denied → exit 1 | `test_missing_each_gate_import_name_blocks` |
| Unusable `check` symbol → exit 1; zero side effects | `test_unusable_gate_symbol_blocks_with_zero_side_effects` |
| Arbitrary gate exception → exit 1; zero side effects | `test_arbitrary_gate_exception_blocks_with_zero_side_effects` |
| Subverted return → invariant breach exit 1 | `test_subverted_gate_return_is_invariant_breach` |
| Import-time `SystemExit(0)` (both names) remapped → exit **1** | `test_import_time_systemexit_zero_is_remapped_to_one` |
| Import-time `KeyboardInterrupt` (both names) → exit **1** + type-only log | `test_import_time_keyboardinterrupt_is_remapped_to_one` |
| Check-time `SystemExit(0)` remapped → exit **1** | `test_check_time_systemexit_zero_is_remapped_to_one` |
| Check-time `KeyboardInterrupt` remapped → exit **1** | `test_check_time_keyboardinterrupt_is_remapped_to_one` |
| `--help` → exit **1** from gate (not argparse 0); no usage dump | `test_help_exits_one_from_gate_not_argparse` |
| Malformed/unknown argv → exit **1** from gate (not argparse 2) | `test_malformed_argv_exits_one_from_gate_not_argparse` |
| No CLI arg / env var opens the path | `test_no_cli_arg_opens_the_gate`, `test_no_env_var_opens_the_gate` |
| `reload_model` not imported/called on refusal | `test_reload_model_never_imported_or_called_on_refusal` |
| Path-bearing `GateBlocked` text absent from logs | `test_gateblocked_path_secret_absent_from_logs` |
| AST: `main().body[0]` is gate; bootstrap before numpy/src **and** `sys.path` | `test_ast_main_body0_is_gate_and_cli_bootstrap_before_heavy_imports` |
| AST mutation: path mutation before bootstrap is observed RED | `test_ast_sys_path_mutation_before_bootstrap_is_observed_red` |
| Direct CLI `--help`/`--force`, no repo `PYTHONPATH`: exit 1 + fixed log | `test_cli_subprocess_hits_canonical_check_without_repo_pythonpath` |
| Fake-only `PYTHONPATH` poisoned numpy/src; exit 1; no sentinel | `test_cli_subprocess_exits_before_poisoned_numpy_import` |

**Focused suite result (this verification run):** **68 passed**
(`pytest --collect-only`: `test_finetune_from_feedback.py` 6 +
`test_finetune_from_feedback_failclosed.py` 34 + `test_eval_integrity_gate.py` 20 +
`test_auto_retrain_failclosed.py` 8 = **68**).

## Explicit non-claims (boundary)

- **Not** Track E (two-phase evaluation-integrity-v2 / real §8.1.4 metrics).
- **Not** Phase B (signed proof, dynamic swap, retrain enablement).
- **Not** production activation membrane / C1 fixed-hash loaders.
- **Not** a seal of other offline research scripts (`train_classifier_*`,
  `finetune_graph2d_*`, `finetune_agent_llm.py`, …) — they remain registered
  residuals unless they promote through a sealed path.
- **Not** re-opening or changing `POST /api/v1/model/reload` (already 403 on
  main via #516) or `auto_retrain.sh` (already gated via #509).

Re-enablement remains a **code change** that replaces
`eval_integrity_gate.check()` with the real two-phase Track E gate — never a
flag, env var, or artifact token.
