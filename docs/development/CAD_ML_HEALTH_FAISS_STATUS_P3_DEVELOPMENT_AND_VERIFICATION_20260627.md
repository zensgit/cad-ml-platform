# Health router slimming — `faiss_health` status/age extraction (P3 slice)

Date: 2026-06-27
Branch: `claude/health-faiss-status-extract-20260627`
Line: 背景工程 / Priority 3 (router slimming), continuation of #493 (`compute_model_health`)
and #494 (dedup constants/models). Pure-logic extraction, behaviour-preserving.

## Background

The `/goal` for this line is explicit: P1 (real-data closed loop) and P2 (manufacturing
reviewed labels) are both human-gated on data the assistant cannot produce, so the standing
fallback is *"如果数据/人工审核暂时没法推进,再做 deeper P3"* — specifically *"health.py 其他纯
helper"*. The two named P3 targets are already merged (#493 model_health, #494 dedup
constants/models), so this slice takes the next clean pure block in `health.py`.

Explicitly out of scope per the same directive: `precision-L4`, Redis monkeypatch, async
provider (higher risk — left untouched).

## What changed

Extracted the inline status/age derivation of the `faiss_health` handler into a pure,
side-effect-free helper.

New module `src/core/health_faiss_status.py`:

```python
compute_faiss_health(*, available, degraded, last_export_ts, last_import, now=None)
    -> {"status": str, "age_seconds": Optional[int]}
```

- `status` priority is a verbatim port: `degraded > unavailable > ok`.
- `age_seconds` is measured from the last export timestamp, falling back to the last
  import timestamp, `None` when neither is known — verbatim port of the prior inline block.
- `now` is injectable for deterministic tests, defaulting to `time.time()` to preserve
  behaviour (mirrors the `compute_model_health` pattern from #493).

`src/api/v1/health.py`:

- Imports `compute_faiss_health` (re-exported at module scope for testing, same convention
  as `compute_model_health` / `compute_cache_tuning`).
- The handler now calls the helper after `get_degraded_mode_info()` and reads
  `status` / `age_seconds` from the result.
- The now-unused function-local `import time` was removed.

### Deliberately NOT touched

`next_recovery_eta` is intertwined with Prometheus side effects (it sets the gauge to the
ETA value or to `0`, inside a defensive `try/except`). Extracting it would risk changing the
"exception → no gauge write" branch, so it is left exactly as-is. This keeps the slice
strictly behaviour-preserving.

## Verification

Test interpreter: `.venv311` (Python 3.11.15), matching CI.

| Check | Command | Result |
|---|---|---|
| New unit + facade tests | `pytest tests/unit/test_health_faiss_status.py` | 8 passed |
| Sibling regression (#493) | `pytest tests/unit/test_health_model_status.py` | 5 passed |
| FAISS endpoint regression (behaviour guard) | `pytest tests/unit/test_faiss_health_endpoint.py test_faiss_eta_reset_on_recovery.py test_faiss_flapping_suppression.py` | 3 passed |
| flake8 (`.flake8`, the blocking lint gate) | `flake8 <changed files>` | clean |

New tests cover: helper re-export from the router module; status priority across all three
branches incl. `degraded` winning over `unavailable`; age from export ts; age fallback to
import ts; `None` when no timestamps; and `int()` truncation.

### Note on black / isort

`black --check` / `isort --check-only` are **non-blocking** in CI
(`ci-enhanced.yml`: `black --check src/ tests/ || true`). The locally-installed black version
also wants to reformat pre-existing, already-merged files (#493 siblings, untouched lines of
`health.py`), i.e. it disagrees with the version that formatted `main`. To avoid ballooning
the diff and diverging from `main`'s actual formatting, `health.py` was **not** run through the
local formatter; the hand-written edits match the file's existing style and pass the blocking
flake8 gate.

## Risk / impact

Behaviour-preserving router-internal refactor. No public API, response schema, or Prometheus
metric change. Net `health.py`: 25 changed lines (-15 inline branch logic, +helper call +
import); pure logic now unit-tested in isolation.
