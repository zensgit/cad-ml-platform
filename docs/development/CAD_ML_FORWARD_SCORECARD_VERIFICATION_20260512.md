# CAD ML Forward Scorecard Verification

Date: 2026-05-12

## Scope

Verified the `forward-scorecard-v1` slice:

- reusable scorecard builder;
- CLI exporter;
- default JSON/Markdown report generation;
- unit tests for release/gap/blocking behavior.

## Commands Run

### Syntax

```bash
PYTHONPYCACHEPREFIX=/tmp/cad_ml_platform_pycache python3 -m py_compile \
  src/core/benchmark/forward_scorecard.py \
  scripts/export_forward_scorecard.py \
  tests/unit/test_forward_scorecard.py
```

Result: passed.

### Lint

```bash
.venv311/bin/flake8 \
  src/core/benchmark/forward_scorecard.py \
  scripts/export_forward_scorecard.py \
  tests/unit/test_forward_scorecard.py \
  --max-line-length=100
```

Result: passed.

### Unit Tests

```bash
.venv311/bin/python -m pytest -q tests/unit/test_forward_scorecard.py
```

Result: `5 passed, 7 warnings`.

The warnings are existing `ezdxf` / `pyparsing` deprecation warnings:

- `addParseAction` deprecated
- `oneOf` deprecated
- `setResultsName` deprecated
- `infixNotation` deprecated

They are dependency warnings, not scorecard failures.

### Report Generation

```bash
.venv311/bin/python scripts/export_forward_scorecard.py
```

Result: generated:

```text
reports/benchmark/forward_scorecard/latest.json
reports/benchmark/forward_scorecard/latest.md
```

The generated local scorecard currently reports:

```text
overall_status=blocked
```

This is expected for the default local run because no benchmark artifact inputs
were supplied. The scorecard correctly treats absent Hybrid DXF, Graph2D blind,
History Sequence, B-Rep, Qdrant, and knowledge evidence as non-release-ready
instead of inferring readiness.

## Verified Behavior

- `release_ready` is possible only when all components have release-grade
  evidence.
- Fallback model readiness prevents `release_ready` and produces
  `benchmark_ready_with_gap`.
- Missing Hybrid DXF benchmark evidence blocks release claims.
- B-Rep/3D is scored independently from stronger 2D Hybrid DXF signals.
- The CLI writes both JSON and Markdown outputs.
- Markdown includes the release-claim rule and component evidence table.

## Remaining Verification Gaps

- CI has not yet been wired to pass real benchmark artifact paths into
  `scripts/export_forward_scorecard.py`.
- No release gate currently fails PR labels or deployments based on the scorecard
  status.
- Trend validation is pending until multiple scorecard snapshots are produced.
