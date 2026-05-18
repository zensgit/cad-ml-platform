# CAD ML Review Manifest Merge Duplicate Base Block Verification

Date: 2026-05-14

## Scope

Verified approved review-manifest merge blocking for duplicate base benchmark
manifest identities, ambiguous file-name fallback rows, merge audit diagnostics,
Claude Code read-only review, and focused regression coverage.

## Commands

```bash
<targeted snippets> | claude -p --tools "" --max-budget-usd 0.50
```

Result: Claude Code found no clean-base merge regression. It identified a
file-name fallback ambiguity caveat; this was implemented and then covered by
local tests.

```bash
bash -n scripts/ci/build_forward_scorecard_optional.sh
```

Result: passed.

```bash
PYTHONPYCACHEPREFIX=/tmp/cad_ml_platform_pycache python3 -m py_compile \
  scripts/build_manufacturing_review_manifest.py \
  tests/unit/test_build_manufacturing_review_manifest.py \
  tests/unit/test_forward_scorecard_release_gate.py \
  tests/unit/test_evaluation_report_workflow_forward_scorecard.py
```

Result: passed.

```bash
.venv311/bin/flake8 \
  scripts/build_manufacturing_review_manifest.py \
  tests/unit/test_build_manufacturing_review_manifest.py \
  tests/unit/test_forward_scorecard_release_gate.py \
  tests/unit/test_evaluation_report_workflow_forward_scorecard.py
```

Result: passed.

```bash
.venv311/bin/pytest -q \
  tests/unit/test_build_manufacturing_review_manifest.py \
  tests/unit/test_forward_scorecard_release_gate.py \
  tests/unit/test_evaluation_report_workflow_forward_scorecard.py
```

Result: `56 passed, 7 warnings in 6.44s`.

Warnings: existing third-party `ezdxf`/`pyparsing` deprecation warnings surfaced
during manifest tests. No project-code failure.

```bash
git diff --check
```

Result: passed.

```bash
rg -n "[ \t]+$" \
  scripts/build_manufacturing_review_manifest.py \
  scripts/ci/build_forward_scorecard_optional.sh \
  tests/unit/test_build_manufacturing_review_manifest.py \
  tests/unit/test_forward_scorecard_release_gate.py \
  tests/unit/test_evaluation_report_workflow_forward_scorecard.py \
  docs/development/CAD_ML_DETAILED_DEVELOPMENT_TODO_20260512.md \
  docs/development/CAD_ML_REVIEW_MANIFEST_MERGE_DUPLICATE_BASE_BLOCK_DEVELOPMENT_20260514.md \
  docs/development/CAD_ML_REVIEW_MANIFEST_MERGE_DUPLICATE_BASE_BLOCK_VERIFICATION_20260514.md
```

Result: no trailing whitespace matches.

## Coverage

- Clean base benchmark manifests still merge approved review rows normally.
- Duplicate base benchmark manifest identities block approved review-manifest
  merge.
- Duplicate base file names block file-name fallback when a review row omits a
  precise matching `relative_path`.
- Precise `relative_path` matches still merge even when the base manifest
  contains the same file name in another relative path.
- Mixed batches stay blocked when they contain any ambiguous file-name fallback
  row, even if other precise rows merge.
- Blocked merge leaves base rows unchanged.
- Merge summary exposes duplicate base identity count and identifiers.
- Merge audit marks eligible rows as `blocked_duplicate_base_manifest`.
- Merge audit marks file-name fallback ambiguity as `ambiguous_file_name_match`.
- Existing reviewer-template apply/preflight, CI wrapper, and workflow tests
  remain in the focused regression set.

## Residual Risk

- This detects ambiguous base benchmark manifests but does not choose or repair
  the canonical row.
- Release readiness still depends on qualified reviewer labels and later
  threshold tuning.
