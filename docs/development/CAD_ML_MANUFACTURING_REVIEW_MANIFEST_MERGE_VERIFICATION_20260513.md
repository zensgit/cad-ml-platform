# CAD ML Manufacturing Review Manifest Merge Verification

Date: 2026-05-13

## Scope

Validated the approved-only merge path from manufacturing review manifests back into
benchmark manifests. The validation focused on review status gating, reviewer
metadata enforcement, unmatched-row handling, output CSV behavior, and CLI summary
generation.

## Commands

```bash
PYTHONPYCACHEPREFIX=/tmp/cad_ml_platform_pycache python3 -m py_compile \
  scripts/build_manufacturing_review_manifest.py \
  tests/unit/test_build_manufacturing_review_manifest.py
```

```bash
.venv311/bin/flake8 \
  scripts/build_manufacturing_review_manifest.py \
  tests/unit/test_build_manufacturing_review_manifest.py
```

```bash
.venv311/bin/pytest -q tests/unit/test_build_manufacturing_review_manifest.py
```

```bash
git diff --check
```

## Results

- Python compile passed for the review manifest script and touched tests.
- Flake8 passed for the review manifest script and touched tests.
- Targeted pytest passed: `7 passed, 7 warnings in 2.39s`.
- `git diff --check` passed.
- Warnings are existing `ezdxf`/`pyparsing` dependency deprecation warnings imported
  through existing DXF evaluation helpers.

## Verified Behavior

- Merge mode requires `--base-manifest` and `--output-csv`.
- Approved rows with reviewed source and payload labels are merged into matching base
  manifest rows.
- `relative_path` is used before `file_name` when matching review rows to base rows.
- Rows with `review_status=needs_human_review` are skipped even when reviewed columns
  are populated.
- Approved rows without reviewer metadata are skipped when
  `--require-reviewer-metadata` is enabled.
- Approved rows that do not match the base manifest are counted as unmatched and are
  not written into the merged manifest.
- The merged CSV preserves base manifest rows and appends reviewed label/governance
  columns where needed.
- Merge summary JSON reports approved, merged, skipped, missing metadata, and
  unmatched counts.
