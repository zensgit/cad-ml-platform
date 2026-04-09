# Evaluation Comment Support Manifest Validation

## Scope

- Add a single-entry manifest for the `evaluation-comment-support` artifact bundle.
- Wire the manifest into `evaluation-report.yml` step summary and uploaded artifacts.
- Extend regression gates so the manifest generator is covered by existing workflow tests.

## Changed Files

- `scripts/ci/generate_evaluation_comment_support_manifest.py`
- `.github/workflows/evaluation-report.yml`
- `tests/unit/test_generate_evaluation_comment_support_manifest.py`
- `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`
- `tests/unit/test_graph2d_parallel_make_targets.py`
- `Makefile`

## Validation Commands

```bash
pytest -q \
  tests/unit/test_generate_evaluation_comment_support_manifest.py \
  tests/unit/test_evaluation_report_workflow_graph2d_extensions.py \
  tests/unit/test_graph2d_parallel_make_targets.py

TMPDIR=$PWD/.tmp_pytest \
PYTEST_ADDOPTS="--basetemp=$PWD/.tmp_pytest/basetemp" \
make validate-eval-with-history-ci-workflows
```

## Expected Outcome

- The manifest generator reports `ok`, `warning`, or `error` based on support file presence and JSON validity.
- `evaluation-report.yml` generates `evaluation_comment_support_manifest.json/md`.
- The manifest markdown is appended to `GITHUB_STEP_SUMMARY`.
- The manifest JSON and markdown are uploaded inside the `evaluation-comment-support-*` artifact.
- `validate-eval-with-history-ci-workflows` includes the new manifest generator regression test.
