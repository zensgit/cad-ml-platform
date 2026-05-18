# CAD ML Manufacturing Evidence CI Artifact Verification

Date: 2026-05-12

## Scope

Validated the forward scorecard CI wrapper outputs and the evaluation-report workflow
artifact upload path for manufacturing evidence benchmark summaries.

## Commands

```bash
bash -n scripts/ci/build_forward_scorecard_optional.sh
```

```bash
PYTHONPYCACHEPREFIX=/tmp/cad_ml_platform_pycache python3 -m py_compile \
  tests/unit/test_forward_scorecard_release_gate.py \
  tests/unit/test_evaluation_report_workflow_forward_scorecard.py
```

```bash
.venv311/bin/flake8 \
  tests/unit/test_forward_scorecard_release_gate.py \
  tests/unit/test_evaluation_report_workflow_forward_scorecard.py
```

```bash
.venv311/bin/pytest -q \
  tests/unit/test_forward_scorecard_release_gate.py \
  tests/unit/test_evaluation_report_workflow_forward_scorecard.py
```

```bash
.venv311/bin/python -c "import yaml; yaml.safe_load(open('.github/workflows/evaluation-report.yml', encoding='utf-8'))"
```

```bash
git diff --check
```

## Results

- Bash syntax validation passed for the forward scorecard wrapper.
- Python compile passed for touched tests.
- Flake8 passed for touched tests.
- Targeted pytest passed: `10 passed, 7 warnings in 3.32s`.
- Workflow YAML parse passed.
- `git diff --check` passed.
- Warnings are existing `ezdxf`/`pyparsing` dependency deprecation warnings.

## Verified Behavior

- The wrapper reports `manufacturing_evidence_summary_available=true` when it
  consumes a manufacturing evidence summary path.
- The wrapper reports the exact consumed summary path through
  `manufacturing_evidence_summary_json`.
- `evaluation-report.yml` uploads that exact path as
  `manufacturing-evidence-benchmark-summary-${{ github.run_number }}`.
- The upload step runs after `Upload forward scorecard` and before adjacent
  benchmark artifact uploads.
