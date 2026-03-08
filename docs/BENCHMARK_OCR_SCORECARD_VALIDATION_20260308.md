# Benchmark OCR Scorecard Validation

## Goal
- Add OCR review-pack readiness into `scripts/generate_benchmark_scorecard.py`.
- Surface the OCR benchmark component through `.github/workflows/evaluation-report.yml`.
- Keep the scorecard aligned with the benchmark/competitive roadmap by treating OCR
  review load as a first-class benchmark signal.

## Key Changes
- Added scorecard component:
  - `ocr_review`
- Added CLI input:
  - `--ocr-review-summary`
- Added workflow dispatch input:
  - `benchmark_scorecard_ocr_review_summary`
- Added workflow env:
  - `BENCHMARK_SCORECARD_OCR_REVIEW_SUMMARY_JSON`
- Benchmark workflow now feeds OCR review summary from:
  - workflow-dispatch input
  - OCR review-pack step output
  - env fallback
- Benchmark outputs now include:
  - `ocr_status`
- Job summary and PR comment now expose benchmark OCR status.

## OCR Component Semantics
- `missing`: no OCR review summary available
- `ocr_ready`: no OCR review candidates remain
- `mostly_ready`: OCR still has review candidates, but readiness is already high
- `managed_review`: OCR still needs review, but the backlog is manageable
- `review_heavy`: OCR review backlog is still a major benchmark gap

## Validation Commands
```bash
python3 - <<'PY'
import yaml
from pathlib import Path
p = Path('.github/workflows/evaluation-report.yml')
yaml.safe_load(p.read_text(encoding='utf-8'))
print('yaml_ok')
PY

python3 -m py_compile \
  scripts/generate_benchmark_scorecard.py \
  tests/unit/test_generate_benchmark_scorecard.py \
  tests/unit/test_evaluation_report_workflow_graph2d_extensions.py

flake8 \
  scripts/generate_benchmark_scorecard.py \
  tests/unit/test_generate_benchmark_scorecard.py \
  tests/unit/test_evaluation_report_workflow_graph2d_extensions.py \
  --max-line-length=100

pytest -q \
  tests/unit/test_generate_benchmark_scorecard.py \
  tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
```

## Expected Result
- Benchmark scorecard contains `ocr_review` alongside hybrid/history/brep/governance/
  assistant/review_queue.
- OCR readiness can influence the overall benchmark status and recommendations.
- CI workflow exposes the OCR benchmark status in both job summary and PR comment.
