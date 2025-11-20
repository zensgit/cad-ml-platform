# Batch T — Provider Tuning & A/B Harness

Scope
- Introduce configurable PaddleOCR parameter trials and improved bbox scoring integrating line confidence.

Changes
- BBox mapper scoring adjusted: weights (similarity 0.5, numeric proximity 0.25, type hint 0.1, line score 0.15) for dimensions; symbols use similarity 0.7 + line score 0.3.
- Added `scripts/ocr/paddle_ab_tune.py` script to run multiple Paddle configurations and print CSV summary.

Usage
```bash
python scripts/ocr/paddle_ab_tune.py > reports/paddle_ab_results.csv
```

Result Columns
- name: trial name
- latency_ms: end-to-end provider latency
- dimensions_count / symbols_count: proxy recall indicators (heuristic)
- extraction_mode: provider_native|regex_only|error:...

Rationale
- Early heuristic comparison without full dataset; guides which config to prioritize for real benchmarks.

Acceptance
- Script executes without exceptions in environment with Python & PaddleOCR installed (falls back if not available).
- Existing tests remain green.

Future Work
- Integrate real images and compute true recall vs ground truth.
- Add automatic selection logic based on latency-quality Pareto frontier.

