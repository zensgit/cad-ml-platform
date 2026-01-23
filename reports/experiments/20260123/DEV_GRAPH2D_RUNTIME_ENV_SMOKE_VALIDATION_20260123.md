# DEV_GRAPH2D_RUNTIME_ENV_SMOKE_VALIDATION_20260123

## Checks
- Loaded `.env` recommended Graph2D settings and ran a 20-file DXF batch analysis.

## Runtime Output
- Command:
  - `set -o allexport && source .env && set +o allexport && .venv-graph/bin/python scripts/batch_analyze_dxf_local.py --dxf-dir "$DXF_DIR" --output-dir reports/experiments/20260123/dxf_batch_analysis_graph2d_runtime_env --max-files 20 --seed 23 --min-confidence 0.6`
- Result:
  - Total samples: 20
  - Graph2D labels present: 20
  - Allowlist hits (`再沸器`): 2
  - Allowlist + min_conf>=0.7: 2

## Notes
- `batch_results.csv` does not include `allowed` or `passed_threshold`; allowlist gating is inferred.
