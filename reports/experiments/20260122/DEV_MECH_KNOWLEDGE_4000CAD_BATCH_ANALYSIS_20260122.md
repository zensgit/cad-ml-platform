# DEV_MECH_KNOWLEDGE_4000CAD_BATCH_ANALYSIS_20260122

## Summary
- Ran local batch analyze (TestClient) on training DXF files using the parts-upsampled Graph2D model.
- Captured label distribution, confidence buckets, and low-confidence samples for review.

## Inputs
- DXF directory: `/Users/huazhou/Downloads/训练图纸/训练图纸_dxf`
- Sample size: 110 (max-files=200, seed=22)
- Model: `models/graph2d_parts_upsampled_20260122.pth`

## Outputs
- Results: `reports/experiments/20260122/batch_analysis_graph2d/batch_results.csv`
- Label distribution: `reports/experiments/20260122/batch_analysis_graph2d/label_distribution.csv`
- Low-confidence samples: `reports/experiments/20260122/batch_analysis_graph2d/batch_low_confidence.csv`
- Summary: `reports/experiments/20260122/batch_analysis_graph2d/summary.json`

## Key Stats
- Success: 110/110
- Low-confidence (<=0.6): 53
- Confidence buckets: <0.4=0, 0.4-0.6=28, 0.6-0.8=25, >=0.8=57
- Top labels: complex_assembly(28), moderate_component(25), 机械制图(14), 盖(11)

## Commands
- `GRAPH2D_ENABLED=true GRAPH2D_FUSION_ENABLED=true FUSION_ANALYZER_ENABLED=true GRAPH2D_MODEL_PATH=models/graph2d_parts_upsampled_20260122.pth .venv-graph/bin/python scripts/batch_analyze_dxf_local.py --dxf-dir /Users/huazhou/Downloads/训练图纸/训练图纸_dxf --output-dir reports/experiments/20260122/batch_analysis_graph2d --max-files 200 --seed 22 --min-confidence 0.6`

## Notes
- Batch analysis uses FastAPI TestClient (no external server required).
