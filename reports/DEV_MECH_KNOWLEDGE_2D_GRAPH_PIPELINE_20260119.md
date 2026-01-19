# DEV_MECH_KNOWLEDGE_2D_GRAPH_PIPELINE_20260119

## Summary
- Added DWG manifest + DXF conversion tooling and auto-generated geometry rules for dataset labels.
- Introduced a lightweight 2D graph pipeline (dataset/model/inference) and optional API integration for graph2d predictions.
- Added a training script for weakly labeled DXF graphs and a template generator for English synonym mappings.

## Scope
- Manifest + conversion: `scripts/build_dwg_label_manifest.py`, `scripts/convert_dwg_batch.py`
- Rules generation: `scripts/build_geometry_rules_from_manifest.py`, `data/knowledge/geometry_rules.json`
- 2D graph ML: `src/ml/train/dataset_2d.py`, `src/ml/train/model_2d.py`, `src/ml/vision_2d.py`, `scripts/train_2d_graph.py`
- API integration: `src/api/v1/analyze.py` (graph2d prediction + optional fusion)
- Synonyms template: `scripts/build_label_synonyms_template.py`

## Configuration
- `GRAPH2D_ENABLED=true`: enable DXF graph predictions in analysis responses.
- `GRAPH2D_MODEL_PATH=models/graph2d_latest.pth`: load the graph2d checkpoint.
- `GRAPH2D_FUSION_ENABLED=true`: allow graph2d output to feed the FusionAnalyzer.
- `FUSION_ANALYZER_ENABLED=true`: enable fusion decisions alongside rules.

## Notes
- Run `python3 scripts/build_label_synonyms_template.py` to generate
  `data/knowledge/label_synonyms_template.json`. Populate English synonyms and
  re-run `scripts/build_geometry_rules_from_manifest.py --synonyms-json ...` to
  include bilingual keywords.
- Graph2D inference is shadow-mode by default; it only emits `graph2d_prediction`
  when enabled and a model checkpoint is present.
