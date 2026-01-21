# DEV_MECH_KNOWLEDGE_4000CAD_GRAPH2D_SMOKE_BATCH_20260120

## Summary
Ran a 20-sample Graph2D smoke inference against the 4000CAD DXF conversions to
spot-check the refreshed merged checkpoint.

## Inputs
- Manifest: `reports/experiments/20260120/MECH_4000_DWG_LABEL_MANIFEST_MERGED_20260120.csv`
- DXF dir: `/Users/huazhou/Downloads/4000例CAD及三维机械零件练习图纸/机械CAD图纸_dxf`
- Model: `models/graph2d_merged_latest.pth`
- Sample size: 20 (seed=13)

## Command
```
GRAPH2D_MODEL_PATH=models/graph2d_merged_latest.pth \
  ./.venv-graph/bin/python - <<'PY'
import csv
import random
from pathlib import Path
from src.ml.vision_2d import Graph2DClassifier

manifest_path = Path('reports/experiments/20260120/MECH_4000_DWG_LABEL_MANIFEST_MERGED_20260120.csv')
dxf_dir = Path('/Users/huazhou/Downloads/4000例CAD及三维机械零件练习图纸/机械CAD图纸_dxf')
output_path = Path('reports/experiments/20260120/MECH_4000_DWG_GRAPH2D_SMOKE_PRED_20260120.csv')

random.seed(13)
rows = []
with manifest_path.open('r', encoding='utf-8') as handle:
    reader = csv.DictReader(handle)
    for row in reader:
        label = (row.get('label_cn') or '').strip()
        file_name = (row.get('file_name') or '').strip()
        if not label or not file_name:
            continue
        stem = Path(file_name).stem
        dxf_path = dxf_dir / f"{stem}.dxf"
        if not dxf_path.exists():
            dxf_path_upper = dxf_dir / f"{stem}.DXF"
            if dxf_path_upper.exists():
                dxf_path = dxf_path_upper
            else:
                continue
        rows.append({"label_cn": label, "file_name": file_name, "dxf_path": dxf_path})

sample = random.sample(rows, min(20, len(rows)))
classifier = Graph2DClassifier()

output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open('w', encoding='utf-8', newline='') as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=['file_name', 'label_cn', 'graph2d_label', 'confidence', 'status'],
    )
    writer.writeheader()
    for item in sample:
        with open(item['dxf_path'], 'rb') as dxf_handle:
            data = dxf_handle.read()
        result = classifier.predict_from_bytes(data, item['file_name'])
        writer.writerow(
            {
                'file_name': item['file_name'],
                'label_cn': item['label_cn'],
                'graph2d_label': result.get('label'),
                'confidence': result.get('confidence'),
                'status': result.get('status'),
            }
        )

print(f"Wrote {output_path}")
PY
```

## Results
- Output: `reports/experiments/20260120/MECH_4000_DWG_GRAPH2D_SMOKE_PRED_20260120.csv`
- Statuses: 20 ok
- Sample Top-1 match: 4/20 (0.20)
