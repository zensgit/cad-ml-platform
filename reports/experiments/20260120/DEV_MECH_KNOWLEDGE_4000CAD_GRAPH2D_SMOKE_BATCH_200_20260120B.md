# DEV_MECH_KNOWLEDGE_4000CAD_GRAPH2D_SMOKE_BATCH_200_20260120B

## Summary
Ran a 200-sample Graph2D smoke inference after label-merge updates to gauge
Top-1 match quality on the refreshed merged checkpoint.

## Inputs
- Manifest: `reports/experiments/20260120/MECH_4000_DWG_LABEL_MANIFEST_MERGED_20260120.csv`
- DXF dir: `/Users/huazhou/Downloads/4000例CAD及三维机械零件练习图纸/机械CAD图纸_dxf`
- Model: `models/graph2d_merged_latest.pth`
- Sample size: 200 (seed=17)

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
output_path = Path('reports/experiments/20260120/MECH_4000_DWG_GRAPH2D_SMOKE_PRED_200_20260120B.csv')

random.seed(17)
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

sample_size = min(200, len(rows))
sample = random.sample(rows, sample_size)
classifier = Graph2DClassifier()

output_path.parent.mkdir(parents=True, exist_ok=True)
match = 0
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
        predicted = result.get('label')
        if predicted == item['label_cn']:
            match += 1
        writer.writerow(
            {
                'file_name': item['file_name'],
                'label_cn': item['label_cn'],
                'graph2d_label': predicted,
                'confidence': result.get('confidence'),
                'status': result.get('status'),
            }
        )

print(f"Wrote {output_path}")
print(f"Sample size: {sample_size}")
print(f"Top1 match: {match}/{sample_size} ({match / sample_size:.2f})")
PY
```

## Results
- Output: `reports/experiments/20260120/MECH_4000_DWG_GRAPH2D_SMOKE_PRED_200_20260120B.csv`
- Sample size: 200
- Top-1 match: 117/200 (0.58)
