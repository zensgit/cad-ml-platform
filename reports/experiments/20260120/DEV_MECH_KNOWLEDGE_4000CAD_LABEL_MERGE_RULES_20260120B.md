# DEV_MECH_KNOWLEDGE_4000CAD_LABEL_MERGE_RULES_20260120B

## Summary
Applied label-merge rules to collapse generic drawing labels into `机械制图`,
updated the synonyms template accordingly, and rebuilt geometry rules.

## Merge Rules
The following labels are now merged into `机械制图` in the merged manifest and
synonym normalization:
- 机械平面图
- 练习零件图
- 视图
- 三视图
- 三视图练习
- 示意图
- 图框
- 模板
- 零件图
- 站架三视图

## Commands
```
python3 - <<'PY'
import json
import csv
from pathlib import Path

MERGE_LABELS = {
    "机械平面图",
    "练习零件图",
    "视图",
    "三视图",
    "三视图练习",
    "示意图",
    "图框",
    "模板",
    "零件图",
    "站架三视图",
}


def update_manifest(path: Path) -> int:
    with path.open('r', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    updated = 0
    for row in rows:
        label = (row.get('label_cn') or '').strip()
        if label in MERGE_LABELS:
            row['label_cn'] = '机械制图'
            updated += 1

    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return updated


def update_synonyms(path: Path) -> int:
    data = json.loads(path.read_text(encoding='utf-8'))
    mech_key = '机械制图'
    synonyms = data.get(mech_key, [])
    if not isinstance(synonyms, list):
        synonyms = []

    added = 0
    for label in sorted(MERGE_LABELS):
        if label not in synonyms:
            synonyms.append(label)
            added += 1
        if label in data:
            data.pop(label)

    data[mech_key] = synonyms
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return added

manifest_path = Path('reports/experiments/20260120/MECH_4000_DWG_LABEL_MANIFEST_MERGED_20260120.csv')
synonyms_path = Path('data/knowledge/label_synonyms_template.json')

update_manifest(manifest_path)
update_synonyms(synonyms_path)
PY
```

```
python3 scripts/build_geometry_rules_from_manifest.py \
  --manifest reports/experiments/20260120/MECH_4000_DWG_LABEL_MANIFEST_MERGED_20260120.csv \
  --synonyms-json data/knowledge/label_synonyms_template.json
```

## Outputs Updated
- Manifest: `reports/experiments/20260120/MECH_4000_DWG_LABEL_MANIFEST_MERGED_20260120.csv`
- Synonyms: `data/knowledge/label_synonyms_template.json`
- Geometry rules: `data/knowledge/geometry_rules.json`
