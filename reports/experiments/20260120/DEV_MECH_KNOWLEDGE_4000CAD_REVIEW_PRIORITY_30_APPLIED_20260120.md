# DEV_MECH_KNOWLEDGE_4000CAD_REVIEW_PRIORITY_30_APPLIED_20260120

## Summary
Applied the Top-30 manual review decisions to the DXF review sheet and merged
manifest, re-ran auto-review/conflicts, and refreshed the Top-30 pack to reflect
confirmed labels.

## Manual Decisions Applied
- Keep specialized labels: 挡板, 风叶(改), 喷雾器组件（改）, 后筒体（改）, 前筒体（改）
- Keep assembly labels: 装配图 (样本2,3,4,5,7,18,21)
- Merge remaining low-confidence items to 机械制图

## Commands
```
python3 - <<'PY'
import csv
from pathlib import Path

DECISIONS = {
    "PE2540.0.dxf": "挡板",
    "71002小立柱油缸装配图.dxf": "装配图",
    "71001小立柱装配图2.dxf": "装配图",
    "夹具装配图.dxf": "装配图",
    "11001床身装配图.dxf": "装配图",
    "风叶(改).dxf": "风叶(改)",
    "装配图纸.dxf": "装配图",
    "喷雾器组件（改）.dxf": "喷雾器组件（改）",
    "后筒体（改）.dxf": "后筒体（改）",
    "ZHITUI.dxf": "机械制图",
    "夹具二维图.dxf": "机械制图",
    "夹具二维图(2000).dxf": "机械制图",
    "JDC00000.dxf": "机械制图",
    "压下螺丝.dxf": "机械制图",
    "CAD图框.dxf": "机械制图",
    "站架三视图 - 副本.dxf": "机械制图",
    "站架三视图.dxf": "机械制图",
    "52001组合刀架装配图.dxf": "装配图",
    "FU200-02-05.dxf": "机械制图",
    "806控制板.dxf": "机械制图",
    "总装配.dxf": "装配图",
    "JDA00000.dxf": "机械制图",
    "61001.dxf": "机械制图",
    "8.dxf": "机械制图",
    "44002液压系统管路图.dxf": "机械制图",
    "涡轮.dxf": "机械制图",
    "A0模板.dxf": "机械制图",
    "JDD00000.dxf": "机械制图",
    "前筒体（改）.dxf": "前筒体（改）",
    "31001-2.dxf": "机械制图",
}

note_marker = "manual_priority_decision"
extra_marker = "manual_priority_pack_30"


def _update_review_csv(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    updated = 0
    for row in rows:
        file_name = (row.get("file_name") or "").strip()
        if file_name in DECISIONS:
            row["reviewer_label_cn"] = DECISIONS[file_name]
            row["review_status"] = "confirmed"
            existing_notes = (row.get("review_notes") or "").strip()
            for marker in [note_marker, extra_marker]:
                if marker not in existing_notes:
                    existing_notes = f"{existing_notes}; {marker}" if existing_notes else marker
            row["review_notes"] = existing_notes
            updated += 1

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return updated


def _update_manifest(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    updated = 0
    for row in rows:
        stem = (row.get("stem") or "").strip()
        file_name = f"{stem}.dxf" if stem else ""
        if file_name in DECISIONS:
            row["label_cn"] = DECISIONS[file_name]
            updated += 1

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return updated


review_sheet = Path("reports/experiments/20260120/MECH_4000_DWG_REVIEW_SHEET_20260120.csv")
priority_30 = Path("reports/experiments/20260120/MECH_4000_DWG_REVIEW_SHEET_PRIORITY_30_20260120.csv")
priority_30_with_previews = Path(
    "reports/experiments/20260120/MECH_4000_DWG_REVIEW_SHEET_PRIORITY_30_WITH_PREVIEWS_20260120.csv"
)
manifest = Path("reports/experiments/20260120/MECH_4000_DWG_LABEL_MANIFEST_MERGED_20260120.csv")

_update_review_csv(review_sheet)
_update_review_csv(priority_30)
_update_review_csv(priority_30_with_previews)
_update_manifest(manifest)
PY
```

## Outputs Updated
- Review sheet: `reports/experiments/20260120/MECH_4000_DWG_REVIEW_SHEET_20260120.csv`
- Priority list: `reports/experiments/20260120/MECH_4000_DWG_REVIEW_SHEET_PRIORITY_30_20260120.csv`
- Priority list (with previews): `reports/experiments/20260120/MECH_4000_DWG_REVIEW_SHEET_PRIORITY_30_WITH_PREVIEWS_20260120.csv`
- Merged manifest: `reports/experiments/20260120/MECH_4000_DWG_LABEL_MANIFEST_MERGED_20260120.csv`

## Follow-up Steps
- Re-ran auto-review/conflicts and refreshed the Top-30 pack.
