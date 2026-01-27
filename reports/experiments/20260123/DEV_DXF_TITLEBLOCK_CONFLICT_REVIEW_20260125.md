# DEV_DXF_TITLEBLOCK_CONFLICT_REVIEW_20260125

## Objective
Review title-block vs filename label conflicts and formalize precedence rules for hybrid classification.

## Findings
Two DXF files show title-block part names that disagree with filename-derived labels:
- `LTJ012306102-0084调节螺栓v1.dxf`
  - filename label: `调节螺栓`
  - title-block part name: `小减速机轴` → mapped to `轴类`
- `比较_LTJ012306102-0084调节螺栓v1 vs LTJ012306102-0084调节螺栓v2.dxf`
  - filename label: `调节螺栓`
  - title-block part name: `小减速机轴` → mapped to `轴类`

## Review List
- `reports/experiments/20260123/titleblock_conflict_review_list_20260125.csv`
  - Tracks the two remaining conflicts after synonym updates.

## Decision
- Preserve filename precedence when filename confidence is high, and record conflicts for review.
- Title-block override remains opt-in via `TITLEBLOCK_OVERRIDE_ENABLED=true`.

## Implementation
- `src/ml/hybrid_classifier.py`
  - Records `titleblock_filename_conflict` and `titleblock_ignored_filename_high_conf` in `decision_path`.

## Notes
- This decision avoids silent overrides while retaining title-block evidence for fusion.
