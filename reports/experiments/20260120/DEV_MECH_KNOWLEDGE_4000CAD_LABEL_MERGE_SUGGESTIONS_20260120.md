# DEV_MECH_KNOWLEDGE_4000CAD_LABEL_MERGE_SUGGESTIONS_20260120

## Summary
Generated a semi-automatic label merge suggestion list for low-frequency labels
(count < 2) using keyword-based grouping rules.

## Approach
- Labels with count >= 2 are left as-is (`action=keep`).
- Labels with count < 2 are matched against keyword rules (e.g., 装配图/视图/轴/板/盖/泵/阀).
- Suggested group and rule are recorded for review.

## Output
- `reports/MECH_4000_DWG_LABEL_MERGE_SUGGESTIONS_20260120.csv`
  - Columns: `label_cn`, `count`, `suggested_group`, `action`, `rule`, `notes`.

## Next Step
- Review and edit `suggested_group` / `action` as needed.
- Once approved, I will apply the merge map to the manifest and retrain.
