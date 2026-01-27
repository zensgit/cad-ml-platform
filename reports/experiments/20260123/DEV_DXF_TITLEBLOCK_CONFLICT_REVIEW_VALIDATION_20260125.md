# DEV_DXF_TITLEBLOCK_CONFLICT_REVIEW_VALIDATION_20260125

## Validation Summary
- Verified conflict handling paths in hybrid classification for two DXF samples and refreshed the review list.

## Checks
- Decision paths include `titleblock_filename_conflict` and `titleblock_ignored_filename_high_conf` for conflicted samples.
- Review list updated at `reports/experiments/20260123/titleblock_conflict_review_list_20260125.csv`.

## Evidence
```
LTJ012306102-0084调节螺栓v1.dxf
filename_label 调节螺栓
titleblock_label 轴类
decision_path ['filename_extracted', 'titleblock_predicted', 'titleblock_filename_conflict', 'titleblock_ignored_filename_high_conf', 'filename_high_conf_adopted']

比较_LTJ012306102-0084调节螺栓v1 vs LTJ012306102-0084调节螺栓v2.dxf
filename_label 调节螺栓
titleblock_label 轴类
decision_path ['filename_extracted', 'titleblock_predicted', 'titleblock_filename_conflict', 'titleblock_ignored_filename_high_conf', 'filename_high_conf_adopted']
```
