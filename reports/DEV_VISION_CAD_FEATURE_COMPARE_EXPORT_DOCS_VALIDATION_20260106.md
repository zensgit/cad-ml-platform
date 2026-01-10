# DEV_VISION_CAD_FEATURE_COMPARE_EXPORT_DOCS_VALIDATION_20260106

## Scope
Validate documentation updates for compare summary CSV output and compare export usage.

## Command
- `rg -n "output-compare-csv|compare_export" README.md docs/VISION_CAD_FEATURE_TUNING_DESIGN.md`

## Results
- README includes `--output-compare-csv` example and compare export command.
- Tuning design lists `--output-compare-csv` and references the compare export script.

## Files Reviewed
- `README.md`
- `docs/VISION_CAD_FEATURE_TUNING_DESIGN.md`
