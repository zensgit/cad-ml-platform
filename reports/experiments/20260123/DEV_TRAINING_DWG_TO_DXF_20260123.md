# DEV_TRAINING_DWG_TO_DXF_20260123

## Summary
- Converted all DWG files in the training drawings directory to DXF using ODAFileConverter.

## Inputs
- DWG directory: `/Users/huazhou/Downloads/训练图纸/训练图纸`
- Output directory: `/Users/huazhou/Downloads/训练图纸/训练图纸_dxf_oda_20260123`
- Log CSV: `reports/experiments/20260123/MECH_TRAINING_DWG_TO_DXF_LOG_20260123.csv`

## Command
- `ODA_FILE_CONVERTER_EXE="/Applications/ODAFileConverter.app/Contents/MacOS/ODAFileConverter" .venv-graph/bin/python scripts/convert_dwg_batch.py --input-dir "/Users/huazhou/Downloads/训练图纸/训练图纸" --output-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf_oda_20260123" --log-csv "reports/experiments/20260123/MECH_TRAINING_DWG_TO_DXF_LOG_20260123.csv" --recursive`

## Notes
- Output directory is separate to avoid overwriting the existing DXF set.
