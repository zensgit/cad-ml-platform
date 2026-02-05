# DEV_ISO286_DEVIATIONS_REFRESH_20260205

## Summary
Regenerated `data/knowledge/iso286_deviations.json` from the GB/T 1800.2 PDF
using the updated header normalization logic so deviation tables keep
canonical symbols.

## Changes
- Updated `data/knowledge/iso286_deviations.json` with the latest extraction
  output.

## Verification
- `python3 scripts/extract_iso286_deviations.py --pdf "/Users/huazhou/Downloads/GB-T 1800.2-2020 产品几何技术规范（GPS） 线性尺寸公差ISO代号体系 第2部分：标准公差带代号和孔、轴的极限偏差表在线预览.pdf" --out data/knowledge/iso286_deviations.json`
- `python3 -m pytest tests/unit/test_tolerance_limit_deviations.py -q`

## Notes
- Source PDF path is local; replace with the correct path if you regenerate.
