# DEV_ISO286_DEVIATIONS_SCRIPT_NORMALIZATION_20260205

## Summary
Improved ISO 286 PDF extraction to normalize column symbols by stripping non-letter footnote markers
before table keys are built. This keeps table labels stable (e.g., H7, CD6, y6) even when the PDF
header includes footnotes or mixed tokens.

## Changes
- Added ASCII-letter normalization for ISO 286 header symbols in
  `scripts/extract_iso286_deviations.py`.
- Applied symbol normalization while constructing column labels so the JSON output uses canonical
  tolerance symbols.

## Verification
- `python3 scripts/extract_iso286_deviations.py --pdf "/Users/huazhou/Downloads/GB-T 1800.2-2020 产品几何技术规范（GPS） 线性尺寸公差ISO代号体系 第2部分：标准公差带代号和孔、轴的极限偏差表在线预览.pdf" --out /tmp/iso286_deviations_preview_20260205.json`

## Notes
- Output is written to `/tmp/iso286_deviations_preview_20260205.json` for inspection and kept
  out of version control.
