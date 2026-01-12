# DEV_DRAWING_RECOGNITION_TITLE_BLOCK_VALIDATION_20260112

## Scope
Validate the title block parser and schema expansion for drawing recognition.

## Validation Steps
1. Ran a quick parser check via Python to exercise English and Chinese labels:

```bash
python3 - <<'PY'
from src.core.ocr.parsing.title_block_parser import parse_title_block
sample = "Drawing No: DWG-2025-01 Rev A Part Name: Bracket Material: Aluminum Scale 1:2 Sheet 1 of 3 Date 2025-01-12 Weight 2.5kg Company ACME Projection third"
print(parse_title_block(sample))
cn = "图号: A-100 修订: B 名称: 支架 材料: 钢 比例 1:1 页 2/5 日期 2025/01/02 重量 3.2kg 公司: ACME 投影 第三角"
print(parse_title_block(cn))
PY
```

## Results
- Parsed fields extracted cleanly for both English and Chinese samples.
- Sheet normalization converted `1 of 3` to `1/3` as expected.

## Notes
- This validation focuses on parsing behavior; API-level validation is covered in later steps.
