"""OCR structured extraction prompt for local LLM models.

Takes raw OCR text from PaddleOCR and produces structured JSON output
with title block fields (part_name, material, drawing_number, etc.).

Includes few-shot examples for reliable extraction from 7B models.
"""

OCR_EXTRACTION_PROMPT = """\
从以下OCR识别文本中提取标题栏信息，输出JSON格式。

提取字段:
- part_name: 零件名称
- material: 材料
- drawing_number: 图号
- quantity: 数量
- scale: 比例
- revision: 版本号
- date: 日期
- weight: 重量
- surface_finish: 表面处理
- designer: 设计者

示例1:
输入: "零件名称 轴承座 材料 HT250 图号 MK-2025-003 比例 1:1 数量 2 设计 张工 日期 2025-03"
输出:
```json
{{"part_name":"轴承座","material":"HT250","drawing_number":"MK-2025-003","scale":"1:1","quantity":"2","designer":"张工","date":"2025-03"}}
```

示例2:
输入: "名称:连接法兰 材质:304不锈钢 DWG NO.:FL-0812 REV:B 重量:1.2kg Ra3.2"
输出:
```json
{{"part_name":"连接法兰","material":"304不锈钢","drawing_number":"FL-0812","revision":"B","weight":"1.2kg","surface_finish":"Ra3.2"}}
```

规则: 未找到的字段不要输出。只输出JSON，不加解释。

OCR文本:
{ocr_text}

输出:"""

# Minimal version for very constrained contexts
OCR_EXTRACTION_PROMPT_MINIMAL = """\
从OCR文本提取标题栏字段为JSON(part_name,material,drawing_number,quantity,scale,revision,date,weight)。
只输出JSON。

文本: {ocr_text}
输出:"""


def get_ocr_extraction_prompt(
    ocr_text: str,
    max_tokens: int = 4000,
) -> str:
    """Build OCR extraction prompt with the given raw text.

    Args:
        ocr_text: Raw OCR text to extract from
        max_tokens: Token budget; if < 2000 use minimal prompt

    Returns:
        Formatted prompt string ready for LLM
    """
    if max_tokens < 2000:
        return OCR_EXTRACTION_PROMPT_MINIMAL.format(ocr_text=ocr_text)
    return OCR_EXTRACTION_PROMPT.format(ocr_text=ocr_text)
