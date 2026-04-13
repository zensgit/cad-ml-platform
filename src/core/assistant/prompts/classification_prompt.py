"""LLM-assisted drawing classification prompt.

Takes geometric features, OCR text, and filename as input.
Returns classification with reasoning in structured format.
"""

CLASSIFICATION_PROMPT = """\
根据以下CAD图纸信息进行分类。

分类类别:
- mechanical_part: 机械零件图(轴、齿轮、法兰、壳体等)
- assembly: 装配图(多零件组合)
- sheet_metal: 钣金件(折弯、冲压)
- weldment: 焊接件
- electrical: 电气图
- architectural: 建筑图
- other: 其他

输入信息:
文件名: {filename}
OCR文本: {ocr_text}
几何特征: 线段={line_count}, 圆={circle_count}, 弧={arc_count}

示例:
输入: 文件名=shaft_assembly.dxf, OCR=M10螺纹 Ra3.2 45钢, 几何特征: 线段=120,圆=8,弧=15
输出:
```json
{{"category":"mechanical_part","confidence":0.85,"reasoning":"含螺纹标注和表面粗糙度,45钢材料,圆弧特征多,典型轴类零件"}}
```

只输出JSON，包含category、confidence(0-1)和reasoning字段。

文件名: {filename}
OCR文本: {ocr_text}
几何特征: 线段={line_count}, 圆={circle_count}, 弧={arc_count}

输出:"""


def get_classification_prompt(
    filename: str = "",
    ocr_text: str = "",
    line_count: int = 0,
    circle_count: int = 0,
    arc_count: int = 0,
) -> str:
    """Build classification prompt with drawing features.

    Args:
        filename: Drawing filename
        ocr_text: Extracted OCR text (truncated if needed)
        line_count: Number of line segments detected
        circle_count: Number of circles detected
        arc_count: Number of arcs detected

    Returns:
        Formatted prompt string ready for LLM
    """
    # Truncate OCR text for local model context limits
    if len(ocr_text) > 500:
        ocr_text = ocr_text[:500] + "..."

    return CLASSIFICATION_PROMPT.format(
        filename=filename or "unknown",
        ocr_text=ocr_text or "(none)",
        line_count=line_count,
        circle_count=circle_count,
        arc_count=arc_count,
    )
