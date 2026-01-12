# Drawing Recognition Field Confidence Design

## Summary
Adds per-field confidence to drawing recognition responses by propagating OCR line-level scores through the OCR result model and into the drawing API fields.

## Data Model
- `OcrResult.title_block_confidence`: `Dict[str, float]`
  - Stores per-field confidence when available (e.g., from Paddle OCR line scores).

## Provider Behavior
- Paddle OCR:
  - Uses `parse_title_block_with_confidence` to extract both values and confidences.
  - Merges text-based title block parsing without overwriting confidence for existing fields.
- DeepSeek HF:
  - Accepts optional `title_block_confidence` from JSON output when present.
  - Defaults to an empty mapping when absent.

## API Mapping
- `DrawingField.confidence` uses per-field confidence when present.
- Fields without per-field confidence fall back to the overall OCR confidence.

## Limitations
- Per-field confidence depends on OCR provider availability of line scores.
- LLM providers do not emit per-field confidence unless explicitly supported.
