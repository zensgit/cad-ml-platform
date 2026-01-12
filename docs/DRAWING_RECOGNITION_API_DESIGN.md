# Drawing Recognition API Design

## Summary
Introduces a dedicated drawing recognition endpoint that surfaces title block fields and key OCR outputs in a normalized response. The endpoint reuses the existing OCR manager and input validation pipeline.

## Endpoint
- `POST /api/v1/drawing/recognize`

### Request
- Multipart form upload: `file` (PNG/JPG/PDF)
- Query parameter: `provider` (default `auto`)
- Optional header: `Idempotency-Key`

### Response
`DrawingRecognitionResponse`:
- `success`: boolean
- `provider`: OCR provider used
- `confidence`: calibrated confidence if available
- `processing_time_ms`: latency
- `fields`: list of key title block fields
- `dimensions`: parsed dimensions (same schema as OCR)
- `symbols`: parsed symbols (same schema as OCR)
- `error` / `code`: error details when `success=false`

### Field Mapping
Title block fields are mapped in a fixed order:
- `drawing_number`, `revision`, `part_name`, `material`, `scale`, `sheet`, `date`, `weight`, `company`, `projection`

## Behavior
- Uses `validate_and_read` for MIME and PDF validation.
- Delegates extraction to `OcrManager` for provider routing and metrics.
- Supports idempotency caching via `Idempotency-Key`.

## Observability
- Emits structured logs on success and uses existing OCR error counters on input failures.

## Limitations
- Per-field confidence is only available when OCR line scores are provided.
- Fields without per-field confidence fall back to overall OCR confidence.
