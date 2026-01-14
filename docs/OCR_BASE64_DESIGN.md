# OCR_BASE64_DESIGN

## Goal
- Accept base64-encoded OCR inputs to match vision and drawing payload styles.

## Changes
- Add `POST /api/v1/ocr/extract-base64` with a JSON payload containing `image_base64`.
- Reuse the OCR extraction pipeline and validation logic for raw bytes.

## Approach
- Decode base64 payload (support data URL prefixes) and validate size/MIME.
- Feed validated bytes into the OCR manager and return the standard response model.
