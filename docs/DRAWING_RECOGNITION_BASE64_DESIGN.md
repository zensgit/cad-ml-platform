# DRAWING_RECOGNITION_BASE64_DESIGN

## Goal
- Accept base64-encoded drawing images/PDFs for the drawing recognition endpoint.

## Changes
- Add `POST /api/v1/drawing/recognize-base64` with a JSON payload containing `image_base64`.
- Reuse the existing recognition pipeline and idempotency cache.
- Extend input validation to handle raw bytes (base64 decode) with image signature checks.

## Approach
- Decode base64 payload (including data URL prefix) and validate size/MIME.
- Feed validated bytes into the OCR manager and reuse the same response model.
