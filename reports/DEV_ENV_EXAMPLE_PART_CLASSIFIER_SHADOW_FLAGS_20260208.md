# DEV_ENV_EXAMPLE_PART_CLASSIFIER_SHADOW_FLAGS_20260208

## Goal
Document the PartClassifier provider shadow-only controls in `.env.example` so local/staging deployments can enable evaluation safely without changing `classification.part_type`.

## Changes
- `.env.example`
  - Added `PART_CLASSIFIER_PROVIDER_*` flags (enable/name/formats/timeout/max size/cache key inclusion).
  - Added optional V16 runtime tuning flags (`DISABLE_V16_CLASSIFIER`, `V16_SPEED_MODE`, `V16_CACHE_SIZE`, `CAD_CLASSIFIER_MODEL`).

## Verification
- `make lint`
- `.venv/bin/python -m mypy src`

Result:
- Lint + mypy passed locally.

