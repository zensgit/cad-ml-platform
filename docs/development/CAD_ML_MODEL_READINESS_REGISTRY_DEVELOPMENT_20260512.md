# CAD ML Model Readiness Registry Development

Date: 2026-05-12

## Goal

Replace the static model readiness stub with a model-by-model readiness registry
that can support forward-looking CAD intelligence claims without overstating
runtime capability.

This slice intentionally does not eagerly load large checkpoints at application
startup. It reports evidence that is cheap and stable:

- whether a model branch is enabled;
- whether configured checkpoints exist;
- whether a runtime singleton is already loaded;
- checkpoint checksum when the checkpoint exists;
- fallback mode when the model is enabled but not backed by a checkpoint;
- blocking readiness reasons when specific models are required.

## Implemented Files

### `src/models/readiness_registry.py`

Added the central registry with six model entries:

- `v16_classifier`
- `graph2d`
- `uvnet`
- `pointnet`
- `ocr_provider`
- `embedding_model`

Each entry emits:

- `enabled`
- `checkpoint_paths`
- `checkpoint_exists`
- `loaded`
- `status`
- `version`
- `checksum`
- `fallback_mode`
- `error`
- `metadata`

Registry status values:

- `disabled`: branch intentionally disabled.
- `loaded`: runtime object is loaded.
- `available`: checkpoint exists, but the model is lazy or not yet loaded.
- `fallback`: checkpoint is missing, but a deterministic fallback exists.
- `missing`: checkpoint is missing and no fallback is declared.
- `error`: readiness evidence observed an error.

The registry supports:

- `MODEL_READINESS_REQUIRED_MODELS`: comma or space separated model IDs that must
  be `loaded` or `available`.
- `MODEL_READINESS_STRICT=true`: fail readiness when any enabled model is degraded.

Checksum calculation is cached by path, mtime, and size to keep repeated health
checks cheap after the first checksum read.

### `src/models/loader.py`

Replaced the old static `_loaded = True` behavior with a facade over the registry.

Preserved legacy API:

- `load_models()`
- `models_loaded()`

Added:

- `get_model_readiness_snapshot()`
- `models_readiness_check()`

This keeps old imports working while making `/ready` depend on actual readiness
evidence.

### `src/main.py`

Updated `/ready` so the existing `models_loaded` check key now uses
`models_readiness_check()`.

Default local development remains runnable: fallback-only model state is reported
as `degraded`, but it does not return HTTP 503 unless strict or required-model
gates are configured.

### `src/api/health_utils.py`

Extended the existing `/health` payload under:

```text
config.ml.readiness.model_registry
```

The legacy readiness fields remain in place for compatibility:

- `torch_available`
- `graph2d_model_path`
- `graph2d_model_present`
- `v16_disabled`
- `v6_model_path`
- `v14_model_path`
- `v16_models_present`
- `degraded_reasons`
- provider readiness env fields

### `src/api/v1/health.py`

Added:

```text
GET /api/v1/health/model-readiness
```

The endpoint returns the full registry snapshot and requires the same API key
dependency as the other health control-plane endpoints.

### `tests/unit/test_model_readiness_registry.py`

Added unit/API coverage for:

- missing checkpoints becoming degraded fallback states;
- checkpoint presence and checksum reporting;
- required missing model blocking readiness;
- loader compatibility with the legacy `models_loaded` patch surface;
- `/health` model registry exposure;
- `/api/v1/health/model-readiness`.

## Current Boundary

This slice proves readiness evidence and control-plane wiring. It does not yet
turn Graph2D, UV-Net, PointNet, or embedding models into eager startup loads.
That is intentional because eager loading changes startup behavior and should be
done with model-specific failure fixtures.

The next production hardening slice should add explicit loaded/error callbacks
from each model implementation into the registry after checkpoint load attempts.
