# Online Example AI Inputs Validation 2026-03-07

## Summary

Online example data for both `history_sequence` (`.h5`) and `STEP/B-Rep` was
successfully acquired and validated at the input level.

- `.h5` example: fully validated against the current history-sequence pipeline
- `STEP` example: file acquisition validated, but local B-Rep extraction remains
  blocked because `pythonocc-core` is not installed in the current environment
- Docker-based fallback validation was not executed because the Docker daemon is
  unavailable on this machine

## Online Sources

### HPSketch `.h5`

Source:

- https://www.gitlink.org.cn/fzhe/HPSketch

Local checkout:

- `/private/tmp/cad-ai-example-data-20260307/HPSketch`

Reference:

- commit: `6607e7a05`
- license: MIT

Validated sample:

- `/private/tmp/cad-ai-example-data-20260307/HPSketch/data/0000/00000007_1.h5`

### STEP examples

Source:

- https://github.com/Formlabs/foxtrot

Local checkout:

- `/private/tmp/cad-ai-example-data-20260307/foxtrot`

Reference:

- commit: `69fb376`
- license: MIT / Apache-2.0

Validated samples:

- `/private/tmp/cad-ai-example-data-20260307/foxtrot/examples/cube_hole.step`
- `/private/tmp/cad-ai-example-data-20260307/foxtrot/examples/cuboid.step`
- `/private/tmp/cad-ai-example-data-20260307/foxtrot/examples/abstract_pca.step`

## New Validation Script

Added:

- `scripts/validate_online_example_ai_inputs.py`

This script validates:

- `.h5` file structure
- token extraction through `history_sequence_tools`
- prototype inference through `HistorySequenceClassifier`
- `STEP` file presence and local B-Rep availability

Unit coverage:

- `tests/unit/test_validate_online_example_ai_inputs.py`

## Verification

### Unit test

```bash
python3 -m pytest -q tests/unit/test_validate_online_example_ai_inputs.py
```

Result:

- `1 passed`

### Static validation

```bash
flake8 scripts/validate_online_example_ai_inputs.py \
  tests/unit/test_validate_online_example_ai_inputs.py \
  --max-line-length=100

python3 -m py_compile scripts/validate_online_example_ai_inputs.py
```

Result:

- passed

### Example data smoke validation

```bash
python3 scripts/validate_online_example_ai_inputs.py \
  --output reports/experiments/20260307/online_example_ai_inputs_validation.json
```

Generated report:

- `reports/experiments/20260307/online_example_ai_inputs_validation.json`

## Results

### `.h5` validation

- file exists: `true`
- keys: `["vec"]`
- shape: `[5, 21]`
- dtype: `int32`
- extracted tokens: `[6, 2, 10, 15, 5]`

Classifier result:

- `status = ok`
- `label = 轴类`
- `confidence = 0.5068745667645342`
- `source = history_sequence_prototype`

Conclusion:

- the real online `.h5` sample is compatible with the current history-sequence
  tooling and classifier entrypoint

### `STEP` validation

- file exists: `true`
- local `HAS_OCC = false`
- status: `skipped_no_occ`

Conclusion:

- the online STEP sample is available locally and suitable for future B-Rep
  validation
- the current blocker is runtime environment, not sample availability

## Environment Findings

### Local Python

- `h5py`: available
- `pythonocc-core` / `OCC`: unavailable

### Docker

Check result:

- `Cannot connect to the Docker daemon at unix:///Users/huazhou/.docker/run/docker.sock`

Implication:

- existing docker-based `pythonocc-core` validation scripts cannot be executed
  until Docker is available again

## Next Step

When the 3D environment is available again, run one of these paths:

1. local conda/mamba path using `scripts/setup_mac_m4.sh`
2. docker path using `scripts/validate_brep_features_linux_amd64_cached.sh`

At that point the existing online sample
`foxtrot/examples/cube_hole.step` can be used immediately for B-Rep extraction
smoke validation.
