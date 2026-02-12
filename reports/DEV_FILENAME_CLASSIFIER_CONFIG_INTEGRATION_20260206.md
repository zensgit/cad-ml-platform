# DEV_FILENAME_CLASSIFIER_CONFIG_INTEGRATION_20260206

## Scope

- Integrate `FilenameClassifier` with `hybrid_config` defaults.
- Keep environment variable precedence unchanged.
- Add focused unit tests for config-path and threshold override behavior.

## Code Changes

- Updated:
  - `src/ml/filename_classifier.py`
- Added:
  - `tests/unit/test_filename_classifier_config.py`

## Behavior

- `FilenameClassifier` now resolves defaults in this order:
  1. explicit constructor argument
  2. env var (highest priority)
  3. `hybrid_config` YAML value
  4. built-in fallback
- `synonyms_path` now supports:
  1. explicit constructor argument
  2. `FILENAME_SYNONYMS_PATH`
  3. `config/hybrid_classifier.yaml` (`filename.synonyms_path`)
  4. default `data/knowledge/label_synonyms_template.json`

## Verification Commands

```bash
python3 -m black src/ml/filename_classifier.py tests/unit/test_filename_classifier_config.py
python3 -m pytest tests/unit/test_filename_classifier_config.py tests/unit/test_filename_classifier.py tests/unit/test_hybrid_classifier.py -q
python3 -m flake8 src/ml/filename_classifier.py tests/unit/test_filename_classifier_config.py
```

## Results

- Formatting: success.
- Unit tests: `41 passed`.
- Flake8: success.
