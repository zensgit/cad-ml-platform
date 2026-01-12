# DEV_METRICS_DISABLED_TEST_GUARDS_20260112

## Scope
Add guards in metrics-related tests so they skip metric assertions when the metrics endpoint reports `app_metrics_disabled`.

## Tests
```bash
pytest tests/test_health_and_metrics.py -k "metrics_has_vision_and_ocr_counters or metrics_rejected_counter_for_large_base64" -v
pytest tests/test_vision_base64_rejection_reasons.py -v
pytest tests/test_ocr_provider_down.py -v
```

## Results
- `tests/test_health_and_metrics.py` (2 selected): **passed**
- `tests/test_vision_base64_rejection_reasons.py`: **passed**
- `tests/test_ocr_provider_down.py`: **passed**
