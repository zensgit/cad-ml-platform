# DEV_FUSION_ANALYZER_L4_CONFIDENCE_20260117

## Summary
Wired ML classifier confidence into FusionAnalyzer L4 inputs and added unit coverage for
confidence extraction from classifier models.

## Steps
- Added confidence extraction from `predict_proba`/`decision_function` in `src/ml/classifier.py`.
- Passed classifier confidence into FusionAnalyzer L4 predictions in `/api/v1/analyze`.
- Added unit coverage for classifier confidence extraction.
- Ran: `source .venv-graph/bin/activate && pytest tests/unit/test_classifier_confidence.py -v`.

## Results
- Tests passed (3 cases).

## Notes
- If the classifier lacks confidence APIs, L4 confidence defaults to `0.0`.
