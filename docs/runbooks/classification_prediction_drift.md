## Classification Prediction Drift Runbook

### Summary
Alert `ClassificationPredictionDrift` triggers when a single predicted label exceeds 85% of total ML classifications in a 30m window, indicating potential model bias or skewed input distribution.

### Impact
- Reduced reliability of automated routing or process recommendations based on classification.
- Potential model overfitting or upstream data change.

### Diagnosis
1. Query distribution: `sum(rate(classification_prediction_distribution[30m])) by (label,version)`.
2. Compare with 7d baseline (if recorded) or historical dashboard panel.
3. Check recent deployments / model version changes.
4. Inspect sample inputs for diversity (entity counts, formats, materials).
5. Validate incoming feature version shifts (e.g., majority still v1 instead of v3).

### Mitigation
- If input skew: adjust ingestion sources or throttle dominant project.
- If model drift: retrain with balanced dataset; consider rollback to prior model.
- Enable rule-based fallback (already automatic if model unavailable) temporarily.

### Metrics & Queries
```
sum(rate(classification_prediction_distribution[30m])) by (label)
topk(3, sum(rate(classification_prediction_distribution[1h])) by (label))
```

### False Positives
- Legitimate batch focusing on a single part type.
- Low traffic scenarios with small absolute counts.

### Follow-Up
- Document outcome in operations log.
- Revisit alert threshold if persistent legitimate dominance.

### Owners
Primary: ML Engineering
Secondary: CAD Platform Ops

