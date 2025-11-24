## Material Distribution Drift Runbook

### Summary
Alert `MaterialDistributionDrift` fires when a single material dominates >85% of vectors over a 30m window, indicating potential imbalance or ingestion anomaly.

### Impact
- Similarity and recommendations may become biased.
- Downstream ML training (future) risks overfitting.

### Diagnosis Steps
1. Check Grafana panel "Vector Material Distribution" for sudden spikes.
2. Query `topk(5, vector_store_material_total)` for recent dominant materials.
3. Inspect recent uploads: `kubectl logs deploy/cad-ml | grep material=`.
4. Validate ingestion source (batch job, client) for repeated material field.
5. Confirm no recent process rule change forcing default material.

### Mitigation
- If erroneous uploads: pause offending client credentials.
- Rebalance by ingesting diverse material samples if available.
- Adjust process rules to handle missing material explicitly to avoid defaulting.
- Consider temporary lowering similarity weight on material dimension (future configurable).

### Metrics & Queries
```
material_drift_ratio
sum(vector_store_material_total) by (material)
(sum(rate(vector_store_material_total[1h])) by (material)) / sum(rate(vector_store_material_total[1h]))
```

### False Positives
- Legitimate campaign focusing on single material.
- Early-stage system with limited dataset variety.

### Follow-Up
- Record investigation outcome in `docs/OBSERVATION_PERIOD_LOG.md`.
- Update alert threshold if persistent legitimate dominance.

### Owners
Primary: CAD Platform Ops
Secondary: Data Engineering

