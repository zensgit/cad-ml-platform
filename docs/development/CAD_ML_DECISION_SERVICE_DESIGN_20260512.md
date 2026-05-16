# CAD ML DecisionService Design

Date: 2026-05-12

## Goal

Create a single final-decision boundary for CAD classification. The first version is
intentionally conservative: it does not change the existing baseline, Fusion, Hybrid,
Graph2D, or review ordering. It centralizes the final contract and evidence emitted
after those branches have run.

## Contract

`DecisionService` writes the following top-level fields:

- `contract_version`
- `decision_contract_version`
- `fallback_flags`
- `evidence`
- `decision_contract`

`decision_contract` contains:

- `fine_part_type`
- `coarse_part_type`
- `confidence`
- `decision_source`
- `branch_conflicts`
- `evidence`
- `review_reasons`
- `fallback_flags`
- `contract_version`

Current version:

```text
classification_decision.v1
```

## Evidence Sources

The service normalizes available branch outputs into a list of evidence rows:

- `baseline`
- `filename`
- `titleblock`
- `ocr`
- `graph2d`
- `history_sequence`
- `process`
- `hybrid`
- `part_classifier`
- `fusion`
- `brep`
- `knowledge`
- `vector_neighbors`
- `active_learning_history`

Rows use this shape when fields are available:

```json
{
  "source": "graph2d",
  "kind": "prediction",
  "label": "传动件",
  "confidence": 0.86,
  "status": "ok",
  "contribution": 0.22,
  "details": {
    "margin": 0.18,
    "passed_threshold": true
  }
}
```

## Fallback Flags

Fallback flags are stable strings intended for API, active-learning, benchmark, and
assistant consumers. Initial flags include:

- `rules_baseline`
- `ml_unavailable`
- `hybrid_error`
- `hybrid_rejected`
- `graph2d_model_unavailable`
- `part_classifier_timeout`
- `brep_invalid`

## Integration Plan

This slice integrates `DecisionService` into the analyze classification pipeline only.
Batch classify, assistant explanation, and benchmark exporters remain follow-up work
so the first service version can stabilize under focused unit tests.

## Non-Goals

- No model ranking or decision-priority change.
- No new external dependency.
- No claim that B-Rep or Graph2D is release-ready without scorecard evidence.
