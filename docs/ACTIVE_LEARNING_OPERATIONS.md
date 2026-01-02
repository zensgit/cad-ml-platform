#!/usr/bin/env markdown
# Active Learning Operations

This guide documents the active learning loop endpoints and environment flags.

## Enablement

Set the following environment variables:

- `ACTIVE_LEARNING_ENABLED=true`
- `ACTIVE_LEARNING_CONFIDENCE_THRESHOLD=0.6`
- `ACTIVE_LEARNING_STORE=memory|file`
- `ACTIVE_LEARNING_DATA_DIR=/tmp/active_learning`
- `ACTIVE_LEARNING_RETRAIN_THRESHOLD=10`

## API Endpoints

Base path: `/api/v1/active-learning`

### 1) Pending Samples

```
GET /pending?limit=10
```

### 2) Submit Feedback

```
POST /feedback
{
  "sample_id": "uuid",
  "true_type": "bolt",
  "reviewer_id": "user-1"
}
```

### 3) Stats

```
GET /stats
```

### 4) Export Training Data

```
POST /export
{
  "format": "jsonl",
  "only_labeled": true
}
```

## Sampling Logic (Current)

Samples are flagged when classification confidence is below
`ACTIVE_LEARNING_CONFIDENCE_THRESHOLD`. The payload includes alternatives (if any)
and a lightweight score breakdown (rule version + model version).
