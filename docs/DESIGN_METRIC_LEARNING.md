# CAD Metric Learning Design Document

**Version**: v1.0
**Date**: 2025-11-29
**Status**: Implementation Phase 2

## 1. Overview

This document details the design and implementation of the Metric Learning module for the CAD ML Platform. The goal is to improve similarity search quality by learning a task-specific embedding space where semantically similar parts are closer together than in the raw geometric feature space.

## 2. Architecture

### 2.1 Module Structure

```
src/
  ml/
    metric_learning/       # Training & Modeling
      dataset.py           # TripletDataset loading
      model.py             # MetricMLP network definition
      losses.py            # TripletCosineLoss
      train.py             # Training script
    metric_embedder.py     # Inference wrapper
```

### 2.2 Model Architecture (`MetricMLP`)

A lightweight Multi-Layer Perceptron (MLP) designed for low-latency inference.

- **Input**: Raw geometric features (v5: 26 dim, v6: 32 dim).
- **Hidden Layers**:
  - Linear(Input -> 256) + BN + ReLU + Dropout(0.1)
  - Linear(256 -> 128) + BN + ReLU + Dropout(0.1)
- **Output Layer**:
  - Linear(128 -> 64)
  - **L2 Normalization**: Critical for Cosine Similarity to work effectively as a distance metric.

### 2.3 Loss Function (`TripletCosineLoss`)

Optimizes the embedding space such that:
`Distance(Anchor, Positive) < Distance(Anchor, Negative) - Margin`

- **Distance Metric**: Cosine Distance (`1 - CosineSimilarity`).
- **Margin**: Default 0.2.
- **Formula**: `L = max(0, (1 - sim(a, p)) - (1 - sim(a, n)) + margin)`

## 3. Data Pipeline

### 3.1 Data Format
- **Source**: `scripts/export_metric_learning_data.py`
- **Format**: Parquet (preferred) or CSV.
- **Schema**:
  - `anchor_features`: List[float]
  - `pos_features`: List[float]
  - `neg_features`: List[float]

### 3.2 Dataset Loader (`TripletDataset`)
- Loads data into memory (efficient for <1M vectors).
- Converts lists to PyTorch tensors.
- Returns `(anchor, positive, negative)` tuples.

## 4. Training Pipeline

### 4.1 Script: `src/ml/metric_learning/train.py`
- **Input**: Path to training data.
- **Process**:
  1. Splits data into Train (80%) and Validation (20%).
  2. Trains for N epochs using Adam optimizer.
  3. Evaluates on Validation set using **Triplet Accuracy** (`sim(a,p) > sim(a,n)`).
  4. Saves best model (`best_model.pth`) and exports to ONNX (`model.onnx`).

## 5. Inference Integration

### 5.1 `MetricEmbedder`
- **Role**: Wraps the trained model for easy use in the API.
- **Behavior**:
  - Loads model on initialization.
  - `embed(features)`: Transforms raw features -> 64d embedding.
  - **Fallback**: Returns original features if model fails or is missing (safe degradation).

### 5.2 Feature Extractor Integration
- **Location**: `src/core/feature_extractor.py`
- **Configuration**:
  - `FEATURE_EMBEDDER_BACKEND`: `none` (default) or `ml_embed_v1`.
  - `METRIC_MODEL_PATH`: Path to `.pth` file.
  - `FEATURE_EMBEDDER_SHADOW_MODE`: If `true`, computes embedding but returns original features (for testing/logging).
- **Logic**:
  - If backend is enabled, `extract()` calls `embedder.embed(geometric)`.
  - Replaces the geometric vector with the 64d embedding.

## 6. Deployment Strategy

### 6.1 Shadow Mode
1. Set `FEATURE_EMBEDDER_BACKEND=ml_embed_v1`.
2. Set `FEATURE_EMBEDDER_SHADOW_MODE=true`.
3. Monitor logs/metrics for errors and latency.
4. Verify embedding distribution (L2 norm ~ 1.0).

### 6.2 Active Rollout
1. Set `FEATURE_EMBEDDER_SHADOW_MODE=false`.
2. New vectors will be stored as 64d embeddings.
3. **Note**: Existing 24d/30d vectors in the store will remain as is. Similarity search between 64d query and 24d target will fail/degrade.
   - **Migration Required**: Run a migration script to update existing vectors using the model.
   - **Alternative**: Use a separate vector store index for the new embeddings.

## 7. Future Work

- **Vector Store Versioning**: Explicitly handle mixed-dimension stores.
- **Online Learning**: Update model with user feedback (clicks).

## 8. Migration

A migration script `scripts/migrate_vectors_to_embedding.py` is provided to batch update existing vectors.

```bash
# Dry run (default)
python scripts/migrate_vectors_to_embedding.py --model models/metric_learning/best_model.pth

# Apply changes
python scripts/migrate_vectors_to_embedding.py --model models/metric_learning/best_model.pth --apply
```

**Note**: Since the current system uses in-memory storage, this script demonstrates the logic. In a persistent system, it should be run against the database or snapshot file.

## 9. Observability

### 9.1 Metrics (Prometheus)
- `feature_embedding_generation_seconds`: Histogram of embedding latency.
- `feature_embedding_errors_total`: Counter of errors (model_missing, inference_error).
- `vector_query_latency_seconds`: Existing metric, now covers embedding-based queries.

### 9.2 Health & Stats
- `/vectors/stats` and `/vectors/distribution`:
  - Added `embedding_versions` field to track rollout progress (e.g., `{"none": 100, "ml_embed_v1": 50}`).


