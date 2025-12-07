# Metric Learning for CAD Similarity

This module implements a learned embedding space for CAD similarity retrieval, enhancing the baseline geometric features (v5/v6) with task-specific metric learning.

## Overview

The system maps high-dimensional geometric feature vectors (26d for v5, 32d for v6) into a compact, semantically meaningful embedding space (64d) optimized for retrieval tasks.

### Key Features
- **Learned Embeddings**: MLP-based projection optimized with Triplet Loss.
- **Shadow Mode**: Safe rollout mechanism to compute embeddings in background without affecting production traffic.
- **Backward Compatibility**: Designed to work alongside existing feature extraction pipelines.
- **Observability**: Integrated Prometheus metrics for latency, L2 norms, and embedding distribution.

## Usage

### 1. Data Preparation
Export triplets from existing vector store or labeled data:
```bash
python scripts/export_metric_learning_data.py --output data/triplets.csv --dim 26
```

### 2. Training
Train the embedding model:
```bash
python src/ml/metric_learning/train.py --data data/triplets.csv --epochs 50 --output models/metric_learning/best_model.pth
```

### 3. Deployment
Enable the metric embedder in `config/settings.yaml` or via environment variables:

```bash
# Enable Metric Learning Backend
export FEATURE_EMBEDDER_BACKEND=ml_embed_v1
export METRIC_MODEL_PATH=models/metric_learning/best_model.pth

# Optional: Enable Shadow Mode (compute but do not use for retrieval)
export FEATURE_EMBEDDER_SHADOW_MODE=true
```

## Architecture

### Model Structure
- **Input**: 26d (v5) or 32d (v6) geometric vector.
- **Hidden Layers**: 256 -> 128 (ReLU + BatchNorm + Dropout).
- **Output**: 64d embedding (L2 Normalized).

### Integration Point
The `MetricEmbedder` is integrated into `FeatureExtractor.extract()`. It intercepts the geometric vector and projects it before it is returned to the API/Storage layer.

## Metrics & Monitoring

- `feature_vector_l2_norm`: Monitor the norm of embeddings (should be close to 1.0).
- `feature_extraction_latency_seconds`: Track overhead of the embedding step.
- `feature_version_usage_total`: Track usage of different feature versions.

## Rollback

If issues arise, simply unset `FEATURE_EMBEDDER_BACKEND` or set it to `none` to revert to raw geometric features.
