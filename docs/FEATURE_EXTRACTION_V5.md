# Feature Extraction v5: Rotation & Scale Invariance

## Overview

Feature Version `v5` introduces a significant leap in robustness for the CAD ML Platform. Unlike previous versions (v1-v4) which relied heavily on absolute geometric dimensions (and thus were sensitive to scaling and rotation), v5 implements **invariant feature descriptors**.

This means a part will generate a highly similar feature vector regardless of:
*   **Uniform Scaling**: e.g., scaling a part by 0.5x or 10x.
*   **Rotation**: e.g., rotating a part by 90 degrees (swapping width/height) or arbitrary angles.
*   **Translation**: Moving the part in space (handled by bounding box centering).

## Key Improvements

| Feature | v4 (Legacy) | v5 (New) |
| :--- | :--- | :--- |
| **Scale Invariance** | ❌ Poor (Volume dominates vector) | ✅ Excellent (Normalized ratios) |
| **Rotation Invariance** | ❌ Poor (Sensitive to W/H swap) | ✅ Excellent (Sorted dimensions) |
| **Shape Sensitivity** | ❌ Low (Masked by volume magnitude) | ✅ High (Detects distortion) |
| **Vector Dimension** | 24 | 26 |

### The "Volume Dominance" Problem in v4
In v4, the `volume_estimate` feature (often > 100,000) was orders of magnitude larger than other features (typically < 100). This caused the feature vector to point almost entirely along the "volume axis". As a result, cosine similarity was determined almost exclusively by volume, making the system unable to distinguish between different shapes of similar volume.

**v5 fixes this by normalizing all features to a comparable range (typically [0, 1] or log-scale).**

## Feature Composition (26 Dimensions)

v5 vectors are composed of three invariant groups plus semantic tags:

### 1. Shape Signature (10 dims)
*   **Normalized Dimensions (3)**: Sorted [L, M, S] dimensions normalized by the largest dimension. (Rotation/Scale Invariant)
*   **Dimension Ratios (2)**: M/L and S/M ratios. (Scale Invariant)
*   **Compactness (1)**: Volume / (MaxDim^3). Measures how "filled" the bounding box is.
*   **Sphericity (1)**: Measures how close the shape is to a cube/sphere vs a flat sheet.
*   **Entity Entropy (1)**: Diversity of entity types (e.g., lines vs circles).
*   **Top Entity Ratio (1)**: Dominance of the most common entity type.
*   **Type Diversity (1)**: Log-normalized count of unique entity types.

### 2. Axis-Aligned Invariants (6 dims)
*   **Sorted Aspect Ratios (3)**: Pairwise ratios of W, H, D, sorted to be invariant to axis swapping.
*   **Aspect Variance (1)**: Variance of the aspect ratios.
*   **Diagonal Ratio (1)**: Ratio of max 2D diagonal to 3D diagonal.
*   **Fill Ratio (1)**: Convex Hull Volume / Bounding Box Volume. Measures concavity and "hollowness" (implemented via `scipy.spatial.ConvexHull`).

### 3. Topological Invariants (8 dims)
*   **Log Entity Count (1)**: Log-scale count of total entities.
*   **Log Layer Count (1)**: Log-scale count of layers.
*   **Log Density (1)**: Entities per layer.
*   **Top 3 Type Frequencies (3)**: Normalized frequencies of top 3 entity types.
*   **Gini Coefficient (1)**: Inequality of entity type distribution.
*   **Complexity Score (1)**: Composite score of diversity and quantity.

### 4. Semantic Features (2 dims)
*   **Complexity Flag**: 1.0 if high complexity, else 0.0.
*   **Layer Count**: Raw layer count (kept for compatibility).

## Migration Guide

### Enabling v5
Set the environment variable:
```bash
FEATURE_VERSION=v5
```

### Compatibility
*   **New Vectors**: New analyses will generate 26-dimensional vectors.
*   **Old Vectors (v1-v4)**:
    *   **Cannot be auto-upgraded**: Because v5 relies on new topological and invariant calculations that cannot be derived from the raw v1-v4 feature vectors, old vectors cannot be mathematically transformed to v5.
    *   **Recommendation**: Re-analyze original CAD files to generate v5 vectors.
    *   **Mixed Mode**: The system will error if you try to compare v5 vectors with v1-v4 vectors due to dimension mismatch. You must migrate your entire index or use a separate index for v5.

## Benchmarks

| Scenario | v4 Similarity | v5 Similarity | Result |
| :--- | :--- | :--- | :--- |
| **Base Part** | 1.0000 | 1.0000 | Baseline |
| **Scaled 0.5x** | 1.0000 | 1.0000 | ✅ Perfect Invariance |
| **Rotated 90°** | 1.0000 | 1.0000 | ✅ Perfect Invariance |
| **Rotated 45°** | 1.0000 | 0.9468 | ✅ **v5 reflects AABB change** (v4 falsely 1.0) |
| **Distorted** | 1.0000 | 0.9507 | ✅ **v5 detects distortion** (v4 falsely 1.0) |

## Developer Notes
*   Implementation: `src/core/invariant_features.py`
*   Integration: `src/core/feature_extractor.py` (v5 branch)
*   Tests: `tests/unit/test_invariant_features.py`
*   Benchmark Script: `scripts/benchmark_v4_vs_v5.py`
