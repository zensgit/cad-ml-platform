# v4 Geometric Feature Research

## Objectives
Enhance the geometric feature vector (v4) with topological and complexity metrics to improve similarity search for complex assemblies.

## Proposed Features

### 1. Surface Count (Complexity)
**Definition**: The total number of bounded surfaces (faces) in the 3D model.
**Relevance**: Distinguishes between simple primitives (cube = 6 faces) and complex machined parts (hundreds of faces).

**Implementation Strategy**:
- **Simple Mode (Day 4)**: Parse STEP file structure and count `ADVANCED_FACE` entities.
  - *Pros*: Fast, robust.
  - *Cons*: May count internal or construction geometry if not filtered.
- **Advanced Mode (Future)**: Full B-Rep graph traversal.

### 2. Shape Entropy (Heterogeneity)
**Definition**: Shannon entropy of the distribution of geometric primitive types used in the model.
**Relevance**: Distinguishes between uniform parts (all planes) and heterogeneous parts (planes + cylinders + b-splines).

**Formula**:
$$ H(X) = - \sum_{i=1}^{n} P(x_i) \log_2 P(x_i) $$
Where $P(x_i)$ is the proportion of entity type $i$ (e.g., Plane, Cylinder, Torus).
Normalized: $H_{norm} = H(X) / \log_2(N)$ where $N$ is number of unique types present (or total vocabulary size).

**Smoothing**:
To handle sparse distributions, we will apply Laplace smoothing (Day 4 PM).

## Dataset & Validation
- **Dataset**: `data/cad_v4_dataset` (Synthetic STEP files).
- **Baseline**: v3 features (24 dimensions).
- **Target**: v4 features (26 dimensions = v3 + surface_count + entropy).

## Performance Considerations
- **Latency**: Feature extraction must remain under 200ms for typical parts.
- **Fallback**: If STEP parsing fails, default to v3 values (0.0 for new features).
