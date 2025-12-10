# v4 Geometry Algorithm Research

## Overview
This document covers the research and selection of geometry algorithms for v4 feature extraction, specifically for `surface_count` and `shape_entropy` enhancements.

## Goals
- Implement accurate surface counting for CAD models
- Improve shape entropy calculation with better geometric understanding
- Maintain performance overhead < 5% compared to v3

---

## Test Data Requirements

### Category 1: Empty Entity Files
```yaml
file: empty_entity.step
description: Valid CAD file with no geometric entities
expected_surface_count: 0
expected_shape_entropy: 0.0
use_case: Edge case handling
```

### Category 2: Single Entity (Simple Geometry)
```yaml
files:
  - single_cube.step:
      entities: 1
      surfaces: 6
      volume: known
      description: Unit cube for baseline

  - single_cylinder.step:
      entities: 1
      surfaces: 3  # top, bottom, lateral
      volume: known
      description: Simple cylinder

  - single_sphere.step:
      entities: 1
      surfaces: 1  # continuous surface
      volume: known
      description: Single continuous surface
```

### Category 3: Multi-Entity (Complex Geometry)
```yaml
files:
  - assembly_10_parts.step:
      entities: 10
      description: Medium complexity assembly
      diversity: high  # multiple entity types

  - assembly_50_parts.step:
      entities: 50
      description: High complexity assembly
      diversity: medium

  - mixed_geometry.step:
      entities: 25
      types: [box, cylinder, cone, sphere, freeform]
      description: High diversity for entropy testing
```

---

## Algorithm Options

### Option 1: Open CASCADE (OCC) via PythonOCC

**Overview:**
Open CASCADE Technology (OCCT) is the industry standard for CAD kernel operations.

**Pros:**
- Industry standard, production-proven
- Full BREP (Boundary Representation) support
- Accurate surface extraction
- Handles complex topologies

**Cons:**
- Heavy dependency (~200MB)
- Complex installation (requires conda or manual build)
- Steeper learning curve

**Surface Counting Implementation:**
```python
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Extend.DataExchange import read_step_file

def count_surfaces_occ(file_path: str) -> int:
    """Count surfaces using Open CASCADE."""
    shape = read_step_file(file_path)
    explorer = TopExp_Explorer(shape, TopAbs_FACE)

    count = 0
    while explorer.More():
        count += 1
        explorer.Next()

    return count
```

**Performance Estimate:** ~50-200ms per file (size dependent)

---

### Option 2: Trimesh

**Overview:**
Trimesh is a lightweight Python library for working with triangular meshes.

**Pros:**
- Pure Python, easy installation (`pip install trimesh`)
- Fast for mesh operations
- Good format support (STL, OBJ, PLY, OFF)
- Memory efficient

**Cons:**
- Requires tessellation for BREP formats
- Surface count is approximated via mesh faces
- May miss fine geometric details

**Surface Counting Implementation:**
```python
import trimesh

def count_surfaces_trimesh(file_path: str) -> int:
    """Count surfaces using trimesh (approximation via faces)."""
    mesh = trimesh.load(file_path)

    if isinstance(mesh, trimesh.Scene):
        # Multi-part file
        total_faces = sum(g.faces.shape[0] for g in mesh.geometry.values())
        return total_faces
    else:
        return mesh.faces.shape[0]

def estimate_logical_surfaces(file_path: str) -> int:
    """Estimate logical surfaces via connected components."""
    mesh = trimesh.load(file_path)

    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.geometry.values())

    # Group faces by normal similarity (connected components)
    components = mesh.split(only_watertight=False)
    return len(components)
```

**Performance Estimate:** ~10-50ms per file

---

### Option 3: cadquery

**Overview:**
CadQuery is a Python library for building parametric 3D CAD models.

**Pros:**
- Python-native API
- Built on OCC (so full BREP support)
- Easier API than raw OCC
- Good for programmatic CAD operations

**Cons:**
- Still has OCC dependency
- Primarily for model creation, not analysis
- Less documentation for import/analysis workflows

**Surface Counting Implementation:**
```python
import cadquery as cq

def count_surfaces_cadquery(file_path: str) -> int:
    """Count surfaces using CadQuery."""
    result = cq.importers.importStep(file_path)
    faces = result.faces().vals()
    return len(faces)
```

**Performance Estimate:** ~50-150ms per file

---

### Option 4: Simple Heuristic (Current v3 Approach)

**Overview:**
Estimate surface count based on entity count and type.

**Implementation:**
```python
def count_surfaces_simple(entity_count: int, entity_types: List[str]) -> int:
    """Estimate surfaces using simple heuristics."""
    SURFACE_MULTIPLIERS = {
        "box": 6,
        "cylinder": 3,
        "sphere": 1,
        "cone": 2,
        "prism": 5,  # average estimate
        "unknown": 6,  # default to box-like
    }

    total = 0
    for entity_type in entity_types:
        multiplier = SURFACE_MULTIPLIERS.get(entity_type.lower(), 6)
        total += multiplier

    # Fallback if no type info
    if total == 0:
        return entity_count * 6

    return total
```

**Performance Estimate:** ~1ms (in-memory calculation)

---

## Comparison Matrix

| Feature | OCC | Trimesh | CadQuery | Simple |
|---------|-----|---------|----------|--------|
| Accuracy | High | Medium | High | Low |
| Performance | Slow | Fast | Medium | Instant |
| Installation | Complex | Easy | Medium | None |
| STEP Support | Native | Via mesh | Native | N/A |
| Memory Usage | High | Low | Medium | Minimal |
| Maintenance | Medium | Low | Medium | Low |

---

## Recommendation

### Phase 1: v4 Simple Mode (Day 4 target)
Use **Simple Heuristic** approach:
- Zero additional dependencies
- Performance overhead: negligible
- Accuracy: sufficient for relative comparisons
- Risk: minimal

```python
# src/core/feature_extractor.py

def extract_surface_count_v4(doc: CadDocument, mode: str = "simple") -> int:
    """Extract surface count with v4 algorithm."""
    if mode == "simple":
        return _surface_count_simple(doc)
    elif mode == "advanced":
        return _surface_count_advanced(doc)
    else:
        raise ValueError(f"Unknown mode: {mode}")

def _surface_count_simple(doc: CadDocument) -> int:
    """Simple heuristic-based surface counting."""
    if not doc.entities:
        return 0

    type_multipliers = {
        "box": 6,
        "cylinder": 3,
        "sphere": 1,
        "cone": 2,
    }

    total = 0
    for entity in doc.entities:
        multiplier = type_multipliers.get(entity.type.lower(), 6)
        total += multiplier

    return total
```

### Phase 2: v4 Advanced Mode (Future)
Consider **Trimesh** for advanced mode:
- Easy to add as optional dependency
- Good balance of accuracy and performance
- Use for `FEATURE_V4_SURFACE_ALGORITHM=advanced`

```python
# Future implementation
def _surface_count_advanced(doc: CadDocument) -> int:
    """Advanced surface counting using trimesh (optional)."""
    try:
        import trimesh
    except ImportError:
        logger.warning("trimesh not installed, falling back to simple mode")
        return _surface_count_simple(doc)

    # Implementation with trimesh
    ...
```

### Phase 3: Production Enhancement (Long-term)
Evaluate **CadQuery** for production if:
- High accuracy is critical for business logic
- Team has capacity for OCC maintenance
- CAD file diversity increases significantly

---

## Shape Entropy v4 Enhancement

The v4 shape entropy calculation uses Laplace smoothing to handle edge cases:

```python
from collections import Counter
import math

def calculate_shape_entropy_v4(
    entities: List[Entity],
    smoothing: float = 1.0
) -> float:
    """Calculate normalized shape entropy with Laplace smoothing.

    Args:
        entities: List of CAD entities
        smoothing: Laplace smoothing parameter (default=1.0)

    Returns:
        Normalized entropy value in [0, 1]
    """
    if not entities:
        return 0.0

    type_counts = Counter(e.type for e in entities)
    total = sum(type_counts.values())
    vocab_size = len(type_counts)

    if vocab_size <= 1:
        return 0.0  # Single type = no uncertainty

    # Laplace smoothed entropy
    entropy = 0.0
    for count in type_counts.values():
        p = (count + smoothing) / (total + smoothing * vocab_size)
        entropy -= p * math.log2(p)

    # Normalize to [0, 1]
    max_entropy = math.log2(vocab_size)
    return entropy / max_entropy if max_entropy > 0 else 0.0
```

**Benefits of Laplace Smoothing:**
- Prevents log(0) errors
- Handles unseen categories gracefully
- Produces more stable entropy values

---

## Performance Considerations

### Benchmarking Plan

```python
# tests/performance/test_v4_surface_count.py

import pytest
import time

@pytest.mark.slow
class TestV4SurfaceCountPerformance:

    def test_simple_mode_performance(self, benchmark_files):
        """Simple mode should be < 1ms per file."""
        for file_path in benchmark_files:
            doc = load_cad_document(file_path)
            start = time.perf_counter()
            result = extract_surface_count_v4(doc, mode="simple")
            elapsed = time.perf_counter() - start
            assert elapsed < 0.001, f"Simple mode too slow: {elapsed*1000:.2f}ms"

    def test_v4_vs_v3_overhead(self, benchmark_files):
        """v4 overhead should be < 5% vs v3."""
        v3_times = []
        v4_times = []

        for file_path in benchmark_files:
            doc = load_cad_document(file_path)

            # v3
            start = time.perf_counter()
            extract_features_v3(doc)
            v3_times.append(time.perf_counter() - start)

            # v4
            start = time.perf_counter()
            extract_features_v4(doc)
            v4_times.append(time.perf_counter() - start)

        v3_avg = sum(v3_times) / len(v3_times)
        v4_avg = sum(v4_times) / len(v4_times)
        overhead = (v4_avg - v3_avg) / v3_avg

        assert overhead < 0.05, f"v4 overhead {overhead:.1%} exceeds 5% limit"
```

---

## Environment Variables

```bash
# Feature flag for v4 surface algorithm mode
FEATURE_V4_SURFACE_ALGORITHM=simple  # simple | advanced

# Enable/disable v4 features
FEATURE_V4_ENABLED=1

# Performance thresholds
FEATURE_V4_MAX_OVERHEAD_PERCENT=5
```

---

## Conclusion

**Recommended Approach for Day 4:**
1. Implement **simple mode** with heuristic-based surface counting
2. Add Laplace smoothing to shape entropy
3. Include performance benchmarks
4. Document experimental status

**Future Work:**
1. Evaluate trimesh for advanced mode
2. Consider cadquery for high-accuracy requirements
3. Build comprehensive test dataset from production samples

---

## References

- [Open CASCADE Technology](https://dev.opencascade.org/)
- [Trimesh Documentation](https://trimsh.org/)
- [CadQuery Documentation](https://cadquery.readthedocs.io/)
- [Laplace Smoothing](https://en.wikipedia.org/wiki/Additive_smoothing)
- [Information Entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory))

---

**Document Status:** Research Complete
**Next Steps:** Day 4 Implementation
**Author:** CAD ML Platform Team
**Last Updated:** 2025-12-10
