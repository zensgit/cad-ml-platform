"""Feature extraction operating on CadDocument.

Phase 1 provides lightweight scalar features derived from entity stats & bounding box.
Returns legacy dict shape expected by existing API until response model refactor.

v4 features (Phase 1A):
- surface_count: Total geometric surfaces/faces from entities
- shape_entropy: Shannon entropy of entity type distribution with Laplace smoothing,
  normalized to [0, 1]
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

from src.models.cad_document import CadDocument


def compute_shape_entropy(type_counts: Dict[str, int]) -> float:
    """Compute shape type distribution entropy, normalized to [0, 1].

    Uses Laplace smoothing to avoid zero probabilities:
        p_i = (freq_i + 1) / (N + K)

    Normalizes by max theoretical entropy:
        H_norm = H / log(K)

    Args:
        type_counts: Dictionary mapping entity type names to their counts.

    Returns:
        Normalized entropy in [0, 1]. Returns 0.0 for empty input or single type.
    """
    if not type_counts:
        return 0.0

    K = len(type_counts)  # Number of distinct types
    N = sum(type_counts.values())  # Total count

    if K == 1:
        return 0.0  # Single type → no uncertainty

    denom = N + K
    inv_denom = 1.0 / denom
    log = math.log

    # Shannon entropy with natural log, computed in a single pass
    H = 0.0
    for count in type_counts.values():
        p = (count + 1) * inv_denom
        H -= p * log(p)
    max_H = log(K)  # Maximum entropy for uniform distribution

    return H / max_H if max_H > 0 else 0.0


def compute_surface_count(doc: "CadDocument") -> int:
    """Compute total surface/face count from document.

    Strategy (in priority order):
    1. Use explicit 'surfaces' metadata if available
    2. Use 'facets' metadata (mesh triangles)
    3. Count entities with surface-like kinds (FACE, FACET, SURFACE, PATCH, etc.)
    4. Fallback to facets + solids heuristic

    Args:
        doc: CadDocument instance.

    Returns:
        Surface count (non-negative integer).
    """
    # Priority 1: Explicit surfaces metadata
    if "surfaces" in doc.metadata and doc.metadata["surfaces"] is not None:
        return int(doc.metadata["surfaces"])

    # Priority 2: Count surface-like entities
    surface_kinds = {"FACE", "FACET", "SURFACE", "PATCH", "TRIANGLE", "3DFACE"}
    surface_count = sum(1 for e in doc.entities if e.kind.upper() in surface_kinds)

    if surface_count > 0:
        return surface_count

    # Priority 3: Use facets metadata (common in STL)
    facets = doc.metadata.get("facets")
    if facets is not None and facets > 0:
        return int(facets)

    # Priority 4: Fallback heuristic (solids approximation)
    solids = doc.metadata.get("solids") or 0
    return int(solids)


# Stable slot declarations per version for dynamic introspection
SLOTS_V1 = [
    ("entity_count", "geometric"),
    ("bbox_width", "geometric"),
    ("bbox_height", "geometric"),
    ("bbox_depth", "geometric"),
    ("bbox_volume_estimate", "geometric"),
    ("layer_count", "semantic"),
    ("complexity_high_flag", "semantic"),
]
SLOTS_V2 = [
    ("norm_width", "geometric"),
    ("norm_height", "geometric"),
    ("norm_depth", "geometric"),
    ("width_height_ratio", "geometric"),
    ("width_depth_ratio", "geometric"),
]
SLOTS_V3 = [
    ("solids_count", "geometric"),
    ("facets_count", "geometric"),
    ("avg_volume_per_entity", "geometric"),
    ("solids_ratio", "geometric"),
    ("facets_ratio", "geometric"),
    ("top_kind_freq_1", "geometric"),
    ("top_kind_freq_2", "geometric"),
    ("top_kind_freq_3", "geometric"),
    ("top_kind_freq_4", "geometric"),
    ("top_kind_freq_5", "geometric"),
]

# Planned v4 scaffold (inactive unless FEATURE_VERSION=v4)
SLOTS_V4 = [
    ("surface_count", "geometric"),
    ("shape_entropy", "geometric"),
]


class FeatureExtractor:
    def __init__(self, feature_version: str | None = None):
        """Initialize FeatureExtractor with optional version override.

        Args:
            feature_version: Feature version to use. If None, reads from FEATURE_VERSION env var.
        """
        if feature_version is None:
            feature_version = __import__("os").getenv("FEATURE_VERSION", "v1")
        self.feature_version = feature_version

    def upgrade_vector(
        self, existing: List[float], current_version: str | None = None
    ) -> List[float]:
        """Upgrade an existing combined vector to target version.

        This is a best-effort transformation when original document context is
        unavailable. Strategy:
        - Assume input ordering: base geometric (5) + semantic (2) + optional v2 + optional v3.
        - If upgrading to higher version, pad new slots with 0.0 to maintain determinism.
        - If downgrading (rare), truncate extra tail slots.
        """
        version = self.feature_version
        base_len = 5 + 2  # geometric + semantic base
        v2_len = len(SLOTS_V2)
        v3_len = len(SLOTS_V3)
        v4_len = len(SLOTS_V4)
        # Determine current version heuristically by length
        cur_len = len(existing)
        expected_v1 = base_len
        expected_v2 = base_len + v2_len
        expected_v3 = base_len + v2_len + v3_len
        expected_v4 = base_len + v2_len + v3_len + v4_len
        inferred_version = "v1"
        if cur_len >= expected_v4:
            inferred_version = "v4"
        elif cur_len >= expected_v3:
            inferred_version = "v3"
        elif cur_len >= expected_v2:
            inferred_version = "v2"

        if current_version not in {"v1", "v2", "v3", "v4"}:
            current_version = inferred_version

        expected_map = {
            "v1": expected_v1,
            "v2": expected_v2,
            "v3": expected_v3,
            "v4": expected_v4,
        }
        expected_len = expected_map.get(current_version, expected_v1)

        # Validate length (must match expected); if not, raise to allow caller to mark error
        if cur_len != expected_len:
            raise ValueError(
                f"Unsupported existing vector length {cur_len}; expected {expected_len} for {current_version}"
            )
        # Downgrade
        if version == "v1":
            return existing[:expected_v1]
        if version == "v2":
            if current_version == "v1":
                # pad v2 slots
                return existing + [0.0] * v2_len
            if current_version in {"v3", "v4"}:
                # truncate v3/v4 extension
                return existing[:expected_v2]
            return existing  # already v2
        if version == "v3":
            if current_version == "v1":
                return existing + [0.0] * v2_len + [0.0] * v3_len
            if current_version == "v2":
                return existing + [0.0] * v3_len
            if current_version == "v4":
                # truncate v4 extension
                return existing[:expected_v3]
            return existing  # already v3
        if version == "v4":
            if current_version == "v1":
                return existing + [0.0] * v2_len + [0.0] * v3_len + [0.0] * v4_len
            if current_version == "v2":
                return existing + [0.0] * v3_len + [0.0] * v4_len
            if current_version == "v3":
                return existing + [0.0] * v4_len
            return existing  # already v4
        # Unknown target version -> return as-is
        return existing

    async def extract(self, doc: CadDocument) -> Dict[str, Any]:
        """
        Extract features from CAD document.
        """
        import time

        from src.utils.analysis_metrics import feature_extraction_latency_seconds

        start = time.time()
        entity_count = doc.entity_count()
        bbox = doc.bounding_box
        width = bbox.width
        height = bbox.height
        depth = bbox.depth
        volume = bbox.volume_estimate

        # Count entities by kind for downstream features
        counts: Dict[str, int] = {}
        for entity in doc.entities:
            kind = str(entity.kind).upper()
            counts[kind] = counts.get(kind, 0) + 1

        # Base v1 geometric features
        geometric: List[float] = [
            float(entity_count),
            float(width),
            float(height),
            float(depth),
            float(volume),
        ]

        # Base v1 semantic features
        layers = doc.layers
        try:
            layer_count = len(layers) if layers is not None else 0
        except TypeError:
            layer_count = len(list(layers))
        complexity_flag = 1.0 if doc.complexity_bucket() == "high" else 0.0
        semantic: List[float] = [float(layer_count), float(complexity_flag)]

        if self.feature_version in {"v2", "v3", "v4"}:
            max_dim = max(width, height, depth, 0.0)
            if max_dim > 0:
                norm_width = width / max_dim
                norm_height = height / max_dim
                norm_depth = depth / max_dim
            else:
                norm_width = norm_height = norm_depth = 0.0
            width_height_ratio = width / height if height > 0 else 0.0
            width_depth_ratio = width / depth if depth > 0 else 0.0
            geometric.extend(
                [
                    float(norm_width),
                    float(norm_height),
                    float(norm_depth),
                    float(width_height_ratio),
                    float(width_depth_ratio),
                ]
            )

        if self.feature_version in {"v3", "v4"}:
            solids = int(doc.metadata.get("solids") or 0)
            facets = int(doc.metadata.get("facets") or 0)
            avg_volume_per_entity = volume / entity_count if entity_count > 0 else 0.0
            solids_ratio = solids / entity_count if entity_count > 0 else 0.0
            facets_ratio = facets / entity_count if entity_count > 0 else 0.0

            total = entity_count if entity_count > 0 else 1
            top_counts = sorted(counts.values(), reverse=True)
            top_freqs = [count / total for count in top_counts[:5]]
            while len(top_freqs) < 5:
                top_freqs.append(0.0)

            geometric.extend(
                [
                    float(solids),
                    float(facets),
                    float(avg_volume_per_entity),
                    float(solids_ratio),
                    float(facets_ratio),
                ]
                + [float(freq) for freq in top_freqs]
            )

        if self.feature_version == "v4":
            surface_count = compute_surface_count(doc)
            shape_entropy = compute_shape_entropy(counts)
            geometric.extend([float(surface_count), float(shape_entropy)])

        try:
            dur = time.time() - start
            feature_extraction_latency_seconds.labels(version=self.feature_version).observe(dur)
        except Exception:
            pass

        return {
            "geometric": geometric,
            "semantic": semantic,
            "entity_counts": counts,
        }

    def flatten(self, features: Dict[str, Any]) -> List[float]:
        """Flatten features into canonical vector order.

        Order: base geometric (5) + semantic (2) + geometric extensions (v2/v3/v4).
        """
        geometric = [float(x) for x in features.get("geometric", [])]
        semantic = [float(x) for x in features.get("semantic", [])]
        base_geometric_len = 5
        base_geom = geometric[:base_geometric_len]
        ext_geom = geometric[base_geometric_len:]
        return base_geom + semantic + ext_geom

    def expected_dim(self, version: str) -> int:
        """Return expected vector length for a given feature version."""
        total = len(SLOTS_V1)
        if version in {"v2", "v3", "v4"}:
            total += len(SLOTS_V2)
        if version in {"v3", "v4"}:
            total += len(SLOTS_V3)
        if version == "v4":
            total += len(SLOTS_V4)
        return total

    def reorder_legacy_vector(self, combined: List[float], version: str) -> List[float]:
        """Convert legacy layout (geom_all + semantic) into canonical layout.

        Canonical order: base geometric (5) + semantic (2) + geometric extensions.
        """
        expected_len = self.expected_dim(version)
        if len(combined) != expected_len:
            raise ValueError(
                f"Unsupported existing vector length {len(combined)}; expected {expected_len} for {version}"
            )
        base_geometric_len = 5
        semantic_len = 2
        geom_len = expected_len - semantic_len
        geom_all = combined[:geom_len]
        semantic = combined[geom_len:]
        base_geom = geom_all[:base_geometric_len]
        ext_geom = geom_all[base_geometric_len:]
        return base_geom + semantic + ext_geom

    def rehydrate(self, combined: List[float], version: str) -> Dict[str, List[Any]]:
        """Rehydrate combined cached vector back into geometric/semantic components.

        Assumes combined order: base geometric slots, semantic slots, then version extensions.
        """
        # Base lengths
        base_geometric_len = 5
        base_semantic_len = 2
        idx = 0
        geometric = combined[idx : idx + base_geometric_len]
        idx += base_geometric_len
        semantic = combined[idx : idx + base_semantic_len]
        idx += base_semantic_len
        if version in {"v2", "v3", "v4"}:
            geometric.extend(combined[idx : idx + len(SLOTS_V2)])
            idx += len(SLOTS_V2)
        if version in {"v3", "v4"}:
            geometric.extend(combined[idx : idx + len(SLOTS_V3)])
            idx += len(SLOTS_V3)
        if version == "v4":
            geometric.extend(combined[idx : idx + len(SLOTS_V4)])
        return {"geometric": geometric, "semantic": semantic}

    def slots(self, version: str) -> List[Dict[str, str]]:
        base: List[Dict[str, str]] = [
            {"name": n, "category": c, "version": "v1"} for n, c in SLOTS_V1
        ]
        if version in {"v2", "v3", "v4"}:
            base.extend({"name": n, "category": c, "version": "v2"} for n, c in SLOTS_V2)
        if version in {"v3", "v4"}:
            base.extend({"name": n, "category": c, "version": "v3"} for n, c in SLOTS_V3)
        if version == "v4":
            base.extend({"name": n, "category": c, "version": "v4"} for n, c in SLOTS_V4)
        return base
