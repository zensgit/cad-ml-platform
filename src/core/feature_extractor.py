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

    # Laplace smoothed probabilities
    probs = [(c + 1) / (N + K) for c in type_counts.values()]

    # Shannon entropy with natural log
    H = -sum(p * math.log(p) for p in probs)
    max_H = math.log(K)  # Maximum entropy for uniform distribution

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

    def upgrade_vector(self, existing: List[float]) -> List[float]:
        """Upgrade an existing geometric+semantic combined vector to target version.

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
        current_version = "v1"
        if cur_len >= expected_v4:
            current_version = "v4"
        elif cur_len >= expected_v3:
            current_version = "v3"
        elif cur_len >= expected_v2:
            current_version = "v2"
        # Validate length (must be one of expected set); if not, raise to allow caller to mark error
        if cur_len not in {expected_v1, expected_v2, expected_v3, expected_v4}:
            raise ValueError(
                f"Unsupported existing vector length {cur_len}; expected one of {expected_v1},{expected_v2},{expected_v3},{expected_v4}"
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

    async def extract(self, doc: CadDocument) -> Dict[str, List[Any]]:  # legacy shape
        import time
        from src.utils.analysis_metrics import feature_extraction_latency_seconds
        start = time.time()
        bbox = doc.bounding_box
        geometric: List[float] = [
            doc.entity_count(),
            bbox.width,
            bbox.height,
            bbox.depth,
            bbox.volume_estimate,
        ]
        semantic: List[float] = [len(doc.layers), 1 if doc.complexity_bucket() == "high" else 0]
        version = self.feature_version

        # v2 & above normalized dims + aspect ratios
        if version in {"v2", "v3", "v4"}:
            max_edge = max(bbox.width or 1.0, bbox.height or 1.0, bbox.depth or 1.0)
            norm_width = (bbox.width / max_edge) if max_edge else 0.0
            norm_height = (bbox.height / max_edge) if max_edge else 0.0
            norm_depth = (bbox.depth / max_edge) if max_edge else 0.0
            geometric.extend([
                round(norm_width, 5),
                round(norm_height, 5),
                round(norm_depth, 5),
                round((bbox.width / (bbox.height or 1.0)), 5),
                round((bbox.width / (bbox.depth or 1.0)), 5),
            ])

        # v3 & above geometry enrichment + entity kind frequencies
        if version in {"v3", "v4"}:
            solids = doc.metadata.get("solids") or 0
            facets = doc.metadata.get("facets") or 0
            entity_count = doc.entity_count() or 1
            volume = bbox.volume_estimate or 0.0
            geometric.extend([
                float(solids),
                float(facets),
                round(volume / entity_count, 5),
                round((solids / entity_count) if entity_count else 0.0, 5),
                round((facets / entity_count) if entity_count else 0.0, 5),
            ])
            kind_counts: Dict[str, int] = {}
            for e in doc.entities:
                kind_counts[e.kind] = kind_counts.get(e.kind, 0) + 1
            top_k = sorted(kind_counts.items(), key=lambda x: (-x[1], x[0]))[:5]
            total_entities = doc.entity_count() or 1
            for _, cnt in top_k:
                geometric.append(round(cnt / total_entities, 5))
            if len(top_k) < 5:
                geometric.extend([0.0] * (5 - len(top_k)))

        # v4 real features: surface_count and shape_entropy (Phase 1A)
        if version == "v4":
            # Surface count: from entities or metadata
            surface_count = float(compute_surface_count(doc))

            # Shape entropy: entity type distribution with Laplace smoothing
            kind_counts: Dict[str, int] = {}
            for e in doc.entities:
                kind_counts[e.kind] = kind_counts.get(e.kind, 0) + 1
            shape_entropy = compute_shape_entropy(kind_counts)

            geometric.extend([surface_count, round(shape_entropy, 5)])

        out = {"geometric": geometric, "semantic": semantic}
        try:
            dur = time.time() - start
            feature_extraction_latency_seconds.labels(version=version).observe(dur)
        except Exception:
            pass
        return out

    def rehydrate(self, combined: List[float], version: str) -> Dict[str, List[Any]]:
        """Rehydrate combined cached vector back into geometric/semantic components.

        Assumes ordering identical to extract(): first base geometric slots, then semantic slots, then version extensions.
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
