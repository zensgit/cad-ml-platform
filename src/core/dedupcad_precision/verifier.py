from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from dataclasses import replace
from typing import Any, Dict, Literal, Optional

from .vendor.config import Settings
from .vendor.scoring import weighted_similarity
from .vendor.v2_normalize import normalize_v2

logger = logging.getLogger(__name__)

_TRUE_VALUES = {"1", "true", "yes", "y", "on"}
_FALSE_VALUES = {"0", "false", "no", "n", "off"}


def _env_bool(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    v = raw.strip().lower()
    if v == "":
        return default
    if v in _TRUE_VALUES:
        return True
    if v in _FALSE_VALUES:
        return False
    logger.warning("invalid_bool_env", extra={"name": name, "value": raw})
    return default


@dataclass(frozen=True)
class PrecisionScore:
    score: float
    breakdown: Dict[str, float]
    geom_hash_left: str
    geom_hash_right: str


PrecisionProfile = Literal["strict", "version"]


def _normalize_profile(profile: Optional[str]) -> PrecisionProfile:
    raw = (profile or "").strip().lower()
    if not raw:
        return "strict"
    if raw in {"strict", "version"}:
        return raw  # type: ignore[return-value]
    raise ValueError(f"Unknown precision profile: {profile}")


class PrecisionVerifier:
    """Compute geometric/semantic similarity for v2 JSON (DedupCAD scoring)."""

    def __init__(self, settings: Optional[Settings] = None, *, normalize: bool = True) -> None:
        cfg = settings or Settings()
        # L4 in cad-ml-platform prioritizes precision over recall; keep geom-hash off by default
        # unless explicitly enabled via env.
        if settings is None:
            cfg.entities_geom_hash = _env_bool(
                "CAD_ML_PLATFORM_L4_ENTITIES_GEOM_HASH",
                default=False,
            )
        self.settings = cfg
        self.normalize = normalize

    @staticmethod
    def load_json_bytes(data: bytes) -> Dict[str, Any]:
        if not data:
            raise ValueError("Empty JSON bytes")
        return json.loads(data.decode("utf-8"))

    @staticmethod
    def canonical_json_bytes(obj: Dict[str, Any]) -> bytes:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )

    @classmethod
    def geom_hash(cls, obj: Dict[str, Any]) -> str:
        return hashlib.sha256(cls.canonical_json_bytes(obj)).hexdigest()

    def score_pair(
        self, left_v2: Dict[str, Any], right_v2: Dict[str, Any], *, profile: Optional[str] = None
    ) -> PrecisionScore:
        profile_key = _normalize_profile(profile)
        cfg = self._settings_for_profile(profile_key)
        left = normalize_v2(left_v2, cfg) if self.normalize else left_v2
        right = normalize_v2(right_v2, cfg) if self.normalize else right_v2

        cfg = self._adaptive_settings(cfg, left, right, profile=profile_key)
        score, breakdown = weighted_similarity(left, right, cfg)
        return PrecisionScore(
            score=float(score),
            breakdown={k: float(v) for k, v in breakdown.items()},
            geom_hash_left=self.geom_hash(left),
            geom_hash_right=self.geom_hash(right),
        )

    def _settings_for_profile(self, profile: PrecisionProfile) -> Settings:
        cfg = replace(self.settings)
        if profile == "strict":
            return cfg
        if profile == "version":
            # Version dedup prioritizes recall for near-duplicate drawings (minor edits,
            # export differences, small shifts). Stabilize entities matching with the
            # geometric bag-of-features fallback, and down-weight blocks/dimensions
            # which are often brittle across revisions.
            cfg.entities_geom_hash = True
            cfg.entities_spatial_enable = True
            cfg.use_hungarian = False
            cfg.use_frechet = False
            cfg.use_procrustes = False
            cfg.max_match_entities = min(int(getattr(cfg, "max_match_entities", 64) or 64), 32)
            # In "version" mode, blocks are frequently unstable across exports (proxy entities,
            # missing hashes, re-blocking). Prefer using entities/layers and keep blocks out.
            cfg.w_blocks = 0.0
            cfg.w_dimensions = min(float(getattr(cfg, "w_dimensions", 0.0) or 0.0), 0.1)
            return cfg
        raise ValueError(f"Unknown precision profile: {profile}")

    def _adaptive_settings(
        self,
        cfg_in: Settings,
        left: Dict[str, Any],
        right: Dict[str, Any],
        *,
        profile: PrecisionProfile,
    ) -> Settings:
        """Avoid inflating/penalizing scores when optional sections are missing."""
        cfg = replace(cfg_in)

        def _nonempty(v: Any) -> bool:
            if v is None:
                return False
            if isinstance(v, (list, dict, str, tuple, set)):
                return bool(v)
            return True

        def _has_both(key: str) -> bool:
            return _nonempty(left.get(key)) and _nonempty(right.get(key))

        # Only compare optional sections when both sides provide them (like blocks handling).
        if not _has_both("dimensions"):
            cfg.w_dimensions = 0.0
            cfg.w_dim_extra = 0.0
        if not _has_both("text_content"):
            cfg.w_text = 0.0
        if not _has_both("hatches"):
            cfg.w_hatch_extra = 0.0

        # In "version" profile, treat leaders as annotation: do not penalize when
        # only one side contains LEADER entities.
        if profile == "version":
            try:
                left_has_leader = any(
                    e.get("type") == "LEADER" for e in (left.get("entities") or [])
                )
                right_has_leader = any(
                    e.get("type") == "LEADER" for e in (right.get("entities") or [])
                )
                if left_has_leader != right_has_leader:
                    cfg.enable_entity_leader = False
            except Exception:
                pass
        return cfg
