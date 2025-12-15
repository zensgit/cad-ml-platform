"""
Config for DedupCAD 2.0 standalone service.
Values are read from environment variables with sane defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _get_env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class Settings:
    # Service version (for /version endpoint)
    version: str = _get_env("DEDUPCAD2_VERSION", "0.2.5")
    port: int = int(_get_env("PORT", "58000"))
    auth_token: str | None = _get_env("DEDUPCAD2_TOKEN", "") or None
    max_file_size_mb: int = int(_get_env("DEDUPCAD2_MAX_FILE_MB", "20"))
    cache_dir: str = _get_env("DEDUPCAD2_CACHE_DIR", "standalone-product/dedupcad2/cache")
    # Allow tests to bypass local-only restriction by setting DEDUPCAD2_TEST_MODE=1
    local_only: bool = (
        _get_env("DEDUPCAD2_LOCAL_ONLY", "1") == "1" and _get_env("DEDUPCAD2_TEST_MODE", "0") != "1"
    )
    max_workers: int = int(_get_env("DEDUPCAD2_MAX_WORKERS", "4"))
    # Safer defaults: lower prefilter threshold and disable minhash by default
    quick_threshold: float = float(_get_env("DEDUPCAD2_QUICK_THRESHOLD", "0.2"))
    use_minhash: bool = _get_env("DEDUPCAD2_USE_MINHASH", "0") == "1"
    mh_k: int = int(_get_env("DEDUPCAD2_MH_K", "64"))
    mh_seed: int = int(_get_env("DEDUPCAD2_MH_SEED", "1337"))

    # Converter controls
    converter_order: str = _get_env("DEDUPCAD2_CONVERTER_ORDER", "oda,dwg2dxf")
    converter_timeout_s: int = int(_get_env("DEDUPCAD2_CONVERTER_TIMEOUT_S", "20"))
    converter_retries: int = int(_get_env("DEDUPCAD2_CONVERTER_RETRIES", "2"))
    converter_backoff_s: float = float(_get_env("DEDUPCAD2_CONVERTER_BACKOFF_S", "1.0"))

    # Curve matching refinements
    use_rdp: bool = _get_env("DEDUPCAD2_USE_RDP", "1") == "1"
    rdp_eps: float = float(_get_env("DEDUPCAD2_RDP_EPS", "0.5"))
    use_frechet: bool = _get_env("DEDUPCAD2_USE_FRECHET", "1") == "1"
    frechet_samples: int = int(_get_env("DEDUPCAD2_FRECHET_SAMPLES", "64"))

    # Block inner matching
    block_inner_match: bool = _get_env("DEDUPCAD2_BLOCK_INNER_MATCH", "1") == "1"
    block_cache: bool = _get_env("DEDUPCAD2_BLOCK_CACHE", "1") == "1"

    # Procrustes micro adjustment
    use_procrustes: bool = _get_env("DEDUPCAD2_USE_PROCRUSTES", "1") == "1"
    procrustes_deg: float = float(_get_env("DEDUPCAD2_PROCRUSTES_DEG", "5.0"))
    procrustes_steps: int = int(_get_env("DEDUPCAD2_PROCRUSTES_STEPS", "3"))

    # Global alignment (whole drawing) controls
    global_align_enable: bool = _get_env("DEDUPCAD2_GLOBAL_ALIGN_ENABLE", "0") == "1"
    global_align_min_pairs: int = int(_get_env("DEDUPCAD2_GLOBAL_ALIGN_MIN_PAIRS", "20"))
    global_align_min_coverage: float = float(_get_env("DEDUPCAD2_GLOBAL_ALIGN_MIN_COVERAGE", "0.5"))
    global_align_rmse_max: float = float(_get_env("DEDUPCAD2_GLOBAL_ALIGN_RMSE_MAX", "0.8"))

    # Weighted similarity for v2 JSON
    w_entities: float = float(_get_env("DEDUPCAD2_W_ENTITIES", "1.0"))
    w_layers: float = float(_get_env("DEDUPCAD2_W_LAYERS", "0.2"))
    w_dimensions: float = float(_get_env("DEDUPCAD2_W_DIMENSIONS", "0.5"))
    w_text: float = float(_get_env("DEDUPCAD2_W_TEXT", "0.3"))
    # Enhanced similarity fusion weight (0 disables fusion; still can run in shadow)
    w_enhanced: float = float(_get_env("DEDUPCAD2_W_ENHANCED", "0.0"))
    # Dimension / Hatch extra similarity weights (off by default)
    w_dim_extra: float = float(_get_env("DEDUPCAD2_W_DIM_EXTRA", "0.0"))
    w_hatch_extra: float = float(_get_env("DEDUPCAD2_W_HATCH_EXTRA", "0.0"))
    # Blocks consistency weight
    w_blocks: float = float(_get_env("DEDUPCAD2_W_BLOCKS", "0.8"))
    # Only enable blocks section when both sides have INSERT with block_hash
    blocks_strict: bool = _get_env("DEDUPCAD2_BLOCKS_STRICT", "1") == "1"
    # Block hash parameters
    block_hash_quant_step: float = float(_get_env("DEDUPCAD2_BLOCK_HASH_QUANT_STEP", "0.001"))
    block_hash_enable_arc_spline: bool = (
        _get_env("DEDUPCAD2_BLOCK_HASH_ENABLE_ARC_SPLINE", "1") == "1"
    )
    block_hash_max_entities: int = int(
        _get_env("DEDUPCAD2_BLOCK_HASH_MAX_ENTITIES", "0")
    )  # 0=unlimited
    # Advanced block hash options (default off to preserve compatibility)
    block_hash_pca_align: bool = _get_env("DEDUPCAD2_BLOCK_HASH_PCA_ALIGN", "0") == "1"
    block_hash_mirror_invariant: bool = _get_env("DEDUPCAD2_BLOCK_HASH_MIRROR_INV", "0") == "1"
    block_hash_resample_step: float = float(
        _get_env("DEDUPCAD2_BLOCK_HASH_RESAMPLE_STEP", "0.0")
    )  # 0=disabled
    # Blocks matching options
    blocks_global_align: bool = _get_env("DEDUPCAD2_BLOCKS_GLOBAL_ALIGN", "1") == "1"
    blocks_ransac: bool = _get_env("DEDUPCAD2_BLOCKS_RANSAC", "1") == "1"
    # Use Kabsch-like 2-point hypothesis inside RANSAC for global pose
    blocks_ransac_kabsch: bool = _get_env("DEDUPCAD2_BLOCKS_RANSAC_KABSCH", "1") == "1"
    blocks_ransac_iters: int = int(_get_env("DEDUPCAD2_BLOCKS_RANSAC_ITERS", "64"))
    # Effective position tolerance = max(tol_insert_pos, diag * frac)
    blocks_ransac_pos_scale_fraction: float = float(
        _get_env("DEDUPCAD2_BLOCKS_RANSAC_POS_FRAC", "0.005")
    )
    # Local clustering alignment (per spatial cluster)
    blocks_local_align: bool = _get_env("DEDUPCAD2_BLOCKS_LOCAL_ALIGN", "1") == "1"
    blocks_local_eps_frac: float = float(_get_env("DEDUPCAD2_BLOCKS_LOCAL_EPS_FRAC", "0.02"))
    blocks_local_min_samples: int = int(_get_env("DEDUPCAD2_BLOCKS_LOCAL_MIN_SAMPLES", "3"))

    # Near-hash equivalence for blocks (optional)
    blocks_near_hash: bool = _get_env("DEDUPCAD2_BLOCKS_NEAR_HASH", "0") == "1"
    blocks_near_hash_sig_threshold: float = float(_get_env("DEDUPCAD2_BLOCKS_NEAR_HASH_TH", "0.8"))
    # Near-hash v2 using enriched signature (sig2) with cosine similarity
    blocks_near_hash_v2: bool = _get_env("DEDUPCAD2_BLOCKS_NEAR_HASH_V2", "0") == "1"
    blocks_near_hash_v2_threshold: float = float(
        _get_env("DEDUPCAD2_BLOCKS_NEAR_HASH_V2_TH", "0.85")
    )
    # Optional TF-IDF-like weighting for sig2 cosine (pairwise rare-bin emphasis)
    blocks_near_hash_v2_tfidf: bool = _get_env("DEDUPCAD2_BLOCKS_NEAR_HASH_V2_TFIDF", "0") == "1"
    # Weighted Jaccard similarity for blocks (optional)
    blocks_weighted_jaccard: bool = _get_env("DEDUPCAD2_BLOCKS_WEIGHTED_JACCARD", "0") == "1"
    # Area-weighted Jaccard for blocks (optional)
    blocks_area_weighted_jaccard: bool = _get_env("DEDUPCAD2_BLOCKS_AREA_WEIGHTED", "0") == "1"
    # Hybrid weighting w = area^alpha / freq^beta (set via alpha/beta)
    blocks_weight_alpha: float = float(_get_env("DEDUPCAD2_BLOCKS_WEIGHT_ALPHA", "0.0"))
    blocks_weight_beta: float = float(_get_env("DEDUPCAD2_BLOCKS_WEIGHT_BETA", "1.0"))
    # Auto down-weight blocks when either side lacks block hashes for a pair
    blocks_auto_downweight_when_unavailable: bool = (
        _get_env("DEDUPCAD2_BLOCKS_AUTO_DOWNWEIGHT", "1") == "1"
    )

    # Use advanced entity matching (Hungarian) for entities section
    use_entities_matching: bool = _get_env("DEDUPCAD2_ENTITIES_MATCHING", "1") == "1"
    use_hungarian: bool = _get_env("DEDUPCAD2_HUNGARIAN", "1") == "1"
    # Entities geometric hash fallback (bag-of-features) to stabilize no-block cases
    entities_geom_hash: bool = _get_env("DEDUPCAD2_ENTITIES_GEOM_HASH", "1") == "1"
    # Entities spatial signature (position-aware histogram) for reducing false positives
    # in "version" matching. Disabled by default and enabled via profile/env.
    entities_spatial_enable: bool = _get_env("DEDUPCAD2_ENTITIES_SPATIAL_ENABLE", "0") == "1"
    entities_spatial_grid: int = int(_get_env("DEDUPCAD2_ENTITIES_SPATIAL_GRID", "8"))
    entities_spatial_min_points: int = int(_get_env("DEDUPCAD2_ENTITIES_SPATIAL_MIN_POINTS", "20"))
    entities_spatial_w_max: float = float(_get_env("DEDUPCAD2_ENTITIES_SPATIAL_W_MAX", "0.6"))
    entities_spatial_count_sim_gamma: float = float(
        _get_env("DEDUPCAD2_ENTITIES_SPATIAL_COUNT_GAMMA", "2.0")
    )
    entities_spatial_bbox_q_low: float = float(_get_env("DEDUPCAD2_ENTITIES_SPATIAL_BBOX_Q_LOW", "1.0"))
    entities_spatial_bbox_q_high: float = float(_get_env("DEDUPCAD2_ENTITIES_SPATIAL_BBOX_Q_HIGH", "99.0"))
    # Extended entity enable toggles
    enable_entity_ellipse: bool = _get_env("DEDUPCAD2_ENABLE_ENTITY_ELLIPSE", "1") == "1"
    enable_entity_leader: bool = _get_env("DEDUPCAD2_ENABLE_ENTITY_LEADER", "1") == "1"
    # Text fuzzy matching
    text_fuzzy_enable: bool = _get_env("DEDUPCAD2_TEXT_FUZZY_ENABLE", "1") == "1"
    text_fuzzy_threshold: float = float(_get_env("DEDUPCAD2_TEXT_FUZZY_THRESHOLD", "0.12"))
    text_fuzzy_shadow_collect: bool = _get_env("DEDUPCAD2_TEXT_FUZZY_SHADOW_COLLECT", "1") == "1"
    # Semantic embedding options
    text_embed_enable: bool = _get_env("DEDUPCAD2_TEXT_EMBED_ENABLE", "0") == "1"
    text_embed_model: str = _get_env(
        "DEDUPCAD2_TEXT_EMBED_MODEL", "mini-lm"
    )  # placeholder identifier
    text_embed_cache_dir: str = _get_env(
        "DEDUPCAD2_TEXT_EMBED_CACHE_DIR", "standalone-product/dedupcad2/text_embed_cache"
    )
    # Persistent cache backend (in-memory JSON files by default; can enable sqlite)
    text_embed_cache_backend: str = _get_env(
        "DEDUPCAD2_TEXT_EMBED_CACHE_BACKEND", "files"
    )  # files|sqlite
    text_embed_cache_sqlite_path: str = _get_env(
        "DEDUPCAD2_TEXT_EMBED_CACHE_SQLITE",
        "standalone-product/dedupcad2/text_embed_cache/embed_cache.db",
    )
    text_embed_similarity_threshold: float = float(
        _get_env("DEDUPCAD2_TEXT_EMBED_SIM_THRESHOLD", "0.85")
    )
    text_embed_use_cosine: bool = _get_env("DEDUPCAD2_TEXT_EMBED_USE_COSINE", "1") == "1"
    # Heavy model (sentence-transformers) layered embedding
    text_embed_heavy_enable: bool = _get_env("DEDUPCAD2_TEXT_EMBED_HEAVY_ENABLE", "0") == "1"
    text_embed_heavy_model: str = _get_env(
        "DEDUPCAD2_TEXT_EMBED_HEAVY_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    text_embed_heavy_threshold: float = float(
        _get_env("DEDUPCAD2_TEXT_EMBED_HEAVY_THRESHOLD", "0.90")
    )
    text_embed_heavy_batch_size: int = int(_get_env("DEDUPCAD2_TEXT_EMBED_HEAVY_BATCH_SIZE", "32"))
    text_embed_max_calls_per_request: int = int(
        _get_env("DEDUPCAD2_TEXT_EMBED_MAX_CALLS_PER_REQ", "64")
    )
    # Persistent embedding cache control
    text_embed_cache_backend: str = _get_env(
        "DEDUPCAD2_TEXT_EMBED_CACHE_BACKEND", "files"
    )  # files|sqlite
    text_embed_cache_sqlite_path: str = _get_env(
        "DEDUPCAD2_TEXT_EMBED_CACHE_SQLITE",
        "standalone-product/dedupcad2/text_embed_cache/embed_cache.db",
    )
    text_embed_cache_max_rows: int = int(_get_env("DEDUPCAD2_TEXT_EMBED_CACHE_MAX_ROWS", "50000"))
    text_embed_cache_evict_rows: int = int(
        _get_env("DEDUPCAD2_TEXT_EMBED_CACHE_EVICT_ROWS", "5000")
    )
    # EMA smoothing factor for p95 similarities reported in /stats (0.0 disables smoothing)
    embed_p95_ema_alpha: float = float(_get_env("DEDUPCAD2_EMBED_P95_EMA_ALPHA", "0.3"))
    # Short text handling (TEXT/MTEXT early-stop)
    text_short_len: int = int(_get_env("DEDUPCAD2_TEXT_SHORT_LEN", "3"))
    text_short_penalty: float = float(_get_env("DEDUPCAD2_TEXT_SHORT_PENALTY", "1.0"))
    # Heavy embedding memory control (0 disables auto-clear)
    heavy_batch_clear_threshold: int = int(_get_env("DEDUPCAD2_HEAVY_BATCH_CLEAR_THRESHOLD", "0"))
    heavy_mem_warn_mb: int = int(_get_env("DEDUPCAD2_HEAVY_MEM_WARN_MB", "0"))  # 0 disables warning
    # When blocks are unavailable in a pair, optionally boost entities weight by redirecting blocks weight
    entities_boost_when_no_blocks: bool = _get_env("DEDUPCAD2_ENTITIES_BOOST_NO_BLOCKS", "0") == "1"
    # Neighbor index for blocks assignment to reduce far-pair cost evals
    blocks_neighbor_index: bool = _get_env("DEDUPCAD2_BLOCKS_NEIGHBOR_INDEX", "1") == "1"
    # Neighbor radius multiplier over max(eps, eff_pos_tol)
    blocks_nbr_radius_mul: float = float(_get_env("DEDUPCAD2_BLOCKS_NBR_RADIUS_MUL", "4.0"))
    # Only enable neighbor index when |L|*|R| exceeds this threshold
    blocks_nbr_min_matrix: int = int(_get_env("DEDUPCAD2_BLOCKS_NBR_MIN_MATRIX", "50000"))

    # Advanced matching controls
    max_match_entities: int = int(_get_env("DEDUPCAD2_MAX_MATCH_ENTITIES", "64"))
    # Polyline/Spline/Block tolerances
    tol_polyline_len: float = float(_get_env("DEDUPCAD2_TOL_POLYLINE_LEN", "5.0"))
    tol_spline_ctrl: float = float(_get_env("DEDUPCAD2_TOL_SPLINE_CTRL", "1.0"))
    tol_insert_pos: float = float(_get_env("DEDUPCAD2_TOL_INSERT_POS", "1.0"))
    tol_insert_scale: float = float(_get_env("DEDUPCAD2_TOL_INSERT_SCALE", "0.05"))
    tol_insert_rot_deg: float = float(_get_env("DEDUPCAD2_TOL_INSERT_ROT_DEG", "2.0"))
    tol_dimension_value: float = float(_get_env("DEDUPCAD2_TOL_DIMENSION_VALUE", "0.5"))
    tol_hatch_loops: float = float(_get_env("DEDUPCAD2_TOL_HATCH_LOOPS", "1.0"))
    hatch_pattern_penalty: float = float(_get_env("DEDUPCAD2_HATCH_PATTERN_PENALTY", "0.2"))

    # Salience weights per entity type for subset selection
    sal_w_line: float = float(_get_env("DEDUPCAD2_SALIENCE_W_LINE", "1.0"))
    sal_w_circle: float = float(_get_env("DEDUPCAD2_SALIENCE_W_CIRCLE", "1.2"))
    sal_w_arc: float = float(_get_env("DEDUPCAD2_SALIENCE_W_ARC", "1.1"))
    sal_w_polyline: float = float(_get_env("DEDUPCAD2_SALIENCE_W_POLYLINE", "1.3"))
    sal_w_text: float = float(_get_env("DEDUPCAD2_SALIENCE_W_TEXT", "0.8"))
    sal_w_insert: float = float(_get_env("DEDUPCAD2_SALIENCE_W_INSERT", "1.5"))
    sal_w_ellipse: float = float(_get_env("DEDUPCAD2_SALIENCE_W_ELLIPSE", "1.2"))
    sal_w_leader: float = float(_get_env("DEDUPCAD2_SALIENCE_W_LEADER", "0.9"))

    # Section thresholds (optional enforcement)
    th_entities: float = float(_get_env("DEDUPCAD2_TH_ENTITIES", "0.90"))
    th_layers: float = float(_get_env("DEDUPCAD2_TH_LAYERS", "0.80"))

    def validate(self):  # runtime validation returning list of error strings
        errors = []

        def rng(name, val, lo, hi):
            try:
                if not (lo <= val <= hi):
                    errors.append(f"{name} out_of_range {val} not in [{lo},{hi}]")
            except Exception:
                errors.append(f"{name} invalid")

        rng("text_embed_similarity_threshold", self.text_embed_similarity_threshold, 0.5, 0.99)
        rng("text_embed_heavy_threshold", self.text_embed_heavy_threshold, 0.5, 1.0)
        try:
            if self.text_embed_heavy_threshold < self.text_embed_similarity_threshold:
                errors.append("heavy_threshold < light_threshold")
        except Exception:
            errors.append("heavy_threshold comparison failed")
        if self.text_embed_heavy_batch_size <= 0:
            errors.append("heavy_batch_size <= 0")
        if self.text_short_len < 0:
            errors.append("text_short_len < 0")
        if self.text_short_penalty < 0:
            errors.append("text_short_penalty < 0")
        if self.embed_p95_ema_alpha < 0 or self.embed_p95_ema_alpha > 1:
            errors.append("embed_p95_ema_alpha out_of_range")
        if self.text_embed_cache_max_rows < 0 or self.text_embed_cache_evict_rows < 0:
            errors.append("embed_cache rows negative")
        if self.text_embed_cache_evict_rows > self.text_embed_cache_max_rows:
            errors.append("embed_cache_evict_rows > max_rows")
        return errors

    th_dimensions: float = float(_get_env("DEDUPCAD2_TH_DIMENSIONS", "0.85"))
    th_text: float = float(_get_env("DEDUPCAD2_TH_TEXT", "0.80"))
    th_blocks: float = float(_get_env("DEDUPCAD2_TH_BLOCKS", "0.85"))
    enforce_section_thresholds: bool = _get_env("DEDUPCAD2_ENFORCE_SECTION_THRESHOLDS", "0") == "1"

    # Enhanced similarity (distribution + tolerant counts + layer structure)
    enhanced_similarity_enable: bool = _get_env("DEDUPCAD2_ENHANCED_SIM_ENABLE", "0") == "1"
    entity_count_tolerance: float = float(
        _get_env("DEDUPCAD2_ENH_ENT_COUNT_TOL", "0.10")
    )  # Optimized from 0.05 via parameter tuning
    entity_count_penalty_factor: float = float(
        _get_env("DEDUPCAD2_ENH_ENT_COUNT_PENALTY", "2.5")
    )  # Optimized from 5.0 via parameter tuning
    w_entity_distribution: float = float(_get_env("DEDUPCAD2_W_ENTITY_DISTRIBUTION", "0.3"))
    w_entity_count: float = float(_get_env("DEDUPCAD2_W_ENTITY_COUNT", "0.2"))
    w_layer_structure: float = float(_get_env("DEDUPCAD2_W_LAYER_STRUCTURE", "0.1"))
    # v0.4.0: Shadow mode - run enhanced similarity without affecting results, for A/B testing
    enhanced_similarity_shadow_mode: bool = _get_env("DEDUPCAD2_ENHANCED_SIM_SHADOW", "0") == "1"

    # Tolerances
    tol_numeric_abs: float = float(_get_env("DEDUPCAD2_TOL_NUMERIC_ABS", "0.001"))
    tol_text_case_insensitive: bool = _get_env("DEDUPCAD2_TOL_TEXT_CASEI", "1") == "1"
    tol_text_ignore_ws: bool = _get_env("DEDUPCAD2_TOL_TEXT_IGNORE_WS", "1") == "1"

    # Geometry tolerances (units assumed in DXF)
    tol_line_pos: float = float(_get_env("DEDUPCAD2_TOL_LINE_POS", "0.5"))
    tol_circle_center: float = float(_get_env("DEDUPCAD2_TOL_CIRCLE_CENTER", "0.5"))
    tol_circle_radius: float = float(_get_env("DEDUPCAD2_TOL_CIRCLE_RADIUS", "0.5"))
    tol_arc_angle_deg: float = float(_get_env("DEDUPCAD2_TOL_ARC_ANGLE_DEG", "2.0"))
    # Ellipse tolerances
    tol_ellipse_center: float = float(_get_env("DEDUPCAD2_TOL_ELLIPSE_CENTER", "0.8"))
    tol_ellipse_major_len: float = float(_get_env("DEDUPCAD2_TOL_ELLIPSE_MAJOR_LEN", "1.0"))
    tol_ellipse_major_angle_deg: float = float(_get_env("DEDUPCAD2_TOL_ELLIPSE_MAJOR_ANG", "3.0"))
    tol_ellipse_ratio: float = float(_get_env("DEDUPCAD2_TOL_ELLIPSE_RATIO", "0.05"))
    tol_ellipse_param: float = float(_get_env("DEDUPCAD2_TOL_ELLIPSE_PARAM", "0.15"))
    # Leader tolerances
    tol_leader_pos: float = float(_get_env("DEDUPCAD2_TOL_LEADER_POS", "1.0"))
    tol_leader_len: float = float(_get_env("DEDUPCAD2_TOL_LEADER_LEN", "3.0"))
    layer_mismatch_penalty: float = float(_get_env("DEDUPCAD2_LAYER_MISMATCH_PENALTY", "0.1"))

    # Stats snapshot (disabled when 0)
    stats_snapshot_interval_s: int = int(_get_env("DEDUPCAD2_STATS_SNAPSHOT_INTERVAL_S", "0"))
    stats_snapshot_path: str = _get_env(
        "DEDUPCAD2_STATS_SNAPSHOT_PATH", "outputs/stats_snapshot.json"
    )
    # Prometheus metrics toggle (disabled by default)
    enable_metrics: bool = _get_env("DEDUPCAD2_ENABLE_METRICS", "0") == "1"

    # API Output Settings
    # Maximum number of heatmap points to return in /report-json (M2 feature)
    heatmap_max_points: int = int(_get_env("DEDUPCAD2_HEATMAP_MAX_POINTS", "1000"))
    # PDF report snapshot embedding (M2 Phase 2 P0 feature)
    # When enabled, embed 3 snapshots (left/right/merged) in PDF reports
    pdf_embed_snapshots: bool = _get_env("DEDUPCAD2_PDF_EMBED_SNAPSHOTS", "0") == "1"

    @property
    def max_file_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


def get_settings() -> Settings:
    cfg = Settings()
    # Optional adaptive thresholds auto-load
    try:
        adaptive_path = os.environ.get(
            "DEDUPCAD2_EMBED_THRESHOLD_FILE", "config/embed_thresholds.env"
        )
        if adaptive_path and os.path.exists(adaptive_path):
            with open(adaptive_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip()
                    if k == "DEDUPCAD2_TEXT_EMBED_SIM_THRESHOLD":
                        try:
                            cfg.text_embed_similarity_threshold = float(v)
                        except Exception:
                            pass
                    elif k == "DEDUPCAD2_TEXT_EMBED_HEAVY_THRESHOLD":
                        try:
                            cfg.text_embed_heavy_threshold = float(v)
                        except Exception:
                            pass
    except Exception:
        pass
    return cfg
