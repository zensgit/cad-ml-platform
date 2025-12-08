import os
import pytest


# List of environment variables that may be modified by tests
_ENV_VARS_TO_ISOLATE = [
    "ADMIN_TOKEN",
    "X_API_KEY",
    "MODEL_OPCODE_SCAN",
    "MODEL_OPCODE_MODE",
    "MODEL_OPCODE_STRICT",
    "MODEL_MAX_MB",
    "ALLOWED_MODEL_HASHES",
    "CLASSIFICATION_MODEL_PATH",
    "CLASSIFICATION_MODEL_VERSION",
    "BATCH_SIMILARITY_MAX_IDS",
    "DRIFT_BASELINE_MAX_AGE_SECONDS",
    "DRIFT_BASELINE_AUTO_REFRESH",
    "DRIFT_BASELINE_MIN_COUNT",
    "ANALYSIS_VECTOR_DIM_CHECK",
    "ANALYSIS_MAX_FILE_MB",
    "ANALYSIS_MAX_ENTITIES",
    "VECTOR_STORE_BACKEND",
    "FORMAT_STRICT_MODE",
    "TELEMETRY_STORE_BACKEND",
    "FAISS_RECOVERY_STATE_PATH",
    "ADAPTIVE_RATE_LIMIT_ENABLED",
]


@pytest.fixture(autouse=True)
def env_isolation():
    """Isolate environment variables between tests."""
    backup = {k: os.environ.get(k) for k in _ENV_VARS_TO_ISOLATE}
    try:
        yield
    finally:
        for k, v in backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.fixture(autouse=True)
def vector_store_isolation():
    """Isolate global in-memory vector store between tests.

    Backs up and restores similarity globals to avoid cross-test interference.
    """
    from src.core import similarity  # type: ignore
    backup_store = dict(similarity._VECTOR_STORE)
    backup_meta = dict(similarity._VECTOR_META)
    backup_ts = dict(similarity._VECTOR_TS)
    backup_last_access = dict(similarity._VECTOR_LAST_ACCESS)
    backup_degraded = similarity._VECTOR_DEGRADED
    backup_degraded_reason = similarity._VECTOR_DEGRADED_REASON
    backup_degraded_at = similarity._VECTOR_DEGRADED_AT
    # Backup faiss-specific globals
    backup_faiss_index = similarity._FAISS_INDEX
    backup_faiss_dim = similarity._FAISS_DIM
    backup_faiss_id_map = dict(similarity._FAISS_ID_MAP) if similarity._FAISS_ID_MAP else {}
    backup_faiss_reverse_map = dict(similarity._FAISS_REVERSE_MAP) if similarity._FAISS_REVERSE_MAP else {}
    backup_faiss_next_recovery = getattr(similarity, "_FAISS_NEXT_RECOVERY_TS", None)
    backup_faiss_suppress_until = getattr(similarity, "_FAISS_SUPPRESS_UNTIL_TS", None)
    backup_degradation_history = list(getattr(similarity, "_DEGRADATION_HISTORY", []))
    try:
        yield
    finally:
        similarity._VECTOR_STORE.clear()
        similarity._VECTOR_META.clear()
        similarity._VECTOR_TS.clear()
        similarity._VECTOR_LAST_ACCESS.clear()
        similarity._VECTOR_STORE.update(backup_store)
        similarity._VECTOR_META.update(backup_meta)
        similarity._VECTOR_TS.update(backup_ts)
        similarity._VECTOR_LAST_ACCESS.update(backup_last_access)
        similarity._VECTOR_DEGRADED = backup_degraded
        similarity._VECTOR_DEGRADED_REASON = backup_degraded_reason
        similarity._VECTOR_DEGRADED_AT = backup_degraded_at
        # Restore faiss-specific globals
        similarity._FAISS_INDEX = backup_faiss_index
        similarity._FAISS_DIM = backup_faiss_dim
        similarity._FAISS_ID_MAP = backup_faiss_id_map
        similarity._FAISS_REVERSE_MAP = backup_faiss_reverse_map
        if hasattr(similarity, "_FAISS_NEXT_RECOVERY_TS"):
            similarity._FAISS_NEXT_RECOVERY_TS = backup_faiss_next_recovery
        if hasattr(similarity, "_FAISS_SUPPRESS_UNTIL_TS"):
            similarity._FAISS_SUPPRESS_UNTIL_TS = backup_faiss_suppress_until
        if hasattr(similarity, "_DEGRADATION_HISTORY"):
            similarity._DEGRADATION_HISTORY.clear()
            similarity._DEGRADATION_HISTORY.extend(backup_degradation_history)


@pytest.fixture(autouse=True)
def feature_cache_isolation():
    """Reset feature cache between tests."""
    try:
        from src.core.feature_cache import reset_feature_cache_for_tests
        reset_feature_cache_for_tests()
    except ImportError:
        pass
    yield
    try:
        from src.core.feature_cache import reset_feature_cache_for_tests
        reset_feature_cache_for_tests()
    except ImportError:
        pass


@pytest.fixture
def disable_dimension_enforcement(monkeypatch):
    monkeypatch.setenv("ANALYSIS_VECTOR_DIM_CHECK", "0")
    yield


@pytest.fixture(autouse=True)
def config_cache_isolation():
    """Reset config settings cache between tests."""
    try:
        import src.core.config as cfg
        backup_cache = cfg._settings_cache
    except (ImportError, AttributeError):
        backup_cache = None
    yield
    try:
        import src.core.config as cfg
        cfg._settings_cache = backup_cache
    except (ImportError, AttributeError):
        pass


@pytest.fixture(autouse=True)
def migration_history_isolation():
    """Reset vector migration history between tests."""
    try:
        import src.api.v1.vectors as vectors_mod
        if "_VECTOR_MIGRATION_HISTORY" in vectors_mod.__dict__:
            backup_history = list(vectors_mod._VECTOR_MIGRATION_HISTORY)
        else:
            backup_history = None
    except (ImportError, AttributeError):
        backup_history = None
    yield
    try:
        import src.api.v1.vectors as vectors_mod
        if backup_history is not None:
            if "_VECTOR_MIGRATION_HISTORY" in vectors_mod.__dict__:
                vectors_mod._VECTOR_MIGRATION_HISTORY.clear()
                vectors_mod._VECTOR_MIGRATION_HISTORY.extend(backup_history)
        elif "_VECTOR_MIGRATION_HISTORY" in vectors_mod.__dict__:
            vectors_mod._VECTOR_MIGRATION_HISTORY.clear()
    except (ImportError, AttributeError):
        pass


@pytest.fixture(autouse=True)
def ingestor_isolation():
    """Reset telemetry ingestor singleton between tests."""
    import asyncio
    try:
        from src.core.twin.ingest import reset_ingestor_for_tests
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                pass  # Skip if async loop is running
            else:
                loop.run_until_complete(reset_ingestor_for_tests())
        except RuntimeError:
            asyncio.run(reset_ingestor_for_tests())
    except ImportError:
        pass
    yield
    try:
        from src.core.twin.ingest import reset_ingestor_for_tests
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                pass
            else:
                loop.run_until_complete(reset_ingestor_for_tests())
        except RuntimeError:
            asyncio.run(reset_ingestor_for_tests())
    except ImportError:
        pass


@pytest.fixture(autouse=True)
def local_cache_isolation():
    """Reset in-memory cache between tests."""
    try:
        import src.utils.cache as cache_mod
        backup_cache = dict(cache_mod._local_cache)
    except (ImportError, AttributeError):
        backup_cache = None
    yield
    try:
        import src.utils.cache as cache_mod
        cache_mod._local_cache.clear()
        if backup_cache is not None:
            cache_mod._local_cache.update(backup_cache)
    except (ImportError, AttributeError):
        pass


@pytest.fixture(autouse=True)
def classifier_isolation():
    """Reset ML classifier state between tests."""
    try:
        from src.ml import classifier
        backup_model = classifier._MODEL
        backup_hash = classifier._MODEL_HASH
        backup_loaded_at = classifier._MODEL_LOADED_AT
        backup_last_error = classifier._MODEL_LAST_ERROR
    except (ImportError, AttributeError):
        backup_model = backup_hash = backup_loaded_at = backup_last_error = None
    yield
    try:
        from src.ml import classifier
        classifier._MODEL = backup_model
        classifier._MODEL_HASH = backup_hash
        classifier._MODEL_LOADED_AT = backup_loaded_at
        classifier._MODEL_LAST_ERROR = backup_last_error
    except (ImportError, AttributeError):
        pass


@pytest.fixture(autouse=True)
def analyze_module_isolation():
    """Reset analyze module global state between tests.

    Isolates:
    - _DRIFT_STATE: drift tracking state
    - _CACHE_HIT_EVENTS: sliding window cache hit events
    - _CACHE_MISS_EVENTS: sliding window cache miss events
    """
    try:
        import src.api.v1.analyze as analyze_mod
        # Backup _DRIFT_STATE
        backup_drift = {
            "materials": list(analyze_mod._DRIFT_STATE.get("materials", [])),
            "predictions": list(analyze_mod._DRIFT_STATE.get("predictions", [])),
            "baseline_materials": list(analyze_mod._DRIFT_STATE.get("baseline_materials", [])),
            "baseline_predictions": list(analyze_mod._DRIFT_STATE.get("baseline_predictions", [])),
            "baseline_materials_ts": analyze_mod._DRIFT_STATE.get("baseline_materials_ts"),
            "baseline_predictions_ts": analyze_mod._DRIFT_STATE.get("baseline_predictions_ts"),
        }
        # Backup sliding window events (created via globals().setdefault)
        backup_hit_events = None
        backup_miss_events = None
        if "_CACHE_HIT_EVENTS" in analyze_mod.__dict__:
            from collections import deque
            backup_hit_events = deque(analyze_mod._CACHE_HIT_EVENTS)
        if "_CACHE_MISS_EVENTS" in analyze_mod.__dict__:
            from collections import deque
            backup_miss_events = deque(analyze_mod._CACHE_MISS_EVENTS)
    except (ImportError, AttributeError):
        backup_drift = None
        backup_hit_events = None
        backup_miss_events = None

    yield

    try:
        import src.api.v1.analyze as analyze_mod
        # Restore _DRIFT_STATE
        if backup_drift is not None:
            analyze_mod._DRIFT_STATE.clear()
            analyze_mod._DRIFT_STATE.update(backup_drift)
        # Restore sliding window events
        if "_CACHE_HIT_EVENTS" in analyze_mod.__dict__:
            analyze_mod._CACHE_HIT_EVENTS.clear()
            if backup_hit_events is not None:
                analyze_mod._CACHE_HIT_EVENTS.extend(backup_hit_events)
        if "_CACHE_MISS_EVENTS" in analyze_mod.__dict__:
            analyze_mod._CACHE_MISS_EVENTS.clear()
            if backup_miss_events is not None:
                analyze_mod._CACHE_MISS_EVENTS.extend(backup_miss_events)
    except (ImportError, AttributeError):
        pass


@pytest.fixture(autouse=True)
def ocr_manager_isolation():
    """Reset OcrManager singleton state between tests.

    The global _manager singleton in src.api.v1.ocr holds:
    - _rate_limiters dict (RateLimiter per provider)
    - _circuits dict (CircuitBreaker per provider)
    - _locks dict (asyncio.Lock per image)
    - confidence_fallback EMA state

    Without reset, these persist across tests causing pollution.
    """
    # Reset BEFORE test to ensure clean state
    try:
        import src.api.v1.ocr as ocr_mod
        # Reset the global manager singleton to force fresh creation
        ocr_mod._manager = None
    except (ImportError, AttributeError):
        pass

    yield

    # Also reset after test as cleanup
    try:
        import src.api.v1.ocr as ocr_mod
        ocr_mod._manager = None
    except (ImportError, AttributeError):
        pass


@pytest.fixture(autouse=True)
def adaptive_limiter_isolation():
    """Reset adaptive rate limiter state between tests.

    The global adaptive_manager holds all rate limiter instances which
    accumulate state (error counts, phase, cooldown) across tests.
    Resets BEFORE each test to ensure clean state.
    """
    try:
        from src.core.resilience.adaptive_rate_limiter import adaptive_manager
        # Reset all existing limiters BEFORE test to ensure clean state
        for limiter in adaptive_manager.limiters.values():
            limiter.reset()
    except (ImportError, AttributeError):
        pass

    yield

    # Also reset after test as cleanup
    try:
        from src.core.resilience.adaptive_rate_limiter import adaptive_manager
        for limiter in adaptive_manager.limiters.values():
            limiter.reset()
    except (ImportError, AttributeError):
        pass


@pytest.fixture(autouse=True)
def redis_client_isolation():
    """Reset Redis client state between tests.

    The global _redis_client in cache.py persists across tests. When one test
    initializes Redis, subsequent tests use Redis instead of local fallback.
    This causes rate limiter pollution when different tests share the same
    Redis key 'ocr:rl:paddle'.

    This fixture resets the Redis client BEFORE each test to ensure consistent
    behavior (tests always start with Redis unavailable, using local fallback).
    """
    try:
        import src.utils.cache as cache_mod
        backup_client = cache_mod._redis_client
        cache_mod._redis_client = None  # Reset to None before test
    except (ImportError, AttributeError):
        backup_client = None

    yield

    # Restore after test (though typically we want to leave it as None)
    try:
        import src.utils.cache as cache_mod
        cache_mod._redis_client = backup_client
    except (ImportError, AttributeError):
        pass

