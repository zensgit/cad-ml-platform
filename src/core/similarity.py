"""In-memory similarity vector store (Phase 1).

Provides lightweight cosine similarity for CAD feature vectors.
Later phases can replace with persistent / ANN index (Faiss, Milvus etc.).
"""

from __future__ import annotations

from math import sqrt
from typing import Any, Dict, List, Protocol, runtime_checkable
import os
from src.utils.cache import get_client
from src.utils.analysis_metrics import (
    analysis_vector_count,
    vector_dimension_rejections_total,
    material_drift_ratio,
    vector_store_material_total,
    faiss_index_size,
    faiss_init_errors_total,
    faiss_index_age_seconds,
    vector_query_backend_total,
    similarity_degraded_total,
    faiss_recovery_state_backend,
    faiss_recovery_suppression_remaining_seconds,
)
from src.core.errors_extended import ErrorCode
from src.utils.analysis_metrics import analysis_error_code_total

_VECTOR_STORE: Dict[str, List[float]] = {}
_VECTOR_META: Dict[str, Dict[str, str]] = {}
_VECTOR_LOCK = __import__("threading").RLock()
_LAST_VECTOR_ERROR: Dict[str, str] | None = None

_BACKEND = os.getenv("VECTOR_STORE_BACKEND", "memory")
_TTL_SECONDS = int(os.getenv("VECTOR_TTL_SECONDS", "0"))  # 0 = disabled
_VECTOR_TS: Dict[str, float] = {}
_VECTOR_LAST_ACCESS: Dict[str, float] = {}

# Degraded mode flag: set when Faiss falls back to memory
_VECTOR_DEGRADED: bool = False
_VECTOR_DEGRADED_REASON: str | None = None
_VECTOR_DEGRADED_AT: float | None = None

# Degradation history (limited to last 10 events)
_DEGRADATION_HISTORY: list[Dict[str, Any]] = []
_MAX_DEGRADATION_HISTORY = 10
_FAISS_RECOVERY_LOCK = __import__("threading").Lock()
_FAISS_MANUAL_RECOVERY_IN_PROGRESS = False
_FAISS_RECOVERY_INTERVAL_SECONDS = float(os.getenv("FAISS_RECOVERY_INTERVAL_SECONDS", "300"))
_FAISS_RECOVERY_MAX_BACKOFF = float(os.getenv("FAISS_RECOVERY_MAX_BACKOFF", "3600"))
_FAISS_RECOVERY_BACKOFF_MULTIPLIER = float(os.getenv("FAISS_RECOVERY_BACKOFF_MULTIPLIER", "2"))
_FAISS_NEXT_RECOVERY_TS: float | None = None
_FAISS_SUPPRESS_UNTIL_TS: float | None = None  # flapping suppression window
_FAISS_RECOVERY_FLAP_THRESHOLD = int(os.getenv("FAISS_RECOVERY_FLAP_THRESHOLD", "3"))
_FAISS_RECOVERY_FLAP_WINDOW_SECONDS = int(os.getenv("FAISS_RECOVERY_FLAP_WINDOW_SECONDS", "900"))  # 15m
_FAISS_RECOVERY_SUPPRESSION_SECONDS = int(os.getenv("FAISS_RECOVERY_SUPPRESSION_SECONDS", "300"))  # 5m
_FAISS_RECOVERY_STATE_BACKEND = os.getenv("FAISS_RECOVERY_STATE_BACKEND", "file").lower()
try:
    faiss_recovery_state_backend.labels(backend=_FAISS_RECOVERY_STATE_BACKEND).set(1)
except Exception:
    pass

# Persistence for recovery backoff / suppression state
def _get_recovery_state_path() -> str:
    return os.getenv("FAISS_RECOVERY_STATE_PATH", "data/faiss_recovery_state.json")

def _store_recovery_state(payload: dict) -> None:
    """Store recovery state using configured backend (file|redis)."""
    try:
        if _FAISS_RECOVERY_STATE_BACKEND == "redis":
            client = get_client()
            if client is not None:
                client.set("faiss:recovery_state", __import__("json").dumps(payload))
                return
    except Exception:
        pass
    # Fallback to file
    try:
        import json, os
        path = _get_recovery_state_path()
        dirn = os.path.dirname(path)
        if dirn:
            os.makedirs(dirn, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    except Exception:
        pass

def _persist_recovery_state():  # pragma: no cover (IO side effects)
    try:
        import time
        payload = {
            "next_recovery_ts": _FAISS_NEXT_RECOVERY_TS,
            "suppress_until_ts": _FAISS_SUPPRESS_UNTIL_TS,
            "degraded": _VECTOR_DEGRADED,
            "degraded_reason": _VECTOR_DEGRADED_REASON,
            "degraded_at": _VECTOR_DEGRADED_AT,
            "persisted_at": time.time(),
        }
        _store_recovery_state(payload)
    except Exception:
        pass

def load_recovery_state():  # pragma: no cover (invoked at startup)
    """Load persisted recovery state into globals (best-effort)."""
    global _FAISS_NEXT_RECOVERY_TS, _FAISS_SUPPRESS_UNTIL_TS, _VECTOR_DEGRADED, _VECTOR_DEGRADED_REASON, _VECTOR_DEGRADED_AT
    try:
        import json, os, time
        data = None
        if _FAISS_RECOVERY_STATE_BACKEND == "redis":
            client = get_client()
            if client is not None:
                val = client.get("faiss:recovery_state")
                if val:
                    data = json.loads(val)
        if data is None:
            path = _get_recovery_state_path()
            if not os.path.exists(path):
                return False
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        _FAISS_NEXT_RECOVERY_TS = data.get("next_recovery_ts")
        _FAISS_SUPPRESS_UNTIL_TS = data.get("suppress_until_ts")
        # Only restore degraded flags if still relevant (timestamp within last 24h)
        ts = data.get("degraded_at")
        if ts and time.time() - ts < 86400 and data.get("degraded"):
            _VECTOR_DEGRADED = True
            _VECTOR_DEGRADED_REASON = data.get("degraded_reason")
            _VECTOR_DEGRADED_AT = ts
        return True
    except Exception:
        return False


def register_vector(doc_id: str, vector: List[float], meta: Dict[str, str] | None = None) -> bool:
    """Register a vector.

    Returns True if accepted, False if rejected (e.g. dimension mismatch).
    Dimension enforcement controlled by env ANALYSIS_VECTOR_DIM_CHECK (default=1).
    """
    enforce = os.getenv("ANALYSIS_VECTOR_DIM_CHECK", "1") == "1"
    if enforce and _VECTOR_STORE:
        first_vec = next(iter(_VECTOR_STORE.values()))
        if len(first_vec) != len(vector):
            analysis_error_code_total.labels(code=ErrorCode.DIMENSION_MISMATCH.value).inc()
            vector_dimension_rejections_total.labels(reason="dimension_mismatch_register").inc()
            import logging
            logging.getLogger(__name__).debug(
                "vector_dimension_mismatch", extra={
                    "doc_id": doc_id,
                    "expected_dim": len(first_vec),
                    "received_dim": len(vector),
                }
            )
            # store structured last error for retrieval (avoid breaking existing boolean API)
            global _LAST_VECTOR_ERROR
            _LAST_VECTOR_ERROR = {
                "code": ErrorCode.DIMENSION_MISMATCH.value,
                "stage": "vector_register",
                "id": doc_id,
                "expected": str(len(first_vec)),
                "found": str(len(vector)),
            }
            return False
    with _VECTOR_LOCK:
        _VECTOR_STORE[doc_id] = vector
    import time
    _VECTOR_TS[doc_id] = time.time()
    _VECTOR_LAST_ACCESS[doc_id] = _VECTOR_TS[doc_id]
    if meta:
        with _VECTOR_LOCK:
            _VECTOR_META[doc_id] = meta
    # update material distribution drift metric (dominant ratio)
    try:
        if meta and meta.get("material"):
            vector_store_material_total.labels(material=meta.get("material", "unknown")).inc()
        # compute dominant ratio
        counts: Dict[str, int] = {}
        for m in (m.get("material", "unknown") for m in _VECTOR_META.values()):
            counts[m] = counts.get(m, 0) + 1
        total = sum(counts.values()) or 1
        dominant = max(counts.values()) if counts else 0
        material_drift_ratio.observe(dominant / total)
    except Exception:
        pass
    if _BACKEND == "redis":
        client = get_client()
        if client is not None:
            try:
                # store vector as comma-separated floats and meta as JSON
                client.hset(f"vector:{doc_id}", mapping={
                    "v": ",".join(str(float(x)) for x in vector),
                    "m": __import__("json").dumps(meta or {}),
                    "ts": str(int(_VECTOR_TS[doc_id])),
                })
                analysis_vector_count.labels(backend="redis").inc()
            except Exception:
                pass
    else:
        analysis_vector_count.labels(backend="memory").inc()
    return True


def last_vector_error() -> Dict[str, str] | None:
    """Return the last structured vector registration error (dimension mismatch etc.)."""
    return _LAST_VECTOR_ERROR


def has_vector(doc_id: str) -> bool:
    with _VECTOR_LOCK:
        return doc_id in _VECTOR_STORE


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sqrt(sum(x * x for x in a))
    nb = sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def compute_similarity(reference_id: str, candidate_vector: List[float]) -> Dict[str, float | str]:
    if not has_vector(reference_id):
        return {"reference_id": reference_id, "status": "reference_not_found", "score": 0.0}
    with _VECTOR_LOCK:
        ref = _VECTOR_STORE[reference_id]
    score = _cosine(ref, candidate_vector)
    return {
        "reference_id": reference_id,
        "score": round(score, 4),
        "method": "cosine",
        "dimension": len(candidate_vector),
    }


@runtime_checkable
class VectorStoreProtocol(Protocol):
    """Protocol defining the interface for vector storage backends."""
    def add(self, key: str, vector: List[float]) -> None: ...  # noqa: D401
    def get(self, key: str) -> List[float] | None: ...
    def exists(self, key: str) -> bool: ...
    def query(self, vector: List[float], top_k: int = 5) -> List[tuple[str, float]]: ...
    def meta(self, key: str) -> Dict[str, str] | None: ...  # noqa: D102


class InMemoryVectorStore(VectorStoreProtocol):
    def __init__(self):
        self._store: Dict[str, List[float]] = _VECTOR_STORE  # reuse global for backward compatibility

    def add(self, key: str, vector: List[float]) -> None:
        self._store[key] = vector

    def get(self, key: str) -> List[float] | None:
        if key in self._store:
            return self._store.get(key)
        if _BACKEND == "redis":
            client = get_client()
            if client is not None:
                try:
                    data = client.hgetall(f"vector:{key}")
                    if not data:
                        return None
                    raw = data.get("v")
                    if raw:
                        return [float(x) for x in raw.split(",") if x]
                except Exception:
                    return None
        return None

    def exists(self, key: str) -> bool:
        if key in self._store:
            return True
        if _BACKEND == "redis":
            client = get_client()
            if client is not None:
                try:
                    return client.exists(f"vector:{key}") == 1
                except Exception:
                    return False
        return False

    def query(self, vector: List[float], top_k: int = 5) -> List[tuple[str, float]]:
        results: List[tuple[str, float]] = []
        self._prune()
        if _BACKEND == "redis":
            client = get_client()
            if client is not None:
                try:
                    # scan redis keys pattern vector:*
                    cursor = 0
                    while True:
                        cursor, batch = client.scan(cursor=cursor, match="vector:*", count=100)
                        for full in batch:
                            k = full.split(":", 1)[1]
                            v = self.get(k)
                            if not v or len(v) != len(vector):
                                continue
                            results.append((k, _cosine(vector, v)))
                        if cursor == 0:
                            break
                except Exception:
                    pass
        # memory store always included (combined view)
        with _VECTOR_LOCK:
            for k, v in _VECTOR_STORE.items():
                if len(v) != len(vector):
                    continue
                results.append((k, _cosine(vector, v)))
                import time as _t
                _VECTOR_LAST_ACCESS[k] = _t.time()
        return sorted(results, key=lambda x: x[1], reverse=True)[:top_k]

    def meta(self, key: str) -> Dict[str, str] | None:
        if key in _VECTOR_META:
            return _VECTOR_META.get(key)
        if _BACKEND == "redis":
            client = get_client()
            if client is not None:
                try:
                    data = client.hgetall(f"vector:{key}")
                    if not data:
                        return None
                    raw_meta = data.get("m")
                    if raw_meta:
                        return __import__("json").loads(raw_meta)
                except Exception:
                    return None
        return None

    def _prune(self) -> None:
        if _TTL_SECONDS <= 0:
            return
        import time
        now = time.time()
        expired = [vid for vid, ts in _VECTOR_TS.items() if now - ts > _TTL_SECONDS]
        if not expired:
            return
        with _VECTOR_LOCK:
            for vid in expired:
                _VECTOR_STORE.pop(vid, None)
                _VECTOR_META.pop(vid, None)
                _VECTOR_TS.pop(vid, None)
                _VECTOR_LAST_ACCESS.pop(vid, None)
        if _BACKEND == "redis":
            client = get_client()
            if client is not None:
                try:
                    for vid in expired:
                        client.delete(f"vector:{vid}")
                except Exception:
                    pass



__all__ = [
    "register_vector",
    "compute_similarity",
    "has_vector",
    "VectorStoreProtocol",
    "InMemoryVectorStore",
    "FaissVectorStore",
    "last_vector_error",
    "get_vector_store",
    "reset_default_store",
    "get_degraded_mode_info",
]


async def background_prune_task(interval: float = 30.0) -> None:  # pragma: no cover (loop)
    """Periodic pruning of expired vectors based on TTL."""
    import asyncio
    while True:
        try:
            if _TTL_SECONDS > 0:
                store = InMemoryVectorStore()
                store._prune()  # type: ignore[attr-defined]
                # cold access pruning (optional)
                max_idle = float(os.getenv("VECTOR_MAX_IDLE_SECONDS", "0"))
                if max_idle > 0:
                    import time
                    now = time.time()
                    cold = [vid for vid, ats in _VECTOR_LAST_ACCESS.items() if now - ats > max_idle]
                    with _VECTOR_LOCK:
                        for vid in cold:
                            _VECTOR_STORE.pop(vid, None)
                            _VECTOR_META.pop(vid, None)
                            _VECTOR_TS.pop(vid, None)
                            _VECTOR_LAST_ACCESS.pop(vid, None)
                    if cold:
                        from src.utils.analysis_metrics import vector_cold_pruned_total
                        try:
                            vector_cold_pruned_total.labels(reason="idle").inc(len(cold))
                        except Exception:
                            pass
        except Exception:
            pass
        await asyncio.sleep(interval)
_FAISS_INDEX = None  # type: ignore
_FAISS_DIM: int | None = None
_FAISS_ID_MAP: Dict[int, str] = {}
_FAISS_REVERSE_MAP: Dict[str, int] = {}
_FAISS_AVAILABLE: bool | None = None
_FAISS_PENDING_DELETE: set[str] = set()
_FAISS_LAST_EXPORT_SIZE: int = 0
_FAISS_LAST_EXPORT_TS: float | None = None
_FAISS_LAST_IMPORT: float | None = None
_FAISS_IMPORTED: bool = False
_FAISS_MAX_PENDING_DELETE = int(os.getenv("FAISS_MAX_PENDING_DELETE", "100"))
_FAISS_REBUILD_BACKOFF = float(os.getenv("FAISS_REBUILD_BACKOFF_INITIAL", "5"))  # seconds
_FAISS_REBUILD_BACKOFF_MAX = float(os.getenv("FAISS_REBUILD_BACKOFF_MAX", "300"))
_FAISS_NEXT_REBUILD_TS: float | None = None


class FaissVectorStore(VectorStoreProtocol):
    """FAISS backend using IndexFlatIP (inner product) to approximate cosine after normalization.

    Gracefully degrades to unavailable state if faiss not installed.
    """

    def __init__(self, normalize: bool | None = None):
        global _FAISS_AVAILABLE
        if _FAISS_AVAILABLE is None:
            try:
                import faiss  # type: ignore  # noqa: F401
                _FAISS_AVAILABLE = True
            except Exception:
                _FAISS_AVAILABLE = False
        self._available = _FAISS_AVAILABLE
        self._normalize = normalize if normalize is not None else os.getenv("FEATURE_COSINE_NORMALIZE", "1") == "1"

    def _create_index(self, dim: int) -> Any:
        """Create a new Faiss index based on configuration."""
        import faiss  # type: ignore
        index_type = os.getenv("FAISS_INDEX_TYPE", "flat").lower()

        if index_type == "hnsw":
            m = int(os.getenv("FAISS_HNSW_M", "32"))
            # METRIC_INNER_PRODUCT = 2
            index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = int(os.getenv("FAISS_HNSW_EF_CONSTRUCTION", "40"))
            return index
        else:
            # Default to FlatIP (Brute force)
            return faiss.IndexFlatIP(dim)

    def add(self, key: str, vector: List[float]) -> None:
        if not self._available:
            raise RuntimeError("Faiss backend not available")
        global _FAISS_INDEX, _FAISS_DIM
        import numpy as np  # type: ignore
        dim = len(vector)
        if _FAISS_DIM is None:
            _FAISS_DIM = dim
            try:
                _FAISS_INDEX = self._create_index(dim)
            except Exception:
                faiss_init_errors_total.inc()
                raise RuntimeError("Faiss initialization failed")
        if dim != _FAISS_DIM:
            vector_dimension_rejections_total.labels(reason="dimension_mismatch_faiss_add").inc()
            raise ValueError(f"Dimension mismatch: expected {_FAISS_DIM} got {dim}")
        arr = np.array(vector, dtype="float32")
        if self._normalize:
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm
        _FAISS_INDEX.add(arr.reshape(1, dim))  # type: ignore
        idx = _FAISS_INDEX.ntotal - 1  # type: ignore
        _FAISS_ID_MAP[idx] = key
        _FAISS_REVERSE_MAP[key] = idx
        faiss_index_size.set(_FAISS_INDEX.ntotal)  # type: ignore
        try:
            # age increases until next import/export; here we just ensure gauge not negative
            import time as _t
            if _FAISS_LAST_EXPORT_TS is not None:
                faiss_index_age_seconds.set(_t.time() - _FAISS_LAST_EXPORT_TS)
            elif _FAISS_LAST_IMPORT is not None:
                faiss_index_age_seconds.set(_t.time() - _FAISS_LAST_IMPORT)
        except Exception:
            pass

    def get(self, key: str) -> List[float] | None:
        return _VECTOR_STORE.get(key)

    def exists(self, key: str) -> bool:
        return key in _FAISS_REVERSE_MAP

    def mark_delete(self, key: str) -> None:
        if key in _FAISS_REVERSE_MAP:
            _FAISS_PENDING_DELETE.add(key)
            # Auto rebuild trigger based on threshold
            if len(_FAISS_PENDING_DELETE) >= _FAISS_MAX_PENDING_DELETE:
                import time as _t
                global _FAISS_NEXT_REBUILD_TS, _FAISS_REBUILD_BACKOFF
                if _FAISS_NEXT_REBUILD_TS is None or _t.time() >= _FAISS_NEXT_REBUILD_TS:
                    try:
                        ok = self.rebuild()  # type: ignore[attr-defined]
                        globals()["_FAISS_LAST_REBUILD_STATUS"] = "success" if ok else "error"
                        globals()["_FAISS_LAST_ERROR"] = None if ok else "auto_rebuild_failed"
                        from src.utils.analysis_metrics import faiss_auto_rebuild_total, faiss_rebuild_backoff_seconds
                        faiss_auto_rebuild_total.labels(status="success" if ok else "error").inc()
                        # reset or increase backoff
                        if ok:
                            _FAISS_REBUILD_BACKOFF = float(os.getenv("FAISS_REBUILD_BACKOFF_INITIAL", "5"))
                        else:
                            _FAISS_REBUILD_BACKOFF = min(_FAISS_REBUILD_BACKOFF * 2, _FAISS_REBUILD_BACKOFF_MAX)
                        _FAISS_NEXT_REBUILD_TS = _t.time() + _FAISS_REBUILD_BACKOFF
                        faiss_rebuild_backoff_seconds.set(_FAISS_REBUILD_BACKOFF)
                    except Exception as e:
                        from src.utils.analysis_metrics import faiss_auto_rebuild_total, faiss_rebuild_backoff_seconds
                        faiss_auto_rebuild_total.labels(status="error").inc()
                        _FAISS_REBUILD_BACKOFF = min(_FAISS_REBUILD_BACKOFF * 2, _FAISS_REBUILD_BACKOFF_MAX)
                        _FAISS_NEXT_REBUILD_TS = _t.time() + _FAISS_REBUILD_BACKOFF
                        faiss_rebuild_backoff_seconds.set(_FAISS_REBUILD_BACKOFF)
                        globals()["_FAISS_LAST_REBUILD_STATUS"] = "error"
                        globals()["_FAISS_LAST_ERROR"] = str(e)

    def rebuild(self) -> bool:
        global _FAISS_INDEX, _FAISS_ID_MAP, _FAISS_REVERSE_MAP, _FAISS_PENDING_DELETE, _FAISS_DIM
        if not self._available or _FAISS_INDEX is None or _FAISS_DIM is None:
            return False
        from src.utils.analysis_metrics import faiss_rebuild_total, faiss_rebuild_duration_seconds
        import time
        start = time.time()
        import numpy as np  # type: ignore
        import faiss  # type: ignore  # noqa: F401
        # Collect remaining vectors (excluding pending delete)
        remaining: List[tuple[str, List[float]]] = []
        for vid, vec in _VECTOR_STORE.items():
            if vid in _FAISS_PENDING_DELETE:
                continue
            if len(vec) == _FAISS_DIM:
                remaining.append((vid, vec))
        if not remaining:
            faiss_rebuild_total.labels(status="skipped").inc()
            return False
        try:
            new_index = self._create_index(_FAISS_DIM)
            id_map_local: Dict[int, str] = {}
            reverse_local: Dict[str, int] = {}
            for vid, vec in remaining:
                arr = np.array(vec, dtype="float32")
                if self._normalize:
                    norm = np.linalg.norm(arr)
                    if norm > 0:
                        arr = arr / norm
                new_index.add(arr.reshape(1, _FAISS_DIM))
                idx = new_index.ntotal - 1
                id_map_local[idx] = vid
                reverse_local[vid] = idx
            # Atomic swap
            _FAISS_INDEX = new_index
            _FAISS_ID_MAP = id_map_local
            _FAISS_REVERSE_MAP = reverse_local
            _FAISS_PENDING_DELETE.clear()
            faiss_rebuild_total.labels(status="success").inc()
            faiss_rebuild_duration_seconds.observe(time.time() - start)
            faiss_index_size.set(_FAISS_INDEX.ntotal)  # type: ignore
            try:
                faiss_index_age_seconds.set(0)
            except Exception:
                pass
            globals()["_FAISS_LAST_REBUILD_STATUS"] = "success"
            globals()["_FAISS_LAST_ERROR"] = None
            return True
        except Exception as e:
            faiss_rebuild_total.labels(status="error").inc()
            globals()["_FAISS_LAST_REBUILD_STATUS"] = "error"
            globals()["_FAISS_LAST_ERROR"] = str(e)
            return False

    def query(self, vector: List[float], top_k: int = 5) -> List[tuple[str, float]]:
        if not self._available or _FAISS_INDEX is None or _FAISS_DIM is None:
            return []
        import numpy as np  # type: ignore
        if len(vector) != _FAISS_DIM:
            return []
        arr = np.array(vector, dtype="float32")
        if self._normalize:
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm

        # Configure search parameters for HNSW if applicable
        if hasattr(_FAISS_INDEX, "hnsw"):
            _FAISS_INDEX.hnsw.efSearch = int(os.getenv("FAISS_HNSW_EF_SEARCH", "16"))

        D, I = _FAISS_INDEX.search(arr.reshape(1, _FAISS_DIM), top_k)  # type: ignore  # noqa: E741
        results: List[tuple[str, float]] = []
        for dist, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            vid = _FAISS_ID_MAP.get(int(idx))
            if not vid:
                continue
            results.append((vid, float(dist)))
        vector_query_backend_total.labels(backend="faiss").inc()
        return results

    def export(self, path: str) -> bool:
        if not self._available or _FAISS_INDEX is None:
            return False
        from src.utils.analysis_metrics import faiss_export_total, faiss_export_duration_seconds
        import time, os
        start = time.time()
        try:
            import faiss  # type: ignore
            tmp_path = f"{path}.tmp"
            faiss.write_index(_FAISS_INDEX, tmp_path)  # type: ignore
            os.replace(tmp_path, path)
            faiss_export_total.labels(status="success").inc()
            faiss_export_duration_seconds.observe(time.time() - start)
            global _FAISS_LAST_EXPORT_SIZE
            import time
            _FAISS_LAST_EXPORT_SIZE = _FAISS_INDEX.ntotal  # type: ignore
            time.time()
            return True
        except Exception:
            faiss_export_total.labels(status="error").inc()
            return False

    def import_index(self, path: str) -> bool:
        if not self._available:
            return False
        from src.utils.analysis_metrics import faiss_import_total, faiss_import_duration_seconds
        import time
        start = time.time()
        try:
            import faiss  # type: ignore
            if not os.path.exists(path):
                faiss_import_total.labels(status="skipped").inc()
                return False
            idx = faiss.read_index(path)
            if _FAISS_DIM is not None and idx.d != _FAISS_DIM:  # type: ignore
                faiss_import_total.labels(status="skipped").inc()
                return False
            global _FAISS_INDEX
            _FAISS_INDEX = idx
            faiss_index_size.set(_FAISS_INDEX.ntotal)  # type: ignore
            faiss_import_total.labels(status="success").inc()
            faiss_import_duration_seconds.observe(time.time() - start)
            global _FAISS_IMPORTED
            _FAISS_IMPORTED = True
            return True
        except Exception:
            faiss_import_total.labels(status="error").inc()
            return False


# ============================================================================
# Vector Store Factory and Dependency Injection
# ============================================================================

_DEFAULT_STORE: VectorStoreProtocol | None = None


def reload_vector_store_backend() -> bool:
    """Force reinitialization of cached default store.

    Clears cached store and re-invokes factory. Returns True on success.
    """
    global _DEFAULT_STORE
    _DEFAULT_STORE = None
    try:
        get_vector_store()  # rebuild with current env
        return True
    except Exception:
        return False


def get_vector_store(backend: str | None = None) -> VectorStoreProtocol:
    """Factory function to get the appropriate vector store backend.

    Args:
        backend: Backend type ("memory", "faiss", "redis"). If None, uses VECTOR_STORE_BACKEND env var.

    Returns:
        VectorStoreProtocol implementation based on backend selection.

    Examples:
        >>> store = get_vector_store()  # Uses env var
        >>> store = get_vector_store("faiss")  # Explicit backend
    """
    global _DEFAULT_STORE

    if backend is None:
        backend = os.getenv("VECTOR_STORE_BACKEND", "memory")

    # Return cached default store if backend matches
    if _DEFAULT_STORE is not None and _matches_backend(_DEFAULT_STORE, backend):
        return _DEFAULT_STORE

    # Create new store instance
    if backend == "faiss":
        try:
            store = FaissVectorStore()
            if not store._available:  # type: ignore
                # Degraded path: Faiss library present but unavailable
                global _VECTOR_DEGRADED, _VECTOR_DEGRADED_REASON, _VECTOR_DEGRADED_AT, _DEGRADATION_HISTORY
                _VECTOR_DEGRADED = True
                _VECTOR_DEGRADED_REASON = "Faiss library unavailable"
                import time
                _VECTOR_DEGRADED_AT = time.time()
                similarity_degraded_total.labels(event="degraded").inc()
                _DEGRADATION_HISTORY.append({
                    "timestamp": _VECTOR_DEGRADED_AT,
                    "reason": _VECTOR_DEGRADED_REASON,
                    "backend_requested": "faiss",
                    "backend_actual": "memory",
                })
                if len(_DEGRADATION_HISTORY) > _MAX_DEGRADATION_HISTORY:
                    _DEGRADATION_HISTORY = _DEGRADATION_HISTORY[-_MAX_DEGRADATION_HISTORY:]
                import logging
                logging.getLogger(__name__).warning(
                    "Faiss unavailable, falling back to memory store",
                    extra={
                        "degraded": True,
                        "reason": _VECTOR_DEGRADED_REASON,
                        "backend_requested": "faiss",
                        "backend_actual": "memory",
                    }
                )
                store = InMemoryVectorStore()
            else:
                # Faiss available: if we were previously degraded, record restore event
                if _VECTOR_DEGRADED:
                    import time
                    similarity_degraded_total.labels(event="restored").inc()
                    import logging
                    logging.getLogger(__name__).info(
                        "Faiss restored",
                        extra={
                            "degraded": False,
                            "duration_seconds": (
                                None if not _VECTOR_DEGRADED_AT else round(time.time() - _VECTOR_DEGRADED_AT, 2)
                            ),
                            "previous_reason": _VECTOR_DEGRADED_REASON,
                        }
                    )
                    # Reset degraded flags (keep history for audit)
                    _VECTOR_DEGRADED = False
                    _VECTOR_DEGRADED_REASON = None
                    _VECTOR_DEGRADED_AT = None
        except Exception as e:
            _VECTOR_DEGRADED = True
            _VECTOR_DEGRADED_REASON = f"Faiss initialization failed: {str(e)}"
            import time
            _VECTOR_DEGRADED_AT = time.time()
            similarity_degraded_total.labels(event="degraded").inc()
            _DEGRADATION_HISTORY.append({
                "timestamp": _VECTOR_DEGRADED_AT,
                "reason": _VECTOR_DEGRADED_REASON,
                "backend_requested": "faiss",
                "backend_actual": "memory",
                "error": str(e),
            })
            if len(_DEGRADATION_HISTORY) > _MAX_DEGRADATION_HISTORY:
                _DEGRADATION_HISTORY = _DEGRADATION_HISTORY[-_MAX_DEGRADATION_HISTORY:]
            import logging
            logging.getLogger(__name__).warning(
                "Faiss initialization failed, falling back to memory store",
                extra={
                    "degraded": True,
                    "reason": _VECTOR_DEGRADED_REASON,
                    "backend_requested": "faiss",
                    "backend_actual": "memory",
                    "error": str(e),
                }
            )
            store = InMemoryVectorStore()
    else:
        store = InMemoryVectorStore()

    # Cache as default store
    _DEFAULT_STORE = store
    return store


def _matches_backend(store: VectorStoreProtocol, backend: str) -> bool:
    """Check if a store instance matches the requested backend type."""
    if backend == "faiss":
        # FaissVectorStore may be None if faiss not installed, or may be mocked in tests
        if FaissVectorStore is None or not isinstance(FaissVectorStore, type):
            return False
        return isinstance(store, FaissVectorStore)
    else:
        return isinstance(store, InMemoryVectorStore)


def reset_default_store() -> None:
    """Reset the cached default store (useful for testing)."""
    global _DEFAULT_STORE, _VECTOR_DEGRADED, _VECTOR_DEGRADED_REASON, _VECTOR_DEGRADED_AT, _DEGRADATION_HISTORY
    _DEFAULT_STORE = None
    # Reset degraded mode flags and history on store reset
    _VECTOR_DEGRADED = False
    _VECTOR_DEGRADED_REASON = None
    _VECTOR_DEGRADED_AT = None
    _DEGRADATION_HISTORY = []


def get_degraded_mode_info() -> Dict[str, any]:
    """Get current degraded mode information including history.

    Returns:
        Dictionary with degraded status, reason, timestamp, and history
    """
    return {
        "degraded": _VECTOR_DEGRADED,
        "reason": _VECTOR_DEGRADED_REASON,
        "degraded_at": _VECTOR_DEGRADED_AT,
        "degraded_duration_seconds": (
            None if not _VECTOR_DEGRADED_AT
            else round(__import__("time").time() - _VECTOR_DEGRADED_AT, 2)
        ),
        "history": _DEGRADATION_HISTORY.copy(),  # Return copy to prevent external modification
        "history_count": len(_DEGRADATION_HISTORY),
    }


def attempt_faiss_recovery(now: float | None = None) -> bool:
    """Try to recover from degraded mode by reinitializing Faiss backend.

    Returns True if recovery succeeded (and flags were cleared), False otherwise.
    """
    from src.utils.analysis_metrics import (
        faiss_recovery_attempts_total,
        faiss_degraded_duration_seconds,
        faiss_next_recovery_eta_seconds,
    )
    n = __import__("time").time() if now is None else now
    # Only attempt when degraded and time threshold passed
    if not _VECTOR_DEGRADED:
        faiss_recovery_attempts_total.labels(result="skipped").inc()
        return False
    global _FAISS_NEXT_RECOVERY_TS
    # Suppression window active (flapping protection)
    if _FAISS_SUPPRESS_UNTIL_TS and n < _FAISS_SUPPRESS_UNTIL_TS:
        try:
            from src.utils.analysis_metrics import faiss_recovery_suppressed_total
            faiss_recovery_suppressed_total.labels(reason="flapping").inc()
            # update remaining suppression seconds gauge
            try:
                faiss_recovery_suppression_remaining_seconds.set(_FAISS_SUPPRESS_UNTIL_TS - n)
            except Exception:
                pass
        except Exception:
            pass
        faiss_recovery_attempts_total.labels(result="suppressed").inc()
        return False
    if _FAISS_NEXT_RECOVERY_TS and n < _FAISS_NEXT_RECOVERY_TS:
        faiss_recovery_attempts_total.labels(result="skipped").inc()
        return False
    with _FAISS_RECOVERY_LOCK:
        try:
            # Reset store and try to get faiss
            reset_default_store()
            store = get_vector_store("faiss")
            if FaissVectorStore is not None and isinstance(FaissVectorStore, type) and isinstance(store, FaissVectorStore) and getattr(store, "_available", False):
                # recovered
                faiss_recovery_attempts_total.labels(result="success").inc()
                faiss_degraded_duration_seconds.set(0)
                # Clear ETA gauge immediately on successful recovery
                try:
                    faiss_next_recovery_eta_seconds.set(0)
                except Exception:
                    pass
                # clear degraded flags (preserve history)
                if _VECTOR_DEGRADED:
                    try:
                        similarity_degraded_total.labels(event="restored").inc()
                    except Exception:
                        pass
                # mutate globals with lock
                with _VECTOR_LOCK:
                    globals()["_VECTOR_DEGRADED"] = False
                    globals()["_VECTOR_DEGRADED_REASON"] = None
                    globals()["_VECTOR_DEGRADED_AT"] = None
                # Clear suppression window once recovered
                globals()["_FAISS_SUPPRESS_UNTIL_TS"] = None
                globals()["_FAISS_NEXT_RECOVERY_TS"] = None
                try:
                    faiss_recovery_suppression_remaining_seconds.set(0)
                except Exception:
                    pass
                _persist_recovery_state()
                return True
            else:
                # still degraded; schedule next attempt
                faiss_recovery_attempts_total.labels(result="error").inc()
                backoff = min(max(_FAISS_RECOVERY_INTERVAL_SECONDS, 60.0) * _FAISS_RECOVERY_BACKOFF_MULTIPLIER, _FAISS_RECOVERY_MAX_BACKOFF)
                _FAISS_NEXT_RECOVERY_TS = n + backoff
                _persist_recovery_state()
                try:
                    from src.utils.analysis_metrics import faiss_rebuild_backoff_seconds
                    faiss_rebuild_backoff_seconds.set(backoff)
                    # Reflect scheduled ETA in gauge for observability
                    try:
                        faiss_next_recovery_eta_seconds.set(_FAISS_NEXT_RECOVERY_TS)
                    except Exception:
                        pass
                    # If suppression ended but we are scheduling next attempt, ensure remaining seconds gauge reflects 0
                    if not _FAISS_SUPPRESS_UNTIL_TS or n >= (_FAISS_SUPPRESS_UNTIL_TS or 0):
                        try:
                            faiss_recovery_suppression_remaining_seconds.set(0)
                        except Exception:
                            pass
                except Exception:
                    pass
                if _VECTOR_DEGRADED_AT:
                    faiss_degraded_duration_seconds.set(max(0, n - _VECTOR_DEGRADED_AT))
                return False
        except Exception:
            faiss_recovery_attempts_total.labels(result="error").inc()
            return False


async def faiss_recovery_loop() -> None:  # pragma: no cover (background loop)
    import asyncio, time
    while True:
        try:
            # If a manual recovery is underway, skip this iteration
            if _FAISS_MANUAL_RECOVERY_IN_PROGRESS:
                await asyncio.sleep(1.0)
            else:
                # Flapping detection: count degraded history entries in window
                if _VECTOR_DEGRADED and _DEGRADATION_HISTORY:
                    now = time.time()
                    recent = [h for h in _DEGRADATION_HISTORY if now - h.get("timestamp", 0) <= _FAISS_RECOVERY_FLAP_WINDOW_SECONDS]
                    if len(recent) >= _FAISS_RECOVERY_FLAP_THRESHOLD and (_FAISS_SUPPRESS_UNTIL_TS is None or now >= _FAISS_SUPPRESS_UNTIL_TS):
                        globals()["_FAISS_SUPPRESS_UNTIL_TS"] = now + _FAISS_RECOVERY_SUPPRESSION_SECONDS
                attempt_faiss_recovery()
        except Exception:
            pass
        await asyncio.sleep(max(_FAISS_RECOVERY_INTERVAL_SECONDS, 60.0))


def _detect_flapping_and_set_suppression(now: float | None = None) -> bool:  # pragma: no cover - tested via unit
    """Internal helper to detect flapping degraded events and set suppression window.

    Returns True if suppression window was set during this call, False otherwise.
    """
    import time
    n = time.time() if now is None else now
    if not _VECTOR_DEGRADED or not _DEGRADATION_HISTORY:
        return False
    recent = [h for h in _DEGRADATION_HISTORY if n - h.get("timestamp", 0) <= _FAISS_RECOVERY_FLAP_WINDOW_SECONDS]
    if len(recent) >= _FAISS_RECOVERY_FLAP_THRESHOLD and (_FAISS_SUPPRESS_UNTIL_TS is None or n >= _FAISS_SUPPRESS_UNTIL_TS):
        globals()["_FAISS_SUPPRESS_UNTIL_TS"] = n + _FAISS_RECOVERY_SUPPRESSION_SECONDS
        _persist_recovery_state()
        try:
            faiss_recovery_suppression_remaining_seconds.set(_FAISS_SUPPRESS_UNTIL_TS - n)
        except Exception:
            pass
        return True
    return False
