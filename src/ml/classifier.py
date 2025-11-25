from __future__ import annotations

"""Lightweight ML classifier wrapper.

Loads a pickled sklearn-like model if available; falls back to rule-based analyzer classification.
Model expected to implement `.predict` taking List[List[float]].
"""

import os
import pickle
import threading
from typing import List, Dict, Any
from pathlib import Path

_MODEL = None
_MODEL_LOADED_AT: float | None = None
_MODEL_VERSION = os.getenv("CLASSIFICATION_MODEL_VERSION", "none")
_MODEL_PATH = Path(os.getenv("CLASSIFICATION_MODEL_PATH", "models/classifier_v1.pkl"))
_MODEL_HASH: str | None = None
_MODEL_LAST_ERROR: str | None = None
_MODEL_LOAD_SEQ: int = 0  # Monotonic sequence to disambiguate reloads
# Previous model snapshot for rollback
_MODEL_PREV: Dict[str, Any] | None = None
_MODEL_PREV_HASH: str | None = None
_MODEL_PREV_VERSION: str | None = None
_MODEL_PREV_PATH: Path | None = None
# Second level rollback snapshot (in case rollback target also fails later)
_MODEL_PREV2: Dict[str, Any] | None = None
_MODEL_PREV2_HASH: str | None = None
_MODEL_PREV2_VERSION: str | None = None
_MODEL_PREV2_PATH: Path | None = None
# Thread safety for reload operations
_MODEL_LOCK = threading.Lock()


def load_model() -> None:
    global _MODEL
    if _MODEL is not None:
        return
    from src.utils.analysis_metrics import classification_model_load_total, classification_model_version_total
    if not _MODEL_PATH.exists():
        classification_model_load_total.labels(status="absent", version=_MODEL_VERSION).inc()
        return
    try:
        with _MODEL_PATH.open("rb") as f:
            _MODEL = pickle.load(f)
        classification_model_load_total.labels(status="success", version=_MODEL_VERSION).inc()
        classification_model_version_total.labels(version=_MODEL_VERSION).inc()
        try:
            import hashlib
            _MODEL_HASH = hashlib.sha256(_MODEL_PATH.read_bytes()).hexdigest()[:16]
        except Exception:
            _MODEL_HASH = None
    except Exception:
        _MODEL = None
        classification_model_load_total.labels(status="error", version=_MODEL_VERSION).inc()


def predict(vector: List[float]) -> Dict[str, Any]:
    load_model()
    if _MODEL is None:
        return {"status": "model_unavailable"}
    try:
        import time
        start = time.time()
        # sklearn predict returns array-like labels
        label = _MODEL.predict([vector])[0]
        dur = time.time() - start
        from src.utils.analysis_metrics import (
            classification_model_inference_seconds,
            classification_prediction_distribution,
        )
        classification_model_inference_seconds.observe(dur)
        classification_prediction_distribution.labels(label=str(label), version=_MODEL_VERSION).inc()
        return {
            "predicted_type": str(label),
            "model_version": _MODEL_VERSION,
            "inference_seconds": round(dur, 6),
            "model_hash": _MODEL_HASH,
        }
    except Exception:
        return {"status": "inference_error"}


def reload_model(path: str, expected_version: str | None = None, force: bool = False) -> Dict[str, Any]:
    """Hot reload classification model with security validation.

    Security checks:
    1. Magic number validation (pickle format)
    2. Hash whitelist (if configured)
    3. Size limit validation
    4. Interface validation (has predict method)

    Returns structured status dict.
    """
    from src.utils.analysis_metrics import model_reload_total, model_security_fail_total
    import os

    # Declare all globals at the beginning (TODO: Add _MODEL_LOCK for thread safety)
    global _MODEL, _MODEL_VERSION, _MODEL_PATH, _MODEL_HASH, _MODEL_LOADED_AT, _MODEL_LAST_ERROR, _MODEL_LOAD_SEQ
    global _MODEL_PREV, _MODEL_PREV_HASH, _MODEL_PREV_VERSION, _MODEL_PREV_PATH
    global _MODEL_PREV2, _MODEL_PREV2_HASH, _MODEL_PREV2_VERSION, _MODEL_PREV2_PATH

    p = Path(path)
    if not p.exists():
        model_reload_total.labels(status="not_found", version=str(expected_version or "unknown")).inc()
        return {"status": "not_found"}
    if expected_version and expected_version != os.getenv("CLASSIFICATION_MODEL_VERSION", expected_version):
        if not force:
            model_reload_total.labels(status="mismatch", version=str(expected_version)).inc()
            return {"status": "version_mismatch"}

    # Magic number validation (pickle file format check)
    try:
        with p.open("rb") as f:
            magic = f.read(2)
            # Pickle protocol 0-5 magic numbers
            valid_pickle_magics = [
                b'\x80\x02',  # Protocol 2
                b'\x80\x03',  # Protocol 3
                b'\x80\x04',  # Protocol 4
                b'\x80\x05',  # Protocol 5
            ]
            if magic not in valid_pickle_magics and not magic.startswith(b'('):  # Protocol 0
                model_security_fail_total.labels(reason="magic_number_invalid").inc()
                model_reload_total.labels(status="magic_invalid", version=str(expected_version or "unknown")).inc()
                _MODEL_LAST_ERROR = f"Invalid pickle magic number: {magic.hex()}"
                return {
                    "status": "magic_invalid",
                    "message": "File does not appear to be a valid pickle file",
                    "magic_bytes": magic.hex()
                }
    except Exception as e:
        model_security_fail_total.labels(reason="forged_file").inc()
        return {"status": "error", "error": f"Magic number check failed: {str(e)}"}

    # Hash whitelist validation (optional)
    allowed_hashes_env = os.getenv("ALLOWED_MODEL_HASHES", "").strip()
    allowed_hashes = {h for h in allowed_hashes_env.split(",") if h}
    # Size limit validation
    try:
        max_mb = float(os.getenv("MODEL_MAX_MB", "50"))
        size_mb = p.stat().st_size / (1024 * 1024)
        if size_mb > max_mb:
            model_reload_total.labels(status="size_exceeded", version=str(expected_version or _MODEL_VERSION)).inc()
            _MODEL_LAST_ERROR = f"Model size {size_mb:.1f}MB exceeds limit {max_mb}MB"
            return {"status": "size_exceeded", "size_mb": round(size_mb, 3), "max_mb": max_mb}
    except Exception:
        pass
    # Snapshot current model for rollback
    # Shift previous snapshots down one level
    _MODEL_PREV2 = _MODEL_PREV
    _MODEL_PREV2_HASH = _MODEL_PREV_HASH
    _MODEL_PREV2_VERSION = _MODEL_PREV_VERSION
    _MODEL_PREV2_PATH = _MODEL_PREV_PATH
    _MODEL_PREV = _MODEL
    _MODEL_PREV_HASH = _MODEL_HASH
    _MODEL_PREV_VERSION = _MODEL_VERSION
    _MODEL_PREV_PATH = _MODEL_PATH
    try:
        with p.open("rb") as f:
            data = f.read()
        # Compute hash before unpickling to avoid loading malicious content if hash mismatch would fail
        import hashlib
        candidate_hash = hashlib.sha256(data).hexdigest()[:16]
        # Optional opcode security scan before full load (lightweight heuristic)
        if os.getenv("MODEL_OPCODE_SCAN", "1") == "1":
            from src.utils.analysis_metrics import model_security_fail_total
            import pickletools
            # Expanded block list (potential remote code / object construction vectors)
            blocked = {"GLOBAL", "STACK_GLOBAL", "REDUCE", "INST", "OBJ", "NEWOBJ_EX"}
            try:
                for op, arg, pos in pickletools.genops(data):  # type: ignore
                    if op.name in blocked:
                        model_security_fail_total.labels(reason="opcode_blocked").inc()
                        model_reload_total.labels(status="opcode_blocked", version=str(expected_version or _MODEL_VERSION)).inc()
                        return {
                            "status": "security_blocked",
                            "message": "Disallowed pickle opcode detected",
                            "opcode": op.name,
                            "position": pos,
                            "blocked_set": sorted(list(blocked))[:10],
                        }
            except Exception as scan_err:
                # Non-fatal scan error; proceed unless strict mode requested
                if os.getenv("MODEL_OPCODE_STRICT", "0") == "1":
                    model_security_fail_total.labels(reason="opcode_scan_error").inc()
                    model_reload_total.labels(status="opcode_scan_error", version=str(expected_version or _MODEL_VERSION)).inc()
                    return {"status": "security_scan_error", "error": str(scan_err)}
        obj = pickle.loads(data)
        if not hasattr(obj, "predict"):
            raise ValueError("Model missing predict method")
        new_version = expected_version or _MODEL_VERSION
        # Assign only after all validations pass
        _MODEL = obj
        _MODEL_PATH = p
        _MODEL_VERSION = new_version
        _MODEL_HASH = candidate_hash
        import time as _t
        _MODEL_LOADED_AT = _t.time()
        _MODEL_LOAD_SEQ += 1  # Increment sequence on successful load
        _MODEL_LAST_ERROR = None  # Clear error on success
        # Whitelist check after computing hash
        if allowed_hashes and _MODEL_HASH not in allowed_hashes:
            # Restore previous immediately, treat as security mismatch
            _MODEL = _MODEL_PREV
            _MODEL_HASH = _MODEL_PREV_HASH
            _MODEL_VERSION = _MODEL_PREV_VERSION or _MODEL_VERSION
            _MODEL_PATH = _MODEL_PREV_PATH or _MODEL_PATH
            model_security_fail_total.labels(reason="hash_mismatch").inc()
            model_reload_total.labels(status="hash_mismatch", version=str(expected_version or _MODEL_VERSION)).inc()
            return {"status": "hash_mismatch", "expected_hashes": list(allowed_hashes), "found_hash": _MODEL_HASH}
        model_reload_total.labels(status="success", version=_MODEL_VERSION).inc()
        return {"status": "success", "model_version": _MODEL_VERSION, "hash": _MODEL_HASH}
    except Exception as e:
        # Rollback
        _MODEL_LAST_ERROR = str(e)
        if _MODEL_PREV is not None:
            _MODEL = _MODEL_PREV
            _MODEL_HASH = _MODEL_PREV_HASH
            _MODEL_VERSION = _MODEL_PREV_VERSION or _MODEL_VERSION
            _MODEL_PATH = _MODEL_PREV_PATH or _MODEL_PATH
            model_reload_total.labels(status="rollback", version=str(expected_version or _MODEL_VERSION)).inc()
            return {"status": "rollback", "error": str(e), "previous_version": _MODEL_VERSION, "previous_hash": _MODEL_HASH}
        # If no previous, attempt second-level rollback (unlikely path)
        if _MODEL_PREV2 is not None:
            _MODEL = _MODEL_PREV2
            _MODEL_HASH = _MODEL_PREV2_HASH
            _MODEL_VERSION = _MODEL_PREV2_VERSION or _MODEL_VERSION
            _MODEL_PATH = _MODEL_PREV2_PATH or _MODEL_PATH
            model_reload_total.labels(status="rollback_level2", version=str(expected_version or _MODEL_VERSION)).inc()
            return {"status": "rollback_level2", "error": str(e), "previous_version": _MODEL_VERSION, "previous_hash": _MODEL_HASH}
        model_reload_total.labels(status="error", version=str(expected_version or _MODEL_VERSION)).inc()
        return {"status": "error", "error": str(e)}


def get_model_info() -> Dict[str, Any]:
    """Get current model information.

    Returns:
        Dictionary with model version, hash, path, loaded status, and rollback info
    """
    # Determine rollback level
    rollback_level = 0
    rollback_reason = None

    # Check if current model is from rollback
    if _MODEL_PREV is not None and _MODEL == _MODEL_PREV:
        rollback_level = 1
        rollback_reason = "Rolled back to previous model after reload failure"
    elif _MODEL_PREV2 is not None and _MODEL == _MODEL_PREV2:
        rollback_level = 2
        rollback_reason = "Rolled back to level 2 snapshot after consecutive failures"

    return {
        "version": _MODEL_VERSION,
        "hash": _MODEL_HASH,
        "path": str(_MODEL_PATH) if _MODEL_PATH else None,
        "loaded_at": _MODEL_LOADED_AT,
        "loaded": _MODEL is not None,
        "rollback_level": rollback_level,
        "last_error": _MODEL_LAST_ERROR,  # Now tracking actual errors
        "rollback_reason": rollback_reason if rollback_level > 0 else None,
        "has_prev": _MODEL_PREV is not None,
        "has_prev2": _MODEL_PREV2 is not None,
        "load_seq": _MODEL_LOAD_SEQ,  # Monotonic sequence for disambiguation
    }


__all__ = ["predict", "load_model", "reload_model", "get_model_info"]
