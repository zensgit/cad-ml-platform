from __future__ import annotations

"""Lightweight ML classifier wrapper.

Loads a pickled sklearn-like model if available; falls back to rule-based analyzer classification.
Model expected to implement `.predict` taking List[List[float]].
"""

import os
import pickle
import threading
import logging
from typing import List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

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
# Third level rollback snapshot (deepest recovery point)
_MODEL_PREV3: Dict[str, Any] | None = None
_MODEL_PREV3_HASH: str | None = None
_MODEL_PREV3_VERSION: str | None = None
_MODEL_PREV3_PATH: Path | None = None
# Thread safety for reload operations
_MODEL_LOCK = threading.Lock()
_OPCODE_AUDIT_SET: set[str] = set()
_OPCODE_AUDIT_COUNT: dict[str, int] = {}


def load_model() -> None:
    """Thread-safe initial model load.

    Uses double-checked locking pattern: check without lock first (fast path),
    then acquire lock and check again to prevent race conditions.
    """
    global _MODEL, _MODEL_HASH, _MODEL_LOADED_AT, _MODEL_LOAD_SEQ, _MODEL_VERSION, _MODEL_PATH

    # Refresh model settings from environment; if changed, reset loaded model.
    new_version = os.getenv("CLASSIFICATION_MODEL_VERSION", "none")
    new_path = Path(os.getenv("CLASSIFICATION_MODEL_PATH", "models/classifier_v1.pkl"))
    if _MODEL is not None and new_version == _MODEL_VERSION and new_path == _MODEL_PATH:
        return

    # Slow path: acquire lock for settings refresh and initial load
    with _MODEL_LOCK:
        if new_version != _MODEL_VERSION or new_path != _MODEL_PATH:
            _MODEL_VERSION = new_version
            _MODEL_PATH = new_path
            _MODEL = None
            _MODEL_HASH = None
            _MODEL_LOADED_AT = None
            _MODEL_LOAD_SEQ = 0
        # Double-check: another thread may have loaded while we waited for lock
        if _MODEL is not None:
            return

        from src.utils.analysis_metrics import classification_model_load_total, classification_model_version_total
        if not _MODEL_PATH.exists():
            classification_model_load_total.labels(status="absent", version=_MODEL_VERSION).inc()
            return
        try:
            with _MODEL_PATH.open("rb") as f:
                _MODEL = pickle.load(f)

            import time as _t
            _MODEL_LOADED_AT = _t.time()
            _MODEL_LOAD_SEQ = 0  # Initial load is sequence 0

            classification_model_load_total.labels(status="success", version=_MODEL_VERSION).inc()
            classification_model_version_total.labels(version=_MODEL_VERSION).inc()

            try:
                import hashlib
                _MODEL_HASH = hashlib.sha256(_MODEL_PATH.read_bytes()).hexdigest()[:16]
            except Exception:
                _MODEL_HASH = None

            logger.info(
                "Model initial load successful",
                extra={
                    "status": "initial_load",
                    "version": _MODEL_VERSION,
                    "hash": _MODEL_HASH,
                    "load_seq": _MODEL_LOAD_SEQ,
                }
            )
        except Exception as e:
            _MODEL = None
            classification_model_load_total.labels(status="error", version=_MODEL_VERSION).inc()
            logger.error(
                "Model initial load failed",
                extra={
                    "status": "initial_load_error",
                    "error": str(e),
                }
            )


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


def get_opcode_audit_snapshot() -> Dict[str, Any]:
    """Return snapshot of audited opcodes (audit or whitelist/blacklist scans)."""
    return {
        "opcodes": sorted(list(_OPCODE_AUDIT_SET)),
        "counts": dict(sorted(_OPCODE_AUDIT_COUNT.items(), key=lambda kv: kv[1], reverse=True)),
        "total_samples": sum(_OPCODE_AUDIT_COUNT.values()),
    }


def reload_model(path: str, expected_version: str | None = None, force: bool = False) -> Dict[str, Any]:
    """Hot reload classification model with security validation.

    Security checks:
    1. Magic number validation (pickle format)
    2. Hash whitelist (if configured)
    3. Size limit validation
    4. Interface validation (has predict method)

    Thread-safe: Uses _MODEL_LOCK to prevent concurrent modifications.

    Returns structured status dict.
    """
    from src.utils.analysis_metrics import model_reload_total, model_security_fail_total
    import os

    # Thread safety: use context manager for automatic lock release
    with _MODEL_LOCK:
        # Declare all globals at the beginning
        global _MODEL, _MODEL_VERSION, _MODEL_PATH, _MODEL_HASH, _MODEL_LOADED_AT, _MODEL_LAST_ERROR, _MODEL_LOAD_SEQ
        global _MODEL_PREV, _MODEL_PREV_HASH, _MODEL_PREV_VERSION, _MODEL_PREV_PATH
        global _MODEL_PREV2, _MODEL_PREV2_HASH, _MODEL_PREV2_VERSION, _MODEL_PREV2_PATH
        global _MODEL_PREV3, _MODEL_PREV3_HASH, _MODEL_PREV3_VERSION, _MODEL_PREV3_PATH

        return _reload_model_impl(path, expected_version, force, model_reload_total, model_security_fail_total, os)


def _reload_model_impl(path: str, expected_version: str | None, force: bool,
                       model_reload_total, model_security_fail_total, os) -> Dict[str, Any]:
    """Internal implementation of reload_model (assumes lock is held)."""
    global _MODEL, _MODEL_VERSION, _MODEL_PATH, _MODEL_HASH, _MODEL_LOADED_AT, _MODEL_LAST_ERROR, _MODEL_LOAD_SEQ
    global _MODEL_PREV, _MODEL_PREV_HASH, _MODEL_PREV_VERSION, _MODEL_PREV_PATH
    global _MODEL_PREV2, _MODEL_PREV2_HASH, _MODEL_PREV2_VERSION, _MODEL_PREV2_PATH
    global _MODEL_PREV3, _MODEL_PREV3_HASH, _MODEL_PREV3_VERSION, _MODEL_PREV3_PATH

    # Log reload start with context
    logger.info(
        "Model reload started",
        extra={
            "path": path,
            "expected_version": expected_version,
            "current_version": _MODEL_VERSION,
            "current_load_seq": _MODEL_LOAD_SEQ,
            "force": force,
        }
    )

    from src.core.errors_extended import ErrorCode, create_extended_error
    p = Path(path)
    if not p.exists():
        model_reload_total.labels(status="not_found", version=str(expected_version or "unknown")).inc()
        err = create_extended_error(
            ErrorCode.MODEL_NOT_FOUND,
            "Model file not found",
            stage="model_reload",
            context={"path": str(p)}
        )
        return {"status": "not_found", "error": err.to_dict()}
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
                b'\x80\x01',  # Protocol 1
                b'\x80\x02',  # Protocol 2
                b'\x80\x03',  # Protocol 3
                b'\x80\x04',  # Protocol 4
                b'\x80\x05',  # Protocol 5
            ]
            if magic not in valid_pickle_magics and not magic.startswith(b'('):  # Protocol 0
                model_security_fail_total.labels(reason="magic_number_invalid").inc()
                model_reload_total.labels(status="magic_invalid", version=str(expected_version or "unknown")).inc()
                _MODEL_LAST_ERROR = f"Invalid pickle magic number: {magic.hex()}"

                logger.warning(
                    "Model reload rejected - invalid pickle magic number",
                    extra={
                        "status": "magic_invalid",
                        "magic_bytes": magic.hex(),
                        "error": f"Invalid pickle magic number: {magic.hex()}",
                    }
                )

                err = create_extended_error(
                    ErrorCode.INPUT_FORMAT_INVALID,
                    "File does not appear to be a valid pickle file",
                    stage="model_reload",
                    context={"magic_bytes": magic.hex(), "path": str(p)}
                )
                return {"status": "magic_invalid", "error": err.to_dict()}
    except Exception as e:
        model_security_fail_total.labels(reason="forged_file").inc()
        err = create_extended_error(
            ErrorCode.MODEL_LOAD_ERROR,
            f"Magic number check failed: {str(e)}",
            stage="model_reload",
            context={"path": str(p)}
        )
        return {"status": "error", "error": err.to_dict()}

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
            err = create_extended_error(
                ErrorCode.MODEL_SIZE_EXCEEDED,
                "Model file size exceeded limit",
                stage="model_reload",
                context={"size_mb": round(size_mb, 3), "max_mb": max_mb, "path": str(p)}
            )
            return {"status": "size_exceeded", "error": err.to_dict()}
    except Exception:
        pass
    # Snapshot current model for rollback
    # Shift previous snapshots down one level (PREV2 -> PREV3, PREV -> PREV2, current -> PREV)
    _MODEL_PREV3 = _MODEL_PREV2
    _MODEL_PREV3_HASH = _MODEL_PREV2_HASH
    _MODEL_PREV3_VERSION = _MODEL_PREV2_VERSION
    _MODEL_PREV3_PATH = _MODEL_PREV2_PATH
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
        opcode_mode = os.getenv("MODEL_OPCODE_MODE", "blacklist")  # blacklist|audit|whitelist
        scan_enabled = os.getenv("MODEL_OPCODE_SCAN", "1") == "1"
        audit_set = set()
        audit_count = {}
        if scan_enabled and opcode_mode in {"blacklist", "audit", "whitelist"}:
            from src.utils.analysis_metrics import model_security_fail_total
            import pickletools
            # Expanded block list (potential remote code / object construction vectors)
            blocked = {"GLOBAL", "STACK_GLOBAL", "REDUCE", "INST", "OBJ", "NEWOBJ_EX"}
            try:
                from src.utils.analysis_metrics import model_opcode_audit_total, model_opcode_whitelist_violations_total
                for op, arg, pos in pickletools.genops(data):  # type: ignore
                    # Audit collection (all modes when scan enabled)
                    audit_set.add(op.name)
                    audit_count[op.name] = audit_count.get(op.name, 0) + 1
                    model_opcode_audit_total.labels(opcode=op.name).inc()
                    # Whitelist mode: allow only safe minimal set
                    if opcode_mode == "whitelist":
                        allowed = {"NONE", "INT", "BININT", "BININT1", "BININT2", "LONG", "BINUNICODE", "SHORT_BINUNICODE", "UNICODE", "EMPTY_LIST", "APPEND", "LIST", "EMPTY_TUPLE", "TUPLE", "EMPTY_DICT", "DICT", "NEWTRUE", "NEWFALSE", "BINPUT", "BINPERSID"}
                        if op.name not in allowed:
                            error_msg = f"Disallowed opcode (whitelist) {op.name} at position {pos}"
                            _MODEL_LAST_ERROR = error_msg
                            model_opcode_whitelist_violations_total.labels(opcode=op.name).inc()
                            model_security_fail_total.labels(reason="opcode_whitelist_violation").inc()
                            model_reload_total.labels(status="opcode_whitelist_violation", version=str(expected_version or _MODEL_VERSION)).inc()
                            err = create_extended_error(
                                ErrorCode.INPUT_FORMAT_INVALID,
                                "Disallowed pickle opcode (whitelist mode)",
                                stage="model_reload",
                                context={"opcode": op.name, "position": pos, "path": str(p), "mode": opcode_mode}
                            )
                            return {"status": "opcode_blocked", "error": err.to_dict()}
                    # Blacklist blocking logic
                    if opcode_mode == "blacklist" and op.name in blocked:
                        # Record security error
                        error_msg = f"Disallowed pickle opcode {op.name} at position {pos}"
                        _MODEL_LAST_ERROR = error_msg

                        logger.warning(
                            "Model reload blocked by security scan",
                            extra={
                                "status": "security_blocked",
                                "opcode": op.name,
                                "position": pos,
                                "error": error_msg,
                            }
                        )

                        model_security_fail_total.labels(reason="opcode_blocked").inc()
                        model_reload_total.labels(status="opcode_blocked", version=str(expected_version or _MODEL_VERSION)).inc()
                        err = create_extended_error(
                            ErrorCode.INPUT_FORMAT_INVALID,
                            "Disallowed pickle opcode detected",
                            stage="model_reload",
                            context={"opcode": op.name, "position": pos, "path": str(p)}
                        )
                        return {"status": "opcode_blocked", "error": err.to_dict()}
                # Persist audit info globally for query endpoint (simple assignment; lock still held)
                global _OPCODE_AUDIT_SET, _OPCODE_AUDIT_COUNT
                try:
                    _OPCODE_AUDIT_SET.update(audit_set)
                    for k, v in audit_count.items():
                        _OPCODE_AUDIT_COUNT[k] = _OPCODE_AUDIT_COUNT.get(k, 0) + v
                except NameError:
                    # Initialize if first time
                    _OPCODE_AUDIT_SET = audit_set
                    _OPCODE_AUDIT_COUNT = audit_count
                # In audit mode we never block; in whitelist/blacklist we may have returned earlier.
            except Exception as scan_err:
                # Non-fatal scan error; proceed unless strict mode requested
                if os.getenv("MODEL_OPCODE_STRICT", "0") == "1":
                    model_security_fail_total.labels(reason="opcode_scan_error").inc()
                    model_reload_total.labels(status="opcode_scan_error", version=str(expected_version or _MODEL_VERSION)).inc()
                    err = create_extended_error(
                        ErrorCode.MODEL_LOAD_ERROR,
                        "Opcode scan error",
                        stage="model_reload",
                        context={"error": str(scan_err), "path": str(p)}
                    )
                    return {"status": "opcode_scan_error", "error": err.to_dict()}
        obj = pickle.loads(data)
        if not hasattr(obj, "predict"):
            raise ValueError("Model missing predict method")
        new_version = expected_version or _MODEL_VERSION

        # Whitelist check BEFORE committing the model (check candidate_hash before assignment)
        if allowed_hashes and candidate_hash not in allowed_hashes:
            error_msg = f"Hash {candidate_hash} not in whitelist"
            _MODEL_LAST_ERROR = error_msg
            logger.warning(
                "Model reload hash mismatch - rejected before commit",
                extra={
                    "status": "hash_mismatch",
                    "error": error_msg,
                    "found_hash": candidate_hash,
                    "expected_hashes_count": len(allowed_hashes),
                    "current_version": _MODEL_VERSION,
                    "load_seq": _MODEL_LOAD_SEQ,
                }
            )
            model_security_fail_total.labels(reason="hash_mismatch").inc()
            model_reload_total.labels(status="hash_mismatch", version=str(expected_version or _MODEL_VERSION)).inc()
            err = create_extended_error(
                ErrorCode.INPUT_VALIDATION_FAILED,
                "Hash whitelist validation failed",
                stage="model_reload",
                context={
                    "expected_hashes": list(allowed_hashes),
                    "found_hash": candidate_hash,
                    "path": str(p),
                }
            )
            return {"status": "hash_mismatch", "error": err.to_dict()}

        # All validations passed - commit the model
        _MODEL = obj
        _MODEL_PATH = p
        _MODEL_VERSION = new_version
        _MODEL_HASH = candidate_hash
        import time as _t
        _MODEL_LOADED_AT = _t.time()
        _MODEL_LAST_ERROR = None  # Clear error on success
        _MODEL_LOAD_SEQ += 1  # Increment ONLY after all validations pass

        # Log successful reload (after increment)
        logger.info(
            "Model reload successful",
            extra={
                "status": "success",
                "version": _MODEL_VERSION,
                "hash": _MODEL_HASH,
                "load_seq": _MODEL_LOAD_SEQ,
                "path": str(_MODEL_PATH),
            }
        )
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

            # Log level 1 rollback
            logger.warning(
                "Model reload failed - rolled back to previous version",
                extra={
                    "status": "rollback",
                    "rollback_level": 1,
                    "rollback_reason": "Reload failure triggered automatic rollback",
                    "error": str(e),
                    "recovered_version": _MODEL_VERSION,
                    "recovered_hash": _MODEL_HASH,
                    "load_seq": _MODEL_LOAD_SEQ,  # Preserved from previous load
                }
            )

            model_reload_total.labels(status="rollback", version=str(expected_version or _MODEL_VERSION)).inc()
            err = create_extended_error(
                ErrorCode.MODEL_ROLLBACK,
                "Model loading failed, rolled back to previous version",
                stage="model_reload",
                context={
                    "rollback_version": _MODEL_VERSION,
                    "rollback_hash": _MODEL_HASH,
                    "error": str(e),
                }
            )
            return {"status": "rollback", "error": err.to_dict(), "rollback_version": _MODEL_VERSION, "rollback_hash": _MODEL_HASH}
        # If no previous, attempt second-level rollback (unlikely path)
        if _MODEL_PREV2 is not None:
            _MODEL = _MODEL_PREV2
            _MODEL_HASH = _MODEL_PREV2_HASH
            _MODEL_VERSION = _MODEL_PREV2_VERSION or _MODEL_VERSION
            _MODEL_PATH = _MODEL_PREV2_PATH or _MODEL_PATH

            # Log level 2 rollback
            logger.error(
                "Model reload failed - rolled back to level 2 snapshot",
                extra={
                    "status": "rollback",
                    "rollback_level": 2,
                    "rollback_reason": "Consecutive failures - rolled back to level 2",
                    "error": str(e),
                    "recovered_version": _MODEL_VERSION,
                    "recovered_hash": _MODEL_HASH,
                    "load_seq": _MODEL_LOAD_SEQ,
                }
            )

            model_reload_total.labels(status="rollback_level2", version=str(expected_version or _MODEL_VERSION)).inc()
            err = create_extended_error(
                ErrorCode.MODEL_ROLLBACK,
                "Rolled back to level 2 snapshot after consecutive failures",
                stage="model_reload",
                context={
                    "rollback_version": _MODEL_VERSION,
                    "rollback_hash": _MODEL_HASH,
                    "error": str(e),
                }
            )
            return {"status": "rollback_level2", "error": err.to_dict(), "rollback_version": _MODEL_VERSION, "rollback_hash": _MODEL_HASH}

        # Attempt third-level rollback (deepest recovery)
        if _MODEL_PREV3 is not None:
            _MODEL = _MODEL_PREV3
            _MODEL_HASH = _MODEL_PREV3_HASH
            _MODEL_VERSION = _MODEL_PREV3_VERSION or _MODEL_VERSION
            _MODEL_PATH = _MODEL_PREV3_PATH or _MODEL_PATH

            # Log level 3 rollback
            logger.error(
                "Model reload failed - rolled back to level 3 snapshot (deepest recovery)",
                extra={
                    "status": "rollback",
                    "rollback_level": 3,
                    "rollback_reason": "Multiple consecutive failures - rolled back to level 3",
                    "error": str(e),
                    "recovered_version": _MODEL_VERSION,
                    "recovered_hash": _MODEL_HASH,
                    "load_seq": _MODEL_LOAD_SEQ,
                }
            )

            model_reload_total.labels(status="rollback_level3", version=str(expected_version or _MODEL_VERSION)).inc()
            err = create_extended_error(
                ErrorCode.MODEL_ROLLBACK,
                "Rolled back to level 3 snapshot (deepest recovery) after multiple failures",
                stage="model_reload",
                context={
                    "rollback_version": _MODEL_VERSION,
                    "rollback_hash": _MODEL_HASH,
                    "error": str(e),
                }
            )
            return {"status": "rollback_level3", "error": err.to_dict(), "rollback_version": _MODEL_VERSION, "rollback_hash": _MODEL_HASH}

        # No rollback target available
        logger.error(
            "Model reload failed - no rollback target available",
            extra={
                "status": "error",
                "error": str(e),
                "load_seq": _MODEL_LOAD_SEQ,
            }
        )

        model_reload_total.labels(status="error", version=str(expected_version or _MODEL_VERSION)).inc()
        err = create_extended_error(
            ErrorCode.MODEL_LOAD_ERROR,
            "Model reload error",
            stage="model_reload",
            context={"error": str(e), "path": str(p)}
        )
        return {"status": "error", "error": err.to_dict()}


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
    elif _MODEL_PREV3 is not None and _MODEL == _MODEL_PREV3:
        rollback_level = 3
        rollback_reason = "Rolled back to level 3 snapshot after multiple consecutive failures"

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
        "has_prev3": _MODEL_PREV3 is not None,
        "load_seq": _MODEL_LOAD_SEQ,  # Monotonic sequence for disambiguation
    }


__all__ = ["predict", "load_model", "reload_model", "get_model_info"]
