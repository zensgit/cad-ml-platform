"""Phase-A C4 activation gateway — the single bootstrap + degraded contract.

Every model family activates through this gateway instead of touching the
:class:`~src.core.model_activation.store.ControlledStore` directly. It owns a
process-wide, lazily-initialized controlled store and exposes the C4 *degraded*
contract:

* :func:`activate_file` / :func:`activate_bundle` return the verified artifact on
  success, or ``None`` on ANY refusal. ``None`` is the universal
  "degrade this family" signal — the caller MUST degrade, NEVER raw-load.

Fail-closed / fail-loud posture
-------------------------------
* **UNCONFIGURED store** (``MODEL_ACTIVATION_STORE_ROOT`` unset) → the store is
  never built and ``activate_*`` returns ``None`` (degraded). This is the
  default production posture; it does NOT raise, and the manifest is not even
  parsed (a store-less deployment can activate nothing regardless).
* **CONFIGURED store** (store root set) → the baseline manifest is loaded. A
  present-but-malformed manifest raises :class:`ValueError` at bootstrap and the
  error PROPAGATES through ``activate_*`` (fail LOUD). A configured store with no
  manifest has ZERO pins, so every activation degrades via ``PIN_ABSENT`` — the
  ratified NO-PIN posture. Bootstrap runs OUTSIDE the per-activation
  refusal-to-``None`` handler precisely so this loud failure is never swallowed.

Path safety
-----------
Degraded activations log a path-safe reason (the :class:`RefusalReason` value,
or the ``store_unconfigured`` sentinel) together with the logical /artifact ids.
A ``store_relpath`` or any filesystem path is NEVER logged.

Env vars
--------
* ``MODEL_ACTIVATION_STORE_ROOT``      — server-owned store root (unset ⇒ unconfigured)
* ``MODEL_ACTIVATION_BASELINE_MANIFEST`` — baseline pin manifest (unset ⇒ NO-PIN)
* ``MODEL_ACTIVATION_FREEZE_PARENT``   — trusted freeze parent for BUNDLE activations (optional)
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional

from src.core.model_activation.baseline_manifest import load_baseline_pins
from src.core.model_activation.store import ControlledStore
from src.core.model_activation.types import (
    ActivationRefusal,
    FrozenBundle,
)

logger = logging.getLogger(__name__)

ENV_STORE_ROOT = "MODEL_ACTIVATION_STORE_ROOT"
ENV_FREEZE_PARENT = "MODEL_ACTIVATION_FREEZE_PARENT"

# Path-safe sentinel logged when the store is unconfigured (no store root). No
# RefusalReason value fits "there is no store at all"; this is not an error, it
# is the default degraded posture.
_UNCONFIGURED_REASON = "store_unconfigured"


class _Gateway:
    """Immutable holder for the resolved store (or ``None`` when unconfigured)."""

    __slots__ = ("store",)

    def __init__(self, store: Optional[ControlledStore]) -> None:
        self.store = store


_gateway_lock = threading.Lock()
_gateway: Optional[_Gateway] = None


def _build_gateway() -> _Gateway:
    """Resolve env → controlled store (or unconfigured). May raise ValueError.

    A malformed baseline manifest under a CONFIGURED store raises here (fail
    LOUD). An unset store root returns an unconfigured gateway without parsing
    the manifest (degraded, never raises).
    """
    store_root = os.environ.get(ENV_STORE_ROOT)
    if not store_root:
        return _Gateway(store=None)

    # Configured: load the baseline pins. A malformed manifest fails LOUD here.
    pins = load_baseline_pins()
    freeze_parent = os.environ.get(ENV_FREEZE_PARENT) or None
    store = ControlledStore(store_root, pins, freeze_parent=freeze_parent)
    return _Gateway(store=store)


def _get_gateway() -> _Gateway:
    """Return the process-wide gateway, building it on first use.

    Bootstrap failures (e.g. malformed manifest ValueError) propagate to the
    caller and are re-attempted on the next call — they are never cached as a
    silently-degraded gateway.
    """
    global _gateway
    with _gateway_lock:
        if _gateway is None:
            _gateway = _build_gateway()
        return _gateway


def reset_gateway_for_tests() -> None:
    """Drop the cached gateway so a later call rebuilds from current env.

    Closes any prior store to release held descriptors. Test-only seam so a
    fixture can inject a manifest via env, reset, then activate.
    """
    global _gateway
    with _gateway_lock:
        old = _gateway
        _gateway = None
    if old is not None and old.store is not None:
        try:
            old.store.close()
        except Exception:  # pragma: no cover - best-effort fd release
            pass


def _log_degraded(reason: str, logical_activation_id: str, artifact_id: str) -> None:
    """Log a path-safe degraded-activation reason (never a filesystem path)."""
    logger.warning(
        "model activation degraded: reason=%s logical_activation_id=%s artifact_id=%s",
        reason,
        logical_activation_id,
        artifact_id,
    )


def activate_file(logical_activation_id: str, artifact_id: str) -> Optional[bytes]:
    """Activate a SINGLE_FILE artifact under the C4 degraded contract.

    Returns the verified bytes on success, or ``None`` on ANY refusal (pin
    absent, artifact missing, digest mismatch, store unconfigured, …). ``None``
    means "degrade this family" — the caller must degrade, never raw-load.

    A malformed-manifest bootstrap ``ValueError`` propagates (fail LOUD); it is
    NOT converted to ``None``.
    """
    gateway = _get_gateway()  # bootstrap ValueError intentionally propagates
    store = gateway.store
    if store is None:
        _log_degraded(_UNCONFIGURED_REASON, logical_activation_id, artifact_id)
        return None
    try:
        return store.load_pinned_file(logical_activation_id, artifact_id)
    except ActivationRefusal as refusal:
        _log_degraded(refusal.reason.value, logical_activation_id, artifact_id)
        return None


def activate_bundle(
    logical_activation_id: str, artifact_id: str
) -> Optional[FrozenBundle]:
    """Activate a BUNDLE artifact under the C4 degraded contract.

    Returns the verified :class:`FrozenBundle` on success, or ``None`` on ANY
    refusal (including an unconfigured store or a missing freeze parent).
    Malformed-manifest bootstrap ``ValueError`` propagates (fail LOUD).
    """
    gateway = _get_gateway()  # bootstrap ValueError intentionally propagates
    store = gateway.store
    if store is None:
        _log_degraded(_UNCONFIGURED_REASON, logical_activation_id, artifact_id)
        return None
    try:
        return store.load_pinned_bundle(logical_activation_id, artifact_id)
    except ActivationRefusal as refusal:
        _log_degraded(refusal.reason.value, logical_activation_id, artifact_id)
        return None


__all__ = [
    "ENV_FREEZE_PARENT",
    "ENV_STORE_ROOT",
    "activate_bundle",
    "activate_file",
    "reset_gateway_for_tests",
]
