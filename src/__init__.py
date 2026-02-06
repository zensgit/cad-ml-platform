"""Source package marker for CAD ML Platform."""

from __future__ import annotations

from importlib import import_module


def __getattr__(name: str):
    """Lazily import subpackages to support mock patching in tests."""
    try:
        module = import_module(f"{__name__}.{name}")
    except ModuleNotFoundError as exc:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'") from exc
    globals()[name] = module
    return module
