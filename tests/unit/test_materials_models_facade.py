"""Facade-compatibility guard for the materials router slimming.

The materials response/request models were moved from `src.api.v1.materials`
into `src.api.v1.materials_models` (behavior-preserving slice). The router module
must continue to re-export every model so existing references and any external
`from src.api.v1.materials import <Model>` imports keep working.
"""

from __future__ import annotations

from pydantic import BaseModel

import src.api.v1.materials as router_mod
import src.api.v1.materials_models as models_mod


def _model_names() -> list[str]:
    return [
        name
        for name, obj in vars(models_mod).items()
        if isinstance(obj, type) and issubclass(obj, BaseModel) and obj is not BaseModel
    ]


def test_models_module_holds_the_response_models() -> None:
    names = _model_names()
    # 34 models were extracted; guard against accidental loss.
    assert len(names) >= 30, names


def test_router_module_reexports_every_model() -> None:
    for name in _model_names():
        assert hasattr(router_mod, name), f"{name} not re-exported by materials.py"
        assert getattr(router_mod, name) is getattr(models_mod, name)


def test_router_only_exports_router() -> None:
    # The public surface stays `router`; models are an implementation re-export.
    assert getattr(router_mod, "__all__", None) == ["router"]
