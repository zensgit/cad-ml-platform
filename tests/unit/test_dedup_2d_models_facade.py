"""Facade-compat guard for the dedup 2D model extraction.

The Dedup2D* request/response models moved from src.api.v1.dedup into
src.api.v1.dedup_2d_models. The router must keep re-exporting every model so the
route decorators (response_model=...) and any external imports keep working.
"""

from __future__ import annotations

from pydantic import BaseModel

import src.api.v1.dedup as dedup_router
import src.api.v1.dedup_2d_models as models_mod


def _model_names() -> list[str]:
    return [
        name
        for name, obj in vars(models_mod).items()
        if isinstance(obj, type)
        and issubclass(obj, BaseModel)
        and obj is not BaseModel
        and name.startswith("Dedup2D")
    ]


def test_models_module_holds_the_models() -> None:
    assert len(_model_names()) >= 15, _model_names()


def test_router_reexports_every_model() -> None:
    for name in _model_names():
        assert hasattr(dedup_router, name), f"{name} not re-exported by dedup.py"
        assert getattr(dedup_router, name) is getattr(models_mod, name)
