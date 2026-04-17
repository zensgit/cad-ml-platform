from __future__ import annotations

import io
from unittest.mock import MagicMock

from src.core.legacy_admin_pipeline import (
    run_faiss_rebuild_pipeline,
    run_process_rules_audit_pipeline,
)


def test_run_process_rules_audit_pipeline_with_existing_file() -> None:
    result = run_process_rules_audit_pipeline(
        raw=True,
        load_rules_fn=lambda force_reload=True: {  # noqa: ARG005
            "__meta__": {"version": "v2"},
            "steel": {"simple": ["cut"], "complex": ["cut", "grind"]},
            "aluminum": {"basic": ["mill"]},
        },
        rules_path="/tmp/rules.yaml",
        path_exists=lambda path: True,
        file_opener=lambda path, mode: io.BytesIO(b"rules-data"),  # noqa: ARG005
    )

    assert result["version"] == "v2"
    assert result["source"] == "/tmp/rules.yaml"
    assert len(result["hash"]) == 16
    assert result["materials"] == ["aluminum", "steel"]
    assert result["complexities"]["steel"] == ["complex", "simple"]
    assert result["raw"]["steel"]["simple"] == ["cut"]


def test_run_process_rules_audit_pipeline_raw_false_and_missing_file() -> None:
    result = run_process_rules_audit_pipeline(
        raw=False,
        load_rules_fn=lambda force_reload=True: {"steel": {"simple": ["cut"]}},  # noqa: ARG005
        rules_path="/missing.yaml",
        path_exists=lambda path: False,
    )

    assert result["version"] == "v1"
    assert result["source"] == "embedded-defaults"
    assert result["hash"] is None
    assert result["raw"] == {}


def test_run_faiss_rebuild_pipeline_skips_non_faiss() -> None:
    store_factory = MagicMock()

    result = run_faiss_rebuild_pipeline(
        vector_store_backend="memory",
        store_factory=store_factory,
    )

    assert result == {"rebuilt": False, "reason": "backend_not_faiss"}
    store_factory.assert_not_called()


def test_run_faiss_rebuild_pipeline_uses_store_factory() -> None:
    store = MagicMock()
    store.rebuild.return_value = True

    result = run_faiss_rebuild_pipeline(
        vector_store_backend="faiss",
        store_factory=lambda: store,
    )

    assert result == {"rebuilt": True, "message": "Index rebuilt successfully"}
    store.rebuild.assert_called_once_with()
