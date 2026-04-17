from __future__ import annotations

from src.api.v1.analyze_aux_models import (
    FaissHealthResponse,
    ModelReloadResponse,
    ProcessRulesAuditResponse,
    VectorMigrationStatusResponse,
    VectorUpdateRequest,
)


def test_vector_update_request_fields_round_trip():
    payload = VectorUpdateRequest(
        id="vec-1",
        replace=[0.1, 0.2],
        material="steel",
        complexity="simple",
        format="dxf",
    )

    assert payload.id == "vec-1"
    assert payload.replace == [0.1, 0.2]
    assert payload.append is None
    assert payload.material == "steel"


def test_vector_migration_status_response_defaults():
    response = VectorMigrationStatusResponse()

    assert response.last_migration_id is None
    assert response.pending_vectors is None
    assert response.history is None


def test_process_rules_audit_response_schema():
    response = ProcessRulesAuditResponse(
        version="v1",
        source="embedded-defaults",
        hash=None,
        materials=["steel"],
        complexities={"simple": ["cut"]},
        raw={},
    )

    assert response.version == "v1"
    assert response.source == "embedded-defaults"
    assert response.materials == ["steel"]


def test_model_reload_response_preserves_config():
    response = ModelReloadResponse(status="ok")

    assert response.status == "ok"
    assert ModelReloadResponse.model_config.get("protected_namespaces") == ()


def test_faiss_health_response_schema():
    response = FaissHealthResponse(
        available=True,
        index_size=10,
        dim=256,
        age_seconds=5,
        pending_delete=0,
        max_pending_delete=100,
        normalize=True,
        status="ok",
    )

    assert response.available is True
    assert response.status == "ok"
