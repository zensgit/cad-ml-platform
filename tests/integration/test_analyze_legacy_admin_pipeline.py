from __future__ import annotations

from fastapi.testclient import TestClient

from src.main import app


def test_analyze_faiss_rebuild_route_delegates_to_shared_pipeline(monkeypatch) -> None:
    client = TestClient(app)
    captured = {}

    def _stub_run_faiss_rebuild_pipeline(**kwargs):  # noqa: ANN003, ANN201
        captured.update(kwargs)
        return {"rebuilt": True, "message": "Index rebuilt successfully"}

    monkeypatch.setattr(
        "src.api.v1.analyze_faiss_admin_router.run_faiss_rebuild_pipeline",
        _stub_run_faiss_rebuild_pipeline,
    )

    response = client.post(
        "/api/v1/analyze/vectors/faiss/rebuild",
        headers={"X-API-Key": "test-key"},
    )

    assert response.status_code == 200
    assert response.json()["rebuilt"] is True
    assert captured["vector_store_backend"] in {"memory", "faiss"}
    assert callable(captured["store_factory"])
