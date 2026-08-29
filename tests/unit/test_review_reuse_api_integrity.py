"""Fail-first HTTP integrity contract for ReviewReuse ER1/ER2."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("API_KEY", "test")


def _assert_domain_error(response: Any, status: int, code: str) -> None:
    assert response.status_code == status, response.text
    detail = response.json().get("detail")
    assert isinstance(detail, dict), response.text
    assert detail.get("code") == code, response.text
    assert isinstance(detail.get("message"), str) and detail["message"]


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("API_KEY", "test")
    monkeypatch.delenv("REVIEW_REUSE_DECISIONS_ENABLED", raising=False)
    monkeypatch.delenv("REVIEW_REUSE_REQUIRE_VALIDATED_REVIEWER", raising=False)

    from src.core.review_reuse import service as service_module
    from src.core.review_reuse.store import InMemoryReviewReuseStore

    service_module.reset_review_reuse_store_for_tests(InMemoryReviewReuseStore())
    from src.main import app

    with TestClient(app, headers={"X-API-Key": "test"}) as test_client:
        yield test_client
    service_module.reset_review_reuse_store_for_tests(InMemoryReviewReuseStore())


def _create(
    client: TestClient, *, key: str | None = None, payload: bytes = b"x"
) -> Any:
    data = {"idempotency_key": key} if key is not None else None
    return client.post(
        "/api/v1/review-reuse/tasks",
        files={"file": ("part.dxf", payload, "application/octet-stream")},
        data=data,
    )


def test_canonical_error_status_and_envelope(client: TestClient) -> None:
    empty = client.post(
        "/api/v1/review-reuse/tasks",
        files={"file": ("empty.dxf", b"", "application/octet-stream")},
    )
    _assert_domain_error(empty, 422, "empty_input")

    unsupported = client.post(
        "/api/v1/review-reuse/tasks",
        files={"file": ("part.txt", b"x", "text/plain")},
    )
    _assert_domain_error(unsupported, 415, "unsupported_file_type")

    missing = client.get("/api/v1/review-reuse/tasks/not-a-canonical-uuid")
    _assert_domain_error(missing, 404, "not_found")

    created = _create(client, key="same-key", payload=b"same")
    assert created.status_code == 200, created.text
    conflict = _create(client, key="same-key", payload=b"different")
    _assert_domain_error(conflict, 409, "idempotency_key_conflict")

    task = created.json()
    disabled = client.post(
        f"/api/v1/review-reuse/tasks/{task['task_id']}/decision",
        json={
            "state": "new",
            "candidate_id": None,
            "reason_codes": ["new_part_required"],
            "reason_text": "A new part is required.",
            "idempotency_key": "decision-disabled",
            "expected_revision": task["revision"],
            "evidence_pack_sha256": task["evidence_pack"]["evidence_pack_sha256"],
        },
    )
    _assert_domain_error(disabled, 403, "decisions_disabled")


def test_decision_revision_fields_are_required(client: TestClient) -> None:
    created = _create(client)
    assert created.status_code == 200, created.text
    task_id = created.json()["task_id"]

    missing = client.post(
        f"/api/v1/review-reuse/tasks/{task_id}/decision",
        json={
            "state": "new",
            "reason_codes": ["new_part_required"],
            "reason_text": "Reviewed.",
        },
    )
    _assert_domain_error(missing, 422, "invalid_request")

    malformed = client.post(
        f"/api/v1/review-reuse/tasks/{task_id}/decision",
        json={
            "state": "new",
            "reason_codes": ["new_part_required"],
            "reason_text": "Reviewed.",
            "expected_revision": 0,
            "evidence_pack_sha256": "ABC",
        },
    )
    _assert_domain_error(malformed, 422, "invalid_request")

    for coerced_revision in (True, 1.0, "1"):
        coerced = client.post(
            f"/api/v1/review-reuse/tasks/{task_id}/decision",
            json={
                "state": "new",
                "reason_codes": ["new_part_required"],
                "reason_text": "Reviewed.",
                "expected_revision": coerced_revision,
                "evidence_pack_sha256": "a" * 64,
            },
        )
        _assert_domain_error(coerced, 422, "invalid_request")

    forged_identity = client.post(
        f"/api/v1/review-reuse/tasks/{task_id}/decision",
        json={
            "state": "new",
            "reason_codes": ["new_part_required"],
            "reason_text": "Reviewed.",
            "expected_revision": 1,
            "evidence_pack_sha256": "a" * 64,
            "reviewer_id": "principal-v1-" + ("f" * 64),
        },
    )
    _assert_domain_error(forged_identity, 422, "invalid_request")

    duplicate = client.post(
        f"/api/v1/review-reuse/tasks/{task_id}/decision",
        content=(
            '{"state":"new","state":"reuse","expected_revision":1,'
            '"evidence_pack_sha256":"' + ("a" * 64) + '"}'
        ),
        headers={"Content-Type": "application/json"},
    )
    _assert_domain_error(duplicate, 422, "invalid_request")


def test_store_corruption_maps_to_503(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("API_KEY", "test")
    root = tmp_path / "store"

    from src.core.review_reuse import service as service_module
    from src.core.review_reuse.store import (
        FilesystemReviewReuseStore,
        InMemoryReviewReuseStore,
    )

    store = FilesystemReviewReuseStore(root)
    service_module.reset_review_reuse_store_for_tests(store)
    from src.main import app

    try:
        with TestClient(app, headers={"X-API-Key": "test"}) as test_client:
            created = _create(test_client)
            assert created.status_code == 200, created.text
            task_id = created.json()["task_id"]

            tenant_id = "ak-" + hashlib.sha256(b"test").hexdigest()[:16]
            tenant_digest = hashlib.sha256(tenant_id.encode("utf-8")).hexdigest()
            sidecar = root / f"tenant-v1-{tenant_digest}" / "tenant.json"
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            payload["unexpected"] = "field"
            sidecar.write_text(json.dumps(payload), encoding="utf-8")

            response = test_client.get(f"/api/v1/review-reuse/tasks/{task_id}")
            _assert_domain_error(response, 503, "store_record_corrupt")
            assert "review_reuse_store_failure code=store_record_corrupt" in caplog.text
    finally:
        close = getattr(store, "close", None)
        if close is not None:
            close()
        service_module.reset_review_reuse_store_for_tests(InMemoryReviewReuseStore())


def test_trusted_tenant_claim_cannot_use_fallback_namespace() -> None:
    from src.api.v1.review_reuse import _tenant_identity
    from src.core.review_reuse.service import ReviewReuseError

    request = SimpleNamespace(
        state=SimpleNamespace(
            review_reuse_identity_validated=True,
            tenant_id="ak-forged-tenant",
        )
    )
    with pytest.raises(ReviewReuseError) as raised:
        _tenant_identity(request, "api-key")
    assert raised.value.code == "tenant_invalid"

    request.state.tenant_id = 123
    with pytest.raises(ReviewReuseError) as non_string:
        _tenant_identity(request, "api-key")
    assert non_string.value.code == "tenant_invalid"


def test_openapi_documents_platform_and_domain_error_boundaries(
    client: TestClient,
) -> None:
    schema = client.get("/openapi.json").json()
    responses = schema["paths"]["/api/v1/review-reuse/tasks/{task_id}/decision"][
        "post"
    ]["responses"]
    assert {
        "200",
        "400",
        "401",
        "403",
        "404",
        "409",
        "413",
        "415",
        "422",
        "500",
        "503",
    } <= set(responses)
    auth_schema = responses["401"]["content"]["application/json"]["schema"]
    domain_schema = responses["409"]["content"]["application/json"]["schema"]
    assert auth_schema["$ref"].endswith("/PlatformAuthErrorResponse")
    assert domain_schema["$ref"].endswith("/ReviewReuseErrorResponse")
