from __future__ import annotations

from typing import Any, Dict

import time
import httpx
from fastapi.testclient import TestClient

from src.main import app
from src.api.v1.dedup import (
    get_dedupcad_vision_client,
    get_geom_store,
    get_precision_verifier,
    get_tenant_config_store,
)
from src.core.dedupcad_2d_jobs import reset_dedup2d_job_store
from src.core.dedupcad_precision.verifier import PrecisionScore


class _FakeDedupClient:
    async def health(self) -> Dict[str, Any]:
        return {"status": "healthy", "service": "caddedup-vision", "version": "0.2.0"}

    async def rebuild_indexes(self) -> Dict[str, Any]:
        return {"success": True, "message": "Indexes rebuilt successfully"}

    async def search_2d(self, **kwargs) -> Dict[str, Any]:
        return {
            "success": True,
            "total_matches": 1,
            "duplicates": [
                {
                    "drawing_id": "1",
                    "file_hash": "abc",
                    "file_name": "x.png",
                    "similarity": 0.99,
                    "confidence": 0.9,
                    "match_level": 2,
                    "verdict": "duplicate",
                    "levels": {"l1": {}, "l2": {}},
                    "diff_image_base64": None,
                    "diff_regions": None,
                }
            ],
            "similar": [],
            "final_level": 2,
            "timing": {"total_ms": 12.3},
            "level_stats": {},
            "warnings": [],
            "error": None,
        }

    async def index_add_2d(self, **kwargs) -> Dict[str, Any]:
        return {
            "success": True,
            "drawing_id": 123,
            "file_hash": "abc",
            "message": "Drawing indexed successfully",
            "processing_time_ms": 10.0,
            "s3_key": None,
        }


class _FailingDedupClient:
    async def health(self) -> Dict[str, Any]:
        raise httpx.ConnectError("boom", request=httpx.Request("GET", "http://localhost:58001/health"))


class _FakeDedupClientTwoCandidates:
    async def search_2d(self, **kwargs) -> Dict[str, Any]:
        return {
            "success": True,
            "total_matches": 2,
            "duplicates": [
                {
                    "drawing_id": "1",
                    "file_hash": "same",
                    "file_name": "PartA_v2.png",
                    "similarity": 0.99,
                    "confidence": 0.9,
                    "match_level": 2,
                    "verdict": "duplicate",
                    "levels": {"l1": {}, "l2": {}},
                    "diff_image_base64": None,
                    "diff_regions": None,
                },
                {
                    "drawing_id": "2",
                    "file_hash": "other",
                    "file_name": "PartB_v1.png",
                    "similarity": 0.99,
                    "confidence": 0.9,
                    "match_level": 2,
                    "verdict": "duplicate",
                    "levels": {"l1": {}, "l2": {}},
                    "diff_image_base64": None,
                    "diff_regions": None,
                },
            ],
            "similar": [],
            "final_level": 2,
            "timing": {"total_ms": 12.3},
            "level_stats": {},
            "warnings": [],
            "error": None,
        }


class _FakeGeomStore:
    def __init__(self):
        self.saved: Dict[str, Dict[str, Any]] = {}

    def load(self, file_hash: str) -> Dict[str, Any] | None:
        return self.saved.get(file_hash)

    def exists(self, file_hash: str) -> bool:
        return file_hash in self.saved

    def save(self, file_hash: str, geom_json: Dict[str, Any]):
        self.saved[file_hash] = geom_json


class _FakePrecisionVerifier:
    @staticmethod
    def load_json_bytes(data: bytes) -> Dict[str, Any]:
        return __import__("json").loads(data.decode("utf-8"))

    def score_pair(
        self, left_v2: Dict[str, Any], right_v2: Dict[str, Any], *, profile: str | None = None
    ) -> PrecisionScore:
        return PrecisionScore(
            score=1.0,
            breakdown={"entities": 1.0},
            geom_hash_left="left",
            geom_hash_right="right",
        )


class _FakePrecisionVerifierMidScore(_FakePrecisionVerifier):
    def score_pair(
        self, left_v2: Dict[str, Any], right_v2: Dict[str, Any], *, profile: str | None = None
    ) -> PrecisionScore:
        # Produces a fused score around 0.717 with default weights (0.3*0.99 + 0.7*0.6),
        # which is useful for testing preset-driven threshold behavior.
        return PrecisionScore(
            score=0.6,
            breakdown={"entities": 0.6},
            geom_hash_left="left",
            geom_hash_right="right",
        )


class _FakeTenantConfigStore:
    def __init__(self):
        self.cfg: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def tenant_id(api_key: str) -> str:
        return f"tenant:{api_key}"

    async def get(self, api_key: str) -> Dict[str, Any] | None:
        return self.cfg.get(api_key)

    async def set(self, api_key: str, config_obj: Dict[str, Any]) -> None:
        self.cfg[api_key] = dict(config_obj)

    async def delete(self, api_key: str) -> None:
        self.cfg.pop(api_key, None)


def test_dedup_2d_health_proxy_ok():
    app.dependency_overrides[get_dedupcad_vision_client] = lambda: _FakeDedupClient()
    try:
        client = TestClient(app)
        resp = client.get("/api/v1/dedup/2d/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"
    finally:
        app.dependency_overrides.clear()


def test_dedup_2d_index_rebuild_proxy_ok():
    app.dependency_overrides[get_dedupcad_vision_client] = lambda: _FakeDedupClient()
    try:
        client = TestClient(app)
        resp = client.post("/api/v1/dedup/2d/index/rebuild")
        assert resp.status_code == 200
        assert resp.json()["success"] is True
    finally:
        app.dependency_overrides.clear()


def test_dedup_2d_search_proxy_ok():
    app.dependency_overrides[get_dedupcad_vision_client] = lambda: _FakeDedupClient()
    try:
        client = TestClient(app)
        files = {"file": ("drawing.png", b"fake", "image/png")}
        resp = client.post("/api/v1/dedup/2d/search?mode=balanced&max_results=10", files=files)
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["total_matches"] == 1
        assert data["duplicates"][0]["verdict"] == "duplicate"
    finally:
        app.dependency_overrides.clear()


def test_dedup_2d_search_applies_precision_when_geom_json_provided():
    fake_store = _FakeGeomStore()
    fake_store.saved["abc"] = {"entities": []}

    app.dependency_overrides[get_dedupcad_vision_client] = lambda: _FakeDedupClient()
    app.dependency_overrides[get_geom_store] = lambda: fake_store
    app.dependency_overrides[get_precision_verifier] = lambda: _FakePrecisionVerifier()
    try:
        client = TestClient(app)
        files = {
            "file": ("drawing.png", b"fake", "image/png"),
            "geom_json": ("geom.json", b'{"entities":[]}', "application/json"),
        }
        resp = client.post("/api/v1/dedup/2d/search?mode=precise&max_results=10", files=files)
        assert resp.status_code == 200
        data = resp.json()
        assert data["duplicates"][0]["precision_score"] == 1.0
        assert data["duplicates"][0]["visual_similarity"] == 0.99
        assert data["duplicates"][0]["similarity"] > 0.99
        assert any("precision_verified:" in w for w in data.get("warnings", []))
    finally:
        app.dependency_overrides.clear()


def test_dedup_2d_search_thresholds_can_reclassify_duplicate_to_similar():
    fake_store = _FakeGeomStore()
    fake_store.saved["abc"] = {"entities": []}

    app.dependency_overrides[get_dedupcad_vision_client] = lambda: _FakeDedupClient()
    app.dependency_overrides[get_geom_store] = lambda: fake_store
    app.dependency_overrides[get_precision_verifier] = lambda: _FakePrecisionVerifier()
    try:
        client = TestClient(app)
        files = {
            "file": ("drawing.png", b"fake", "image/png"),
            "geom_json": ("geom.json", b'{"entities":[]}', "application/json"),
        }
        # With default weights, fused sim is ~0.997 (<0.9999), so it should be "similar".
        resp = client.post(
            "/api/v1/dedup/2d/search?mode=precise&max_results=10&duplicate_threshold=0.9999&similar_threshold=0.5",
            files=files,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["duplicates"] == []
        assert len(data["similar"]) == 1
        assert data["similar"][0]["verdict"] == "similar"
    finally:
        app.dependency_overrides.clear()


def test_dedup_2d_search_preset_version_can_classify_mid_similarity_as_similar():
    fake_store = _FakeGeomStore()
    fake_store.saved["abc"] = {"layers": {"0": {"name": "0"}}, "entities": []}

    app.dependency_overrides[get_dedupcad_vision_client] = lambda: _FakeDedupClient()
    app.dependency_overrides[get_geom_store] = lambda: fake_store
    app.dependency_overrides[get_precision_verifier] = lambda: _FakePrecisionVerifierMidScore()
    try:
        client = TestClient(app)
        files = {
            "file": ("drawing.png", b"fake", "image/png"),
            "geom_json": ("geom.json", b'{"layers":{"0":{"name":"0"}},"entities":[]}', "application/json"),
        }
        resp = client.post("/api/v1/dedup/2d/search?preset=version&mode=balanced&max_results=10", files=files)
        assert resp.status_code == 200
        data = resp.json()
        assert data["duplicates"] == []
        assert len(data["similar"]) == 1
        assert data["similar"][0]["verdict"] == "similar"
        assert data["similar"][0]["levels"]["l4"]["precision_profile"] == "version"
    finally:
        app.dependency_overrides.clear()


def test_dedup_2d_search_preset_does_not_override_explicit_thresholds():
    fake_store = _FakeGeomStore()
    fake_store.saved["abc"] = {"layers": {"0": {"name": "0"}}, "entities": []}

    app.dependency_overrides[get_dedupcad_vision_client] = lambda: _FakeDedupClient()
    app.dependency_overrides[get_geom_store] = lambda: fake_store
    app.dependency_overrides[get_precision_verifier] = lambda: _FakePrecisionVerifierMidScore()
    try:
        client = TestClient(app)
        files = {
            "file": ("drawing.png", b"fake", "image/png"),
            "geom_json": ("geom.json", b'{"layers":{"0":{"name":"0"}},"entities":[]}', "application/json"),
        }
        resp = client.post(
            "/api/v1/dedup/2d/search?preset=version&mode=balanced&max_results=10&similar_threshold=0.90",
            files=files,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["duplicates"] == []
        assert data["similar"] == []
    finally:
        app.dependency_overrides.clear()


def test_dedup_2d_search_precision_diff_can_be_requested():
    fake_store = _FakeGeomStore()
    fake_store.saved["abc"] = {"layers": {"0": {"name": "0"}, "A": {"name": "A"}}, "entities": []}

    app.dependency_overrides[get_dedupcad_vision_client] = lambda: _FakeDedupClient()
    app.dependency_overrides[get_geom_store] = lambda: fake_store
    app.dependency_overrides[get_precision_verifier] = lambda: _FakePrecisionVerifier()
    try:
        client = TestClient(app)
        files = {
            "file": ("drawing.png", b"fake", "image/png"),
            "geom_json": ("geom.json", b'{"layers":{"0":{"name":"0"}},"entities":[]}', "application/json"),
        }
        resp = client.post(
            "/api/v1/dedup/2d/search?mode=balanced&max_results=10&precision_compute_diff=true&precision_diff_top_n=1",
            files=files,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["duplicates"]) == 1
        item = data["duplicates"][0]
        assert item["precision_diff"] is not None
        assert item["precision_diff_similarity"] is not None
        assert "l4" in item["levels"]
    finally:
        app.dependency_overrides.clear()


def test_dedup_2d_index_add_stores_geom_json():
    fake_store = _FakeGeomStore()
    app.dependency_overrides[get_dedupcad_vision_client] = lambda: _FakeDedupClient()
    app.dependency_overrides[get_geom_store] = lambda: fake_store
    app.dependency_overrides[get_precision_verifier] = lambda: _FakePrecisionVerifier()
    try:
        client = TestClient(app)
        files = {
            "file": ("drawing.png", b"fake", "image/png"),
            "geom_json": ("geom.json", b'{"entities":[]}', "application/json"),
        }
        resp = client.post("/api/v1/dedup/2d/index/add?user_name=tester&upload_to_s3=false", files=files)
        assert resp.status_code == 200
        assert "geom_json stored" in resp.json()["message"]
        assert "abc" in fake_store.saved
    finally:
        app.dependency_overrides.clear()


def test_dedup_2d_precision_compare_ok():
    app.dependency_overrides[get_precision_verifier] = lambda: _FakePrecisionVerifier()
    try:
        client = TestClient(app)
        files = {
            "left_geom_json": ("left.json", b'{"entities":[]}', "application/json"),
            "right_geom_json": ("right.json", b'{"entities":[]}', "application/json"),
        }
        resp = client.post("/api/v1/dedup/2d/precision/compare", files=files)
        assert resp.status_code == 200
        data = resp.json()
        assert data["score"] == 1.0
        assert data["geom_hash_left"] == "left"
        assert data["geom_hash_right"] == "right"
    finally:
        app.dependency_overrides.clear()


def test_dedup_2d_geom_exists_ok():
    fake_store = _FakeGeomStore()
    file_hash = "a" * 64
    fake_store.saved[file_hash] = {"entities": []}

    app.dependency_overrides[get_geom_store] = lambda: fake_store
    try:
        client = TestClient(app)
        resp = client.get(f"/api/v1/dedup/2d/geom/{file_hash}/exists")
        assert resp.status_code == 200
        assert resp.json() == {"file_hash": file_hash, "exists": True}
    finally:
        app.dependency_overrides.clear()


def test_dedup_2d_health_proxy_unavailable_returns_503():
    app.dependency_overrides[get_dedupcad_vision_client] = lambda: _FailingDedupClient()
    try:
        client = TestClient(app)
        resp = client.get("/api/v1/dedup/2d/health")
        assert resp.status_code == 503
    finally:
        app.dependency_overrides.clear()


def test_dedup_2d_tenant_config_put_requires_admin_token():
    tenant_store = _FakeTenantConfigStore()
    app.dependency_overrides[get_tenant_config_store] = lambda: tenant_store
    try:
        client = TestClient(app)
        resp = client.put("/api/v1/dedup/2d/config", json={"preset": "version"}, headers={"X-API-Key": "t1"})
        assert resp.status_code == 401
    finally:
        app.dependency_overrides.clear()


def test_dedup_2d_tenant_config_roundtrip_get_put_delete():
    tenant_store = _FakeTenantConfigStore()
    app.dependency_overrides[get_tenant_config_store] = lambda: tenant_store
    try:
        client = TestClient(app)
        headers = {"X-API-Key": "t1", "X-Admin-Token": "test"}
        resp = client.put(
            "/api/v1/dedup/2d/config",
            json={"preset": "version", "similar_threshold": 0.75, "precision_top_n": 33},
            headers=headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["tenant_id"] == "tenant:t1"
        assert data["config"]["preset"] == "version"
        assert data["config"]["similar_threshold"] == 0.75
        assert data["config"]["precision_top_n"] == 33

        resp = client.get("/api/v1/dedup/2d/config", headers={"X-API-Key": "t1"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["config"]["preset"] == "version"

        resp = client.delete("/api/v1/dedup/2d/config", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["config"] == {}
    finally:
        app.dependency_overrides.clear()


def test_dedup_2d_search_applies_tenant_preset_when_no_preset_given():
    fake_store = _FakeGeomStore()
    fake_store.saved["abc"] = {"layers": {"0": {"name": "0"}}, "entities": []}

    tenant_store = _FakeTenantConfigStore()
    tenant_store.cfg["test"] = {"preset": "version"}

    app.dependency_overrides[get_dedupcad_vision_client] = lambda: _FakeDedupClient()
    app.dependency_overrides[get_geom_store] = lambda: fake_store
    app.dependency_overrides[get_precision_verifier] = lambda: _FakePrecisionVerifierMidScore()
    app.dependency_overrides[get_tenant_config_store] = lambda: tenant_store
    try:
        client = TestClient(app)
        files = {
            "file": ("drawing.png", b"fake", "image/png"),
            "geom_json": ("geom.json", b'{"layers":{"0":{"name":"0"}},"entities":[]}', "application/json"),
        }
        resp = client.post("/api/v1/dedup/2d/search?mode=balanced&max_results=10", files=files)
        assert resp.status_code == 200
        data = resp.json()
        assert data["duplicates"] == []
        assert len(data["similar"]) == 1
        assert data["similar"][0]["verdict"] == "similar"
    finally:
        app.dependency_overrides.clear()


def test_dedup_2d_search_explicit_preset_overrides_tenant_preset():
    fake_store = _FakeGeomStore()
    fake_store.saved["abc"] = {"layers": {"0": {"name": "0"}}, "entities": []}

    tenant_store = _FakeTenantConfigStore()
    tenant_store.cfg["test"] = {"preset": "strict"}

    app.dependency_overrides[get_dedupcad_vision_client] = lambda: _FakeDedupClient()
    app.dependency_overrides[get_geom_store] = lambda: fake_store
    app.dependency_overrides[get_precision_verifier] = lambda: _FakePrecisionVerifierMidScore()
    app.dependency_overrides[get_tenant_config_store] = lambda: tenant_store
    try:
        client = TestClient(app)
        files = {
            "file": ("drawing.png", b"fake", "image/png"),
            "geom_json": ("geom.json", b'{"layers":{"0":{"name":"0"}},"entities":[]}', "application/json"),
        }
        resp = client.post("/api/v1/dedup/2d/search?preset=version&mode=balanced&max_results=10", files=files)
        assert resp.status_code == 200
        data = resp.json()
        assert data["duplicates"] == []
        assert len(data["similar"]) == 1
        assert data["similar"][0]["verdict"] == "similar"
    finally:
        app.dependency_overrides.clear()


def test_dedup_2d_search_version_gate_file_name_filters_cross_drawing_candidates():
    fake_store = _FakeGeomStore()
    fake_store.saved["same"] = {"entities": []}
    fake_store.saved["other"] = {"entities": []}

    app.dependency_overrides[get_dedupcad_vision_client] = lambda: _FakeDedupClientTwoCandidates()
    app.dependency_overrides[get_geom_store] = lambda: fake_store
    app.dependency_overrides[get_precision_verifier] = lambda: _FakePrecisionVerifier()
    try:
        client = TestClient(app)
        files = {
            "file": ("PartA_v1.png", b"fake", "image/png"),
            "geom_json": ("geom.json", b'{"entities":[]}', "application/json"),
        }
        resp = client.post(
            "/api/v1/dedup/2d/search?preset=version&mode=precise&max_results=10&version_gate=file_name",
            files=files,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["duplicates"]) == 1
        assert data["duplicates"][0]["file_hash"] == "same"
        assert any("precision_version_gate_filtered:1" == w for w in data.get("warnings", []))
    finally:
        app.dependency_overrides.clear()


def test_dedup_2d_search_async_returns_job_and_completes():
    reset_dedup2d_job_store()
    app.dependency_overrides[get_dedupcad_vision_client] = lambda: _FakeDedupClient()
    try:
        client = TestClient(app)
        files = {"file": ("drawing.png", b"fake", "image/png")}
        resp = client.post("/api/v1/dedup/2d/search?mode=balanced&max_results=10&async=true", files=files)
        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"]
        poll_url = data["poll_url"]

        for _ in range(50):
            job_resp = client.get(poll_url)
            assert job_resp.status_code == 200
            job = job_resp.json()
            if job["status"] == "completed":
                assert job["result"]["success"] is True
                assert job["result"]["total_matches"] == 1
                assert job["result"]["duplicates"][0]["verdict"] == "duplicate"
                break
            time.sleep(0.01)
        else:
            raise AssertionError("async job did not complete in time")
    finally:
        app.dependency_overrides.clear()


class _SlowDedupClient(_FakeDedupClient):
    async def search_2d(self, **kwargs) -> Dict[str, Any]:
        import anyio

        await anyio.sleep(0.2)
        return await super().search_2d(**kwargs)


def test_dedup_2d_search_async_cancel():
    reset_dedup2d_job_store()
    app.dependency_overrides[get_dedupcad_vision_client] = lambda: _SlowDedupClient()
    try:
        client = TestClient(app)
        files = {"file": ("drawing.png", b"fake", "image/png")}
        resp = client.post("/api/v1/dedup/2d/search?mode=balanced&max_results=10&async=true", files=files)
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]

        cancel = client.post(f"/api/v1/dedup/2d/jobs/{job_id}/cancel")
        assert cancel.status_code == 200
        assert cancel.json()["canceled"] is True

        # Ensure the job is canceled or already completed (race if very fast).
        status = client.get(f"/api/v1/dedup/2d/jobs/{job_id}")
        assert status.status_code == 200
        assert status.json()["status"] in {"canceled", "completed"}
    finally:
        app.dependency_overrides.clear()
