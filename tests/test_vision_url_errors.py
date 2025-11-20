import httpx
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


class MockTransport(httpx.AsyncBaseTransport):
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:  # type: ignore
        url = str(request.url)
        if "404" in url:
            return httpx.Response(404, request=request)
        if "403" in url:
            return httpx.Response(403, request=request)
        if "timeout" in url:
            raise httpx.TimeoutException("timeout")
        return httpx.Response(200, request=request, content=b"fakeimagebytes")


def test_vision_url_404(monkeypatch):
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **kw: httpx.AsyncClient(transport=MockTransport()))
    resp = client.post(
        "/api/v1/vision/analyze",
        json={"image_url": "http://example.com/404.png", "include_description": False, "include_ocr": False},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is False
    assert data.get("code") == "INPUT_ERROR"


def test_vision_url_403(monkeypatch):
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **kw: httpx.AsyncClient(transport=MockTransport()))
    resp = client.post(
        "/api/v1/vision/analyze",
        json={"image_url": "http://example.com/403.png", "include_description": False, "include_ocr": False},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is False
    assert data.get("code") == "INPUT_ERROR"


def test_vision_url_timeout(monkeypatch):
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **kw: httpx.AsyncClient(transport=MockTransport()))
    resp = client.post(
        "/api/v1/vision/analyze",
        json={"image_url": "http://example.com/timeout.png", "include_description": False, "include_ocr": False},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is False
    assert data.get("code") == "INPUT_ERROR"
