from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_analyze_batch_classify_preserves_alignment_with_unsupported_files():
    files = [
        ("files", ("bad.txt", b"not_cad", "text/plain")),
        ("files", ("good1.dxf", b"0\\nSECTION\\n2\\nHEADER\\n0\\nENDSEC\\n0\\nEOF\\n", "application/dxf")),
        ("files", ("good2.dwg", b"dummy_dwg", "application/acad")),
    ]

    resp = client.post("/api/v1/analyze/batch-classify", files=files)
    assert resp.status_code == 200

    data = resp.json()
    assert data["total"] == 3
    assert data["failed"] >= 1  # at least the unsupported file
    assert data["success"] + data["failed"] == data["total"]

    results = data["results"]
    assert len(results) == 3

    assert results[0]["file_name"] == "bad.txt"
    assert "Unsupported format" in (results[0].get("error") or "")

    assert results[1]["file_name"] == "good1.dxf"

    assert results[2]["file_name"] == "good2.dwg"
