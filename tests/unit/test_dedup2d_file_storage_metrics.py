import pytest

from src.core.dedup2d_file_storage import LocalDedup2DFileStorage


def _metrics_text() -> str | None:
    try:
        from prometheus_client import generate_latest
    except Exception:
        return None
    return generate_latest().decode()


def _metric_value(metric_name: str, label_fragment: str) -> float | None:
    text = _metrics_text()
    if text is None:
        return None
    for line in text.splitlines():
        if line.startswith(metric_name) and label_fragment in line:
            try:
                return float(line.rsplit(" ", 1)[-1])
            except Exception:
                return 0.0
    return 0.0


@pytest.mark.asyncio
async def test_local_storage_metrics_increment(tmp_path, monkeypatch) -> None:
    text = _metrics_text()
    if text is None:
        pytest.skip("prometheus_client not available")

    monkeypatch.setenv("DEDUP2D_FILE_STORAGE_DIR", str(tmp_path))
    storage = LocalDedup2DFileStorage()

    upload_label = 'backend="local",status="success"'
    download_label = 'backend="local",status="success"'
    delete_label = 'backend="local",status="success"'
    upload_bytes_label = 'backend="local"'

    before_upload = _metric_value("dedup2d_file_uploads_total", upload_label) or 0.0
    before_download = _metric_value("dedup2d_file_downloads_total", download_label) or 0.0
    before_delete = _metric_value("dedup2d_file_deletes_total", delete_label) or 0.0
    before_upload_count = (
        _metric_value("dedup2d_file_upload_bytes_count", upload_bytes_label) or 0.0
    )

    file_ref = await storage.save_bytes(
        job_id="job-123",
        file_name="sample.png",
        content_type="image/png",
        data=b"hello",
    )

    data = await storage.load_bytes(file_ref)
    assert data == b"hello"

    await storage.delete(file_ref)

    after_upload = _metric_value("dedup2d_file_uploads_total", upload_label) or 0.0
    after_download = _metric_value("dedup2d_file_downloads_total", download_label) or 0.0
    after_delete = _metric_value("dedup2d_file_deletes_total", delete_label) or 0.0
    after_upload_count = (
        _metric_value("dedup2d_file_upload_bytes_count", upload_bytes_label) or 0.0
    )

    assert after_upload > before_upload
    assert after_download > before_download
    assert after_delete > before_delete
    assert after_upload_count > before_upload_count
