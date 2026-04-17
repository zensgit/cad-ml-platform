import asyncio

import pytest
from fastapi import HTTPException

from src.core.document_pipeline import run_document_pipeline
from src.models.cad_document import CadDocument, CadEntity


class _ParseAdapter:
    def __init__(self, doc):
        self._doc = doc

    async def parse(self, content: bytes, file_name: str):  # type: ignore[override]
        return self._doc


class _SlowAdapter:
    async def parse(self, content: bytes, file_name: str):  # type: ignore[override]
        await asyncio.sleep(0.1)
        return CadDocument(file_name=file_name, format="step")


class _LegacyAdapter:
    async def convert(self, content: bytes, file_name: str):  # type: ignore[override]
        return {"legacy": True}


class _BrokenAdapter:
    async def parse(self, content: bytes, file_name: str):  # type: ignore[override]
        raise ValueError("boom")


class _Factory:
    adapter = None

    @classmethod
    def get_adapter(cls, fmt):
        assert cls.adapter is not None
        return cls.adapter


@pytest.mark.asyncio
async def test_run_document_pipeline_success_attaches_metadata(monkeypatch):
    doc = CadDocument(
        file_name="sample.step",
        format="step",
        entities=[CadEntity(kind="FACE")],
    )
    _Factory.adapter = _ParseAdapter(doc)
    monkeypatch.setenv("FORMAT_STRICT_MODE", "0")

    context = await run_document_pipeline(
        file_name="sample.step",
        content=b"ISO-10303-21\nHEADER;\nENDSEC;\n",
        started_at=0.0,
        material="steel",
        project_id="p1",
        adapter_factory_cls=_Factory,
    )

    assert context["file_format"] == "step"
    assert context["doc"] is doc
    assert doc.metadata["material"] == "steel"
    assert doc.metadata["project_id"] == "p1"
    assert context["unified_data"]["entity_count"] == 1
    assert context["parse_stage_duration"] >= 0.0


@pytest.mark.asyncio
async def test_run_document_pipeline_supports_legacy_convert(monkeypatch):
    _Factory.adapter = _LegacyAdapter()
    monkeypatch.setenv("FORMAT_STRICT_MODE", "0")

    context = await run_document_pipeline(
        file_name="legacy.dxf",
        content=b"0\nSECTION\n2\nHEADER\n0\nENDSEC\n",
        started_at=0.0,
        adapter_factory_cls=_Factory,
    )

    assert context["file_format"] == "dxf"
    assert context["doc"].metadata["legacy"] is True
    assert context["unified_data"]["format"] == "dxf"


@pytest.mark.asyncio
async def test_run_document_pipeline_parse_timeout(monkeypatch):
    _Factory.adapter = _SlowAdapter()
    monkeypatch.setenv("PARSE_TIMEOUT_SECONDS", "0.01")
    monkeypatch.setenv("FORMAT_STRICT_MODE", "0")

    with pytest.raises(HTTPException) as exc_info:
        await run_document_pipeline(
            file_name="slow.step",
            content=b"ISO-10303-21\nHEADER;\nENDSEC;\n",
            started_at=0.0,
            adapter_factory_cls=_Factory,
        )

    assert exc_info.value.status_code == 504
    assert exc_info.value.detail["code"] == "TIMEOUT"


@pytest.mark.asyncio
async def test_run_document_pipeline_strict_deep_validation(monkeypatch):
    doc = CadDocument(file_name="sample.step", format="step")
    _Factory.adapter = _ParseAdapter(doc)
    monkeypatch.setenv("FORMAT_STRICT_MODE", "1")
    monkeypatch.setattr(
        "src.core.document_pipeline.deep_format_validate",
        lambda data, file_format: (False, "missing_header"),
    )

    with pytest.raises(HTTPException) as exc_info:
        await run_document_pipeline(
            file_name="bad.step",
            content=b"ISO-10303-21\nHEADER;\nENDSEC;\n",
            started_at=0.0,
            adapter_factory_cls=_Factory,
        )

    assert exc_info.value.status_code == 415
    assert exc_info.value.detail["context"]["reason"] == "missing_header"


@pytest.mark.asyncio
async def test_run_document_pipeline_matrix_validation(monkeypatch):
    doc = CadDocument(file_name="sample.step", format="step")
    _Factory.adapter = _ParseAdapter(doc)
    monkeypatch.setenv("FORMAT_STRICT_MODE", "1")
    monkeypatch.setattr(
        "src.core.document_pipeline.deep_format_validate",
        lambda data, file_format: (True, "ok"),
    )
    monkeypatch.setattr(
        "src.core.document_pipeline.matrix_validate",
        lambda data, file_format, project_id=None: (False, "missing_token:HEADER"),
    )

    with pytest.raises(HTTPException) as exc_info:
        await run_document_pipeline(
            file_name="bad.step",
            content=b"ISO-10303-21\nHEADER;\nENDSEC;\n",
            started_at=0.0,
            adapter_factory_cls=_Factory,
        )

    assert exc_info.value.status_code == 415
    assert exc_info.value.detail["context"]["reason"] == "missing_token:HEADER"


@pytest.mark.asyncio
async def test_run_document_pipeline_entity_limit(monkeypatch):
    doc = CadDocument(
        file_name="big.step",
        format="step",
        entities=[CadEntity(kind="FACE"), CadEntity(kind="EDGE")],
    )
    _Factory.adapter = _ParseAdapter(doc)
    monkeypatch.setenv("FORMAT_STRICT_MODE", "0")
    monkeypatch.setenv("ANALYSIS_MAX_ENTITIES", "1")

    with pytest.raises(HTTPException) as exc_info:
        await run_document_pipeline(
            file_name="big.step",
            content=b"ISO-10303-21\nHEADER;\nENDSEC;\n",
            started_at=0.0,
            adapter_factory_cls=_Factory,
        )

    assert exc_info.value.status_code == 422
    assert exc_info.value.detail["code"] == "VALIDATION_FAILED"


@pytest.mark.asyncio
async def test_run_document_pipeline_parse_exception_falls_back(monkeypatch):
    _Factory.adapter = _BrokenAdapter()
    monkeypatch.setenv("FORMAT_STRICT_MODE", "0")

    context = await run_document_pipeline(
        file_name="broken.step",
        content=b"ISO-10303-21\nHEADER;\nENDSEC;\n",
        started_at=0.0,
        adapter_factory_cls=_Factory,
    )

    assert context["file_format"] == "step"
    assert context["doc"].entity_count() == 0
    assert context["unified_data"]["complexity"] == "low"
