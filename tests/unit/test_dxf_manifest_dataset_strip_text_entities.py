from __future__ import annotations
from pathlib import Path

import ezdxf


def _write_min_dxf(path: Path) -> None:
    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()
    msp.add_line((0, 0), (100, 0))
    msp.add_text("TITLEBLOCK: 人孔", dxfattribs={"height": 2.5}).set_placement((80, 20))
    doc.saveas(str(path))


def test_manifest_dataset_strip_text_entities_removes_text_nodes(
    tmp_path: Path, monkeypatch
) -> None:
    dxf_path = tmp_path / "sample.dxf"
    _write_min_dxf(dxf_path)

    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "file_name,relative_path,label_cn,label_confidence,label_status\n"
        "sample.dxf,sample.dxf,人孔,0.95,matched\n",
        encoding="utf-8",
    )

    from src.ml.train.dataset_2d import DXFManifestDataset, DXF_NODE_FEATURES

    is_text_idx = DXF_NODE_FEATURES.index("is_text")

    monkeypatch.delenv("DXF_STRIP_TEXT_ENTITIES", raising=False)
    monkeypatch.delenv("DXF_MANIFEST_DATASET_CACHE", raising=False)

    ds = DXFManifestDataset(str(manifest), str(tmp_path))
    graph, _label = ds[0]
    x = graph["x"]
    assert x.size(0) >= 1
    assert float(x[:, is_text_idx].sum().item()) >= 1.0

    monkeypatch.setenv("DXF_STRIP_TEXT_ENTITIES", "true")
    ds2 = DXFManifestDataset(str(manifest), str(tmp_path))
    graph2, _label2 = ds2[0]
    x2 = graph2["x"]
    assert x2.size(0) >= 1
    assert float(x2[:, is_text_idx].sum().item()) == 0.0


def test_manifest_dataset_cache_key_includes_strip_flag(
    tmp_path: Path, monkeypatch
) -> None:
    dxf_path = tmp_path / "sample.dxf"
    _write_min_dxf(dxf_path)

    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "file_name,relative_path,label_cn,label_confidence,label_status\n"
        "sample.dxf,sample.dxf,人孔,0.95,matched\n",
        encoding="utf-8",
    )

    from src.ml.train.dataset_2d import DXFManifestDataset

    monkeypatch.delenv("DXF_MANIFEST_DATASET_CACHE", raising=False)
    ds = DXFManifestDataset(str(manifest), str(tmp_path))

    monkeypatch.setenv("DXF_STRIP_TEXT_ENTITIES", "false")
    key1 = ds._graph_cache_key(dxf_path)

    monkeypatch.setenv("DXF_STRIP_TEXT_ENTITIES", "true")
    key2 = ds._graph_cache_key(dxf_path)

    assert key1 != key2
