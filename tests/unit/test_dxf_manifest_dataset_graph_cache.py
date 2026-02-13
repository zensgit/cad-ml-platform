from __future__ import annotations

import csv
from pathlib import Path

import ezdxf

from src.ml.train.dataset_2d import DXFManifestDataset, DXF_NODE_DIM


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_dxf_manifest_dataset_graph_cache_tracks_hits(tmp_path, monkeypatch):
    monkeypatch.setenv("DXF_MANIFEST_DATASET_CACHE", "memory")
    monkeypatch.setenv("DXF_MANIFEST_DATASET_CACHE_MAX_ITEMS", "10")

    # Create a minimal DXF file.
    dxf_dir = tmp_path / "dxfs"
    dxf_dir.mkdir()
    dxf_path = dxf_dir / "a.dxf"
    doc = ezdxf.new()
    msp = doc.modelspace()
    msp.add_line((0.0, 0.0), (10.0, 0.0))
    doc.saveas(str(dxf_path))

    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        rows=[
            {
                "file_name": "a.dxf",
                "relative_path": "a.dxf",
                "label_cn": "bucket_a",
            }
        ],
    )

    dataset = DXFManifestDataset(
        str(manifest),
        str(dxf_dir),
        node_dim=DXF_NODE_DIM,
        return_edge_attr=False,
    )

    _graph1, _label1 = dataset[0]
    stats_after_first = dataset.get_cache_stats()
    assert stats_after_first["enabled"] is True
    assert stats_after_first["misses"] == 1
    assert stats_after_first["hits"] == 0

    _graph2, _label2 = dataset[0]
    stats_after_second = dataset.get_cache_stats()
    assert stats_after_second["hits"] == 1

