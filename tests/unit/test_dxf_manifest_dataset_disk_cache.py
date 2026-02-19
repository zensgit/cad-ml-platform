from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import patch

import ezdxf
import pytest

pytest.importorskip("torch")
from src.ml.train.dataset_2d import DXFManifestDataset, DXF_NODE_DIM


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_dxf_manifest_dataset_disk_cache_avoids_reparsing(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("DXF_MANIFEST_DATASET_CACHE", "disk")
    monkeypatch.setenv("DXF_MANIFEST_DATASET_CACHE_DIR", str(cache_dir))

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

    # First read should parse DXF and populate disk cache.
    dataset1 = DXFManifestDataset(
        str(manifest),
        str(dxf_dir),
        node_dim=DXF_NODE_DIM,
        return_edge_attr=True,
    )
    _graph1, _label1 = dataset1[0]

    # Second dataset instance should load from disk cache without calling ezdxf.readfile.
    dataset2 = DXFManifestDataset(
        str(manifest),
        str(dxf_dir),
        node_dim=DXF_NODE_DIM,
        return_edge_attr=True,
    )

    with patch("src.ml.train.dataset_2d.ezdxf.readfile", side_effect=RuntimeError("should_not_read")):
        _graph2, _label2 = dataset2[0]
