from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import patch

import ezdxf
import pytest

pytest.importorskip("torch")
from src.ml.train.dataset_2d import DXF_NODE_DIM, DXFManifestDataset


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_disk_cache_does_not_cache_labels_across_manifests(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("DXF_MANIFEST_DATASET_CACHE", "disk")
    monkeypatch.setenv("DXF_MANIFEST_DATASET_CACHE_DIR", str(cache_dir))

    dxf_dir = tmp_path / "dxfs"
    dxf_dir.mkdir()

    # Two DXF files so we can force label-id assignment differences by row order.
    for name in ("a.dxf", "b.dxf"):
        path = dxf_dir / name
        doc = ezdxf.new()
        msp = doc.modelspace()
        msp.add_line((0.0, 0.0), (10.0, 0.0))
        doc.saveas(str(path))

    # Manifest 1: bucket_b appears first, so bucket_a gets label_id=1.
    manifest1 = tmp_path / "manifest1.csv"
    _write_manifest(
        manifest1,
        rows=[
            {"file_name": "b.dxf", "relative_path": "b.dxf", "label_cn": "bucket_b"},
            {"file_name": "a.dxf", "relative_path": "a.dxf", "label_cn": "bucket_a"},
        ],
    )

    ds1 = DXFManifestDataset(
        str(manifest1),
        str(dxf_dir),
        node_dim=DXF_NODE_DIM,
        return_edge_attr=False,
    )
    idx_a1 = next(i for i, s in enumerate(ds1.samples) if s["file_name"] == "a.dxf")
    _graph1, label1 = ds1[idx_a1]
    assert int(label1.item()) == 1

    # Manifest 2: bucket_a appears first, so bucket_a gets label_id=0.
    manifest2 = tmp_path / "manifest2.csv"
    _write_manifest(
        manifest2,
        rows=[
            {"file_name": "a.dxf", "relative_path": "a.dxf", "label_cn": "bucket_a"},
            {"file_name": "b.dxf", "relative_path": "b.dxf", "label_cn": "bucket_b"},
        ],
    )

    ds2 = DXFManifestDataset(
        str(manifest2),
        str(dxf_dir),
        node_dim=DXF_NODE_DIM,
        return_edge_attr=False,
    )
    idx_a2 = next(i for i, s in enumerate(ds2.samples) if s["file_name"] == "a.dxf")

    # If disk cache incorrectly cached the label tensor, this would return 1 (stale)
    # and also trigger a DXF re-parse if the graph wasn't cached. We require graph
    # reuse while recomputing the label for the new manifest.
    with patch(
        "src.ml.train.dataset_2d.ezdxf.readfile",
        side_effect=RuntimeError("should_not_read"),
    ):
        _graph2, label2 = ds2[idx_a2]

    assert int(label2.item()) == 0
