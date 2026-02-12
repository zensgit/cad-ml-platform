from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest

# Skip tests if torch is not available
pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch not installed in this environment"
)


def _write_min_dxf(path: Path) -> None:
    import ezdxf

    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    msp.add_line((0, 0), (100, 0))
    msp.add_line((100, 0), (100, 100))
    doc.saveas(str(path))


def test_manifest_dataset_reads_relative_path(tmp_path: Path) -> None:
    from src.ml.train.dataset_2d import DXFManifestDataset

    dxf_root = tmp_path / "dxf_root"
    label_dir = dxf_root / "传动件"
    label_dir.mkdir(parents=True)

    file_path = label_dir / "case1.dxf"
    _write_min_dxf(file_path)

    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["file_name", "relative_path", "label_cn"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "file_name": file_path.name,
                "relative_path": str(file_path.relative_to(dxf_root)),
                "label_cn": "传动件",
            }
        )

    dataset = DXFManifestDataset(str(manifest), str(dxf_root))
    graph, label = dataset[0]

    assert graph.get("file_name") == file_path.name
    assert int(label.item()) == 0
    assert graph["x"].shape[0] >= 1


def test_manifest_dataset_reads_file_name_as_relative_path(tmp_path: Path) -> None:
    from src.ml.train.dataset_2d import DXFManifestDataset

    dxf_root = tmp_path / "dxf_root"
    label_dir = dxf_root / "轴类"
    label_dir.mkdir(parents=True)

    file_path = label_dir / "case2.DXF"
    _write_min_dxf(file_path)

    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file_name", "label_cn"])
        writer.writeheader()
        writer.writerow(
            {
                "file_name": str(file_path.relative_to(dxf_root)),
                "label_cn": "轴类",
            }
        )

    dataset = DXFManifestDataset(str(manifest), str(dxf_root))
    graph, label = dataset[0]

    assert graph.get("file_name") == str(file_path.relative_to(dxf_root))
    assert int(label.item()) == 0
    assert graph["x"].shape[0] >= 1

