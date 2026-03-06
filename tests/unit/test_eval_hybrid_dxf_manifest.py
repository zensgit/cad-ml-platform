from pathlib import Path

from scripts.eval_hybrid_dxf_manifest import (
    _load_manifest_cases,
    _score_rows,
)


def test_load_manifest_cases_prefers_relative_path(tmp_path: Path) -> None:
    dxf_dir = tmp_path / "dxf"
    dxf_dir.mkdir()
    nested = dxf_dir / "nested"
    nested.mkdir()
    (nested / "part1.dxf").write_text("0", encoding="utf-8")
    (dxf_dir / "part2.dxf").write_text("0", encoding="utf-8")

    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "\n".join(
            [
                "file_name,label_cn,relative_path,source_dir",
                "part1.dxf,过滤组件,nested/part1.dxf,",
                "part2.dxf,轴承件,,",
            ]
        ),
        encoding="utf-8",
    )

    cases = _load_manifest_cases(manifest, dxf_dir)

    assert len(cases) == 2
    assert cases[0].file_path == nested / "part1.dxf"
    assert cases[1].file_path == dxf_dir / "part2.dxf"


def test_score_rows_normalizes_fine_labels_to_coarse() -> None:
    rows = [
        {
            "true_label": "设备",
            "hybrid_label": "再沸器",
            "graph2d_label": "设备",
        },
        {
            "true_label": "传动件",
            "hybrid_label": "搅拌轴组件",
            "graph2d_label": "",
        },
        {
            "true_label": "法兰",
            "hybrid_label": "人孔",
            "graph2d_label": "人孔法兰",
        },
    ]
    alias_map = {"other": "其他"}

    summary = _score_rows(
        rows,
        branch_to_column={
            "hybrid_label": "hybrid_label",
            "graph2d_label": "graph2d_label",
        },
        alias_map=alias_map,
    )

    assert summary["hybrid_label"]["evaluated"] == 3
    assert summary["hybrid_label"]["correct"] == 2
    assert summary["hybrid_label"]["accuracy"] == 2 / 3

    assert summary["graph2d_label"]["evaluated"] == 2
    assert summary["graph2d_label"]["correct"] == 2
    assert summary["graph2d_label"]["missing_pred"] == 1
