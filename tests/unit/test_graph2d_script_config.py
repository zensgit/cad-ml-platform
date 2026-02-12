from __future__ import annotations

import os
from argparse import Namespace
from pathlib import Path


def test_train_script_loads_section_defaults(tmp_path: Path) -> None:
    cfg = tmp_path / "train.yaml"
    cfg.write_text(
        "\n".join(
            [
                "train_2d_graph:",
                "  epochs: 11",
                "  dxf_max_nodes: 321",
                "  scheduler: warmup_cosine",
            ]
        ),
        encoding="utf-8",
    )
    from scripts.train_2d_graph import _load_yaml_defaults

    defaults = _load_yaml_defaults(str(cfg), "train_2d_graph")
    assert defaults["epochs"] == 11
    assert defaults["dxf_max_nodes"] == 321
    assert defaults["scheduler"] == "warmup_cosine"


def test_eval_script_loads_section_defaults(tmp_path: Path) -> None:
    cfg = tmp_path / "eval.yaml"
    cfg.write_text(
        "\n".join(
            [
                "eval_2d_graph:",
                "  batch_size: 8",
                "  val_split: 0.1",
            ]
        ),
        encoding="utf-8",
    )
    from scripts.eval_2d_graph import _load_yaml_defaults

    defaults = _load_yaml_defaults(str(cfg), "eval_2d_graph")
    assert defaults["batch_size"] == 8
    assert defaults["val_split"] == 0.1


def test_train_script_sampling_env_override(monkeypatch) -> None:
    from scripts.train_2d_graph import _apply_dxf_sampling_env

    keys = [
        "DXF_MAX_NODES",
        "DXF_SAMPLING_STRATEGY",
        "DXF_SAMPLING_SEED",
        "DXF_TEXT_PRIORITY_RATIO",
    ]
    for key in keys:
        monkeypatch.delenv(key, raising=False)

    ns = Namespace(
        dxf_max_nodes=123,
        dxf_sampling_strategy="hybrid",
        dxf_sampling_seed=7,
        dxf_text_priority_ratio=0.4,
    )
    _apply_dxf_sampling_env(ns)
    assert os.getenv("DXF_MAX_NODES") == "123"
    assert os.getenv("DXF_SAMPLING_STRATEGY") == "hybrid"
    assert os.getenv("DXF_SAMPLING_SEED") == "7"
    assert os.getenv("DXF_TEXT_PRIORITY_RATIO") == "0.4"

    for key in keys:
        os.environ.pop(key, None)
