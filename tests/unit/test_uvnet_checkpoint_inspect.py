from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from scripts.uvnet_checkpoint_inspect import main as uvnet_checkpoint_inspect_main
from src.ml.train.model import UVNetGraphModel


def test_uvnet_checkpoint_inspect_reports_configured_and_resolved_grid_surface(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model = UVNetGraphModel(
        node_input_dim=4,
        edge_input_dim=2,
        hidden_dim=8,
        embedding_dim=16,
        num_classes=3,
        use_face_grid_features=True,
        use_edge_grid_features=True,
        grid_encoder_kind="cnn_pool",
        grid_fusion_mode="concat_projection",
    )
    checkpoint_path = tmp_path / "uvnet_inspect.pth"
    torch.save(
        {
            "config": {
                **model.get_config(),
                "grid_branch_surface_kind": "cnn_pool_concat_projection_dual_branch",
                "grid_tower_topology_kind": "graph_grid_dual_tower_projection",
            },
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )

    rc = uvnet_checkpoint_inspect_main(
        ["--path", str(checkpoint_path), "--nodes", "3"]
    )

    assert rc == 0
    output = capsys.readouterr().out
    assert "Configured grid branch surface: cnn_pool_concat_projection_dual_branch" in output
    assert "Resolved grid branch surface: cnn_pool_concat_projection_dual_branch" in output
    assert "Configured grid tower topology: graph_grid_dual_tower_projection" in output
    assert "Resolved grid tower topology: graph_grid_dual_tower_projection" in output
    assert "Checkpoint load mode: strict" in output


def test_uvnet_checkpoint_inspect_writes_summary_json(tmp_path: Path) -> None:
    model = UVNetGraphModel(
        node_input_dim=4,
        edge_input_dim=2,
        hidden_dim=8,
        embedding_dim=16,
        num_classes=3,
        use_face_grid_features=True,
        use_edge_grid_features=True,
        grid_encoder_kind="cnn_pool",
        grid_fusion_mode="concat_projection",
    )
    checkpoint_path = tmp_path / "uvnet_inspect_summary.pth"
    summary_path = tmp_path / "uvnet_inspect_summary.json"
    torch.save(
        {
            "config": model.get_config(),
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )

    rc = uvnet_checkpoint_inspect_main(
        [
            "--path",
            str(checkpoint_path),
            "--nodes",
            "3",
            "--summary-json",
            str(summary_path),
        ]
    )

    assert rc == 0
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["surface_kind"] == "uvnet_checkpoint_inspect_summary"
    assert payload["model_surface_contract"]["configured_grid_tower_topology_kind"] == (
        "graph_grid_dual_tower_projection"
    )
    assert payload["model_surface_contract"]["resolved_grid_branch_surface_kind"] == (
        "cnn_pool_concat_projection_dual_branch"
    )
    assert payload["model_surface_contract"]["resolved_grid_tower_topology_kind"] == (
        "graph_grid_dual_tower_projection"
    )
    assert payload["model_surface_contract"]["checkpoint_load_mode"] == "strict"
