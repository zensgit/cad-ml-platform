"""Tests for the UV-Net checkpoint inspector against the REAL model contract.

These tests build checkpoints from the actual ``UVNetGraphModel`` (no fictional
grid parameters, no never-implemented compat APIs — see the 33cb0f65 false-green
diagnosis on PR #523) and assert:

* a valid checkpoint strict-loads and forward-passes end to end (exit 0);
* the JSON summary reports only real facts (config / strict-load / shapes);
* a checkpoint that does NOT strictly match the current architecture exits
  non-zero (the mismatch is the finding);
* a checkpoint missing its state_dict, or a missing file, exits non-zero.

``importorskip`` keeps torch-less lanes green, but the required self-hosted
lane runs an explicit ``python -c "import torch"`` gate before executing this
file (ci-tiered-tests.yml), so it can never silently skip there again.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from scripts.uvnet_checkpoint_inspect import (  # noqa: E402
    main as uvnet_checkpoint_inspect_main,
)
from src.ml.train.model import UVNetGraphModel  # noqa: E402


def _small_model() -> UVNetGraphModel:
    # Real constructor surface only.
    return UVNetGraphModel(
        node_input_dim=4,
        edge_input_dim=2,
        hidden_dim=8,
        embedding_dim=16,
        num_classes=3,
        dropout_rate=0.0,
        use_edge_attr=True,
    )


def _save_checkpoint(model: UVNetGraphModel, path: Path) -> None:
    torch.save(
        {"config": model.get_config(), "model_state_dict": model.state_dict()},
        path,
    )


def test_valid_checkpoint_strict_loads_and_forward_passes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model = _small_model()
    checkpoint_path = tmp_path / "uvnet_inspect.pth"
    _save_checkpoint(model, checkpoint_path)

    rc = uvnet_checkpoint_inspect_main(["--path", str(checkpoint_path), "--nodes", "3"])

    assert rc == 0
    output = capsys.readouterr().out
    assert "Strict load: ok" in output
    assert "Logits shape: (1, 3)" in output
    assert "Embedding shape: (1, 16)" in output


def test_summary_json_reports_only_real_facts(tmp_path: Path) -> None:
    model = _small_model()
    checkpoint_path = tmp_path / "uvnet_inspect_summary.pth"
    summary_path = tmp_path / "uvnet_inspect_summary.json"
    _save_checkpoint(model, checkpoint_path)

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
    assert payload["strict_load"] == {"mode": "strict", "ok": True}
    assert payload["forward_shapes"]["logits"] == [1, 3]
    assert payload["forward_shapes"]["embedding"] == [1, 16]
    # Config is echoed verbatim from the checkpoint (real constructor facts).
    assert payload["config"]["num_classes"] == 3
    assert payload["config"]["embedding_dim"] == 16
    # No fictional grid-capability claims may reappear in the summary.
    assert not any("grid" in key for key in payload)
    assert not any("grid" in key for key in payload["config"])


def test_architecture_mismatch_fails_strict_load_nonzero(tmp_path: Path) -> None:
    # State dict from a DIFFERENT architecture (hidden_dim/num_classes differ):
    # strict load must fail and the tool must exit non-zero — no compat shim.
    donor = UVNetGraphModel(
        node_input_dim=4,
        edge_input_dim=2,
        hidden_dim=32,
        embedding_dim=64,
        num_classes=7,
        dropout_rate=0.0,
    )
    target_config = _small_model().get_config()
    checkpoint_path = tmp_path / "uvnet_inspect_mismatch.pth"
    torch.save(
        {"config": target_config, "model_state_dict": donor.state_dict()},
        checkpoint_path,
    )

    rc = uvnet_checkpoint_inspect_main(["--path", str(checkpoint_path), "--nodes", "3"])

    assert rc == 1


def test_missing_state_dict_nonzero(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "uvnet_inspect_nostate.pth"
    torch.save({"config": _small_model().get_config()}, checkpoint_path)

    rc = uvnet_checkpoint_inspect_main(["--path", str(checkpoint_path)])

    assert rc == 1


def test_missing_file_nonzero(tmp_path: Path) -> None:
    rc = uvnet_checkpoint_inspect_main(["--path", str(tmp_path / "does_not_exist.pth")])

    assert rc == 1


def test_non_dict_checkpoint_nonzero(tmp_path: Path) -> None:
    # A .pth that is not a dict (e.g. a bare tensor) must fail cleanly, not traceback.
    checkpoint_path = tmp_path / "uvnet_inspect_nondict.pth"
    torch.save(torch.zeros(3), checkpoint_path)

    rc = uvnet_checkpoint_inspect_main(["--path", str(checkpoint_path)])

    assert rc == 1
