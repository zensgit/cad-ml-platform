import pytest

from src.ml.vision_3d import HAS_TORCH

if HAS_TORCH:
    import torch

    from src.ml.train.model import UVNetGraphModel
    from src.ml.train.trainer import UVNetTrainer


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
def test_trainer_rejects_node_dim_mismatch() -> None:
    model = UVNetGraphModel(node_input_dim=3, num_classes=2)
    trainer = UVNetTrainer(model, device="cpu")

    inputs = {
        "x": torch.randn(4, 5),
        "edge_index": torch.zeros((2, 0), dtype=torch.long),
        "batch": torch.zeros(4, dtype=torch.long),
    }
    targets = torch.tensor([0])

    with pytest.raises(ValueError, match="Node feature dim mismatch"):
        trainer.train_epoch([(inputs, targets)])
