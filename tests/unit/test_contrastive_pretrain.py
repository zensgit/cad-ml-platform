"""Tests for contrastive pretraining pipeline (B3).

Tests NT-Xent loss, projection head, encoder, checkpoint save/load, and
dry-run mode.
"""

from __future__ import annotations

import os
import tempfile

import torch
import pytest

from src.ml.train.model_2d import (
    EdgeGraphSageClassifier,
    GraphEncoder,
    ProjectionHead,
    SimpleGraphClassifier,
    nt_xent_loss,
)


# --------------------------------------------------------------------------- #
# NT-Xent Loss
# --------------------------------------------------------------------------- #

class TestNTXentLoss:
    def test_similar_views_lower_loss(self):
        """Loss should be lower for similar views than random ones."""
        z = torch.randn(8, 128)
        z_similar = z + torch.randn_like(z) * 0.01  # Small perturbation
        z_random = torch.randn(8, 128)
        loss_similar = nt_xent_loss(z, z_similar, temperature=0.5)
        loss_random = nt_xent_loss(z, z_random, temperature=0.5)
        assert loss_similar.item() < loss_random.item()

    def test_random_views_higher_loss(self):
        """Loss should be higher for random (uncorrelated) views."""
        z1 = torch.randn(16, 128)
        z2 = torch.randn(16, 128)
        loss = nt_xent_loss(z1, z2, temperature=0.5)
        assert loss.item() > 0.0

    def test_batch_size_one(self):
        """Should handle batch size of 1."""
        z1 = torch.randn(1, 128)
        z2 = torch.randn(1, 128)
        loss = nt_xent_loss(z1, z2, temperature=0.5)
        assert not torch.isnan(loss)

    def test_empty_batch(self):
        """Should return 0 for empty batch."""
        z1 = torch.zeros(0, 128)
        z2 = torch.zeros(0, 128)
        loss = nt_xent_loss(z1, z2, temperature=0.5)
        assert loss.item() == 0.0

    def test_temperature_effect(self):
        """Lower temperature should produce sharper distributions."""
        z1 = torch.randn(8, 128)
        z2 = torch.randn(8, 128)
        loss_high_temp = nt_xent_loss(z1, z2, temperature=1.0)
        loss_low_temp = nt_xent_loss(z1, z2, temperature=0.1)
        # Different temperatures should give different losses
        assert loss_high_temp.item() != loss_low_temp.item()

    def test_gradient_flows(self):
        """Gradients should flow through the loss."""
        z1 = torch.randn(4, 128, requires_grad=True)
        z2 = torch.randn(4, 128, requires_grad=True)
        loss = nt_xent_loss(z1, z2, temperature=0.5)
        loss.backward()
        assert z1.grad is not None
        assert z2.grad is not None


# --------------------------------------------------------------------------- #
# ProjectionHead
# --------------------------------------------------------------------------- #

class TestProjectionHead:
    def test_forward_shape(self):
        head = ProjectionHead(input_dim=64, hidden_dim=64, output_dim=128)
        x = torch.randn(8, 64)
        out = head(x)
        assert out.shape == (8, 128)

    def test_single_sample(self):
        head = ProjectionHead(input_dim=64, hidden_dim=64, output_dim=128)
        x = torch.randn(1, 64)
        out = head(x)
        assert out.shape == (1, 128)

    def test_gradient_flows(self):
        head = ProjectionHead(input_dim=64, hidden_dim=64, output_dim=128)
        x = torch.randn(4, 64, requires_grad=True)
        out = head(x)
        out.sum().backward()
        assert x.grad is not None


# --------------------------------------------------------------------------- #
# GraphEncoder
# --------------------------------------------------------------------------- #

class TestGraphEncoder:
    def test_edge_sage_forward(self):
        enc = GraphEncoder(node_dim=19, edge_dim=7, hidden_dim=64, model_type="edge_sage")
        x = torch.randn(10, 19)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        edge_attr = torch.randn(3, 7)
        out = enc(x, edge_index, edge_attr=edge_attr)
        assert out.shape == (1, 64)  # Single graph, mean-pooled

    def test_gcn_forward(self):
        enc = GraphEncoder(node_dim=19, hidden_dim=64, model_type="gcn")
        x = torch.randn(10, 19)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        out = enc(x, edge_index)
        assert out.shape == (1, 64)

    def test_batched_forward(self):
        enc = GraphEncoder(node_dim=19, edge_dim=7, hidden_dim=64, model_type="edge_sage")
        # Two graphs: 5 nodes + 3 nodes
        x = torch.randn(8, 19)
        edge_index = torch.tensor([[0, 1, 5, 6], [1, 2, 6, 7]], dtype=torch.long)
        edge_attr = torch.randn(4, 7)
        batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1], dtype=torch.long)
        out = enc(x, edge_index, edge_attr=edge_attr, batch=batch)
        assert out.shape == (2, 64)  # 2 graphs

    def test_empty_graph(self):
        enc = GraphEncoder(node_dim=19, edge_dim=7, hidden_dim=64, model_type="edge_sage")
        x = torch.zeros(0, 19)
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, 7)
        out = enc(x, edge_index, edge_attr=edge_attr)
        # Empty graph: EdgeSageLayer returns x as-is (0 nodes), pooling returns [1, in_dim]
        # Just verify it doesn't crash and returns a 2D tensor
        assert out.dim() == 2


# --------------------------------------------------------------------------- #
# Checkpoint save/load
# --------------------------------------------------------------------------- #

class TestCheckpointSaveLoad:
    def test_save_and_load_encoder(self):
        enc = GraphEncoder(node_dim=19, edge_dim=7, hidden_dim=64, model_type="edge_sage")
        proj = ProjectionHead(64, 64, 128)

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            path = f.name

        try:
            # Save
            torch.save({
                "encoder_state_dict": enc.state_dict(),
                "proj_head_state_dict": proj.state_dict(),
                "node_dim": 19,
                "edge_dim": 7,
                "hidden_dim": 64,
                "model_type": "edge_sage",
            }, path)

            # Load into new encoder
            enc2 = GraphEncoder(node_dim=19, edge_dim=7, hidden_dim=64, model_type="edge_sage")
            ckpt = torch.load(path, map_location="cpu")
            enc2.load_state_dict(ckpt["encoder_state_dict"])

            # Should produce same output
            x = torch.randn(5, 19)
            ei = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
            ea = torch.randn(2, 7)
            enc.eval()
            enc2.eval()
            with torch.no_grad():
                out1 = enc(x, ei, edge_attr=ea)
                out2 = enc2(x, ei, edge_attr=ea)
            assert torch.allclose(out1, out2, atol=1e-6)
        finally:
            os.unlink(path)

    def test_load_pretrained_into_classifier(self):
        """Pretrained encoder weights should load into EdgeGraphSageClassifier."""
        enc = GraphEncoder(node_dim=19, edge_dim=7, hidden_dim=64, model_type="edge_sage")

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            path = f.name

        try:
            torch.save({
                "encoder_state_dict": enc.state_dict(),
                "node_dim": 19,
                "edge_dim": 7,
                "hidden_dim": 64,
                "model_type": "edge_sage",
            }, path)

            clf = EdgeGraphSageClassifier(19, 7, 64, 10)
            clf.load_pretrained(path)

            # Verify the sage layer weights match after loading
            for key in ["sage1.msg.weight", "sage1.self_lin.weight",
                        "sage2.msg.weight", "sage2.self_lin.weight"]:
                assert torch.equal(
                    enc.state_dict()[key],
                    clf.state_dict()[key],
                ), f"Mismatch in {key}"
        finally:
            os.unlink(path)


# --------------------------------------------------------------------------- #
# get_encoder
# --------------------------------------------------------------------------- #

class TestGetEncoder:
    def test_simple_classifier_get_encoder(self):
        clf = SimpleGraphClassifier(19, 64, 10)
        enc = clf.get_encoder()
        assert isinstance(enc, GraphEncoder)
        assert enc.model_type == "gcn"

    def test_edge_sage_classifier_get_encoder(self):
        clf = EdgeGraphSageClassifier(19, 7, 64, 10)
        enc = clf.get_encoder()
        assert isinstance(enc, GraphEncoder)
        assert enc.model_type == "edge_sage"


# --------------------------------------------------------------------------- #
# Dry-run mode (integration-ish)
# --------------------------------------------------------------------------- #

class TestDryRunMode:
    def test_pretrain_dry_run(self):
        """Verify pretraining script runs in dry-run mode."""
        from scripts.pretrain_graph2d_contrastive import (
            UnlabeledDXFGraphDataset,
            collate_graphs,
            pretrain,
        )

        dataset = UnlabeledDXFGraphDataset(dry_run=True, dry_run_size=8)
        assert len(dataset) == 8

        loader = torch.utils.data.DataLoader(
            dataset, batch_size=4, collate_fn=collate_graphs
        )
        enc = GraphEncoder(node_dim=19, edge_dim=7, hidden_dim=32, model_type="edge_sage")
        proj = ProjectionHead(32, 32, 128)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "test_pretrained.pth")
            result = pretrain(
                enc, proj, loader,
                epochs=2, lr=0.001, temperature=0.5,
                output_path=out_path,
                node_dim=19, edge_dim=7, hidden_dim=32,
            )
            assert os.path.exists(out_path)
            assert result["best_loss"] >= 0

    def test_finetune_dry_run(self):
        """Verify finetuning script runs in dry-run mode."""
        from scripts.finetune_graph2d_from_pretrained import (
            FinetuneDataset,
            finetune,
        )

        dataset = FinetuneDataset(dry_run=True, dry_run_size=16, dry_run_num_classes=5)
        assert len(dataset) == 16

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "test_finetuned.pth")
            result = finetune(
                pretrained_path=None,
                dataset=dataset,
                epochs=3,
                batch_size=8,
                patience=2,
                output_path=out_path,
                node_dim=19, edge_dim=7, hidden_dim=32,
                model_type="edge_sage",
            )
            assert os.path.exists(out_path)
            assert result["num_classes"] == 5
