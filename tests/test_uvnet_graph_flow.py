"""
Test suite for UV-Net Graph Model flow.

Verifies:
1. Model initialization (Dual-Path check).
2. Forward pass with dummy graph data.
3. Batch collation logic.
4. Training step execution.
"""

import sys
import unittest
from unittest.mock import MagicMock

import torch

# Add src to path
sys.path.append(".")

from src.ml.train.model import UVNetGraphModel
from src.ml.train.trainer import GraphBatchCollate, UVNetTrainer


class TestUVNetGraphFlow(unittest.TestCase):
    def setUp(self):
        self.node_dim = 12
        self.hidden_dim = 16
        self.embedding_dim = 32
        self.num_classes = 5
        self.model = UVNetGraphModel(
            node_input_dim=self.node_dim,
            hidden_dim=self.hidden_dim,
            embedding_dim=self.embedding_dim,
            num_classes=self.num_classes,
        )

    def test_model_initialization(self):
        """Test if model initializes layers correctly based on backend."""
        print(f"\nTesting Model Init (Backend: {'PyG' if self.model.has_pyg else 'Pure Torch'})...")
        self.assertIsInstance(self.model, UVNetGraphModel)
        # Check if config returns correct backend status
        config = self.model.get_config()
        self.assertEqual(config["node_input_dim"], self.node_dim)
        self.assertEqual(config["backend"], "pyg" if self.model.has_pyg else "pure_torch")

    def test_forward_pass_single_graph(self):
        """Test forward pass with a single random graph."""
        print("Testing Forward Pass (Single)...")
        num_nodes = 10
        num_edges = 20

        # Random features
        x = torch.randn(num_nodes, self.node_dim)
        # Random edge index [2, E]
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        # Batch vector (all zeros for single graph)
        batch = torch.zeros(num_nodes, dtype=torch.long)

        logits, embedding = self.model(x, edge_index, batch)

        self.assertEqual(logits.shape, (1, self.num_classes))
        self.assertEqual(embedding.shape, (1, self.embedding_dim))
        print("  -> Forward pass successful.")

    def test_batch_collation_and_training_step(self):
        """Test custom collation logic and a full training step."""
        print("Testing Batch Collation & Training Step...")
        
        # Create 3 dummy samples
        batch_size = 3
        dataset = []
        for _ in range(batch_size):
            num_nodes = torch.randint(5, 15, (1,)).item()
            num_edges = torch.randint(10, 30, (1,)).item()
            dataset.append({
                "x": torch.randn(num_nodes, self.node_dim),
                "edge_index": torch.randint(0, num_nodes, (2, num_edges)),
                "y": torch.randint(0, self.num_classes, (1,)).item()
            })

        # Collate
        collate_fn = GraphBatchCollate()
        batch_inputs, batch_targets = collate_fn(dataset)

        # Check Collation Output
        self.assertIn("x", batch_inputs)
        self.assertIn("edge_index", batch_inputs)
        self.assertIn("batch", batch_inputs)
        
        # Total nodes check
        total_nodes = sum(d["x"].size(0) for d in dataset)
        self.assertEqual(batch_inputs["x"].size(0), total_nodes)
        self.assertEqual(batch_inputs["batch"].size(0), total_nodes)
        self.assertEqual(batch_targets.size(0), batch_size)

        # Run Training Step through Trainer
        trainer = UVNetTrainer(self.model, device="cpu")
        # Mock dataloader that yields this one batch
        mock_loader = [(batch_inputs, batch_targets)]
        
        metrics = trainer.train_epoch(mock_loader)
        
        print(f"  -> Training metrics: {metrics}")
        self.assertIn("loss", metrics)
        self.assertIn("accuracy", metrics)
        self.assertIsNotNone(metrics["loss"])


if __name__ == "__main__":
    unittest.main()
