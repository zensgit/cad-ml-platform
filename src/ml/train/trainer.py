"""
Trainer for UV-Net Graph Model.

Handles training loop with support for Graph Batching (Dual-Path).
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.ml.train.model import UVNetGraphModel

logger = logging.getLogger(__name__)

# Try to import PyG loader
HAS_PYG = False
try:
    from torch_geometric.loader import DataLoader as PyGDataLoader
    from torch_geometric.data import Data, Batch

    HAS_PYG = True
except ImportError:
    # We will define a custom collate_fn for pure torch
    pass


# --- Custom Collate for Pure PyTorch Fallback ---
class GraphBatchCollate:
    """
    Collates a list of graph data dictionaries/objects into a single batch
    by concatenating nodes and shifting edge indices.

    Contract: Input samples must have keys 'x', 'edge_index', 'label' (optional).
    """

    def __call__(self, batch_list: List[Any]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        # Initialize lists
        x_list = []
        edge_index_list = []
        batch_idx_list = []
        labels = []

        node_offset = 0

        for i, sample in enumerate(batch_list):
            # Extract fields (support both dict and object access)
            if isinstance(sample, dict):
                x = sample["x"]
                edge_index = sample["edge_index"]
                y = sample.get("y", None)
            else:
                # Assume attribute access (like PyG Data object or tuple)
                # This depends on how Dataset is implemented.
                # Assuming Dataset returns (data_dict, label) or similar
                if isinstance(sample, tuple):
                    x = sample[0]["x"]
                    edge_index = sample[0]["edge_index"]
                    y = sample[1]
                else:
                    x = sample.x
                    edge_index = sample.edge_index
                    y = sample.y

            num_nodes = x.size(0)

            # Append Node Features
            x_list.append(x)

            # Shift and Append Edge Indices
            # edge_index is [2, E]
            edge_index_shifted = edge_index + node_offset
            edge_index_list.append(edge_index_shifted)

            # Create Batch Index Vector [0, 0, 1, 1, 1, ...]
            batch_idx_list.append(torch.full((num_nodes,), i, dtype=torch.long))

            # Labels
            if y is not None:
                if isinstance(y, torch.Tensor):
                    labels.append(y)
                else:
                    labels.append(torch.tensor([y], dtype=torch.long))

            node_offset += num_nodes

        # Concatenate everything
        x_batch = torch.cat(x_list, dim=0)
        edge_index_batch = torch.cat(edge_index_list, dim=1)
        batch_vec = torch.cat(batch_idx_list, dim=0)
        y_batch = torch.cat(labels, dim=0) if labels else None

        return {
            "x": x_batch,
            "edge_index": edge_index_batch,
            "batch": batch_vec,
        }, y_batch


class UVNetTrainer:
    def __init__(
        self,
        model: UVNetGraphModel,
        device: str = "cpu",
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.criterion = nn.NLLLoss()  # Expecting LogSoftmax output
        self.history: Dict[str, List[float]] = {"loss": [], "acc": []}

    def train_epoch(self, dataloader: Any) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_data in dataloader:
            # Handle PyG vs Custom Collate structure
            if HAS_PYG and isinstance(dataloader, PyGDataLoader):
                # PyG Batch object
                batch_data = batch_data.to(self.device)
                x = batch_data.x
                edge_index = batch_data.edge_index
                batch_idx = batch_data.batch
                targets = batch_data.y
            else:
                # Custom Collate returns (inputs_dict, labels)
                inputs, targets = batch_data
                x = inputs["x"].to(self.device)
                edge_index = inputs["edge_index"].to(self.device)
                batch_idx = inputs["batch"].to(self.device)
                targets = targets.to(self.device)

            self.optimizer.zero_grad()

            # Forward
            log_probs, _ = self.model(x, edge_index, batch_idx)

            # Loss
            loss = self.criterion(log_probs, targets)
            loss.backward()
            self.optimizer.step()

            # Metrics
            total_loss += loss.item() * targets.size(0)
            preds = log_probs.argmax(dim=1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)

        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        self.history["loss"].append(avg_loss)
        self.history["acc"].append(accuracy)

        return {"loss": avg_loss, "accuracy": accuracy}

    def evaluate(self, dataloader: Any) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_data in dataloader:
                if HAS_PYG and isinstance(dataloader, PyGDataLoader):
                    batch_data = batch_data.to(self.device)
                    x = batch_data.x
                    edge_index = batch_data.edge_index
                    batch_idx = batch_data.batch
                    targets = batch_data.y
                else:
                    inputs, targets = batch_data
                    x = inputs["x"].to(self.device)
                    edge_index = inputs["edge_index"].to(self.device)
                    batch_idx = inputs["batch"].to(self.device)
                    targets = targets.to(self.device)

                log_probs, _ = self.model(x, edge_index, batch_idx)
                loss = self.criterion(log_probs, targets)

                total_loss += loss.item() * targets.size(0)
                preds = log_probs.argmax(dim=1)
                correct += preds.eq(targets).sum().item()
                total += targets.size(0)

        return {
            "val_loss": total_loss / total if total > 0 else 0.0,
            "val_accuracy": correct / total if total > 0 else 0.0,
        }

    def save_checkpoint(self, path: str):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "history": self.history,
                "config": self.model.get_config(),
            },
            path,
        )
        logger.info(f"Model saved to {path}")


def get_graph_dataloader(dataset, batch_size=32, shuffle=True):
    """Factory to get the correct DataLoader based on dependencies."""
    if HAS_PYG:
        return PyGDataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        # Use standard torch DataLoader with custom collate
        collate_fn = GraphBatchCollate()
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
        )