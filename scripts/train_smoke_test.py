"""
Smoke Test for UV-Net Training on Apple Silicon (M4 Pro).
Runs a minimal training loop to verify MPS acceleration and Graph Data flow.
"""

import os
import sys
import time
import torch
from torch.utils.data import Dataset

# Add src to path
sys.path.append(".")

from src.ml.train.model import UVNetGraphModel
from src.ml.train.trainer import UVNetTrainer, get_graph_dataloader
from src.ml.utils import get_best_device

try:
    from src.core.geometry.engine import BREP_GRAPH_EDGE_FEATURES, BREP_GRAPH_NODE_FEATURES
except Exception:
    BREP_GRAPH_NODE_FEATURES = tuple()
    BREP_GRAPH_EDGE_FEATURES = tuple()

DEFAULT_NODE_DIM = len(BREP_GRAPH_NODE_FEATURES) or 15
DEFAULT_NODE_SCHEMA = BREP_GRAPH_NODE_FEATURES or None
DEFAULT_EDGE_SCHEMA = BREP_GRAPH_EDGE_FEATURES or None

class MockGraphDataset(Dataset):
    """Generates synthetic graph data to test the pipeline without external files."""
    def __init__(self, num_samples=100, node_dim=DEFAULT_NODE_DIM):
        self.num_samples = num_samples
        self.node_dim = node_dim

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Create a random graph
        num_nodes = torch.randint(10, 50, (1,)).item()
        num_edges = torch.randint(20, 100, (1,)).item()
        
        x = torch.randn(num_nodes, self.node_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        label = torch.randint(0, 10, (1,)).item()
        
        # Return graph data dict and label
        return {"x": x, "edge_index": edge_index}, label

def run_smoke_test():
    print("🛠️  Starting CAD-ML Smoke Test on M4 Pro...")
    
    device = get_best_device()
    print(f"📍 Device detected: {device.upper()}")
    
    if device != "mps":
        print("⚠️  Warning: MPS not detected. Training will be slow on CPU.")

    # 1. Initialize Model (UV-Net Graph Architecture)
    node_dim = DEFAULT_NODE_DIM  # Matches GeometryEngine.BREP_GRAPH_NODE_FEATURES
    num_classes = 10
    model = UVNetGraphModel(
        node_input_dim=node_dim,
        num_classes=num_classes,
        node_schema=DEFAULT_NODE_SCHEMA,
        edge_schema=DEFAULT_EDGE_SCHEMA,
    )
    
    # 2. Initialize Trainer
    trainer = UVNetTrainer(model, device=device, learning_rate=0.01)
    
    # 3. Create Data Loader
    # In real use, this would be ABCDataset pointing to your STEP files
    dataset = MockGraphDataset(num_samples=200, node_dim=node_dim)
    dataloader = get_graph_dataloader(dataset, batch_size=32, shuffle=True)
    
    print(f"📊 Dataset size: {len(dataset)} samples")
    print(f"🚀 Training for 5 epochs...")
    
    start_time = time.time()
    
    for epoch in range(1, 6):
        epoch_start = time.time()
        metrics = trainer.train_epoch(dataloader)
        elapsed = time.time() - epoch_start
        
        print(f"Epoch {epoch}/5 | Loss: {metrics['loss']:.4f} | Acc: {metrics['accuracy']:.2%} | Time: {elapsed:.2f}s")
    
    total_time = time.time() - start_time
    print(f"\n✅ Smoke test complete in {total_time:.2f} seconds.")
    
    # Save a test checkpoint
    os.makedirs("models", exist_ok=True)
    checkpoint_path = "models/smoke_test_model.pth"
    trainer.save_checkpoint(checkpoint_path)
    print(f"💾 Mock model saved to {checkpoint_path}")

if __name__ == "__main__":
    run_smoke_test()
