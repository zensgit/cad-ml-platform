"""
2D DXF Training Script (Smoke Test).

Trains a GNN to detect if a "Slot" feature exists in a DXF drawing.
Uses synthetic data generated on-the-fly.
"""

import sys
import os
import time
import torch

# Add src to path
sys.path.append(".")

from src.ml.train.model import UVNetGraphModel
from src.ml.train.trainer import UVNetTrainer, get_graph_dataloader
from src.ml.train.dataset_2d import DXFDataset, DXF_NODE_DIM
from scripts.generate_synthetic_dxf_dataset import DxfGenerator
from src.ml.utils import get_best_device

def run_2d_training():
    print("🎨 Starting 2D DXF Feature Recognition Training...")
    
    device = get_best_device()
    print(f"📍 Device: {device.upper()}")

    # 1. Generate Data
    data_dir = "data/synthetic_dxf_train"
    if not os.path.exists(data_dir):
        print(f"⚙️  Generating synthetic data in {data_dir}...")
        gen = DxfGenerator(data_dir)
        # Generate enough data for convergence
        gen.generate_batch(500) 
    else:
        print(f"📂 Using existing data in {data_dir}")

    # 2. Setup Dataset & Loader
    dataset = DXFDataset(data_dir)
    # Filter out empty/failed parses
    dataset.samples = [s for s in dataset.samples if os.path.exists(os.path.join(data_dir, s["file"]))]
    
    # 80/20 Split
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = get_graph_dataloader(train_set, batch_size=32, shuffle=True)
    test_loader = get_graph_dataloader(test_set, batch_size=32, shuffle=False)
    
    print(f"📊 Data: {len(train_set)} train, {len(test_set)} test")

    # 3. Model (Reuse UVNet GNN architecture)
    # Task: Binary Classification (Has Slot vs No Slot)
    model = UVNetGraphModel(
        node_input_dim=DXF_NODE_DIM, 
        hidden_dim=32,
        embedding_dim=64,
        num_classes=2  # 0: No Slot, 1: Has Slot
    )
    
    trainer = UVNetTrainer(model, device=device, learning_rate=0.005)

    # 4. Train Loop
    epochs = 10
    print(f"🚀 Training for {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
        # Train
        train_metrics = trainer.train_epoch(train_loader)
        
        # Validation
        val_metrics = trainer.evaluate(test_loader)
        
        print(f"Epoch {epoch:02d} | "
              f"Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.2%} | "
              f"Val Loss: {val_metrics['val_loss']:.4f} | Val Acc: {val_metrics['val_accuracy']:.2%}")

    # 5. Save
    os.makedirs("models/2d", exist_ok=True)
    save_path = "models/2d/dxf_slot_detector.pth"
    trainer.save_checkpoint(save_path)
    print(f"💾 Model saved to {save_path}")

if __name__ == "__main__":
    run_2d_training()
