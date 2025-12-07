#!/usr/bin/env python3
"""
Hybrid 3D Model Training Script (Phase 5).

Trains a hybrid network that combines:
1. v7 Features (160-dim): Geometric + Visual
2. PointNet Features (1024-dim): Native 3D Point Cloud

Architecture:
[v7 Input (160)] --> [MLP] --> [Emb A (64)]
[3D Input (N,3)] --> [PointNet] --> [Emb B (64)]
[Fusion] --> Concatenate(Emb A, Emb B) --> [Final MLP] --> [Class Logits]
"""

import argparse
import logging
import os
import sys
import numpy as np
from typing import List, Tuple, Optional

# Add project root to path
sys.path.append(os.getcwd())

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("train_hybrid_3d")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from src.core.deep_3d.pointnet_arch import PointNetPP
except ImportError:
    logger.error("PyTorch is required for this script.")
    sys.exit(1)

class HybridDataset(Dataset):
    """
    Dataset loading both v7 features and 3D point clouds.
    Assumes data is organized as:
    data_dir/
      features/ {id}.npy  (160-dim vector)
      points/   {id}.npy  (N, 3 point cloud)
      labels.csv          (id, label_idx)
    """
    def __init__(self, data_dir: str, num_points: int = 2048):
        self.data_dir = data_dir
        self.num_points = num_points
        self.samples = [] # List of (id, label)
        
        labels_path = os.path.join(data_dir, "labels.csv")
        if os.path.exists(labels_path):
            with open(labels_path, "r") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) >= 2:
                        self.samples.append((parts[0], int(parts[1])))
        else:
            logger.warning(f"No labels.csv found in {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sid, label = self.samples[idx]
        
        # Load v7 features
        feat_path = os.path.join(self.data_dir, "features", f"{sid}.npy")
        if os.path.exists(feat_path):
            v7_feat = np.load(feat_path).astype(np.float32)
        else:
            v7_feat = np.zeros(160, dtype=np.float32)
            
        # Load Point Cloud
        pc_path = os.path.join(self.data_dir, "points", f"{sid}.npy")
        if os.path.exists(pc_path):
            points = np.load(pc_path).astype(np.float32)
            # Resample if needed
            if len(points) > self.num_points:
                indices = np.random.choice(len(points), self.num_points, replace=False)
                points = points[indices]
            elif len(points) < self.num_points:
                # Pad with zeros or repeat
                indices = np.random.choice(len(points), self.num_points, replace=True)
                points = points[indices]
        else:
            points = np.zeros((self.num_points, 3), dtype=np.float32)
            
        # Transpose points for PointNet (N, 3) -> (3, N)
        points = points.T
            
        return {
            "v7": torch.from_numpy(v7_feat),
            "points": torch.from_numpy(points),
            "label": torch.tensor(label, dtype=torch.long)
        }

class HybridNetwork(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        
        # Branch 1: v7 Features
        self.v7_mlp = nn.Sequential(
            nn.Linear(160, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64)
        )
        
        # Branch 2: PointNet
        self.pointnet = PointNetPP(feature_dim=64) # Output 64 dim embedding
        
        # Fusion
        self.classifier = nn.Sequential(
            nn.Linear(64 + 64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, v7, points):
        emb_v7 = self.v7_mlp(v7)
        emb_3d = self.pointnet(points)
        
        # Concatenate
        combined = torch.cat([emb_v7, emb_3d], dim=1)
        
        return self.classifier(combined)

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create dummy dataset if dry run
    if args.dry_run and not os.path.exists(os.path.join(args.data_dir, "labels.csv")):
        logger.info("Dry run with dummy data...")
        dataset = [] # Dummy
        # We'll mock the dataloader loop
    else:
        dataset = HybridDataset(args.data_dir)
        
    if len(dataset) == 0 and not args.dry_run:
        logger.error("Dataset empty. Exiting.")
        return

    # Model
    num_classes = 10 # Example
    model = HybridNetwork(num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    
    epochs = args.epochs
    
    if args.dry_run:
        logger.info("Dry run: Simulating 1 epoch...")
        # Simulate forward pass
        v7 = torch.randn(4, 160).to(device)
        points = torch.randn(4, 3, 2048).to(device)
        labels = torch.tensor([0, 1, 0, 1]).to(device)
        
        optimizer.zero_grad()
        outputs = model(v7, points)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        logger.info(f"Simulated Loss: {loss.item():.4f}")
        return

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            v7 = batch["v7"].to(device)
            points = batch["points"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            outputs = model(v7, points)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        
    # Save
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(model.state_dict(), args.output_path)
    logger.info(f"Model saved to {args.output_path}")

def main():
    parser = argparse.ArgumentParser(description="Train Hybrid 3D Model")
    parser.add_argument("--data-dir", default="data/training_3d", help="Path to training data")
    parser.add_argument("--output-path", default="models/hybrid_3d/best_model.pth", help="Output model path")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true", help="Run simulation only")
    
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()
