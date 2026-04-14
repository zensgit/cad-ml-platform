#!/usr/bin/env python3
"""Train a statistical-feature MLP classifier on cached graph data (B6.0c).

Extracts ~30 hand-crafted features from graph topology (no GNN needed):
  - Node/edge counts, density
  - Degree statistics (mean, max, std)
  - Node feature statistics (per-dim mean/std)
  - Edge attribute statistics (per-dim mean/std)

This MLP learns *different error patterns* from the GNN, making it
valuable as an ensemble member even at lower standalone accuracy (~85%).

Usage:
    python scripts/train_stat_mlp.py \
        --manifest data/graph_cache/cache_manifest.csv \
        --output models/stat_mlp_24class.pth \
        --epochs 100
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_stat_features(data: dict) -> torch.Tensor:
    """Extract ~30 statistical features from a cached graph dict.

    Returns a 1-D float tensor of fixed dimension.
    """
    x = data["x"]            # [N, node_dim]
    ei = data["edge_index"]  # [2, E]
    ea = data.get("edge_attr")  # [E, edge_dim] or None

    N = x.size(0)
    E = ei.size(1) if ei.numel() > 0 else 0
    node_dim = x.size(1)

    feats = []

    # 1. Basic counts (3)
    feats.append(float(N))
    feats.append(float(E))
    feats.append(E / max(N * (N - 1), 1))  # density

    # 2. Degree statistics (4) — vectorized (B6.2a)
    if E > 0 and N > 0:
        src_nodes = ei[0].clamp(0, N - 1)
        deg = torch.zeros(N).scatter_add_(0, src_nodes, torch.ones(E))
        feats.append(float(deg.mean()))
        feats.append(float(deg.max()))
        feats.append(float(deg.std(correction=0)) if N > 1 else 0.0)
        feats.append(float((deg == 0).sum()) / N)
    else:
        feats.extend([0.0, 0.0, 0.0, 1.0])

    # 3. Node feature statistics (node_dim * 2 = 38)
    if N > 1:
        feats.extend(x.mean(dim=0).tolist())
        feats.extend(x.std(dim=0, correction=0).tolist())
    elif N == 1:
        feats.extend(x[0].tolist())
        feats.extend([0.0] * node_dim)
    else:
        feats.extend([0.0] * node_dim * 2)

    # 4. Edge attribute statistics (edge_dim * 2 = 14)
    if ea is not None and ea.size(0) > 1:
        feats.extend(ea.mean(dim=0).tolist())
        feats.extend(ea.std(dim=0, correction=0).tolist())
    elif ea is not None and ea.size(0) == 1:
        feats.extend(ea[0].tolist())
        feats.extend([0.0] * 7)
    else:
        feats.extend([0.0] * 14)

    # 5. B6.3: Extended graph topology features (+20 dims)
    # 5a. Degree distribution quantiles (4)
    if E > 0 and N > 0:
        src_nodes = ei[0].clamp(0, N - 1)
        deg = torch.zeros(N).scatter_add_(0, src_nodes, torch.ones(E))
        sorted_deg = deg.sort().values
        feats.append(float(sorted_deg[int(0.25 * N)]))   # Q25
        feats.append(float(sorted_deg[int(0.75 * N)]))   # Q75
        feats.append(float(sorted_deg[int(0.90 * N)]))   # Q90
        feats.append(float((deg > deg.mean() + 2 * deg.std(correction=0)).sum()) / N if N > 1 else 0.0)  # hub ratio
    else:
        feats.extend([0.0, 0.0, 0.0, 0.0])

    # 5b. Node feature quantiles — per-dim Q25/Q75 (node_dim * 2 = 38... too many)
    # Instead: aggregate node feature range and skew (4)
    if N > 1:
        x_range = (x.max(dim=0).values - x.min(dim=0).values)
        feats.append(float(x_range.mean()))        # avg feature range
        feats.append(float(x_range.max()))         # max feature range
        x_nz = (x.abs() > 1e-6).float().mean(dim=0)  # non-zero ratio per dim
        feats.append(float(x_nz.mean()))           # avg non-zero ratio
        feats.append(float((x_nz > 0.5).sum()))    # num dims with >50% non-zero
    else:
        feats.extend([0.0, 0.0, 0.0, 0.0])

    # 5c. Edge attribute range stats (4)
    if ea is not None and ea.size(0) > 1:
        ea_range = ea.max(dim=0).values - ea.min(dim=0).values
        feats.append(float(ea_range.mean()))
        feats.append(float(ea_range.max()))
        ea_nz = (ea.abs() > 1e-6).float().mean(dim=0)
        feats.append(float(ea_nz.mean()))
        feats.append(float((ea_nz > 0.5).sum()))
    else:
        feats.extend([0.0, 0.0, 0.0, 0.0])

    # 5d. Graph structure ratios (4)
    feats.append(float(E / max(N, 1)))              # edges per node
    feats.append(float(N) / 200.0)                  # node utilization (max_nodes=200)
    if E > 0 and N > 1:
        # Self-loop ratio
        self_loops = (ei[0] == ei[1]).sum().item()
        feats.append(float(self_loops) / max(E, 1))
        # Bidirectional edge estimate
        ei_set = set()
        bi_count = 0
        for j in range(min(E, 1000)):  # cap for speed
            s, t = ei[0, j].item(), ei[1, j].item()
            if (t, s) in ei_set:
                bi_count += 1
            ei_set.add((s, t))
        feats.append(float(bi_count) / max(len(ei_set), 1))
    else:
        feats.extend([0.0, 0.0])

    # 5e. Connected component estimate (4)
    # Simple BFS-based component count (fast for small graphs)
    if N > 0 and E > 0:
        visited = [False] * N
        adj = [[] for _ in range(N)]
        for j in range(min(E, 2000)):  # cap for speed
            s, t = ei[0, j].item(), ei[1, j].item()
            if 0 <= s < N and 0 <= t < N:
                adj[s].append(t)
                adj[t].append(s)
        components = 0
        max_comp = 0
        for start in range(N):
            if visited[start]:
                continue
            components += 1
            stack = [start]
            comp_size = 0
            while stack:
                node = stack.pop()
                if visited[node]:
                    continue
                visited[node] = True
                comp_size += 1
                stack.extend(adj[node])
            max_comp = max(max_comp, comp_size)
        feats.append(float(components))
        feats.append(float(max_comp) / max(N, 1))  # largest component ratio
        feats.append(float(components) / max(N, 1))  # fragmentation
        feats.append(1.0 if components == 1 else 0.0)  # is_connected
    else:
        feats.extend([0.0, 0.0, 0.0, 0.0])

    return torch.tensor(feats, dtype=torch.float32)


STAT_FEAT_DIM = 3 + 4 + 19 * 2 + 7 * 2 + 4 + 4 + 4 + 4 + 4  # = 79


# ── Dataset ───────────────────────────────────────────────────────────────────

class StatFeatureDataset(Dataset):
    def __init__(self, manifest_csv: str):
        self.samples: List[Tuple[str, int]] = []
        self.label_map: Dict[str, int] = {}

        with open(manifest_csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                label = (row.get("taxonomy_v2_class") or row.get("label") or "").strip()
                cache_path = row.get("cache_path", "").strip()
                if not label or not cache_path:
                    continue
                if label not in self.label_map:
                    self.label_map[label] = len(self.label_map)
                self.samples.append((cache_path, self.label_map[label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label_idx = self.samples[idx]
        try:
            data = torch.load(path, map_location="cpu", weights_only=True)
            feats = extract_stat_features(data)
        except Exception:
            feats = torch.zeros(STAT_FEAT_DIM)
        return feats, label_idx


# ── Model ─────────────────────────────────────────────────────────────────────

class StatMLP(nn.Module):
    """3-layer MLP for graph statistical feature classification."""

    def __init__(self, input_dim: int = STAT_FEAT_DIM, hidden_dim: int = 128,
                 num_classes: int = 24, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ── Training ──────────────────────────────────────────────────────────────────

def train(manifest: str, output: str, epochs: int = 100, lr: float = 1e-3,
          batch_size: int = 64, patience: int = 15):
    dataset = StatFeatureDataset(manifest)
    num_classes = len(dataset.label_map)
    n = len(dataset)
    n_val = max(1, int(0.2 * n))
    train_ds, val_ds = random_split(dataset, [n - n_val, n_val],
                                     generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = StatMLP(input_dim=STAT_FEAT_DIM, num_classes=num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Compute feature mean/std for normalization
    logger.info("Computing feature statistics for normalization...")
    all_feats = []
    for feats, _ in train_loader:
        all_feats.append(feats)
    all_feats = torch.cat(all_feats, dim=0)
    feat_mean = all_feats.mean(dim=0)
    feat_std = all_feats.std(dim=0).clamp(min=1e-6)
    logger.info("Feature stats: mean range [%.2f, %.2f], std range [%.4f, %.2f]",
                feat_mean.min(), feat_mean.max(), feat_std.min(), feat_std.max())

    def normalize(x):
        return (x - feat_mean) / feat_std

    best_val_acc = 0.0
    no_improve = 0

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_correct = train_total = 0
        for feats, labels in train_loader:
            optimizer.zero_grad()
            logits = model(normalize(feats))
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += len(labels)
        scheduler.step()

        # Val
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for feats, labels in val_loader:
                logits = model(normalize(feats))
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += len(labels)

        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)

        improved = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            improved = " ✓"
            torch.save({
                "model_state": model.state_dict(),
                "label_map": dataset.label_map,
                "input_dim": STAT_FEAT_DIM,
                "hidden_dim": 128,
                "num_classes": num_classes,
                "best_val_acc": best_val_acc,
                "feat_mean": feat_mean,
                "feat_std": feat_std,
            }, output)
        else:
            no_improve += 1

        if epoch % 5 == 0 or epoch <= 3 or improved:
            print(f"  {epoch:>3d}  train={train_acc:.3f}  val={val_acc:.3f}{improved}")

        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch} (patience={patience})")
            break

    print(f"\nBest val acc: {best_val_acc*100:.1f}%")
    print(f"Saved to: {output}")
    return best_val_acc


def main():
    parser = argparse.ArgumentParser(description="Train stat-feature MLP (B6.0c)")
    parser.add_argument("--manifest", default="data/graph_cache/cache_manifest.csv")
    parser.add_argument("--output", default="models/stat_mlp_24class.pth")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    train(args.manifest, args.output, args.epochs, args.lr, args.batch_size, args.patience)


if __name__ == "__main__":
    main()
