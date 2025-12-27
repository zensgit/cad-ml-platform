"""
ABC Dataset Loader.

Handles loading of STEP files from the ABC Dataset structure for training.
"""

import logging
import os
from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class ABCDataset(Dataset):
    """
    PyTorch Dataset for ABC CAD models.
    """

    def __init__(self, root_dir: str, transform=None):
        """
        Args:
            root_dir (str): Directory with STEP files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = []

        if os.path.exists(root_dir):
            self.file_list = [
                os.path.join(root_dir, f)
                for f in os.listdir(root_dir)
                if f.lower().endswith((".step", ".stp"))
            ]
        else:
            logger.warning(f"ABC Dataset root {root_dir} not found. Using empty set.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        step_path = self.file_list[idx]

        # Real Feature Extraction
        try:
            # Lazy import to avoid circular dependencies
            from src.core.geometry.engine import get_geometry_engine

            # Note: In a high-performance training loop, you might want to pre-process
            # these features offline and save them as .pt or .npy files to avoid
            # re-parsing STEP files (which is slow) every epoch.
            # For this MVP, we parse on-the-fly or assume a cached version exists.

            geo = get_geometry_engine()
            with open(step_path, "rb") as f:
                content = f.read()

            shape = geo.load_step(content, os.path.basename(step_path))
            if shape:
                feats = geo.extract_brep_features(shape)
                # Convert dict features to tensor
                # This is a simplified encoding. Real UV-Net extracts a graph.
                # Here we map our scalar features to a vector for the scaffold model.

                # Feature Vector Construction (Must match input_dim=12 in model.py)
                # [faces, edges, vertices, volume, area, plane_count, cyl_count, ...]

                surfaces = feats.get("surface_types", {})
                vector = [
                    float(feats.get("faces", 0)),
                    float(feats.get("edges", 0)),
                    float(feats.get("vertices", 0)),
                    float(feats.get("volume", 0)),
                    float(feats.get("surface_area", 0)),
                    float(surfaces.get("plane", 0)),
                    float(surfaces.get("cylinder", 0)),
                    float(surfaces.get("cone", 0)),
                    float(surfaces.get("sphere", 0)),
                    float(surfaces.get("torus", 0)),
                    float(surfaces.get("bspline", 0)),
                    float(feats.get("solids", 0)),
                ]

                # Normalize (Naive) - In prod use standard scaler
                vector = [torch.tensor(v).float() for v in vector]
                # Mocking point cloud shape (12, 1024)
                sample = torch.stack(vector).unsqueeze(1).repeat(1, 1024)

            else:
                # Fallback for failed parse
                sample = torch.zeros(12, 1024)

        except Exception as e:
            logger.error(f"Error processing {step_path}: {e}")
            sample = torch.zeros(12, 1024)

        # Mock Label: In reality, ABC dataset needs external labels or self-supervised task
        # Here we just use a dummy label for the pipeline to run
        label = torch.randint(0, 10, (1,)).item()

        if self.transform:
            sample = self.transform(sample)

        return sample, label


def get_dataloader(data_dir: str, batch_size: int = 32, shuffle: bool = True):
    dataset = ABCDataset(data_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
