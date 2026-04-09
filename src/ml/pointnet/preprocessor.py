"""
Point Cloud Preprocessor.

Handles loading, normalisation, and augmentation of 3D point cloud data
from STL, OBJ, PLY, and XYZ files. Uses trimesh for mesh-based formats
and numpy for all mathematical operations.
"""

import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

HAS_TRIMESH = False
try:
    import trimesh

    HAS_TRIMESH = True
except ImportError:
    logger.warning("trimesh not found. Mesh loading (STL/OBJ/PLY) will be unavailable.")

HAS_TORCH = False
try:
    import torch

    HAS_TORCH = True
except ImportError:
    pass


class PointCloudPreprocessor:
    """Load, preprocess, normalise, and augment 3D point clouds.

    Supports STL, OBJ, PLY (mesh or point cloud), and raw XYZ text files.
    All loaders return a fixed-size (num_points, 3) numpy array.
    """

    SUPPORTED_FORMATS = [".stl", ".obj", ".ply", ".xyz"]

    def __init__(self, num_points: int = 2048, normalize_default: bool = True):
        """Initialise preprocessor.

        Args:
            num_points: Default number of points to sample/pad to.
            normalize_default: Whether to normalise points by default when loading.
        """
        self.num_points = num_points
        self.normalize_default = normalize_default

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    def load(self, file_path: str, num_points: Optional[int] = None) -> np.ndarray:
        """Load a point cloud from any supported file format.

        Args:
            file_path: Path to the 3D file.
            num_points: Override default point count.

        Returns:
            (num_points, 3) float64 numpy array.

        Raises:
            ValueError: If the file extension is not supported.
            FileNotFoundError: If the file does not exist.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        n = num_points or self.num_points

        loaders = {
            ".stl": self.load_from_stl,
            ".obj": self.load_from_obj,
            ".ply": self.load_from_ply,
            ".xyz": self.load_from_xyz,
        }

        loader = loaders.get(ext)
        if loader is None:
            raise ValueError(
                f"Unsupported file format: {ext}. "
                f"Supported formats: {self.SUPPORTED_FORMATS}"
            )
        return loader(file_path, num_points=n)

    def load_from_stl(self, file_path: str, num_points: int = 2048) -> np.ndarray:
        """Load an STL file and sample points from its mesh surface.

        Args:
            file_path: Path to the STL file.
            num_points: Number of points to sample.

        Returns:
            (num_points, 3) numpy array of sampled surface points.
        """
        if not HAS_TRIMESH:
            raise ImportError(
                "trimesh is required for STL loading. Install with: pip install trimesh"
            )
        mesh = trimesh.load(file_path, force="mesh")
        points = self._sample_mesh(mesh, num_points)
        if self.normalize_default:
            points = self.normalize(points)
        return points

    def load_from_obj(self, file_path: str, num_points: int = 2048) -> np.ndarray:
        """Load an OBJ file and sample points from its mesh surface.

        Args:
            file_path: Path to the OBJ file.
            num_points: Number of points to sample.

        Returns:
            (num_points, 3) numpy array of sampled surface points.
        """
        if not HAS_TRIMESH:
            raise ImportError(
                "trimesh is required for OBJ loading. Install with: pip install trimesh"
            )
        mesh = trimesh.load(file_path, force="mesh")
        points = self._sample_mesh(mesh, num_points)
        if self.normalize_default:
            points = self.normalize(points)
        return points

    def load_from_ply(self, file_path: str, num_points: int = 2048) -> np.ndarray:
        """Load a PLY file. Handles both mesh and point cloud PLY files.

        If the PLY contains faces it is treated as a mesh and surface-sampled.
        Otherwise the raw vertices are used as the point cloud.

        Args:
            file_path: Path to the PLY file.
            num_points: Number of points to produce.

        Returns:
            (num_points, 3) numpy array.
        """
        if not HAS_TRIMESH:
            raise ImportError(
                "trimesh is required for PLY loading. Install with: pip install trimesh"
            )
        loaded = trimesh.load(file_path)

        if isinstance(loaded, trimesh.Trimesh) and len(loaded.faces) > 0:
            points = self._sample_mesh(loaded, num_points)
        elif isinstance(loaded, trimesh.PointCloud):
            points = np.asarray(loaded.vertices, dtype=np.float64)
            points = self._adjust_point_count(points, num_points)
        elif isinstance(loaded, trimesh.Trimesh):
            # Mesh with no faces -- use vertices directly
            points = np.asarray(loaded.vertices, dtype=np.float64)
            points = self._adjust_point_count(points, num_points)
        else:
            # Scene or other container -- try to extract vertices
            points = np.asarray(loaded.vertices, dtype=np.float64) if hasattr(loaded, "vertices") else np.zeros((num_points, 3), dtype=np.float64)
            points = self._adjust_point_count(points, num_points)

        if self.normalize_default:
            points = self.normalize(points)
        return points

    def load_from_xyz(self, file_path: str, num_points: int = 2048) -> np.ndarray:
        """Load a raw XYZ text file (whitespace-delimited x y z per line).

        Lines starting with '#' or empty lines are skipped. If the file has
        more than 3 columns, only the first 3 are used.

        Args:
            file_path: Path to the XYZ text file.
            num_points: Number of points to produce.

        Returns:
            (num_points, 3) numpy array.
        """
        rows = []
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    except ValueError:
                        continue

        if len(rows) == 0:
            logger.warning("No valid points found in %s, returning zeros.", file_path)
            points = np.zeros((num_points, 3), dtype=np.float64)
        else:
            points = np.array(rows, dtype=np.float64)
            points = self._adjust_point_count(points, num_points)

        if self.normalize_default:
            points = self.normalize(points)
        return points

    # ------------------------------------------------------------------
    # Preprocessing utilities
    # ------------------------------------------------------------------

    @staticmethod
    def normalize(points: np.ndarray) -> np.ndarray:
        """Centre point cloud at origin and scale to unit sphere.

        Args:
            points: (N, 3) array.

        Returns:
            (N, 3) normalised array.
        """
        points = points.copy()
        centroid = points.mean(axis=0)
        points -= centroid
        max_dist = np.max(np.linalg.norm(points, axis=1))
        if max_dist > 0:
            points /= max_dist
        return points

    @staticmethod
    def augment(
        points: np.ndarray,
        rotation: bool = True,
        jitter: bool = True,
        scale: bool = True,
    ) -> np.ndarray:
        """Apply random data augmentation to a point cloud.

        Args:
            points: (N, 3) array.
            rotation: Random rotation around Y-axis.
            jitter: Gaussian noise (std=0.01, clipped at 0.05).
            scale: Random uniform scale in [0.8, 1.2].

        Returns:
            (N, 3) augmented array (same shape as input).
        """
        points = points.copy()

        if rotation:
            theta = np.random.uniform(0, 2 * np.pi)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            rotation_matrix = np.array(
                [
                    [cos_t, 0, sin_t],
                    [0, 1, 0],
                    [-sin_t, 0, cos_t],
                ],
                dtype=points.dtype,
            )
            points = points @ rotation_matrix.T

        if scale:
            s = np.random.uniform(0.8, 1.2)
            points *= s

        if jitter:
            noise = np.clip(np.random.normal(0, 0.01, size=points.shape), -0.05, 0.05)
            points += noise

        return points

    def to_tensor(self, points: np.ndarray) -> "torch.Tensor":
        """Convert numpy array to a PyTorch float tensor.

        Args:
            points: numpy array of any shape.

        Returns:
            torch.Tensor with dtype float32.

        Raises:
            ImportError: If PyTorch is not installed.
        """
        if not HAS_TORCH:
            raise ImportError(
                "PyTorch is required for tensor conversion. "
                "Install with: pip install torch"
            )
        return torch.from_numpy(points.astype(np.float32))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_mesh(mesh: "trimesh.Trimesh", num_points: int) -> np.ndarray:
        """Sample points uniformly from mesh surface.

        Args:
            mesh: trimesh mesh object.
            num_points: Number of points to sample.

        Returns:
            (num_points, 3) array.
        """
        points, _ = trimesh.sample.sample_surface(mesh, num_points)
        return np.asarray(points, dtype=np.float64)

    @staticmethod
    def _adjust_point_count(points: np.ndarray, num_points: int) -> np.ndarray:
        """Subsample or pad a point cloud to exactly num_points.

        Subsampling is done via random choice without replacement (if possible).
        Padding repeats random existing points.

        Args:
            points: (M, 3) array with M existing points.
            num_points: Desired count.

        Returns:
            (num_points, 3) array.
        """
        n = len(points)
        if n == 0:
            return np.zeros((num_points, 3), dtype=np.float64)
        if n == num_points:
            return points
        if n > num_points:
            indices = np.random.choice(n, num_points, replace=False)
            return points[indices]
        # Pad by repeating random points
        pad_indices = np.random.choice(n, num_points - n, replace=True)
        return np.concatenate([points, points[pad_indices]], axis=0)
