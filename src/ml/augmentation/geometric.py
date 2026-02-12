"""
Geometric augmentations for CAD data.

Provides transformations that preserve CAD semantic meaning.
"""

from __future__ import annotations

import logging
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class BoundingBox2D:
    """2D bounding box for geometric calculations."""
    min_x: float = 0.0
    min_y: float = 0.0
    max_x: float = 1.0
    max_y: float = 1.0

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        return self.max_y - self.min_y

    @property
    def center(self) -> Tuple[float, float]:
        return (self.min_x + self.width / 2, self.min_y + self.height / 2)

    @classmethod
    def from_points(cls, points: List[Tuple[float, float]]) -> "BoundingBox2D":
        if not points:
            return cls()
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return cls(min(xs), min(ys), max(xs), max(ys))


class GeometricAugmentation(ABC):
    """Base class for geometric augmentations."""

    def __init__(self, p: float = 0.5):
        """
        Initialize augmentation.

        Args:
            p: Probability of applying this augmentation
        """
        self.p = p

    @abstractmethod
    def transform_point(self, x: float, y: float) -> Tuple[float, float]:
        """Transform a single point."""
        pass

    def transform_points(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Transform a list of points."""
        return [self.transform_point(x, y) for x, y in points]

    def __call__(self, data: Any) -> Any:
        """Apply augmentation to data."""
        if random.random() > self.p:
            return data
        return self._apply(data)

    @abstractmethod
    def _apply(self, data: Any) -> Any:
        """Apply augmentation (internal)."""
        pass

    def get_params(self) -> Dict[str, Any]:
        """Get current transformation parameters."""
        return {"p": self.p}


class RandomRotation(GeometricAugmentation):
    """Random rotation augmentation."""

    def __init__(
        self,
        angle_range: Tuple[float, float] = (-180.0, 180.0),
        center: Optional[Tuple[float, float]] = None,
        p: float = 0.5,
    ):
        """
        Initialize random rotation.

        Args:
            angle_range: (min_angle, max_angle) in degrees
            center: Rotation center (None for origin)
            p: Probability of applying
        """
        super().__init__(p)
        self.angle_range = angle_range
        self.center = center
        self._current_angle = 0.0

    def _sample_angle(self) -> float:
        self._current_angle = random.uniform(*self.angle_range)
        return self._current_angle

    def transform_point(self, x: float, y: float) -> Tuple[float, float]:
        """Rotate point around center."""
        angle_rad = math.radians(self._current_angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        cx, cy = self.center or (0.0, 0.0)

        # Translate to origin
        x -= cx
        y -= cy

        # Rotate
        new_x = x * cos_a - y * sin_a
        new_y = x * sin_a + y * cos_a

        # Translate back
        return new_x + cx, new_y + cy

    def _apply(self, data: Any) -> Any:
        self._sample_angle()

        if isinstance(data, dict):
            # Handle graph data
            if "x" in data:
                data = self._transform_node_features(data)
            if "points" in data:
                data["points"] = self.transform_points(data["points"])
            return data

        elif isinstance(data, list):
            # Assume list of points
            return self.transform_points(data)

        return data

    def _transform_node_features(self, data: Dict) -> Dict:
        """Transform node features containing coordinates."""
        try:
            import torch
            x = data["x"]
            if isinstance(x, torch.Tensor) and x.shape[1] >= 2:
                # Assume first two features are coordinates
                coords = x[:, :2].numpy()
                transformed = [self.transform_point(p[0], p[1]) for p in coords]
                x[:, :2] = torch.tensor(transformed)
                data["x"] = x
        except (ImportError, IndexError):
            pass
        return data

    def get_params(self) -> Dict[str, Any]:
        return {
            "p": self.p,
            "angle_range": self.angle_range,
            "current_angle": self._current_angle,
        }


class RandomScale(GeometricAugmentation):
    """Random scaling augmentation."""

    def __init__(
        self,
        scale_range: Tuple[float, float] = (0.8, 1.2),
        uniform: bool = True,
        center: Optional[Tuple[float, float]] = None,
        p: float = 0.5,
    ):
        """
        Initialize random scaling.

        Args:
            scale_range: (min_scale, max_scale)
            uniform: If True, use same scale for x and y
            center: Scale center (None for origin)
            p: Probability of applying
        """
        super().__init__(p)
        self.scale_range = scale_range
        self.uniform = uniform
        self.center = center
        self._scale_x = 1.0
        self._scale_y = 1.0

    def _sample_scale(self) -> Tuple[float, float]:
        self._scale_x = random.uniform(*self.scale_range)
        if self.uniform:
            self._scale_y = self._scale_x
        else:
            self._scale_y = random.uniform(*self.scale_range)
        return self._scale_x, self._scale_y

    def transform_point(self, x: float, y: float) -> Tuple[float, float]:
        cx, cy = self.center or (0.0, 0.0)
        return (x - cx) * self._scale_x + cx, (y - cy) * self._scale_y + cy

    def _apply(self, data: Any) -> Any:
        self._sample_scale()

        if isinstance(data, dict):
            if "x" in data:
                data = self._transform_node_features(data)
            if "points" in data:
                data["points"] = self.transform_points(data["points"])
            return data

        elif isinstance(data, list):
            return self.transform_points(data)

        return data

    def _transform_node_features(self, data: Dict) -> Dict:
        """Transform coordinate features."""
        try:
            import torch
            x = data["x"]
            if isinstance(x, torch.Tensor) and x.shape[1] >= 2:
                coords = x[:, :2].numpy()
                transformed = [self.transform_point(p[0], p[1]) for p in coords]
                x[:, :2] = torch.tensor(transformed)
                data["x"] = x
        except (ImportError, IndexError):
            pass
        return data

    def get_params(self) -> Dict[str, Any]:
        return {
            "p": self.p,
            "scale_range": self.scale_range,
            "scale_x": self._scale_x,
            "scale_y": self._scale_y,
        }


class RandomTranslation(GeometricAugmentation):
    """Random translation augmentation."""

    def __init__(
        self,
        translate_range: Tuple[float, float] = (-0.1, 0.1),
        relative: bool = True,
        p: float = 0.5,
    ):
        """
        Initialize random translation.

        Args:
            translate_range: (min, max) translation amount
            relative: If True, translate_range is relative to bounding box size
            p: Probability of applying
        """
        super().__init__(p)
        self.translate_range = translate_range
        self.relative = relative
        self._dx = 0.0
        self._dy = 0.0

    def _sample_translation(self, bbox: Optional[BoundingBox2D] = None) -> Tuple[float, float]:
        tx = random.uniform(*self.translate_range)
        ty = random.uniform(*self.translate_range)

        if self.relative and bbox:
            tx *= bbox.width
            ty *= bbox.height

        self._dx = tx
        self._dy = ty
        return self._dx, self._dy

    def transform_point(self, x: float, y: float) -> Tuple[float, float]:
        return x + self._dx, y + self._dy

    def _apply(self, data: Any) -> Any:
        bbox = None
        if isinstance(data, dict) and "points" in data:
            bbox = BoundingBox2D.from_points(data["points"])

        self._sample_translation(bbox)

        if isinstance(data, dict):
            if "x" in data:
                data = self._transform_node_features(data)
            if "points" in data:
                data["points"] = self.transform_points(data["points"])
            return data

        elif isinstance(data, list):
            return self.transform_points(data)

        return data

    def _transform_node_features(self, data: Dict) -> Dict:
        try:
            import torch
            x = data["x"]
            if isinstance(x, torch.Tensor) and x.shape[1] >= 2:
                x[:, 0] += self._dx
                x[:, 1] += self._dy
                data["x"] = x
        except (ImportError, IndexError):
            pass
        return data

    def get_params(self) -> Dict[str, Any]:
        return {
            "p": self.p,
            "translate_range": self.translate_range,
            "dx": self._dx,
            "dy": self._dy,
        }


class RandomFlip(GeometricAugmentation):
    """Random flip augmentation."""

    def __init__(
        self,
        horizontal: bool = True,
        vertical: bool = True,
        p: float = 0.5,
    ):
        """
        Initialize random flip.

        Args:
            horizontal: Allow horizontal flipping
            vertical: Allow vertical flipping
            p: Probability of applying
        """
        super().__init__(p)
        self.horizontal = horizontal
        self.vertical = vertical
        self._flip_h = False
        self._flip_v = False
        self._center = (0.0, 0.0)

    def _sample_flip(self) -> Tuple[bool, bool]:
        self._flip_h = self.horizontal and random.random() < 0.5
        self._flip_v = self.vertical and random.random() < 0.5
        return self._flip_h, self._flip_v

    def transform_point(self, x: float, y: float) -> Tuple[float, float]:
        cx, cy = self._center
        if self._flip_h:
            x = 2 * cx - x
        if self._flip_v:
            y = 2 * cy - y
        return x, y

    def _apply(self, data: Any) -> Any:
        self._sample_flip()

        # Calculate center from data
        if isinstance(data, dict) and "points" in data:
            bbox = BoundingBox2D.from_points(data["points"])
            self._center = bbox.center
        elif isinstance(data, list):
            bbox = BoundingBox2D.from_points(data)
            self._center = bbox.center

        if isinstance(data, dict):
            if "x" in data:
                data = self._transform_node_features(data)
            if "points" in data:
                data["points"] = self.transform_points(data["points"])
            return data

        elif isinstance(data, list):
            return self.transform_points(data)

        return data

    def _transform_node_features(self, data: Dict) -> Dict:
        try:
            import torch
            x = data["x"]
            if isinstance(x, torch.Tensor) and x.shape[1] >= 2:
                cx, cy = self._center
                if self._flip_h:
                    x[:, 0] = 2 * cx - x[:, 0]
                if self._flip_v:
                    x[:, 1] = 2 * cy - x[:, 1]
                data["x"] = x
        except (ImportError, IndexError):
            pass
        return data

    def get_params(self) -> Dict[str, Any]:
        return {
            "p": self.p,
            "flip_h": self._flip_h,
            "flip_v": self._flip_v,
        }


class RandomShear(GeometricAugmentation):
    """Random shear augmentation."""

    def __init__(
        self,
        shear_range: Tuple[float, float] = (-0.2, 0.2),
        p: float = 0.5,
    ):
        """
        Initialize random shear.

        Args:
            shear_range: (min, max) shear factor
            p: Probability of applying
        """
        super().__init__(p)
        self.shear_range = shear_range
        self._shear_x = 0.0
        self._shear_y = 0.0

    def _sample_shear(self) -> Tuple[float, float]:
        self._shear_x = random.uniform(*self.shear_range)
        self._shear_y = random.uniform(*self.shear_range)
        return self._shear_x, self._shear_y

    def transform_point(self, x: float, y: float) -> Tuple[float, float]:
        new_x = x + self._shear_x * y
        new_y = y + self._shear_y * x
        return new_x, new_y

    def _apply(self, data: Any) -> Any:
        self._sample_shear()

        if isinstance(data, dict):
            if "x" in data:
                data = self._transform_node_features(data)
            if "points" in data:
                data["points"] = self.transform_points(data["points"])
            return data

        elif isinstance(data, list):
            return self.transform_points(data)

        return data

    def _transform_node_features(self, data: Dict) -> Dict:
        try:
            import torch
            x = data["x"]
            if isinstance(x, torch.Tensor) and x.shape[1] >= 2:
                coords = x[:, :2].clone()
                x[:, 0] = coords[:, 0] + self._shear_x * coords[:, 1]
                x[:, 1] = coords[:, 1] + self._shear_y * coords[:, 0]
                data["x"] = x
        except (ImportError, IndexError):
            pass
        return data

    def get_params(self) -> Dict[str, Any]:
        return {
            "p": self.p,
            "shear_x": self._shear_x,
            "shear_y": self._shear_y,
        }


class AffineTransform(GeometricAugmentation):
    """General affine transformation."""

    def __init__(
        self,
        matrix: Optional[List[List[float]]] = None,
        p: float = 1.0,
    ):
        """
        Initialize affine transform.

        Args:
            matrix: 2x3 affine transformation matrix
            p: Probability of applying
        """
        super().__init__(p)
        self.matrix = matrix or [[1, 0, 0], [0, 1, 0]]

    def transform_point(self, x: float, y: float) -> Tuple[float, float]:
        m = self.matrix
        new_x = m[0][0] * x + m[0][1] * y + m[0][2]
        new_y = m[1][0] * x + m[1][1] * y + m[1][2]
        return new_x, new_y

    def _apply(self, data: Any) -> Any:
        if isinstance(data, dict):
            if "x" in data:
                data = self._transform_node_features(data)
            if "points" in data:
                data["points"] = self.transform_points(data["points"])
            return data

        elif isinstance(data, list):
            return self.transform_points(data)

        return data

    def _transform_node_features(self, data: Dict) -> Dict:
        try:
            import torch
            x = data["x"]
            if isinstance(x, torch.Tensor) and x.shape[1] >= 2:
                coords = x[:, :2].numpy()
                transformed = [self.transform_point(p[0], p[1]) for p in coords]
                x[:, :2] = torch.tensor(transformed)
                data["x"] = x
        except (ImportError, IndexError):
            pass
        return data

    @classmethod
    def from_rotation(cls, angle_deg: float, center: Tuple[float, float] = (0, 0)) -> "AffineTransform":
        """Create rotation transform."""
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        cx, cy = center

        matrix = [
            [cos_a, -sin_a, cx * (1 - cos_a) + cy * sin_a],
            [sin_a, cos_a, cy * (1 - cos_a) - cx * sin_a],
        ]
        return cls(matrix)

    @classmethod
    def from_scale(cls, sx: float, sy: float, center: Tuple[float, float] = (0, 0)) -> "AffineTransform":
        """Create scale transform."""
        cx, cy = center
        matrix = [
            [sx, 0, cx * (1 - sx)],
            [0, sy, cy * (1 - sy)],
        ]
        return cls(matrix)

    def get_params(self) -> Dict[str, Any]:
        return {
            "p": self.p,
            "matrix": self.matrix,
        }
