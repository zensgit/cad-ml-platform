"""
PointNet Module for 3D Point Cloud Analysis.

Provides direct point cloud processing for STL, OBJ, PLY, and XYZ files
without requiring parametric B-Rep representation.
"""

from src.ml.pointnet.model import PointNetClassifier, PointNetFeatureExtractor

__all__ = ["PointNetClassifier", "PointNetFeatureExtractor"]
