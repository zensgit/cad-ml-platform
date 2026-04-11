"""
PointNet Model Architecture.

Implements PointNet (Qi et al., 2017) for point cloud classification
and feature extraction. Falls back to stub classes when PyTorch is
not available.
"""

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

HAS_TORCH = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:
    logger.warning("PyTorch not found. PointNet module running in stub mode.")


def _make_stub(name: str):
    """Create a stub class that raises ImportError on instantiation."""

    class _Stub:
        __qualname__ = name
        __name__ = name

        def __init__(self, *args, **kwargs):
            raise ImportError(
                f"{name} requires PyTorch. Install with: pip install torch"
            )

    _Stub.__qualname__ = name
    _Stub.__name__ = name
    return _Stub


if HAS_TORCH:

    class TNet(nn.Module):
        """Spatial transformer network for input/feature alignment.

        Predicts a k x k affine transformation matrix from the input
        point cloud, initialised to the identity so the network starts
        from a pass-through transform.
        """

        def __init__(self, k: int = 3):
            super().__init__()
            self.k = k

            # Shared MLPs (implemented as 1-D convolutions)
            self.conv1 = nn.Conv1d(k, 64, 1)
            self.conv2 = nn.Conv1d(64, 128, 1)
            self.conv3 = nn.Conv1d(128, 1024, 1)

            self.bn1 = nn.BatchNorm1d(64)
            self.bn2 = nn.BatchNorm1d(128)
            self.bn3 = nn.BatchNorm1d(1024)

            # Fully connected layers
            self.fc1 = nn.Linear(1024, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, k * k)

            self.bn4 = nn.BatchNorm1d(512)
            self.bn5 = nn.BatchNorm1d(256)

            # Initialise final layer bias to identity matrix
            nn.init.zeros_(self.fc3.weight)
            nn.init.eye_(self.fc3.bias.view(k, k))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Predict transformation matrix.

            Args:
                x: (B, k, N) input features.

            Returns:
                (B, k, k) transformation matrix.
            """
            batch_size = x.size(0)

            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))

            # Max pool over points
            x = torch.max(x, 2)[0]  # (B, 1024)

            x = F.relu(self.bn4(self.fc1(x)))
            x = F.relu(self.bn5(self.fc2(x)))
            x = self.fc3(x)  # (B, k*k)

            x = x.view(batch_size, self.k, self.k)
            return x

    class PointNetEncoder(nn.Module):
        """PointNet encoder: extracts a global feature vector from a point cloud.

        Architecture follows the original PointNet paper with optional
        feature-space transformer.
        """

        def __init__(
            self,
            input_dim: int = 3,
            global_feat_dim: int = 1024,
            feature_transform: bool = True,
        ):
            super().__init__()
            self.input_dim = input_dim
            self.global_feat_dim = global_feat_dim
            self.feature_transform = feature_transform

            # Input transform
            self.input_tnet = TNet(k=input_dim)

            # Shared MLPs (first block)
            self.conv1 = nn.Conv1d(input_dim, 64, 1)
            self.conv2 = nn.Conv1d(64, 128, 1)
            self.conv3 = nn.Conv1d(128, 128, 1)

            self.bn1 = nn.BatchNorm1d(64)
            self.bn2 = nn.BatchNorm1d(128)
            self.bn3 = nn.BatchNorm1d(128)

            # Feature transform (optional)
            self.feat_tnet: Optional[TNet] = None
            if feature_transform:
                self.feat_tnet = TNet(k=128)

            # Shared MLPs (second block)
            self.conv4 = nn.Conv1d(128, 512, 1)
            self.conv5 = nn.Conv1d(512, global_feat_dim, 1)

            self.bn4 = nn.BatchNorm1d(512)
            self.bn5 = nn.BatchNorm1d(global_feat_dim)

        def forward(
            self, x: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
            """Encode point cloud to global feature.

            Args:
                x: (B, N, input_dim) point cloud.

            Returns:
                Tuple of:
                - global_feat: (B, global_feat_dim) global feature vector
                - input_transform: (B, input_dim, input_dim) input transform matrix
                - feat_transform: (B, 128, 128) feature transform matrix or None
            """
            batch_size, num_points, _ = x.shape

            # Transpose for conv1d: (B, input_dim, N)
            x = x.transpose(2, 1)

            # Input transform
            input_transform = self.input_tnet(x)  # (B, input_dim, input_dim)

            # Apply input transform
            x = x.transpose(2, 1)  # (B, N, input_dim)
            x = torch.bmm(x, input_transform)  # (B, N, input_dim)
            x = x.transpose(2, 1)  # (B, input_dim, N)

            # First MLP block
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))  # (B, 128, N)

            # Feature transform
            feat_transform = None
            if self.feat_tnet is not None:
                feat_transform = self.feat_tnet(x)  # (B, 128, 128)
                x = x.transpose(2, 1)  # (B, N, 128)
                x = torch.bmm(x, feat_transform)  # (B, N, 128)
                x = x.transpose(2, 1)  # (B, 128, N)

            # Second MLP block
            x = F.relu(self.bn4(self.conv4(x)))
            x = self.bn5(self.conv5(x))  # (B, global_feat_dim, N)

            # Max pool over points
            global_feat = torch.max(x, 2)[0]  # (B, global_feat_dim)

            return global_feat, input_transform, feat_transform

    class PointNetClassifier(nn.Module):
        """PointNet for point cloud classification.

        Maps a raw point cloud to class logits via the PointNet encoder
        followed by fully connected classification layers.
        """

        def __init__(
            self,
            num_classes: int = 8,
            input_dim: int = 3,
            global_feat_dim: int = 1024,
            feature_transform: bool = True,
            dropout: float = 0.3,
        ):
            super().__init__()
            self.num_classes = num_classes
            self.encoder = PointNetEncoder(
                input_dim=input_dim,
                global_feat_dim=global_feat_dim,
                feature_transform=feature_transform,
            )

            self.fc1 = nn.Linear(global_feat_dim, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, num_classes)

            self.bn1 = nn.BatchNorm1d(512)
            self.bn2 = nn.BatchNorm1d(256)

            self.dropout = nn.Dropout(p=dropout)

        def forward(
            self, x: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
            """Classify point cloud.

            Args:
                x: (B, N, 3) point cloud.

            Returns:
                Tuple of:
                - logits: (B, num_classes)
                - global_feat: (B, global_feat_dim)
                - feat_transform: (B, 128, 128) or None
            """
            global_feat, input_transform, feat_transform = self.encoder(x)

            out = F.relu(self.bn1(self.fc1(global_feat)))
            out = self.dropout(out)
            out = F.relu(self.bn2(self.fc2(out)))
            out = self.dropout(out)
            logits = self.fc3(out)

            return logits, global_feat, feat_transform

    class PointNetFeatureExtractor(nn.Module):
        """Extract a fixed-size L2-normalised feature vector from a point cloud.

        Useful for similarity search and retrieval tasks where a compact,
        normalised embedding is needed.
        """

        def __init__(
            self,
            input_dim: int = 3,
            feature_dim: int = 256,
            global_feat_dim: int = 1024,
            feature_transform: bool = True,
        ):
            super().__init__()
            self.feature_dim = feature_dim
            self.encoder = PointNetEncoder(
                input_dim=input_dim,
                global_feat_dim=global_feat_dim,
                feature_transform=feature_transform,
            )

            self.fc1 = nn.Linear(global_feat_dim, 512)
            self.fc2 = nn.Linear(512, feature_dim)

            self.bn1 = nn.BatchNorm1d(512)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Extract normalised feature vector.

            Args:
                x: (B, N, input_dim) point cloud.

            Returns:
                (B, feature_dim) L2-normalised feature vector.
            """
            global_feat, _, _ = self.encoder(x)

            out = F.relu(self.bn1(self.fc1(global_feat)))
            out = self.fc2(out)

            # L2 normalise
            out = F.normalize(out, p=2, dim=1)
            return out

else:
    # Stubs for environments without PyTorch
    TNet = _make_stub("TNet")
    PointNetEncoder = _make_stub("PointNetEncoder")
    PointNetClassifier = _make_stub("PointNetClassifier")
    PointNetFeatureExtractor = _make_stub("PointNetFeatureExtractor")
