"""
UV-Net Model Definition (Scaffold).

Structure based on 'UV-Net: Learning from Boundary Representations'.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UVNetModel(nn.Module):
    def __init__(self, num_classes=11, input_dim=12):
        super(UVNetModel, self).__init__()

        # Simplified PointNet-like structure for the scaffold
        # UV-Net uses a Face-Adjacency Graph, but the interface is similar (Embedding -> Class)

        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(p=0.3)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        # x shape: (Batch, Features, Points/Faces)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Global Max Pooling
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)

    def get_embedding(self, x):
        """Extract the global feature vector (1024 dim)."""
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        return x.view(-1, 1024)
