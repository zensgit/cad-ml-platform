#!/bin/bash

# CAD ML Platform - Apple Silicon (M4 Pro) Setup Script
# This script sets up a Conda environment optimized for macOS MPS acceleration.

set -e

ENV_NAME="cad-ml-m4"
PYTHON_VERSION="3.11"

echo "🚀 Starting environment setup for Apple Silicon (M4 Pro)..."

# 1. Check for Conda
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Miniconda or Anaconda first."
    exit 1
fi

# 2. Create Conda Environment
echo "📦 Creating conda environment: $ENV_NAME (Python $PYTHON_VERSION)..."
conda create -y -n $ENV_NAME python=$PYTHON_VERSION

# 3. Install PyTorch (Official stable with MPS support)
echo "🔥 Installing PyTorch for macOS..."
conda install -y -n $ENV_NAME pytorch torchvision -c pytorch

# 4. Install PyTorch Geometric and its dependencies
echo "🕸️ Installing PyTorch Geometric (PyG)..."
# Using pip inside conda to get the latest PyG versions which handle ARM64 best
conda run -n $ENV_NAME pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv

# 5. Install CAD dependencies
echo "📐 Installing CAD & Geometry dependencies..."
# pythonocc-core is available on conda-forge for macOS ARM64
conda install -y -n $ENV_NAME -c conda-forge pythonocc-core ezdxf trimesh

# 6. Install other project requirements
echo "📋 Installing general dependencies..."
conda run -n $ENV_NAME pip install fastapi uvicorn pydantic-settings prometheus-client scikit-learn pandas tqdm pyyaml

echo "✅ Setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To verify GPU (MPS) acceleration, run:"
echo "  python3 -c 'import torch; print(\"MPS Available:\", torch.backends.mps.is_available())'"
