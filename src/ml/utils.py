"""
Unified device selection helper for Apple Silicon (MPS), NVIDIA (CUDA), and CPU.
"""

import logging
import torch

logger = logging.getLogger(__name__)

def get_best_device() -> str:
    """
    Returns the most performant available device string.
    Order: cuda > mps > cpu
    """
    if torch.cuda.is_available():
        return "cuda"
    
    # Check for Apple Silicon / Metal Performance Shaders
    if torch.backends.mps.is_available():
        # Optional: check if built with mps support
        if torch.backends.mps.is_built():
            return "mps"
        else:
            logger.warning("MPS is available but not built into this PyTorch install.")
            
    return "cpu"

def move_to_device(batch_data: any, device: str):
    """
    Recursively move tensors in a batch (dict, list, or PyG Data) to device.
    """
    if isinstance(batch_data, torch.Tensor):
        return batch_data.to(device)
    elif isinstance(batch_data, dict):
        return {k: move_to_device(v, device) for k, v in batch_data.items()}
    elif isinstance(batch_data, list):
        return [move_to_device(v, device) for v in batch_data]
    elif hasattr(batch_data, "to"):
        # Handles PyG Data and Batch objects
        return batch_data.to(device)
    return batch_data
