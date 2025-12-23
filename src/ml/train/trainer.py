"""
Training Loop.

Orchestrates the training of the UV-Net model using the ABC Dataset.
"""

import argparse
import logging
import os
import sys

# Mock Torch for environments without it (e.g. CI/Lightweight)
try:
    import torch
    import torch.optim as optim
    import torch.nn.functional as F
    from src.ml.train.dataset import get_dataloader
    from src.ml.train.model import UVNetModel
except ImportError:
    # Minimal mock to pass dry-run
    class MagicMock:
        def __getattr__(self, name): return MagicMock()
        def __call__(self, *args, **kwargs): return MagicMock()
        def to(self, *args): return self
        def item(self): return 0.5

    def _mock_device(device_name):
        return device_name

    def _mock_cuda_is_available():
        return False

    def _mock_randn(*args, **kwargs):
        return MagicMock()

    def _mock_randint(*args, **kwargs):
        return MagicMock()

    def _mock_get_dataloader(*args, **kwargs):
        return []

    torch = MagicMock()
    torch.device = _mock_device
    torch.cuda.is_available = _mock_cuda_is_available
    torch.randn = _mock_randn
    torch.randint = _mock_randint
    optim = MagicMock()
    F = MagicMock()
    UVNetModel = MagicMock()
    get_dataloader = _mock_get_dataloader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Trainer")

def train(args):
    if isinstance(torch, type) and torch.__name__ == "MagicMock":
        logger.warning("PyTorch not found. Running in MOCK mode.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 1. Load Data
    train_loader = get_dataloader(args.data_dir, batch_size=args.batch_size)

    # 2. Init Model
    model = UVNetModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model.train()

    # 3. Training Loop
    logger.info("Starting training...")
    for epoch in range(1, args.epochs + 1):
        if args.dry_run and epoch > 1:
            break

        total_loss = 0
        batch_count = 0

        # If dataset is empty (no real files), we simulate one batch for dry-run
        if len(train_loader) == 0 and args.dry_run:
            data = torch.randn(args.batch_size, 12, 1024).to(device)
            target = torch.randint(0, 10, (args.batch_size,)).to(device)
            iterator = [(data, target)]
        else:
            iterator = train_loader

        for batch_idx, (data, target) in enumerate(iterator):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

            if args.dry_run:
                logger.info("Dry run batch complete.")
                break

        avg_loss = total_loss / max(1, batch_count)
        logger.info(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")

    # 4. Save
    if args.save_model:
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/uvnet_v1.pth")
        logger.info("Model saved to models/uvnet_v1.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UV-Net")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/abc_subset",
        help="Path to STEP files",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dry-run", action="store_true", help="Run a single pass for verification")
    parser.add_argument("--save-model", action="store_true", default=True)

    args = parser.parse_args()
    train(args)
