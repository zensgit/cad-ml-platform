#!/usr/bin/env python3
"""Run knowledge distillation: compress teacher model to lighter student.

Distils a large CAD classifier ensemble (teacher) into a smaller student
network.  Works end-to-end even without a real teacher model on disk — it
will generate synthetic data and dummy teacher logits when nothing is
available.

Usage:
    python scripts/train_knowledge_distillation.py
    python scripts/train_knowledge_distillation.py --teacher models/cad_classifier_v15_ensemble.pt --temperature 4.0
    python scripts/train_knowledge_distillation.py --epochs 100 --lr 0.0005
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# We need torch for the distillation loop, but degrade gracefully.
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

MODELS_DIR = Path("models")

# Default teacher search order (most recent / best first)
_TEACHER_CANDIDATES = [
    "cad_classifier_v15_ensemble.pt",
    "cad_classifier_v14_ensemble.pt",
    "cad_classifier_v14.pt",
    "cad_classifier_v13.pt",
]


# -----------------------------------------------------------------------
# Model definitions
# -----------------------------------------------------------------------


def _make_teacher_stub(input_dim: int, num_classes: int) -> nn.Module:
    """Build a simple feed-forward network mimicking the teacher architecture."""
    return nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes),
    )


def _make_student(input_dim: int, num_classes: int) -> nn.Module:
    """Build a compact student model (roughly 4x smaller than teacher)."""
    return nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, num_classes),
    )


# -----------------------------------------------------------------------
# Teacher discovery
# -----------------------------------------------------------------------


def find_teacher_model() -> Optional[str]:
    """Find the best available teacher model in the ``models/`` directory.

    Returns the path as a string, or ``None`` if nothing is found.
    """
    for candidate in _TEACHER_CANDIDATES:
        path = MODELS_DIR / candidate
        if path.exists():
            return str(path)
    return None


# -----------------------------------------------------------------------
# Synthetic data (fallback when no real dataset is available)
# -----------------------------------------------------------------------


def _generate_synthetic_data(
    num_samples: int,
    input_dim: int,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate random feature vectors and class labels for demonstration."""
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, num_classes, (num_samples,))
    return X, y


# -----------------------------------------------------------------------
# Distillation
# -----------------------------------------------------------------------


def run_distillation(args: argparse.Namespace) -> dict[str, Any]:
    """Full knowledge distillation pipeline.

    Steps
    -----
    1. Locate or load teacher model.
    2. Create a smaller student model.
    3. Load or generate training data.
    4. Run distillation training loop.
    5. Evaluate student vs teacher accuracy.
    6. Save student model.
    7. Report compression ratio.
    """
    if not HAS_TORCH:
        print("ERROR: PyTorch is required for knowledge distillation.")
        print("Install it with:  pip install torch")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("  Knowledge Distillation")
    print("=" * 60)
    print(f"  Device       : {device}")
    print(f"  Temperature  : {args.temperature}")
    print(f"  Alpha        : {args.alpha}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    if args.demo:
        print("  Mode         : DEMO (synthetic data, dev/test only)")
        logger.warning("Running in DEMO mode — synthetic data. Do NOT use in production.")
    else:
        print("  Mode         : PRODUCTION (real data required)")

    input_dim = 64
    num_classes = 10
    num_samples = 2000

    # ------------------------------------------------------------------
    # 1. Teacher model
    # ------------------------------------------------------------------
    teacher_path = args.teacher or find_teacher_model()
    teacher_loaded_from_disk = False

    if teacher_path and Path(teacher_path).exists():
        print(f"\n[1/7] Loading teacher from {teacher_path} ...")
        try:
            teacher_state = torch.load(teacher_path, map_location=device, weights_only=False)
            # Infer architecture from state dict if possible
            if isinstance(teacher_state, dict) and "state_dict" in teacher_state:
                teacher_state = teacher_state["state_dict"]
            teacher = _make_teacher_stub(input_dim, num_classes).to(device)
            try:
                teacher.load_state_dict(teacher_state)
                teacher_loaded_from_disk = True
                print("       Teacher loaded successfully.")
            except (RuntimeError, KeyError):
                if not args.demo:
                    logger.error(
                        "Teacher state dict mismatch — cannot continue without real weights. "
                        "Use --demo to run with random weights in demo mode."
                    )
                    sys.exit(1)
                print("       State dict mismatch — using teacher with random weights (demo mode).")
        except Exception as exc:
            if not args.demo:
                logger.error(
                    "Could not load teacher (%s). Use --demo to run without a real teacher.", exc
                )
                sys.exit(1)
            print(f"       Could not load teacher ({exc}); using random weights (demo mode).")
            teacher = _make_teacher_stub(input_dim, num_classes).to(device)
    else:
        if not args.demo:
            logger.error(
                "No teacher model found. Use --demo to run with a random-weight teacher "
                "for demonstration purposes only."
            )
            sys.exit(1)
        print("\n[1/7] No teacher model found — using random-weight teacher (demo mode).")
        teacher = _make_teacher_stub(input_dim, num_classes).to(device)

    teacher.eval()

    # ------------------------------------------------------------------
    # 2. Student model
    # ------------------------------------------------------------------
    print("[2/7] Creating student model ...")
    student = _make_student(input_dim, num_classes).to(device)

    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())
    print(f"       Teacher params: {teacher_params:,}")
    print(f"       Student params: {student_params:,}")
    print(f"       Compression   : {teacher_params / max(student_params, 1):.1f}x")

    # ------------------------------------------------------------------
    # 3. Training data
    # ------------------------------------------------------------------
    if not args.demo:
        logger.error(
            "No real training data available. Use --demo for synthetic demo mode. "
            "In production, provide a real dataset."
        )
        sys.exit(1)
    print(f"[3/7] Generating {num_samples} synthetic training samples (demo mode) ...")
    X, y = _generate_synthetic_data(num_samples, input_dim, num_classes)
    X, y = X.to(device), y.to(device)

    # ------------------------------------------------------------------
    # 4. Distillation training loop
    # ------------------------------------------------------------------
    print(f"[4/7] Training student for {args.epochs} epochs ...")

    optimizer = optim.Adam(student.parameters(), lr=args.lr)
    T = args.temperature
    alpha = args.alpha

    loss_history: list[float] = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        student.train()

        # Forward pass
        with torch.no_grad():
            teacher_logits = teacher(X)
        student_logits = student(X)

        # Hard (cross-entropy) loss
        hard_loss = F.cross_entropy(student_logits, y)

        # Soft (KL-divergence) loss
        soft_teacher = F.log_softmax(teacher_logits / T, dim=1)
        soft_student = F.log_softmax(student_logits / T, dim=1)
        kl_loss = F.kl_div(soft_student, soft_teacher.exp(), reduction="batchmean")

        loss = alpha * hard_loss + (1.0 - alpha) * (T * T) * kl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        if epoch == 1 or epoch % max(1, args.epochs // 10) == 0 or epoch == args.epochs:
            print(f"       epoch {epoch:4d}/{args.epochs}  loss={loss_val:.4f}  "
                  f"hard={hard_loss.item():.4f}  kl={kl_loss.item():.4f}")

    elapsed = time.time() - t0
    print(f"       Distillation finished in {elapsed:.1f}s")

    # ------------------------------------------------------------------
    # 5. Evaluate student vs teacher accuracy
    # ------------------------------------------------------------------
    print("[5/7] Evaluating student vs teacher ...")
    student.eval()

    with torch.no_grad():
        teacher_preds = teacher(X).argmax(dim=1)
        student_preds = student(X).argmax(dim=1)

    teacher_acc = (teacher_preds == y).float().mean().item()
    student_acc = (student_preds == y).float().mean().item()
    agreement = (teacher_preds == student_preds).float().mean().item()

    print(f"       Teacher accuracy : {teacher_acc:.4f}")
    print(f"       Student accuracy : {student_acc:.4f}")
    print(f"       Agreement        : {agreement:.4f}")

    # ------------------------------------------------------------------
    # 6. Save student model
    # ------------------------------------------------------------------
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(student.state_dict(), output_path)
    print(f"[6/7] Student model saved to {output_path}")

    # ------------------------------------------------------------------
    # 7. Compression report
    # ------------------------------------------------------------------
    teacher_size_mb = teacher_params * 4 / (1024 * 1024)  # float32
    student_size_mb = student_params * 4 / (1024 * 1024)
    if output_path.exists():
        student_size_mb = output_path.stat().st_size / (1024 * 1024)

    print("\n[7/7] Compression Report")
    print(f"       Teacher params   : {teacher_params:>10,}")
    print(f"       Student params   : {student_params:>10,}")
    print(f"       Param reduction  : {teacher_params / max(student_params, 1):.1f}x")
    print(f"       Student file size: {student_size_mb:.3f} MB")
    print(f"       Teacher accuracy : {teacher_acc:.4f}")
    print(f"       Student accuracy : {student_acc:.4f}")

    metrics: dict[str, Any] = {
        "teacher_path": teacher_path,
        "teacher_loaded_from_disk": teacher_loaded_from_disk,
        "teacher_params": teacher_params,
        "student_params": student_params,
        "compression_ratio": round(teacher_params / max(student_params, 1), 2),
        "epochs": args.epochs,
        "temperature": T,
        "alpha": alpha,
        "final_loss": loss_history[-1] if loss_history else float("nan"),
        "teacher_accuracy": round(teacher_acc, 4),
        "student_accuracy": round(student_acc, 4),
        "agreement": round(agreement, 4),
        "elapsed_seconds": round(elapsed, 2),
        "output": str(output_path),
    }

    print("\n" + "=" * 60)
    print("  Distillation complete.")
    print("=" * 60)
    return metrics


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Knowledge distillation: compress teacher to student model.",
    )
    parser.add_argument(
        "--teacher",
        default=None,
        help="Path to teacher model (.pt). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--output",
        default="models/cad_classifier_distilled.pt",
        help="Where to save the distilled student model.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=4.0,
        help="Distillation temperature (higher = softer probabilities).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Weight of hard loss vs soft loss (1.0 = all hard).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate for the student optimizer.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help=(
            "DEV/TEST ONLY: run with synthetic training data and accept a "
            "random-weight teacher when no real model is available. "
            "Must NOT be used in production."
        ),
    )
    args = parser.parse_args()
    run_distillation(args)


if __name__ == "__main__":
    main()
