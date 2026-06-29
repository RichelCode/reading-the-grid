"""Train the ELPV fault classifier.

Transfer-learning training loop over the frozen-backbone ResNet18 defined in
``src/model.py``, using the stratified dataloaders from ``src/dataset.py``. The loss is
class-weighted to address the dataset's class imbalance. The checkpoint with the best
validation F1 on the faulty class is saved to ``models/best_model.pth``; final metrics are
reported on the held-out test set.

Run with:  python src/train.py [--epochs N] [--batch-size N]
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from dataset import get_dataloaders
from model import build_model, get_device

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "labels_binary.csv"
MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pth"

# Faulty is the positive class for precision/recall/F1 reporting.
POSITIVE_CLASS = 1
NUM_CLASSES = 2
SEED = 42


def set_seeds(seed: int = SEED) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_class_weights(train_df, device: torch.device) -> torch.Tensor:
    """Compute CrossEntropyLoss weights inversely proportional to class frequency.

    Weights are derived from the training split only to avoid leaking validation/test
    distribution into the loss. weight[c] = total / (num_classes * count[c]).
    """
    counts = train_df["label"].value_counts().sort_index()
    total = len(train_df)
    weights = torch.tensor(
        [total / (NUM_CLASSES * counts[c]) for c in range(NUM_CLASSES)],
        dtype=torch.float32,
        device=device,
    )
    return weights


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run a single training epoch and return the mean training loss."""
    model.train()
    running_loss = 0.0
    n_samples = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        n_samples += images.size(0)
    return running_loss / n_samples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Evaluate the model and return (mean loss, true labels, predicted labels)."""
    model.eval()
    running_loss = 0.0
    n_samples = 0
    all_true: list[int] = []
    all_pred: list[int] = []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        n_samples += images.size(0)

        preds = outputs.argmax(dim=1)
        all_true.extend(labels.cpu().tolist())
        all_pred.extend(preds.cpu().tolist())

    mean_loss = running_loss / n_samples
    return mean_loss, np.array(all_true), np.array(all_pred)


def faulty_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float, float]:
    """Return (accuracy, precision, recall, F1) with faulty as the positive class."""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[POSITIVE_CLASS],
        average="binary",
        pos_label=POSITIVE_CLASS,
        zero_division=0,
    )
    return accuracy, precision, recall, f1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the ELPV fault classifier.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seeds()

    device = get_device()
    print(f"Device: {device}")

    train_loader, val_loader, test_loader, train_df, _, _ = get_dataloaders(
        CSV_PATH, batch_size=args.batch_size, seed=SEED
    )

    model = build_model(num_classes=NUM_CLASSES, freeze_backbone=True).to(device)

    class_weights = compute_class_weights(train_df, device)
    print(f"Class weights (healthy, faulty): "
          f"{class_weights.cpu().numpy().round(4).tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimize only the trainable head parameters; the backbone is frozen.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    best_val_f1 = -1.0
    print(f"\nTraining for {args.epochs} epochs.\n")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_true, val_pred = evaluate(model, val_loader, criterion, device)
        val_acc, val_prec, val_rec, val_f1 = faulty_metrics(val_true, val_pred)

        print(f"Epoch {epoch:2d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
              f"val_acc={val_acc:.4f} | faulty P={val_prec:.4f} "
              f"R={val_rec:.4f} F1={val_f1:.4f}")

        # Checkpoint on improved validation F1 for the faulty class.
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"         New best val F1={val_f1:.4f} -> saved "
                  f"{BEST_MODEL_PATH.relative_to(PROJECT_ROOT)}")

    # Reload the best checkpoint and report held-out test metrics.
    print("\nLoading best checkpoint for final test evaluation.")
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    _, test_true, test_pred = evaluate(model, test_loader, criterion, device)
    test_acc, test_prec, test_rec, test_f1 = faulty_metrics(test_true, test_pred)
    cm = confusion_matrix(test_true, test_pred, labels=[0, 1])

    print("\n=== Test set (held-out) ===")
    print(f"Accuracy : {test_acc:.4f}")
    print(f"Faulty precision: {test_prec:.4f}")
    print(f"Faulty recall   : {test_rec:.4f}")
    print(f"Faulty F1       : {test_f1:.4f}")
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(f"                 pred_healthy  pred_faulty")
    print(f"  true_healthy   {cm[0, 0]:12d}  {cm[0, 1]:11d}")
    print(f"  true_faulty    {cm[1, 0]:12d}  {cm[1, 1]:11d}")


if __name__ == "__main__":
    main()
