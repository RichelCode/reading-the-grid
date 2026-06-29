"""Fine-tune the ELPV fault classifier by unfreezing the last ResNet block.

Continues from the frozen-backbone checkpoint produced by ``src/train.py``. The final
ResNet18 block (``layer4``) is unfrozen alongside the classification head and trained at a
low learning rate, while all earlier layers remain frozen. Training, evaluation, loss
weighting, and metric conventions are shared with ``src/train.py``. The best-F1 checkpoint
is written to ``models/best_model_finetuned.pth``; the original ``models/best_model.pth``
is left untouched for comparison.

Run with:  python src/finetune.py [--epochs N] [--batch-size N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

from dataset import get_dataloaders
from model import build_model, get_device
from train import (
    BEST_MODEL_PATH,
    CSV_PATH,
    MODELS_DIR,
    NUM_CLASSES,
    PROJECT_ROOT,
    SEED,
    compute_class_weights,
    evaluate,
    faulty_metrics,
    set_seeds,
    train_one_epoch,
)

FINETUNED_MODEL_PATH = MODELS_DIR / "best_model_finetuned.pth"

# Low learning rate for fine-tuning to avoid destroying pretrained features in layer4.
FINETUNE_LR = 1e-5


def unfreeze_layer4(model: nn.Module) -> None:
    """Unfreeze the final ResNet block in place.

    The classification head (``fc``) is already trainable from ``build_model``; this adds
    ``layer4`` so the deepest convolutional features adapt to EL imagery while earlier
    layers stay frozen.
    """
    for param in model.layer4.parameters():
        param.requires_grad = True


def count_trainable(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune the ELPV fault classifier (unfreeze layer4)."
    )
    parser.add_argument("--epochs", type=int, default=8, help="Number of fine-tuning epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seeds()

    device = get_device()
    print(f"Device: {device}")

    # Require the frozen-backbone checkpoint as the starting point.
    if not BEST_MODEL_PATH.exists():
        sys.exit(
            f"[error] Starting checkpoint not found at "
            f"{BEST_MODEL_PATH.relative_to(PROJECT_ROOT)}.\n"
            "        Run `python src/train.py` first to produce best_model.pth."
        )

    train_loader, val_loader, test_loader, train_df, _, _ = get_dataloaders(
        CSV_PATH, batch_size=args.batch_size, seed=SEED
    )

    # Rebuild the same architecture and load the trained weights as the starting point.
    model = build_model(num_classes=NUM_CLASSES, freeze_backbone=True).to(device)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    print(f"Loaded starting checkpoint: {BEST_MODEL_PATH.relative_to(PROJECT_ROOT)}")

    trainable_before = count_trainable(model)
    unfreeze_layer4(model)
    trainable_after = count_trainable(model)
    print(f"Trainable parameters: {trainable_before:,} (head only) -> "
          f"{trainable_after:,} (head + layer4)")

    class_weights = compute_class_weights(train_df, device)
    print(f"Class weights (healthy, faulty): "
          f"{class_weights.cpu().numpy().round(4).tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=FINETUNE_LR)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    best_val_f1 = -1.0
    print(f"\nFine-tuning for {args.epochs} epochs at lr={FINETUNE_LR}.\n")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_true, val_pred = evaluate(model, val_loader, criterion, device)
        val_acc, val_prec, val_rec, val_f1 = faulty_metrics(val_true, val_pred)

        print(f"Epoch {epoch:2d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
              f"val_acc={val_acc:.4f} | faulty P={val_prec:.4f} "
              f"R={val_rec:.4f} F1={val_f1:.4f}")

        # Checkpoint on improved validation F1, to a separate file from the frozen model.
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), FINETUNED_MODEL_PATH)
            print(f"         New best val F1={val_f1:.4f} -> saved "
                  f"{FINETUNED_MODEL_PATH.relative_to(PROJECT_ROOT)}")

    # Reload the best fine-tuned checkpoint and report held-out test metrics.
    print("\nLoading best fine-tuned checkpoint for final test evaluation.")
    model.load_state_dict(torch.load(FINETUNED_MODEL_PATH, map_location=device))
    _, test_true, test_pred = evaluate(model, test_loader, criterion, device)
    test_acc, test_prec, test_rec, test_f1 = faulty_metrics(test_true, test_pred)
    from sklearn.metrics import confusion_matrix

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
