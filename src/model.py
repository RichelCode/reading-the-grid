"""Model construction and device selection for the ELPV fault classifier.

The classifier is a transfer-learning setup: an ImageNet-pretrained ResNet18 backbone with
a replaced classification head. Freezing the backbone trains only the head, which is
appropriate given the relatively small dataset.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes: int = 2, freeze_backbone: bool = True) -> nn.Module:
    """Build a ResNet18 classifier with a replaced final layer.

    Args:
        num_classes: Number of output classes.
        freeze_backbone: When True, freeze all pretrained parameters so only the new
            classification head is trained.

    Returns:
        The configured model.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the final fully-connected layer. Parameters created here have
    # requires_grad=True by default, so the head remains trainable even when the
    # backbone is frozen.
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def get_device() -> torch.device:
    """Return the best available device, preferring MPS, then CUDA, then CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (trainable, frozen) parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, frozen


if __name__ == "__main__":
    device = get_device()
    print(f"Device: {device}")

    model = build_model(num_classes=2, freeze_backbone=True).to(device)

    trainable, frozen = _count_parameters(model)
    total = trainable + frozen
    print(f"Trainable parameters: {trainable:,} ({trainable / total:.2%})")
    print(f"Frozen parameters:    {frozen:,} ({frozen / total:.2%})")

    # Dummy forward pass to confirm the output shape.
    dummy = torch.randn(1, 3, 224, 224, device=device)
    model.eval()
    with torch.no_grad():
        output = model(dummy)
    print(f"Output shape: {tuple(output.shape)}")
