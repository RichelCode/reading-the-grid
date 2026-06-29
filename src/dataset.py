"""Dataset, transforms, and dataloader construction for the ELPV fault classifier.

The pipeline reads the prepared label table at ``data/labels_binary.csv`` and produces
stratified train/validation/test dataloaders over the ELPV solar-cell images. Images are
300x300 grayscale PNGs; they are converted to 3-channel RGB and resized to 224x224 to
match the input expected by an ImageNet-pretrained ResNet.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ImageNet normalization statistics. Required because the backbone is pretrained on
# ImageNet and expects inputs in this distribution.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGE_SIZE = 224


class ELPVDataset(Dataset):
    """Wraps a label DataFrame as a PyTorch Dataset of (image, label) pairs.

    Args:
        df: DataFrame with at least ``full_path`` and ``label`` columns.
        transform: torchvision transform applied to each PIL image.
    """

    def __init__(self, df: pd.DataFrame, transform: transforms.Compose) -> None:
        # Reset the index so positional __getitem__ access is independent of the
        # original DataFrame index after splitting.
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        record = self.df.iloc[idx]
        # Source images are single-channel; convert to RGB so the 3-channel pretrained
        # backbone receives the expected input shape.
        image = Image.open(record["full_path"]).convert("RGB")
        image = self.transform(image)
        label = int(record["label"])
        return image, label


def get_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """Return (train, eval) transform pipelines.

    Both pipelines resize to 224x224 and apply ImageNet normalization. The train pipeline
    adds light augmentation (horizontal flip, small rotation) to reduce overfitting; the
    eval pipeline is deterministic.
    """
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        normalize,
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform, eval_transform


def get_dataloaders(
    csv_path: str | Path,
    batch_size: int = 32,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build stratified train/val/test dataloaders from the label CSV.

    The split is 70/15/15 and stratified on ``label`` to preserve the class ratio across
    all three partitions. Only the training loader is shuffled.

    Returns:
        train_loader, val_loader, test_loader, train_df, val_df, test_df
    """
    df = pd.read_csv(csv_path)

    # First split off the test set (15%), then split the remaining 85% into train (70%)
    # and validation (15%). The second split fraction is 15/85 to yield 15% of the total.
    train_val_df, test_df = train_test_split(
        df,
        test_size=0.15,
        stratify=df["label"],
        random_state=seed,
    )
    val_fraction = 0.15 / 0.85
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_fraction,
        stratify=train_val_df["label"],
        random_state=seed,
    )

    train_transform, eval_transform = get_transforms()

    train_dataset = ELPVDataset(train_df, train_transform)
    val_dataset = ELPVDataset(val_df, eval_transform)
    test_dataset = ELPVDataset(test_df, eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_df, val_df, test_df


def _class_balance(df: pd.DataFrame) -> str:
    """Format per-class counts and percentages for a split."""
    counts = df["label"].value_counts().sort_index()
    total = len(df)
    healthy = int(counts.get(0, 0))
    faulty = int(counts.get(1, 0))
    return (f"healthy={healthy} ({healthy / total:.1%}), "
            f"faulty={faulty} ({faulty / total:.1%})")


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    CSV_PATH = PROJECT_ROOT / "data" / "labels_binary.csv"

    train_loader, val_loader, test_loader, train_df, val_df, test_df = get_dataloaders(
        CSV_PATH
    )

    print("Split sizes and class balance:")
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"  {name:5s}: {len(split_df):4d} images  |  {_class_balance(split_df)}")

    # Pull one training batch to confirm tensor and label shapes.
    images, labels = next(iter(train_loader))
    print(f"\nBatch image tensor shape: {tuple(images.shape)}")
    print(f"Batch label tensor shape: {tuple(labels.shape)}")
    print(f"Sample labels: {labels[:8].tolist()}")
