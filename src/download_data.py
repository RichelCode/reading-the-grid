"""Download and prepare the ELPV solar-cell dataset for fault classification.

This script is the first step of the pipeline. It is *data prep only* — no model is
trained here. It performs four things:

1. Ensures the ELPV dataset is present locally. The dataset is a public GitHub repo
   (https://github.com/zae-bayern/elpv-dataset) containing electroluminescence (EL)
   images of solar cells plus a ``labels.csv``. If ``data/elpv-dataset/`` already
   exists, the download is skipped; otherwise the repo is cloned.

2. Loads the whitespace-delimited ``labels.csv`` (columns: image path, defect
   probability, module type) into a pandas DataFrame with explicit column names.

3. Binarizes the continuous defect probability into a 0/1 label using a single,
   clearly-named threshold (``FAULTY_THRESHOLD``). The raw probabilities only ever take
   the four values {0.0, 0.3333, 0.6667, 1.0} (four annotators voting defect/no-defect),
   so the threshold cleanly splits {0.0, 0.3333} -> healthy and {0.6667, 1.0} -> faulty.

4. Prints a class-balance summary, verifies the image files exist on disk, and writes a
   prepared label table to ``data/labels_binary.csv`` so every later step (training,
   evaluation, the app) just reads that one file.

Run with:  python src/download_data.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

# --- Configuration -----------------------------------------------------------------

# Anchor every path to the project root (the parent of this file's ``src/`` directory)
# so the script behaves identically whether you run it from the repo root or elsewhere.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = DATA_DIR / "elpv-dataset"
# As of the current repo layout, the labels file and images live under this subpath.
# (The image paths inside labels.csv, e.g. "images/cell0001.png", are relative to the
# directory that contains labels.csv.)
LABELS_FILE = DATASET_DIR / "src" / "elpv_dataset" / "data" / "labels.csv"
OUTPUT_CSV = DATA_DIR / "labels_binary.csv"

DATASET_URL = "https://github.com/zae-bayern/elpv-dataset.git"

# Defect-probability threshold for the binary label.
#   probability >= 0.5  -> faulty   (1)   covers {0.6667, 1.0}
#   probability <  0.5  -> healthy  (0)   covers {0.0,    0.3333}
# 0.5 is the standard ELPV convention: a cell is "faulty" once a majority of the four
# expert annotators flagged a defect.
FAULTY_THRESHOLD = 0.5


# --- Steps -------------------------------------------------------------------------

def ensure_dataset() -> None:
    """Clone the ELPV dataset if it is not already present."""
    if DATASET_DIR.exists():
        print(f"[skip] Dataset already exists at {DATASET_DIR.relative_to(PROJECT_ROOT)} "
              "— not downloading.")
        return

    print(f"[clone] Downloading ELPV dataset into "
          f"{DATASET_DIR.relative_to(PROJECT_ROOT)} ...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Use git clone (the dataset ships images via Git LFS-free PNGs in the repo).
    # check=True raises if git fails so we don't proceed with a half-downloaded repo.
    subprocess.run(
        ["git", "clone", DATASET_URL, str(DATASET_DIR)],
        check=True,
    )
    print("[clone] Done.")


def load_labels() -> pd.DataFrame:
    """Load the whitespace-delimited labels.csv into a DataFrame with named columns."""
    labels_file = LABELS_FILE
    if not labels_file.exists():
        # Fallback: the repo layout has moved before — locate labels.csv anywhere in the
        # clone rather than failing outright.
        matches = [p for p in DATASET_DIR.rglob("labels.csv") if ".git" not in p.parts]
        if not matches:
            sys.exit(
                f"[error] Expected labels file not found at {LABELS_FILE}, and no "
                "labels.csv was found anywhere in the clone.\n"
                "        The clone may have failed or the repo layout changed."
            )
        labels_file = matches[0]
        print(f"[load]  labels.csv not at the expected path; using "
              f"{labels_file.relative_to(PROJECT_ROOT)} instead.")

    # The file has no header and is separated by arbitrary whitespace.
    df = pd.read_csv(
        labels_file,
        sep=r"\s+",
        header=None,
        names=["image_path", "probability", "type"],
    )
    print(f"[load]  Read {len(df)} rows from {labels_file.relative_to(PROJECT_ROOT)}.")
    # The image paths in labels.csv are relative to the directory containing it.
    df.attrs["images_base"] = labels_file.parent
    return df


def binarize(df: pd.DataFrame) -> pd.DataFrame:
    """Add a binary ``label`` column derived from the continuous defect probability."""
    # {0.0, 0.3333} -> 0 (healthy); {0.6667, 1.0} -> 1 (faulty)
    df["label"] = (df["probability"] >= FAULTY_THRESHOLD).astype(int)

    # Store the absolute on-disk path so downstream code never has to guess the prefix.
    images_base = df.attrs["images_base"]
    df["full_path"] = df["image_path"].apply(lambda p: str(images_base / p))
    return df


def summarize_and_verify(df: pd.DataFrame) -> None:
    """Print class balance and confirm every referenced image exists on disk."""
    total = len(df)
    n_faulty = int(df["label"].sum())
    n_healthy = total - n_faulty

    print("\n=== Dataset summary ===")
    print(f"Total images : {total}")
    print(f"Healthy (0)  : {n_healthy:5d}  ({n_healthy / total:6.1%})")
    print(f"Faulty  (1)  : {n_faulty:5d}  ({n_faulty / total:6.1%})")

    # Verify the image files actually exist — a missing-file check now saves confusing
    # failures during training later.
    missing = [p for p in df["full_path"] if not Path(p).is_file()]
    if missing:
        print(f"\n[warn] {len(missing)} image file(s) referenced in labels.csv are "
              f"missing on disk. First few:")
        for p in missing[:5]:
            print(f"       - {p}")
    else:
        print(f"\n[ok]   All {total} image files exist on disk.")


def save_output(df: pd.DataFrame) -> None:
    """Write the prepared label table for downstream steps to consume."""
    out = df[["full_path", "label", "probability"]]
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[save]  Wrote prepared labels to "
          f"{OUTPUT_CSV.relative_to(PROJECT_ROOT)} ({len(out)} rows).")


def main() -> None:
    ensure_dataset()
    df = load_labels()
    df = binarize(df)
    summarize_and_verify(df)
    save_output(df)
    print("\nData preparation complete.")


if __name__ == "__main__":
    main()
