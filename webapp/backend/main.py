"""FastAPI backend for the Reading the Grid web app.

Serves the fine-tuned ELPV fault classifier behind a small API and serves the built React
frontend as static files. In production (Docker) this process serves both the API and the
frontend on a single port; in local development the frontend runs on the Vite dev server
and proxies ``/api`` requests here.

Model access:
    The model architecture lives in the repo's top-level ``src/model.py`` and the fine-tuned
    weights at ``models/best_model_finetuned.pth``. Paths are resolved relative to the repo
    root for local development, and are overridable via the ``RTG_SRC_DIR`` and
    ``RTG_CHECKPOINT`` environment variables.

    DOCKER BUILD MUST PROVIDE, one of:
      - copy the repo's ``src/`` (at minimum ``model.py``) and ``models/best_model_finetuned.pth``
        into the image so the default fallbacks below resolve, or
      - set ``RTG_SRC_DIR`` and ``RTG_CHECKPOINT`` to wherever those are copied.
    Note ``model.py`` only depends on torch/torchvision (already in requirements.txt). We do
    NOT import ``src/dataset.py`` here because it pulls in pandas and scikit-learn, which the
    inference image does not need; the eval transform is mirrored inline instead.
"""

from __future__ import annotations

import base64
import io
import os
import sys
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# --- Path resolution ---------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent
# In local dev this file lives at webapp/backend/main.py, so parents[1] is the repo root. In
# the container the backend is copied to /app, which has no second parent, so fall back to
# BACKEND_DIR; the BACKEND_DIR-relative candidates in the resolvers below then locate src/
# and the checkpoint at /app/src and /app/models.
REPO_ROOT = BACKEND_DIR.parents[1] if len(BACKEND_DIR.parents) > 1 else BACKEND_DIR


def _resolve_src_dir() -> Path:
    """Locate the directory containing model.py (env override, then sensible fallbacks)."""
    candidates = []
    if os.getenv("RTG_SRC_DIR"):
        candidates.append(Path(os.environ["RTG_SRC_DIR"]))
    candidates += [REPO_ROOT / "src", BACKEND_DIR / "src"]
    for candidate in candidates:
        if (candidate / "model.py").exists():
            return candidate
    raise RuntimeError(
        "Could not locate model.py. Set RTG_SRC_DIR or copy src/ into the image. "
        f"Looked in: {[str(c) for c in candidates]}"
    )


def _resolve_checkpoint() -> Path:
    """Locate the fine-tuned checkpoint (env override, then sensible fallbacks)."""
    candidates = []
    if os.getenv("RTG_CHECKPOINT"):
        candidates.append(Path(os.environ["RTG_CHECKPOINT"]))
    candidates += [
        REPO_ROOT / "models" / "best_model_finetuned.pth",
        BACKEND_DIR / "models" / "best_model_finetuned.pth",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise RuntimeError(
        "Could not locate best_model_finetuned.pth. Set RTG_CHECKPOINT or copy the "
        f"checkpoint into the image. Looked in: {[str(c) for c in candidates]}"
    )


SRC_DIR = _resolve_src_dir()
CHECKPOINT = _resolve_checkpoint()
sys.path.insert(0, str(SRC_DIR))

from model import build_model, get_device  # noqa: E402

# --- Inference configuration -------------------------------------------------------
CLASS_NAMES = ["healthy", "faulty"]
FAULTY_INDEX = 1
DEFAULT_THRESHOLD = 0.5

# Mirrors the eval pipeline in src/dataset.py::get_transforms (source of truth). Defined
# inline so the backend does not import dataset.py and its pandas/scikit-learn dependencies.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])
# Resize-only transform produces the un-normalized [0, 1] image the Grad-CAM heatmap is
# overlaid on, so the overlay colors render correctly.
display_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


@lru_cache(maxsize=1)
def load_model() -> tuple[torch.nn.Module, torch.device]:
    """Build the model, load the fine-tuned weights, and cache the result.

    Cached so the weights are read from disk and moved to the device only once for the
    lifetime of the process, not on every request.
    """
    device = get_device()
    model = build_model(num_classes=2, freeze_backbone=True)
    # Unfreeze layer4 to match the fine-tuned checkpoint and to let gradients reach the
    # Grad-CAM target layer.
    for param in model.layer4.parameters():
        param.requires_grad = True
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.to(device)
    model.eval()
    return model, device


def _to_data_url(array: np.ndarray) -> str:
    """Encode an HxWx3 image (float [0,1] or uint8) as a base64 PNG data URL."""
    if array.dtype != np.uint8:
        array = (np.clip(array, 0.0, 1.0) * 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def run_prediction(image: Image.Image) -> dict:
    """Run inference and Grad-CAM on one image and assemble the response payload."""
    model, device = load_model()

    input_tensor = eval_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(input_tensor), dim=1)[0]

    # We return the raw faulty-class probability (not just a label) so the frontend can
    # apply a user-adjustable decision threshold without re-running the model: the
    # expensive forward pass and heatmap happen once, and the slider is pure client math.
    faulty_probability = float(probs[FAULTY_INDEX])
    predicted_label = (
        "faulty" if faulty_probability >= DEFAULT_THRESHOLD else "healthy"
    )

    # Grad-CAM for the predicted class, targeting the last conv block, matching the
    # Streamlit app and notebook. The context manager releases the hooks after use.
    predicted_index = int(probs.argmax())
    with GradCAM(model=model, target_layers=[model.layer4[-1]]) as cam:
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=[ClassifierOutputTarget(predicted_index)],
        )[0]

    display_rgb = display_transform(image).permute(1, 2, 0).numpy().astype(np.float32)
    overlay = show_cam_on_image(display_rgb, grayscale_cam, use_rgb=True)

    return {
        "faulty_probability": faulty_probability,
        "predicted_label": predicted_label,
        "original_image": _to_data_url(display_rgb),
        "gradcam_overlay": _to_data_url(overlay),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the model cache at startup so the first request is not penalized by the load.
    load_model()
    yield


app = FastAPI(title="Reading the Grid API", lifespan=lifespan)

# Permissive CORS so the Vite dev server (a different origin during development) can call
# the API directly if it is not using the proxy. The app has no auth and stores nothing.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    """Liveness probe used by the frontend's online/offline indicator."""
    return {"status": "ok"}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    """Classify an uploaded electroluminescence cell image.

    Returns the raw faulty-class probability, the label at the default 0.5 threshold, and
    the original image and Grad-CAM overlay as base64 PNG data URLs.
    """
    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        image.load()  # force decode so a truncated/invalid file fails here, not later
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image. Upload a PNG or JPG.",
        )

    # Inference is synchronous and CPU/GPU-bound; run it off the event loop.
    return await run_in_threadpool(run_prediction, image)


# Serve the built frontend when it exists. The Docker build copies the compiled React app
# to ./static; in local dev that directory is absent, so this mount is skipped and the
# Vite dev server serves the UI instead. Mounted last so it does not shadow /api routes.
STATIC_DIR = BACKEND_DIR / "static"
if STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
