# Reading the Grid — web app

A two-part web app for the Reading the Grid fault classifier: a FastAPI backend that runs
the fine-tuned ResNet18 and computes a Grad-CAM overlay, and a React + TypeScript +
Tailwind frontend for uploading a cell image and viewing the result. Packaged as a single
Docker container for Hugging Face Spaces.

## Layout

```
webapp/
├── backend/          FastAPI app
│   ├── main.py       /api/health and /api/predict (inference + Grad-CAM), static serving
│   └── requirements.txt
├── frontend/         React + TypeScript + Tailwind (Vite)
│   ├── src/App.tsx   upload → prediction, Grad-CAM overlay, client-side threshold slider
│   └── ...
└── Dockerfile        builds the frontend, then serves it and the API on one port
```

The backend loads the model definition from the repo's top-level `src/model.py` and the
fine-tuned weights from `models/best_model_finetuned.pth`. Both paths are resolved relative
to the repo root and are overridable via the `RTG_SRC_DIR` and `RTG_CHECKPOINT` environment
variables.

## API

- `GET /api/health` — liveness check, returns `{"status": "ok"}`.
- `POST /api/predict` — multipart form upload with a `file` field (PNG or JPG). Returns:

  ```json
  {
    "faulty_probability": 0.87,
    "predicted_label": "faulty",
    "original_image": "data:image/png;base64,...",
    "gradcam_overlay": "data:image/png;base64,..."
  }
  ```

  The response carries the raw faulty-class probability rather than only a label, so the
  frontend applies the decision threshold client-side — the forward pass and heatmap run
  once and the slider is pure client math.

## Local development

The backend and frontend run as two processes. The Vite dev server proxies `/api` calls to
the backend, so the browser always uses same-origin `/api` paths.

**1. Backend** (Python 3.12) — from `webapp/backend`:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

The model is loaded and cached at startup; the first launch reads the checkpoint from
`models/best_model_finetuned.pth`. Run `python src/train.py` then `python src/finetune.py`
first if that file does not exist (weights are git-ignored — see the top-level README).
The API is then at http://localhost:8000/api/health.

**2. Frontend** — from `webapp/frontend`:

```bash
npm install
npm run dev
```

Open the printed URL (default http://localhost:5173), drop an EL cell image onto the page,
and view the predicted status alongside the Grad-CAM overlay.

## Production build (single container)

The `Dockerfile` is a two-stage build that mirrors the single-port serving model used on
Hugging Face Spaces:

1. **Frontend stage** (`node:22-alpine`) runs `npm ci` and `npm run build`, producing
   static files in `frontend/dist`.
2. **Backend stage** (`python:3.12-slim`) installs the Python dependencies, copies the
   backend source, the built frontend into `backend/static`, and the model definition
   (`src/`) plus the fine-tuned checkpoint. FastAPI serves the static files at the site
   root and the API under `/api`, all on port **7860**.

Because the backend imports from `src/` and loads the checkpoint from `models/`, the build
context is the **repo root**, not `webapp/`. Build and run from the repo root:

```bash
# from the repo root
docker build -f webapp/Dockerfile -t reading-the-grid-web .
docker run -p 7860:7860 reading-the-grid-web
```

Then open http://localhost:7860.

## Deploying to Hugging Face Spaces

Create a Docker Space and push the repository as the build context, with `webapp/Dockerfile`
as the Dockerfile. Spaces serves the container on port 7860 automatically. The app requires
no secrets or persistent storage.

Note the fine-tuned checkpoint (`models/best_model_finetuned.pth`) is git-ignored in this
repository, so it is not pushed by a normal `git push`. Commit it to the Space via
[git-lfs](https://git-lfs.com/) (or regenerate it there by running training) so the image
build can copy it in.
