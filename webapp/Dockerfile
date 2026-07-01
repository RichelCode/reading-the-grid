# Single-container build for Hugging Face Docker Spaces.
# The React frontend is compiled to static files in stage 1, then copied into the Python
# image in stage 2 where FastAPI serves both those files and the API on one port (7860,
# the Spaces convention).
#
# Build context is the REPO ROOT, not webapp/, because the backend imports the model
# definition from src/ and loads the fine-tuned checkpoint from models/ — both live
# outside webapp/. Build from the repo root with:
#     docker build -f webapp/Dockerfile -t reading-the-grid-web .

# --- Stage 1: build the React frontend to static files ---
FROM node:22-alpine AS frontend
WORKDIR /frontend
# Install dependencies first so this layer is cached unless the lockfile changes.
COPY webapp/frontend/package.json webapp/frontend/package-lock.json ./
RUN npm ci
# Build the app; output goes to /frontend/dist.
COPY webapp/frontend/ ./
RUN npm run build

# --- Stage 2: Python backend that also serves the built frontend ---
FROM python:3.12-slim AS app
WORKDIR /app

# Runtime libraries required by opencv-python (used by the Grad-CAM overlay). Installing
# them here avoids the common libGL import error at inference time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for layer caching.
COPY webapp/backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend source, then drop the compiled frontend into ./static, which main.py
# serves as the site root.
COPY webapp/backend/ ./
COPY --from=frontend /frontend/dist ./static

# Provide the model definition and the fine-tuned checkpoint at the locations main.py
# resolves by default (BACKEND_DIR/src and BACKEND_DIR/models). model.py only depends on
# torch/torchvision, so copying the whole src/ adds no extra runtime requirements.
COPY src/ ./src/
COPY models/best_model_finetuned.pth ./models/best_model_finetuned.pth

# Spaces routes external traffic to this port.
EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
