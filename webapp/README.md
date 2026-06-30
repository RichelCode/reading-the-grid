# Reading the Grid — web app

A two-part web app for the Reading the Grid fault classifier: a FastAPI backend and a
React + TypeScript + Tailwind frontend, packaged as a single Docker container for
Hugging Face Spaces. This is the scaffold — the backend returns placeholder responses and
the model is wired in a later step.

## Layout

```
webapp/
├── backend/          FastAPI app
│   ├── main.py       API routes (/api/health, /api/hello) and static file serving
│   └── requirements.txt
├── frontend/         React + TypeScript + Tailwind (Vite)
│   ├── src/App.tsx   single page: backend status indicator + hello message
│   └── ...
├── Dockerfile        builds the frontend, then serves it and the API on one port
└── .dockerignore
```

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

The API is then at http://localhost:8000/api/health.

**2. Frontend** — from `webapp/frontend`:

```bash
npm install
npm run dev
```

Open the printed URL (default http://localhost:5173). The page shows a green "Backend
online" indicator and the message returned by `/api/hello`.

## Production build (single container)

The `Dockerfile` is a two-stage build that mirrors the single-port serving model used on
Hugging Face Spaces:

1. **Frontend stage** (`node:22-alpine`) runs `npm ci` and `npm run build`, producing
   static files in `frontend/dist`.
2. **Backend stage** (`python:3.12-slim`) installs the Python dependencies, copies the
   backend source, and copies the built frontend into `backend/static`. FastAPI serves
   those static files at the site root and the API under `/api`, all on port **7860**.

Because both are served from the same origin in production, no CORS or proxy configuration
is needed there.

Build and run locally:

```bash
# from webapp/
docker build -t reading-the-grid-web .
docker run -p 7860:7860 reading-the-grid-web
```

Then open http://localhost:7860.

## Deploying to Hugging Face Spaces

Create a Docker Space and push this `webapp/` directory as the build context (the
`Dockerfile` at its root). Spaces serves the container on port 7860 automatically. The app
requires no secrets, tokens, or persistent storage.
