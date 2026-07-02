#!/usr/bin/env bash
# Deploy the Reading the Grid webapp to a Hugging Face Docker Space.
#
# Assembles a clean Space repo (Dockerfile at root, Space README, frontend source, backend,
# model definition, and the fine-tuned checkpoint), then creates the Space and uploads it.
# Requires an authenticated hf session (hf auth login) with write and repo-creation rights.
#
# Usage:
#   SPACE_ID="RichelCode/reading-the-grid" PRIVATE=0 bash deploy/deploy_to_hf.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPACE_ID="${SPACE_ID:-RichelCode/reading-the-grid}"
PRIVATE="${PRIVATE:-0}"
PY="$ROOT/venv/bin/python"
STAGE="$(mktemp -d)/space"

echo "Assembling Space staging at: $STAGE"
mkdir -p "$STAGE/src" "$STAGE/models"

# Dockerfile at Space root (build context = Space root); Space README with HF frontmatter.
cp "$ROOT/webapp/Dockerfile" "$STAGE/Dockerfile"
cp "$ROOT/deploy/README_space.md" "$STAGE/README.md"

# The HF Hub stores files above a size threshold as LFS objects regardless of .gitattributes
# (our JPEG photos are ~250-450 KB, so they land in LFS). HF's build only *smudges* LFS
# content into the build context for files that .gitattributes explicitly marks as LFS;
# files left un-tracked stay as 131-byte pointers and render broken. So we must LFS-track the
# JPEGs here (exactly like the checkpoint) to force HF to materialize the real image bytes at
# build time. The example PNGs are small enough to stay regular files and need no rule.
cat > "$STAGE/.gitattributes" <<'EOF'
*.pth filter=lfs diff=lfs merge=lfs -text
*.jpg filter=lfs diff=lfs merge=lfs -text
*.jpeg filter=lfs diff=lfs merge=lfs -text
EOF

# Keep the Docker build context lean.
cat > "$STAGE/.dockerignore" <<'EOF'
**/node_modules
**/dist
**/__pycache__
**/*.pyc
EOF

# Frontend source (no node_modules/dist/image-sources), backend (no static/__pycache__).
rsync -a --exclude node_modules --exclude dist --exclude image-sources --exclude '.DS_Store' \
  "$ROOT/webapp/frontend/" "$STAGE/webapp/frontend/"
rsync -a --exclude static --exclude __pycache__ --exclude '.venv' --exclude '.DS_Store' \
  "$ROOT/webapp/backend/" "$STAGE/webapp/backend/"

# Model definition and the fine-tuned checkpoint.
rsync -a --exclude __pycache__ "$ROOT/src/" "$STAGE/src/"
cp "$ROOT/models/best_model_finetuned.pth" "$STAGE/models/best_model_finetuned.pth"

echo "Staging tree (excluding public assets):"
( cd "$STAGE" && find . -type f -not -path './webapp/frontend/public/*' | sort )
echo "Total staging size:"; du -sh "$STAGE"

if [ "${ASSEMBLE_ONLY:-0}" = "1" ]; then
  echo "ASSEMBLE_ONLY set; skipping create/upload."
  exit 0
fi

SPACE_ID="$SPACE_ID" PRIVATE="$PRIVATE" STAGE="$STAGE" "$PY" - <<'PY'
import os
from huggingface_hub import HfApi

api = HfApi()
space_id = os.environ["SPACE_ID"]
private = os.environ["PRIVATE"] == "1"
stage = os.environ["STAGE"]

url = api.create_repo(space_id, repo_type="space", space_sdk="docker",
                      private=private, exist_ok=True)
print("Space repo:", url)
api.upload_folder(folder_path=stage, repo_id=space_id, repo_type="space",
                  commit_message="Deploy Reading the Grid webapp")
print("Upload complete; the Space will build automatically.")
PY

echo "Done. Space: https://huggingface.co/spaces/$SPACE_ID"
