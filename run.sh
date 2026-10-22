#!/usr/bin/env bash
#
# run.sh — normal, fully-offline run for PrivateGPT (local / air-gapped).
#
# Run this every time AFTER you have done the one-time ./first_run.sh.
# No internet is used: offline env flags are set by the app, the air-gap guard
# is active, and the server binds to localhost only.
#
# Usage:   bash run.sh
#
set -euo pipefail

# Always run from the project root (the folder this script lives in).
cd "$(dirname "$0")"

echo "=============================================================="
echo " PrivateGPT — offline local run"
echo " Open http://localhost:8001 once it says 'startup complete'."
echo " Press Ctrl+C to stop."
echo "=============================================================="

# --- sanity checks ---------------------------------------------------------
if ! command -v poetry >/dev/null 2>&1; then
  echo "ERROR: 'poetry' is not installed. Run ./first_run.sh first."
  exit 1
fi

# Probe for 'uvicorn' (a real installed dependency). Importing 'private_gpt'
# would pass even with nothing installed, because the package folder is local.
if ! poetry run python -c "import uvicorn" >/dev/null 2>&1; then
  echo "ERROR: dependencies not installed yet. Run ./first_run.sh first,"
  echo "or:  poetry install --extras \"ui llms-llama-cpp embeddings-huggingface vector-stores-qdrant\""
  exit 1
fi

if [ ! -d models ] || [ -z "$(ls -A models 2>/dev/null || true)" ]; then
  echo "WARNING: ./models looks empty — model weights may not be downloaded."
  echo "If startup fails, run ./first_run.sh once to fetch them."
fi

# Local profile; the app itself enforces offline mode + air-gap + localhost.
export PGPT_PROFILES=local
# Loopback-only bind (this is already the default; set explicitly for clarity).
export PGPT_SERVER_HOST=127.0.0.1

URL="http://localhost:8001"

# Open the browser automatically once the server is actually accepting
# connections. Runs in the background; disable with PGPT_NO_BROWSER=1.
open_browser_when_ready() {
  for _ in $(seq 1 120); do          # wait up to ~120s for startup
    if curl -s -o /dev/null "$URL"; then
      break
    fi
    sleep 1
  done
  if command -v open >/dev/null 2>&1; then
    open "$URL"                       # macOS
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$URL"                   # Linux
  fi
}
if [ "${PGPT_NO_BROWSER:-0}" != "1" ]; then
  open_browser_when_ready &
fi

exec poetry run python -m private_gpt
