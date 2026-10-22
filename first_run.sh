#!/usr/bin/env bash
#
# first_run.sh — ONE-TIME setup run for PrivateGPT (local / air-gapped).
#
# This is the only run allowed to reach the internet: it downloads the local
# LLM + embedding model weights into ./models. After it finishes once, use
# ./run.sh for every normal (fully offline) run.
#
# Usage:   bash first_run.sh
#
set -euo pipefail

# Always run from the project root (the folder this script lives in).
cd "$(dirname "$0")"

echo "=============================================================="
echo " PrivateGPT — FIRST RUN (one-time, downloads models)"
echo "=============================================================="

# --- sanity checks ---------------------------------------------------------
if ! command -v poetry >/dev/null 2>&1; then
  echo "ERROR: 'poetry' is not installed."
  echo "Install it first, e.g.:  brew install poetry"
  exit 1
fi

# Install dependencies with the fully-local stack if not already installed.
# NOTE: we probe for 'uvicorn' (a real installed dependency), NOT 'private_gpt'
# — the private_gpt/ folder imports fine even with nothing installed, which
# would falsely skip the install.
if ! poetry run python -c "import uvicorn" >/dev/null 2>&1; then
  echo "Installing dependencies (local stack)… this can take a few minutes."
  poetry install --extras "ui llms-llama-cpp embeddings-huggingface vector-stores-qdrant"
else
  echo "Dependencies already installed — skipping 'poetry install'."
fi

# One-time exceptions: allow HF downloads and skip the air-gap guard so the
# weights can be fetched. run.sh does NOT set these.
export PGPT_ALLOW_DOWNLOADS=1
export PGPT_DISABLE_AIRGAP=1
export PGPT_PROFILES=local

# --- download the local models (embedding + LLM GGUF) ----------------------
# 'make run' does NOT download models in llamacpp mode; scripts/setup does.
# This is the step that actually fetches the weights into ./models.
echo
echo "Downloading local models (embedding + LLM GGUF) into ./models …"
echo "This is a few GB and only happens once."
echo "--------------------------------------------------------------"
poetry run python scripts/setup

echo
echo "Models downloaded. Starting the server to verify it boots."
echo "Wait for 'Application startup complete', open http://localhost:8001,"
echo "then press Ctrl+C. After this, always use:  ./run.sh"
echo "--------------------------------------------------------------"

exec poetry run python -m private_gpt
