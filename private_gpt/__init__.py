"""private-gpt."""

import logging
import os

# Set to 'DEBUG' to have extensive logging turned on, even for libraries
ROOT_LOG_LEVEL = "INFO"

PRETTY_LOG_FORMAT = (
    "%(asctime)s.%(msecs)03d [%(levelname)-8s] %(name)+25s - %(message)s"
)
logging.basicConfig(level=ROOT_LOG_LEVEL, format=PRETTY_LOG_FORMAT, datefmt="%H:%M:%S")
logging.captureWarnings(True)

# Disable gradio analytics
# This is done this way because gradio does not solely rely on what values are
# passed to gr.Blocks(enable_analytics=...) but also on the environment
# variable GRADIO_ANALYTICS_ENABLED. `gradio.strings` actually reads this env
# directly, so to fully disable gradio analytics we need to set this env var.
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

# Disable chromaDB telemetry
# It is already disabled, see PR#1144
# os.environ["ANONYMIZED_TELEMETRY"] = "False"

# adding tiktoken cache path within repo to be able to run in offline environment.
os.environ["TIKTOKEN_CACHE_DIR"] = "tiktoken_cache"

# --------------------------------------------------------------------------- #
# STRICT AIRGAP / CONFIDENTIALITY HARDENING
# These env vars must be set *before* heavy libraries (huggingface_hub,
# transformers, chromadb, ...) are imported, which is why they live at the top
# of the package __init__. They force offline behaviour and kill known
# telemetry channels so that nothing about your documents leaves the machine.
# The one-time exception is the very first model download; see docs/AIRGAP.md.
# Set PGPT_ALLOW_DOWNLOADS=1 for that first run only.
# --------------------------------------------------------------------------- #
if os.environ.get("PGPT_ALLOW_DOWNLOADS", "") not in ("1", "true", "True"):
    # HuggingFace stack: never reach the Hub.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

# Telemetry / analytics kill switches (harmless if a lib isn't installed).
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")  # chromadb
os.environ.setdefault("HAYSTACK_TELEMETRY_ENABLED", "False")
os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("SCARF_NO_ANALYTICS", "true")
os.environ.setdefault("POSTHOG_DISABLED", "1")
