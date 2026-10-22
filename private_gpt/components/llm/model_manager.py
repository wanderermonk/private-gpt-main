"""Discover the local models available to switch between.

Two sources, both local:
  * GGUF files sitting in the ``models/`` folder (llama.cpp).
  * Models already pulled into a local Ollama server.

Listing uses only the standard library (a plain HTTP GET to Ollama's
``/api/tags``), so it adds no dependency that could phone home and degrades
gracefully to an empty list when Ollama isn't running.

A model is identified by a string id:
    "gguf:<filename>"     e.g. "gguf:Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    "ollama:<name>"       e.g. "ollama:llama3.1:latest"
"""

from __future__ import annotations

import json
import logging
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)


def list_gguf_models(models_path: Path) -> list[str]:
    """Return .gguf filenames in the models folder (non-recursive), sorted."""
    try:
        return sorted(
            p.name for p in Path(models_path).glob("*.gguf") if p.is_file()
        )
    except OSError as exc:
        logger.warning("Could not list GGUF models in %s: %s", models_path, exc)
        return []


def list_ollama_models(api_base: str, timeout: float = 1.5) -> list[str]:
    """Return model names from a local Ollama server, or [] if unavailable.

    Deliberately fast-failing and quiet: if Ollama isn't running we just show
    no Ollama models rather than blocking the UI.
    """
    if not api_base:
        return []
    url = api_base.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # connection refused, timeout, bad json, ...
        logger.info("Ollama not available at %s (%s)", api_base, exc)
        return []
    models = data.get("models", []) if isinstance(data, dict) else []
    names = [m.get("name") for m in models if isinstance(m, dict) and m.get("name")]
    return sorted(names)


def available_models(
    models_path: Path,
    ollama_api_base: str | None = None,
    include_ollama: bool = True,
) -> list[dict[str, str]]:
    """Return a list of {"id", "label", "source"} for every selectable model."""
    out: list[dict[str, str]] = []
    for name in list_gguf_models(models_path):
        out.append({"id": f"gguf:{name}", "label": f"GGUF · {name}", "source": "gguf"})
    if include_ollama and ollama_api_base:
        for name in list_ollama_models(ollama_api_base):
            out.append(
                {"id": f"ollama:{name}", "label": f"Ollama · {name}", "source": "ollama"}
            )
    return out


def parse_model_id(model_id: str) -> tuple[str, str]:
    """Split "gguf:foo" -> ("gguf", "foo"). Raises ValueError on a bad id."""
    if ":" not in model_id:
        raise ValueError(f"Malformed model id (expected 'source:name'): {model_id!r}")
    source, name = model_id.split(":", 1)
    if source not in ("gguf", "ollama"):
        raise ValueError(f"Unknown model source {source!r} in {model_id!r}")
    if not name:
        raise ValueError(f"Empty model name in {model_id!r}")
    return source, name
