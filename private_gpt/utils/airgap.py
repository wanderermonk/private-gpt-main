"""Strict air-gap / confidentiality enforcement.

This module is the run-time guard that refuses to start PrivateGPT if the
configuration would allow document content (or embeddings derived from it) to
leave the machine. It is deliberately conservative: anything that is not
clearly local is treated as a violation.

What it checks (``enforce_airgap``):
  * The LLM backend is local (llamacpp / ollama / mock) - never a cloud API.
  * The embedding backend is local (huggingface / ollama / mock).
  * When using Ollama, its API base points at localhost / a private address.
  * No cloud provider API keys are populated (openai / azure / gemini / ...),
    which would indicate a misconfigured or leaky profile.

What it does (``secure_data_dir``):
  * Creates the local data folder and tightens its POSIX permissions to
    0700 (owner-only) so other OS users cannot read ingested documents or
    the vector store. (On Windows this is a no-op; see docs/AIRGAP.md.)

Important honesty note: this is defence in depth at the *application* layer.
It cannot stop a process running as your own user, or as root/administrator,
from reading these files. Full protection requires OS-level disk encryption
(FileVault on macOS) and not running untrusted software as your user.
"""

import logging
import os
import socket
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Backends that keep inference on this machine.
_LOCAL_LLM_MODES = {"llamacpp", "ollama", "mock"}
_LOCAL_EMBEDDING_MODES = {"huggingface", "ollama", "mock"}

# Settings sections whose non-empty ``api_key`` means a cloud provider is
# configured. (Attribute access is defensive - missing sections are ignored.)
_CLOUD_KEY_SECTIONS = ("openai", "azopenai", "gemini", "sagemaker")


class AirgapViolation(RuntimeError):
    """Raised when the configuration is not safe for air-gapped operation."""


def _is_local_host(host: str | None) -> bool:
    """True if ``host`` resolves to loopback or a private/link-local address."""
    if not host:
        return False
    host = host.strip().lower()
    if host in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
        return True
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        # Can't resolve (we are offline) -> treat a bare hostname as NOT local
        # unless it was one of the literals above.
        return False
    for *_rest, sockaddr in infos:
        ip = sockaddr[0]
        try:
            import ipaddress

            addr = ipaddress.ip_address(ip)
            if not (addr.is_loopback or addr.is_private or addr.is_link_local):
                return False
        except ValueError:
            return False
    return True


def _is_local_url(url: str | None) -> bool:
    if not url:
        return False
    try:
        parsed = urlparse(url)
    except (ValueError, TypeError):
        return False
    return _is_local_host(parsed.hostname)


def _nonempty(value: object) -> bool:
    return bool(value) and str(value).strip() != ""


def check_airgap(settings) -> list[str]:
    """Return a list of human-readable violations (empty == safe).

    Kept separate from ``enforce_airgap`` so it can be unit-tested and reused
    (e.g. shown in the UI) without raising.
    """
    violations: list[str] = []

    llm_mode = getattr(getattr(settings, "llm", None), "mode", None)
    if llm_mode not in _LOCAL_LLM_MODES:
        violations.append(
            f"llm.mode = {llm_mode!r} is a cloud/remote backend. "
            f"Allowed for air-gap: {sorted(_LOCAL_LLM_MODES)}."
        )

    emb_mode = getattr(getattr(settings, "embedding", None), "mode", None)
    if emb_mode not in _LOCAL_EMBEDDING_MODES:
        violations.append(
            f"embedding.mode = {emb_mode!r} is a cloud/remote backend. "
            f"Allowed for air-gap: {sorted(_LOCAL_EMBEDDING_MODES)}."
        )

    # If Ollama is the active backend, its endpoints must be local.
    if llm_mode == "ollama" or emb_mode == "ollama":
        ollama = getattr(settings, "ollama", None)
        if ollama is not None:
            for attr in ("api_base", "embedding_api_base"):
                url = getattr(ollama, attr, None)
                if url and not _is_local_url(url):
                    violations.append(
                        f"ollama.{attr} = {url!r} is not a local address."
                    )

    # Any populated cloud key is suspicious even if that backend isn't active.
    for section in _CLOUD_KEY_SECTIONS:
        sec = getattr(settings, section, None)
        if sec is None:
            continue
        for attr in ("api_key", "embedding_api_key"):
            if _nonempty(getattr(sec, attr, None)):
                violations.append(
                    f"{section}.{attr} is set; remove it for air-gapped mode "
                    f"(a populated cloud key risks data egress)."
                )

    return violations


def enforce_airgap(settings) -> None:
    """Raise :class:`AirgapViolation` if the configuration is not air-gap safe.

    Can be disabled with ``PGPT_DISABLE_AIRGAP=1`` (NOT recommended; only for
    an intentional, supervised online run such as the first model download).
    """
    if os.environ.get("PGPT_DISABLE_AIRGAP", "") in ("1", "true", "True"):
        logger.warning(
            "AIRGAP ENFORCEMENT DISABLED via PGPT_DISABLE_AIRGAP - "
            "document data may be able to leave this machine."
        )
        return

    violations = check_airgap(settings)
    if violations:
        bullet = "\n  - ".join(violations)
        raise AirgapViolation(
            "Refusing to start: configuration is not safe for air-gapped / "
            "confidential operation.\n  - "
            + bullet
            + "\n\nFix settings.yaml (use local llamacpp/ollama + huggingface "
            "embeddings and clear cloud keys), or set PGPT_DISABLE_AIRGAP=1 "
            "for an intentional, supervised online run."
        )
    logger.info("Air-gap check passed: LLM and embeddings are local-only.")


def secure_data_dir(data_path: Path) -> None:
    """Create and lock down the local data folder to owner-only (0700).

    Best-effort: logs a warning instead of failing the whole app if the
    permissions can't be changed (e.g. on Windows / a network mount).
    """
    try:
        data_path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("Could not create data folder %s: %s", data_path, exc)
        return
    if os.name != "posix":
        logger.info(
            "Data folder %s created; skipping POSIX chmod on non-posix OS. "
            "Rely on OS disk encryption for confidentiality.",
            data_path,
        )
        return
    try:
        os.chmod(data_path, 0o700)
        logger.info("Secured data folder %s to owner-only (0700).", data_path)
    except OSError as exc:
        logger.warning("Could not chmod data folder %s: %s", data_path, exc)
