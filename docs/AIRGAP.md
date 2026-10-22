# Air-gapped / confidential operation

This install is hardened so that ingested documents stay on this machine.

## What the app enforces

- **Offline model libraries.** `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`,
  `HF_DATASETS_OFFLINE` are set before any library loads (see
  `private_gpt/__init__.py`). No calls to the HuggingFace Hub at runtime.
- **Telemetry off.** Gradio, ChromaDB, Haystack, PostHog, Scarf analytics are
  disabled via env vars.
- **Startup air-gap check.** `private_gpt/utils/airgap.py` refuses to start if
  the LLM or embedding backend is a cloud provider, if an Ollama endpoint is
  non-local, or if any cloud API key is populated.
- **Localhost only.** The server binds to `127.0.0.1` (see `__main__.py`), so
  nothing on your network can reach it.
- **Data folder locked.** `local_data/` is chmod `0700` (owner only).

## First run (one-time model download)

The local model + embedding weights must be downloaded once. For that single
supervised run only:

```bash
PGPT_ALLOW_DOWNLOADS=1 PGPT_DISABLE_AIRGAP=1 PGPT_PROFILES=local make run
```

After the weights are cached under `models/`, run normally — fully offline:

```bash
PGPT_PROFILES=local make run
```

## What the app CANNOT do (your responsibility)

Application code cannot stop another program running as **your own user** (or
as root/admin) from reading these files, nor stop someone copying them off the
machine. For real confidentiality:

1. **Enable full-disk encryption** — FileVault on macOS
   (System Settings → Privacy & Security → FileVault).
2. **Don't run untrusted software** as your user while documents are ingested.
3. Keep `local_data/` on the encrypted internal disk, not a synced/cloud folder
   (Dropbox, iCloud Drive, Google Drive) or a network share.

## Environment-variable reference

| Variable | Purpose | Default |
| --- | --- | --- |
| `PGPT_ALLOW_DOWNLOADS` | Allow HF Hub downloads (first run only) | unset (offline) |
| `PGPT_DISABLE_AIRGAP` | Skip the startup air-gap check (not recommended) | unset (enforced) |
| `PGPT_SERVER_HOST` | Bind address | `127.0.0.1` |
