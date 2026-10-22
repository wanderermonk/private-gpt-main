# start a fastapi server with uvicorn

import os

import uvicorn

from private_gpt.main import app
from private_gpt.settings.settings import settings

# Air-gap default: bind to loopback only so nothing on the local network can
# reach the server (and, with it, your ingested documents). Override with
# PGPT_SERVER_HOST only if you deliberately need LAN access.
bind_host = os.environ.get("PGPT_SERVER_HOST", "127.0.0.1")

# Set log_config=None to do not use the uvicorn logging configuration, and
# use ours instead. For reference, see below:
# https://github.com/tiangolo/fastapi/discussions/7457#discussioncomment-5141108
uvicorn.run(app, host=bind_host, port=settings().server.port, log_config=None)
