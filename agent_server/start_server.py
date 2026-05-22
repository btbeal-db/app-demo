# =============================================================================
# start_server.py — boot the AgentServer + run Lakebase DDL at startup
# =============================================================================
#
# Two things this file does:
#   1. Build the AgentServer and expose its FastAPI ASGI app as `app`.
#   2. Wrap the FastAPI lifespan so that `setup_lakebase()` runs exactly
#      once when each uvicorn worker boots — this creates the checkpoint
#      tables in Lakebase if they don't already exist (idempotent). Per-
#      request handlers then assume the schema is ready.
# -----------------------------------------------------------------------------

from contextlib import asynccontextmanager

from mlflow.genai.agent_server import AgentServer

import agent_server.agent  # noqa: F401 — registers @invoke/@stream handlers
from agent_server.agent import setup_lakebase

server = AgentServer("ResponsesAgent", enable_chat_proxy=True)
app = server.app

_original_lifespan = app.router.lifespan_context


@asynccontextmanager
async def _lifespan(asgi_app):
    # Idempotent DDL — creates checkpoint tables on first boot.
    await setup_lakebase()
    async with _original_lifespan(asgi_app):
        yield


app.router.lifespan_context = _lifespan


def main():
    server.run(app_import_string="agent_server.start_server:app")
