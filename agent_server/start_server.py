# =============================================================================
# start_server.py — boot the MLflow AgentServer
# =============================================================================
#
# This is the process entrypoint that `uv run start-server` calls (the script
# alias is defined in pyproject.toml under [project.scripts]).
#
# The job of this file is small:
#   1. Import `agent_server.agent` so that its @invoke/@stream decorators run
#      and register themselves with the AgentServer registry. (The decorators
#      are global — importing the module is what wires them up.)
#   2. Build an AgentServer and expose its FastAPI ASGI app as a *module-level*
#      variable named `app`. Uvicorn workers need that to fork properly.
#   3. Provide a `main()` that starts the uvicorn server.
# -----------------------------------------------------------------------------

from mlflow.genai.agent_server import AgentServer

# Importing this module triggers the @invoke / @stream decorators in agent.py,
# which register the handlers with AgentServer's internal registry. Without
# this import, AgentServer would start but have no routes to serve.
import agent_server.agent  # noqa: F401

# AgentServer args:
#   • "ResponsesAgent" — the agent kind. Tells the server to expose the
#     /responses endpoint with Responses-API request/response shapes.
#   • enable_chat_proxy=True — also exposes /chat/completions, which lets the
#     Apps built-in chat UI and OpenAI-compatible clients talk to the agent
#     without you writing a separate handler.
server = AgentServer("ResponsesAgent", enable_chat_proxy=True)

# Module-level ASGI app. This is referenced by `app_import_string` below and
# by any external uvicorn invocation. Defining it at module scope (not inside
# main) lets uvicorn spawn multiple workers, each re-importing the module
# and getting their own copy of the app.
app = server.app


def main():
    # `app_import_string` tells AgentServer how to point uvicorn at the ASGI
    # app — uvicorn imports `agent_server.start_server` in each worker and
    # looks up the `app` attribute.
    server.run(app_import_string="agent_server.start_server:app")
