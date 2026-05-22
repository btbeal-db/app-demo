from mlflow.genai.agent_server import AgentServer

import agent_server.agent  # noqa: F401 — registers @invoke/@stream handlers

server = AgentServer("ResponsesAgent", enable_chat_proxy=True)
app = server.app


def main():
    server.run(app_import_string="agent_server.start_server:app")
