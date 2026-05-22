# agent_app.py
#
# Minimal Databricks Apps agent:
# - exposes a /responses API through MLflow AgentServer
# - uses ResponsesAgent so Databricks/MLflow can trace requests automatically
# - uses ChatDatabricks inside the agent so the LLM layer is cleaner to swap
#   within the Databricks / LangChain ecosystem
#
# Why these pieces:
# - ResponsesAgent is still the recommended outer interface for Apps
# - AgentServer provides the /responses endpoint and built-in observability
# - MLflow tracing works automatically for Agent Framework deployments on Databricks Apps
# - ChatDatabricks is a nicer fit if you want a framework-oriented, less OpenAI-SDK-centric
#   implementation inside the agent itself

import os
import mlflow

from databricks_langchain import ChatDatabricks
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Optional but useful in local dev / notebooks:
# if MLFLOW_EXPERIMENT_ID is present, point MLflow at that experiment explicitly.
# In Apps, traces are stored in the agent's MLflow experiment.
experiment_id = os.getenv("MLFLOW_EXPERIMENT_ID")
if experiment_id:
    mlflow.set_experiment(experiment_id=experiment_id)


class SimpleEchoAgent(ResponsesAgent):
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        # This is where your agent logic runs.
        # MLflow traces this request/response path automatically because the app is
        # built using Agent Framework + ResponsesAgent.

        # ChatDatabricks gives you a framework-style LLM abstraction.
        # Swapping models is then mostly just changing the endpoint / model name,
        # or later replacing this with another LangChain chat model if your app architecture evolves.
        llm = ChatDatabricks(
            endpoint=os.getenv("LLM_ENDPOINT_NAME", "databricks-claude-sonnet-4-5"),
            temperature=0.0,
        )

        # Convert the Responses-style input into LangChain message objects.
        lc_messages = [
            SystemMessage(content="You are a concise helper. Answer directly.")
        ]

        for item in request.input:
            raw = item.model_dump() if hasattr(item, "model_dump") else dict(item)
            role = raw.get("role")
            content = raw.get("content", "")

            if isinstance(content, list):
                text_parts = [c.get("text", "") for c in content if c.get("type") == "input_text"]
                content = "".join(text_parts)

            if role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))

        # This is the actual model call.
        # The call itself is part of the traced agent execution because it happens inside predict().
        result = llm.invoke(lc_messages)
        text = result.content if isinstance(result.content, str) else str(result.content)

        # Return a ResponsesAgentResponse so the app can expose a proper /responses API.
        return ResponsesAgentResponse.from_dict(
            {
                "output": [
                    {
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text}],
                    }
                ]
            }
        )
