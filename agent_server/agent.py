# =============================================================================
# agent.py — the actual agent logic
# =============================================================================
#
# On Databricks Apps, MLflow's AgentServer doesn't take an agent instance in
# its constructor. Instead, it serves *module-level* async functions decorated
# with @invoke() and @stream(). Importing this module is what registers them
# with the server — see start_server.py.
#
# Why this shape?
#   • @stream is where the real work lives: an async generator that yields
#     ResponsesAgentStreamEvent objects. Async means one slow LLM call doesn't
#     block other requests.
#   • @invoke is the non-streaming entrypoint. The convention (recommended by
#     Databricks) is to implement it by collecting the events from @stream,
#     so you only write the logic once.
#
# The request/response shape is OpenAI Responses-API compatible
# (ResponsesAgentRequest / ResponsesAgentResponse). That's what makes this
# agent pluggable into AI Playground, evaluation jobs, and the Apps chat UI.
# -----------------------------------------------------------------------------

import logging
import os
from typing import AsyncGenerator

import mlflow
from databricks_langchain import ChatDatabricks
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from mlflow.genai.agent_server import invoke, stream
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

# Silence a noisy autolog warning that fires on every request. Pure cosmetics.
logging.getLogger("mlflow.utils.autologging_utils").setLevel(logging.ERROR)

# Turn on automatic tracing for LangChain calls. Every ChatDatabricks invocation
# below becomes a child span inside the agent trace, with prompt/response and
# timing captured automatically. The spans get exported to the experiment
# wired via MLFLOW_EXPERIMENT_ID (set in databricks.yml).
mlflow.langchain.autolog()

# The LLM endpoint is read from env so the bundle can swap models without a
# code change. See databricks.yml — the env var is populated from the
# `serving_endpoint` resource declaration.
LLM_ENDPOINT = os.getenv("LLM_ENDPOINT_NAME", "databricks-claude-sonnet-4-5")
SYSTEM_PROMPT = "You are a concise helper. Answer directly."


def _to_lc_messages(request: ResponsesAgentRequest):
    """Convert a Responses-API request into LangChain message objects.

    The Responses API uses a structured `input` array where each item has a
    `role` (user/assistant) and `content` that can be a string or a list of
    typed parts (e.g. `{"type": "input_text", "text": "..."}`). LangChain
    chat models expect a flat list of `HumanMessage` / `AIMessage` /
    `SystemMessage`, so we walk the input once and translate.
    """
    msgs = [SystemMessage(content=SYSTEM_PROMPT)]
    for item in request.input:
        # `item` may be a pydantic model or a raw dict depending on the caller —
        # normalize to dict.
        raw = item.model_dump() if hasattr(item, "model_dump") else dict(item)
        role = raw.get("role")
        content = raw.get("content", "")

        # Responses-API content can be a list of typed parts. Flatten the
        # `input_text` parts into a single string for the LLM.
        if isinstance(content, list):
            content = "".join(
                c.get("text", "") for c in content if c.get("type") == "input_text"
            )

        if role == "user":
            msgs.append(HumanMessage(content=content))
        elif role == "assistant":
            msgs.append(AIMessage(content=content))
    return msgs


@invoke()
async def invoke_handler(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    """Non-streaming endpoint — collects the streaming output and returns it.

    Implementing @invoke on top of @stream means there's only one place that
    knows how to talk to the LLM. The AgentServer routes a normal POST
    /responses to this function.
    """
    outputs = [
        event.item
        async for event in stream_handler(request)
        if event.type == "response.output_item.done"
    ]
    return ResponsesAgentResponse(output=outputs)


@stream()
async def stream_handler(
    request: ResponsesAgentRequest,
) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    """Streaming endpoint — yields ResponsesAgentStreamEvent objects.

    This is the only place that actually calls the LLM. AgentServer routes
    `POST /responses?stream=true` to this function (server-sent events).

    For this minimal demo we call the LLM once with `ainvoke` (single async
    completion) and emit a single `response.output_item.done` event. A more
    sophisticated agent would forward LLM streaming tokens as
    `response.output_text.delta` events, then close with `output_item.done`.
    """
    llm = ChatDatabricks(endpoint=LLM_ENDPOINT, temperature=0.0)
    result = await llm.ainvoke(_to_lc_messages(request))

    # ChatDatabricks may return string or list[part] depending on the model —
    # collapse to a plain string for the output_text part below.
    text = result.content if isinstance(result.content, str) else str(result.content)

    # The `item` shape is the Responses-API canonical assistant message:
    # an `id`, `type=message`, `role=assistant`, and `content` parts. Apps
    # downstream (Playground, eval) parse this shape.
    item = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text}],
    }
    yield ResponsesAgentStreamEvent(type="response.output_item.done", item=item)
