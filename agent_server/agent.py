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

logging.getLogger("mlflow.utils.autologging_utils").setLevel(logging.ERROR)
mlflow.langchain.autolog()

LLM_ENDPOINT = os.getenv("LLM_ENDPOINT_NAME", "databricks-claude-sonnet-4-5")
SYSTEM_PROMPT = "You are a concise helper. Answer directly."


def _to_lc_messages(request: ResponsesAgentRequest):
    msgs = [SystemMessage(content=SYSTEM_PROMPT)]
    for item in request.input:
        raw = item.model_dump() if hasattr(item, "model_dump") else dict(item)
        role = raw.get("role")
        content = raw.get("content", "")
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
    llm = ChatDatabricks(endpoint=LLM_ENDPOINT, temperature=0.0)
    result = await llm.ainvoke(_to_lc_messages(request))
    text = result.content if isinstance(result.content, str) else str(result.content)

    item = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text}],
    }
    yield ResponsesAgentStreamEvent(type="response.output_item.done", item=item)
