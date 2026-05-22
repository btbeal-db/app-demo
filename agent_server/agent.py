# =============================================================================
# agent.py — stateful agent backed by Databricks Lakebase Postgres
# =============================================================================
#
# The agent is a 1-node LangGraph graph compiled per-request with an
# `AsyncCheckpointSaver` pointed at a Databricks Lakebase Autoscaling
# project. Lakebase is fully-managed Postgres on Databricks, so the
# checkpointer survives restarts, scales horizontally with the app, and
# uses the app's own service-principal OAuth identity to connect (no
# hard-coded credentials, automatic token refresh handled by the SDK).
#
# Why `AsyncCheckpointSaver` and not raw `AsyncPostgresSaver`?
#   • Lakebase issues short-lived OAuth tokens (~1h) instead of static
#     passwords. `AsyncCheckpointSaver` from `databricks-langchain`
#     wraps psycopg + Databricks SDK so the token refresh is transparent.
#   • It also resolves the autoscaling endpoint hostname (injected via
#     `LAKEBASE_AUTOSCALING_ENDPOINT` by DABs) into a project/branch path
#     under the hood. We just hand it the env var.
#
# Why the context manager per request?
#   • Matches the canonical Databricks app-template pattern.
#   • Connections are pooled inside the saver, so this is cheap.
#   • Compiling the LangGraph builder is also cheap; the expensive work
#     (LLM call) dominates.
# -----------------------------------------------------------------------------

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import mlflow
from databricks_langchain import AsyncCheckpointSaver, ChatDatabricks
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph
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

# Lakebase connection config — populated by DABs from the `postgres:`
# resource declaration in databricks.yml (see env block there).
LAKEBASE_ENDPOINT = os.environ["LAKEBASE_AUTOSCALING_ENDPOINT"]
LAKEBASE_SCHEMA = os.getenv("LAKEBASE_AGENT_MEMORY_SCHEMA", "app_demo")


# -----------------------------------------------------------------------------
# Lakebase checkpointer (async context manager)
# -----------------------------------------------------------------------------
@asynccontextmanager
async def _lakebase_checkpointer():
    """Yield a LangGraph checkpointer backed by Lakebase.

    Each entry opens a pooled connection (token-refreshed under the hood).
    Use at request scope: `async with _lakebase_checkpointer() as ckpt: ...`.
    """
    async with AsyncCheckpointSaver(
        autoscaling_endpoint=LAKEBASE_ENDPOINT,
        schema=LAKEBASE_SCHEMA,
    ) as ckpt:
        yield ckpt


async def setup_lakebase() -> None:
    """One-time DDL — creates the checkpoint tables under `LAKEBASE_SCHEMA`.
    Called from the FastAPI lifespan in start_server.py at app startup.
    Idempotent."""
    async with _lakebase_checkpointer() as ckpt:
        await ckpt.setup()


# -----------------------------------------------------------------------------
# Graph definition
# -----------------------------------------------------------------------------
async def call_model(state: MessagesState) -> dict:
    """The only node in the graph — LLM call with the system prompt
    prepended on a fresh thread, the existing history (rehydrated from
    the checkpointer) on a continuation."""
    llm = ChatDatabricks(endpoint=LLM_ENDPOINT, temperature=0.0)
    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT), *messages]
    reply = await llm.ainvoke(messages)
    return {"messages": [reply]}


def _build_graph(checkpointer):
    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_edge(START, "call_model")
    builder.add_edge("call_model", END)
    return builder.compile(checkpointer=checkpointer)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _thread_id(request: ResponsesAgentRequest) -> str:
    """Resolve the LangGraph thread_id from the incoming request.

    Preferred:  request.context.conversation_id  (Responses-API canonical;
                populated automatically by Databricks chat UIs / Playground)
    Fallback:   request.custom_inputs["thread_id"]  (free-form extension
                channel, easiest to send from curl/scripts)
    Default:    "default"  (smoke-test convenience)
    """
    if request.context and request.context.conversation_id:
        return request.context.conversation_id
    if request.custom_inputs and isinstance(request.custom_inputs, dict):
        if tid := request.custom_inputs.get("thread_id"):
            return tid
    return "default"


def _latest_user_text(request: ResponsesAgentRequest) -> str:
    """Pull just the latest user turn. LangGraph replays prior turns from
    the checkpointer when a known thread_id is reused — we never resend
    history."""
    for item in reversed(request.input):
        raw = item.model_dump() if hasattr(item, "model_dump") else dict(item)
        if raw.get("role") != "user":
            continue
        content = raw.get("content", "")
        if isinstance(content, list):
            content = "".join(
                c.get("text", "") for c in content if c.get("type") == "input_text"
            )
        return content
    return ""


def _ai_text(msg: BaseMessage) -> str:
    return msg.content if isinstance(msg.content, str) else str(msg.content)


# -----------------------------------------------------------------------------
# Handlers
# -----------------------------------------------------------------------------
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
    thread_id = _thread_id(request)
    mlflow.update_current_trace(metadata={"mlflow.trace.session": thread_id})

    config = {"configurable": {"thread_id": thread_id}}
    user_text = _latest_user_text(request)

    async with _lakebase_checkpointer() as ckpt:
        graph = _build_graph(ckpt)
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=user_text)]},
            config=config,
        )

    assistant_msg = result["messages"][-1]
    item = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": _ai_text(assistant_msg)}],
    }
    yield ResponsesAgentStreamEvent(type="response.output_item.done", item=item)
