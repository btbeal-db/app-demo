# =============================================================================
# agent.py — stateful agent using a LangGraph graph + InMemorySaver
# =============================================================================
#
# This is the stateful variant of the demo. The agent is now a 1-node
# LangGraph graph compiled with a `checkpointer`, which means the conversation
# state is persisted across requests keyed by a `thread_id`. Re-send the same
# thread_id on a subsequent request and LangGraph replays the prior turns
# before calling the LLM, so the model "remembers" the conversation without
# the caller having to resend the history.
#
# Why a graph for what is effectively a one-shot LLM call?
#   • Checkpointing is a property of the *graph*, not of the model — to opt
#     into LangGraph's thread-keyed memory you need a compiled graph with a
#     `checkpointer=` argument.
#   • The same shape extends cleanly: add a tool-calling node, a router, a
#     retrieval step, etc., and the persistence story doesn't change.
#
# Where this is appropriate:
#   • Single-replica demos, local dev, "show me thread memory works" tests.
# Where this *breaks*:
#   • Multiple workers/replicas — InMemorySaver is per-process, so a request
#     that lands on worker A then worker B sees an empty thread.
#   • Any restart (redeploy, autoscale, platform-side restart) wipes state.
#
# For production: swap `InMemorySaver()` for `AsyncPostgresSaver` pointed at
# a Databricks Lakebase instance (see README "Promoting to Lakebase").
# -----------------------------------------------------------------------------

import logging
import os
from typing import AsyncGenerator, Optional

import mlflow
from databricks_langchain import ChatDatabricks
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
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


# -----------------------------------------------------------------------------
# Graph definition
# -----------------------------------------------------------------------------
async def call_model(state: MessagesState) -> dict:
    """The only node in the graph — turn the conversation so far into an
    LLM call and append the reply to state."""
    llm = ChatDatabricks(endpoint=LLM_ENDPOINT, temperature=0.0)

    # Ensure the system prompt is present exactly once at the head of the
    # message list. On a brand-new thread, `state["messages"]` is just the
    # one HumanMessage we passed in. On a continuation, it's the full
    # history rehydrated by the checkpointer.
    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT), *messages]

    reply = await llm.ainvoke(messages)
    # MessagesState uses an additive reducer — returning `messages` here
    # appends to (rather than replaces) the persisted message list.
    return {"messages": [reply]}


def _build_graph():
    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_edge(START, "call_model")
    builder.add_edge("call_model", END)

    # Module-level checkpointer. With multiple uvicorn workers each worker
    # gets its *own* InMemorySaver — see the warning at the top of this
    # file. For real deployments swap this for AsyncPostgresSaver.
    checkpointer = InMemorySaver()
    return builder.compile(checkpointer=checkpointer)


GRAPH = _build_graph()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _thread_id(request: ResponsesAgentRequest) -> str:
    """Resolve the LangGraph thread_id from the incoming request.

    Preferred:  request.context.conversation_id  (the Responses-API canonical
                place for a multi-turn conversation key — the workspace
                Playground and chat UIs populate this automatically).
    Fallback:   request.custom_inputs["thread_id"]  (free-form extension
                channel, easiest to send from curl/scripts).
    Default:    "default"  (so a caller who omits the id still gets *a*
                memory, shared globally — useful for the quickest smoke
                test, not appropriate for real workloads).
    """
    if request.context and request.context.conversation_id:
        return request.context.conversation_id
    if request.custom_inputs and isinstance(request.custom_inputs, dict):
        if tid := request.custom_inputs.get("thread_id"):
            return tid
    return "default"


def _latest_user_text(request: ResponsesAgentRequest) -> str:
    """Pull the latest user turn out of the Responses-API input list.

    The caller does NOT need to resend prior turns when a thread_id is set —
    LangGraph replays them from the checkpointer. We only forward the most
    recent user message to the graph.
    """
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

    # Tag the MLflow trace with the same id so traces from one conversation
    # group together in the experiment UI's session view.
    mlflow.update_current_trace(metadata={"mlflow.trace.session": thread_id})

    # LangGraph reads/writes checkpoints scoped to this thread_id.
    config = {"configurable": {"thread_id": thread_id}}

    user_text = _latest_user_text(request)
    result = await GRAPH.ainvoke(
        {"messages": [HumanMessage(content=user_text)]},
        config=config,
    )

    # The last message in returned state is the assistant reply.
    assistant_msg = result["messages"][-1]
    item = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": _ai_text(assistant_msg)}],
    }
    yield ResponsesAgentStreamEvent(type="response.output_item.done", item=item)
