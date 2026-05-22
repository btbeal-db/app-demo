# app-demo — stateful agent on Databricks Apps (LangGraph + Lakebase branch)

A working Databricks Apps agent with **durable** conversation memory. A
LangGraph graph compiled with `AsyncCheckpointSaver` persists thread
state into a Databricks Lakebase Postgres database, so requests carrying
the same `thread_id` continue the same conversation — across worker
restarts, replicas, and redeploys.

Same `/responses` endpoint, same MLflow tracing, same DAB deploy
story — plus thread-keyed memory that doesn't evaporate when the
container restarts.

## What's in the repo

```
.
├── agent_server/
│   ├── __init__.py        # marks the dir as a Python package
│   ├── agent.py           # LangGraph graph + AsyncCheckpointSaver + handlers
│   └── start_server.py    # AgentServer boot + Lakebase DDL lifespan
├── databricks.yml         # DAB config: app, env, resource permissions, postgres
├── pyproject.toml         # deps + start-server console script
├── .gitignore
├── README.md
└── SETUP.md               # end-to-end deploy walkthrough
```

### Why each file is required

| File | Why it has to exist |
|---|---|
| `agent_server/agent.py` | The agent. Wraps a `ChatDatabricks` LLM call in a 1-node `MessagesState` LangGraph compiled with `AsyncCheckpointSaver`. Exposes `@invoke()` and `@stream()` handlers — the only two functions MLflow's `AgentServer` looks for. Reads `LAKEBASE_AUTOSCALING_ENDPOINT` (injected by DABs) to connect. |
| `agent_server/start_server.py` | The process entrypoint. Imports `agent_server.agent` to register handlers, builds an `AgentServer`, and wraps the FastAPI lifespan so `setup_lakebase()` (the idempotent DDL that creates the checkpoint tables) runs once when each worker boots. |
| `agent_server/__init__.py` | Makes `agent_server` an importable Python package. Empty but required. |
| `pyproject.toml` | Tells Apps which dependencies to install via `uv` (note `databricks-langchain[memory]` for `AsyncCheckpointSaver` + psycopg). Declares the `start-server` console script that `databricks.yml`'s command runs. |
| `databricks.yml` | The deploy contract. Names the app, sets the start command and env vars, and declares every external resource the app needs (LLM endpoint, MLflow experiment, trace tables, **Lakebase postgres branch/database**) with the permission the app SP should get on each. |
| `.gitignore` | Standard hygiene. |

## Try it — thread memory

Send a thread_id via `custom_inputs.thread_id`. The same thread continues
the same conversation across **restarts** (try it — stop the app, start
it back up, hit the same thread and you'll still get the same memory).

**From your shell (curl)**:

```bash
TOKEN=$(databricks auth token --profile=<profile> | jq -r .access_token)
URL="https://<your-app>.databricksapps.com/responses"

# Turn 1 — introduce yourself
curl -sS -X POST "$URL" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{
    "input": [{"role":"user","content":[{"type":"input_text","text":"My name is Brennan and my favorite color is teal."}]}],
    "custom_inputs": {"thread_id": "demo-1"}
  }'

# Turn 2 — same thread, the agent recalls from Lakebase
curl -sS -X POST "$URL" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{
    "input": [{"role":"user","content":[{"type":"input_text","text":"What is my name and color?"}]}],
    "custom_inputs": {"thread_id": "demo-1"}
  }'
# → "Your name is Brennan and your favorite color is teal."

# A different thread has no memory of demo-1
curl -sS -X POST "$URL" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{
    "input": [{"role":"user","content":[{"type":"input_text","text":"What is my name?"}]}],
    "custom_inputs": {"thread_id": "demo-2"}
  }'
# → "I don't know your name."
```

Note that you only send the **latest** user message — LangGraph rehydrates
the prior turns from Lakebase using the `thread_id`.

**From a Databricks notebook**:

```python
import requests, uuid
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
app_url = "https://<your-app>.databricksapps.com/responses"
token = w.config.authenticate()["Authorization"].split(" ", 1)[1]
thread_id = str(uuid.uuid4())

def ask(text):
    r = requests.post(
        app_url,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={
            "input": [{"role": "user", "content": [{"type": "input_text", "text": text}]}],
            "custom_inputs": {"thread_id": thread_id},
        },
    )
    return r.json()["output"][0]["content"][0]["text"]

print(ask("My name is Brennan."))
print(ask("What is my name?"))   # → "Your name is Brennan."
```

## How the Lakebase wiring works

1. **`databricks.yml`** declares a `postgres:` resource pointing at a
   Lakebase Autoscaling project/branch/database with permission
   `CAN_CONNECT_AND_CREATE`.
2. **DABs** grants the app's service principal that permission and injects
   the connection endpoint URI into the container as the
   `LAKEBASE_AUTOSCALING_ENDPOINT` env var via
   `value_from: state_db`.
3. **`agent_server/agent.py`** reads that env var and constructs an
   `AsyncCheckpointSaver(autoscaling_endpoint=…)`. The
   `databricks-langchain` library mints short-lived OAuth tokens from
   the SP identity behind the scenes and refreshes them automatically.
4. **`agent_server/start_server.py`** wraps the FastAPI lifespan to call
   `checkpointer.setup()` once at startup — idempotent DDL that
   creates `checkpoints`, `checkpoint_writes`, `checkpoint_blobs`, and
   `checkpoint_migrations` under the configured schema.
5. **`@stream` handler** opens the checkpointer per request as an async
   context manager, compiles the graph with it, and lets LangGraph
   handle the load/save by thread_id.

Restart-durable, horizontally-scalable, no hard-coded credentials.

## Local development

`AsyncCheckpointSaver` needs to talk to a real Lakebase endpoint, so
true local-only development isn't possible against this branch without
provisioning a tunnel. The pragmatic loop is "edit → `bundle deploy` →
`bundle run`" (a sub-10s round trip on this codebase). See SETUP.md
for the full iterate-and-debug commands.
