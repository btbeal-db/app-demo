# app-demo — stateful agent on Databricks Apps (LangGraph branch)

A minimal but **stateful** Databricks Apps agent. The base of `main`
demonstrates the simplest possible stateless agent; this branch
(`feat/stateful-langgraph`) adds a LangGraph graph compiled with an
`InMemorySaver` so that requests carrying the same `thread_id` continue
the same conversation. Same `/responses` endpoint, same MLflow tracing,
same DAB deploy story — plus thread-keyed memory.

## What's in the repo

```
.
├── agent_server/
│   ├── __init__.py        # marks the dir as a Python package
│   ├── agent.py           # the agent logic — @invoke / @stream handlers
│   └── start_server.py    # boots MLflow's AgentServer (FastAPI + uvicorn)
├── databricks.yml         # DAB config: app, env, resource permissions
├── pyproject.toml         # deps + start-server console script
├── .gitignore
└── README.md
```

### Why each file is required

| File | Why it has to exist |
|---|---|
| `agent_server/agent.py` | The actual agent. Defines async `@invoke()` and `@stream()` handlers — the only two functions MLflow's `AgentServer` looks for. `/responses` (non-streaming) routes to `@invoke`; `/responses?stream=true` and the Apps chat UI route to `@stream`. |
| `agent_server/start_server.py` | The process entrypoint. Imports `agent_server.agent` to trigger handler registration, builds an `AgentServer`, and exposes the FastAPI ASGI app + a `main()` for uvicorn. |
| `agent_server/__init__.py` | Makes `agent_server` an importable Python package so the start-server script can resolve `agent_server.start_server:main`. Empty file but required. |
| `pyproject.toml` | Tells the Apps runtime which dependencies to install via `uv`, and declares the `start-server` console script that `databricks.yml`'s command line runs. Without it, Apps logs `"No dependencies file found. Skipping installation."` and the app crashes at import. |
| `databricks.yml` | The deploy contract. Names the app, says where the source lives, sets the start command and env vars, and declares the resources the app needs at runtime (LLM endpoint, MLflow experiment, trace table) along with the permission the app's service principal should get on each. Without it, you'd be wiring permissions by hand in the UI. |
| `.gitignore` | Standard hygiene — keeps `.venv/`, `uv.lock`, build artifacts, and `.databricks/` cache out of the repo. |

## Deploy

See **[SETUP.md](./SETUP.md)** for the definitive walkthrough — including
which resources the app needs, what permissions DABs handles for you,
and the two `USE CATALOG` / `USE SCHEMA` grants you still have to apply
by hand.

TL;DR for an already-set-up workspace:

```bash
databricks bundle validate --profile <p>
databricks bundle deploy   --profile <p>
databricks bundle run app_demo --profile <p>
```

## Try it — thread memory

Send a thread_id via `custom_inputs.thread_id`. The same thread continues
the same conversation; a different thread starts fresh.

**From your shell (curl)**:

```bash
TOKEN=$(databricks auth token --profile=<profile> | jq -r .access_token)
URL="https://<your-app>.databricksapps.com/responses"

# Turn 1 — introduce yourself
curl -sS -X POST "$URL" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{
    "input": [{"role":"user","content":[{"type":"input_text","text":"My name is Brennan."}]}],
    "custom_inputs": {"thread_id": "demo-1"}
  }'

# Turn 2 — same thread, ask the agent what it remembers
curl -sS -X POST "$URL" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{
    "input": [{"role":"user","content":[{"type":"input_text","text":"What is my name?"}]}],
    "custom_inputs": {"thread_id": "demo-1"}
  }'
# → "Your name is Brennan."

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
the prior turns from the checkpointer.

**From a Databricks notebook** (uses your workspace auth):

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

## Tracing

`mlflow.langchain.autolog()` in `agent.py` plus the `MLFLOW_EXPERIMENT_ID` env
var wired in `databricks.yml` means every request becomes a trace under the
configured experiment. Open the experiment in the workspace → **Traces** tab
to inspect inputs, outputs, latencies, and the underlying LLM call.

> MLflow 3 GenAI tracing persists spans into Unity Catalog tables
> (`<catalog>.<schema>.<exp_id>_otel_{spans,annotations}`), which means the
> app's service principal needs permissions beyond just `CAN_MANAGE` on the
> experiment. `databricks.yml` declares the four table-level grants;
> `USE CATALOG` / `USE SCHEMA` are applied via SQL once per workspace —
> see **[SETUP.md § 3d](./SETUP.md#3d-grant-the-sp-use-catalog-and-use-schema)**.

## When this is not enough — promoting to Lakebase

`InMemorySaver` is fine for one-process demos. The moment Apps scales to
multiple workers or replicas, threads stop being durable: a request that
lands on worker A then worker B sees an empty thread. Every redeploy /
autoscale / platform restart wipes the dict.

The migration to a real backing store is small. Lakebase is a managed
Postgres instance, so `langgraph.checkpoint.postgres.aio.AsyncPostgresSaver`
plugs in directly.

**Agent code** — swap `_build_graph()`:

```python
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

# In start_server (async startup) or first-request bootstrap:
checkpointer = await AsyncPostgresSaver.from_conn_string(
    os.environ["PGURI"]
).__aenter__()
await checkpointer.setup()  # one-time DDL — idempotent
GRAPH = _build_graph(checkpointer=checkpointer)
```

**Bundle** — add a Lakebase resource to `databricks.yml`:

```yaml
resources:
  - name: state_db
    database:
      instance_name: <your-lakebase-instance>
      database_name: databricks_postgres
      permission: CAN_CONNECT_AND_CREATE
```

Lakebase is one of the resource types DABs natively supports (unlike the
catalog/schema gap documented in [SETUP.md](./SETUP.md)). The app gets
`PGHOST` / `PGUSER` / `PGPASSWORD` / `PGDATABASE` env vars injected at
runtime; assemble them into the connection string in code.

That single swap turns this into a horizontally-scalable, restart-durable
stateful agent. No other changes to the graph, the handlers, or the
caller-side thread-id protocol.

## Local development

```bash
uv sync
uv run start-server
# In another shell:
curl -X POST http://127.0.0.1:8000/responses ...
```
