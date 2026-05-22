# app-demo — the simplest agent on Databricks Apps

A minimal, working example of a Databricks Apps agent. It exposes an
OpenAI Responses-API-compatible `/responses` endpoint backed by Claude
Sonnet 4.5 (via `ChatDatabricks`), with MLflow tracing into a workspace
experiment, deployed end-to-end via a Declarative Automation Bundle.

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

## Try it

**From your shell (curl)**:

```bash
TOKEN=$(databricks auth token --profile=<profile> | jq -r .access_token)
curl -sS -X POST "https://<your-app>.databricksapps.com/responses" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"input":[{"role":"user","content":[{"type":"input_text","text":"What is 2+2?"}]}]}'
```

**From a Databricks notebook** (paste into a cell — uses your notebook auth automatically):

```python
import requests
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
app_url = "https://<your-app>.databricksapps.com/responses"
token = w.config.authenticate()["Authorization"].split(" ", 1)[1]

r = requests.post(
    app_url,
    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    json={"input": [{"role": "user", "content": [{"type": "input_text", "text": "What is 2+2?"}]}]},
)
print(r.json())
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

## Local development

```bash
uv sync
uv run start-server
# In another shell:
curl -X POST http://127.0.0.1:8000/responses ...
```
