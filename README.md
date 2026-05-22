# app-demo

Minimal Databricks Apps agent — a `ResponsesAgent` that calls Claude Sonnet 4.5
via `ChatDatabricks`, served by the MLflow GenAI `AgentServer` with
`@invoke()` / `@stream()` handlers.

## Layout

```
agent_server/
  agent.py          # @invoke + @stream handlers
  start_server.py   # AgentServer ASGI app + uvicorn entrypoint
databricks.yml      # DABs config — app, env, resources
pyproject.toml      # uv-resolved deps + start-server script
```

## Deploy

```bash
databricks bundle validate --profile=<profile>
databricks bundle deploy --profile=<profile>
databricks bundle run app_demo --profile=<profile>
```

## Local

```bash
uv sync
uv run start-server
```
