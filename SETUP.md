# SETUP — getting this app running end-to-end

This is the definitive checklist for deploying this repo into a Databricks
workspace. It enumerates every external resource the app touches, what
permission it needs on each, which grants DABs handles for you, and which
ones you still have to apply by hand.

## 1. Prerequisites

You need all of the following before the first deploy:

| Prereq | How to check / install |
|---|---|
| Databricks CLI ≥ 0.260 | `databricks --version` — [install docs](https://docs.databricks.com/aws/en/dev-tools/cli/install) |
| An authenticated CLI profile | `databricks auth profiles` — your target profile must show `Valid: YES`. Otherwise `databricks auth login --host <workspace-url> --profile <profile>` |
| `jq` (for the curl snippets) | `brew install jq` or equivalent |
| `uv` (optional, only for local dev) | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Workspace has **Databricks Apps** enabled | Workspace UI → Compute → Apps |
| Workspace has **Unity Catalog** enabled | Workspace UI → Catalog (should not be empty) |
| Workspace has a usable LLM serving endpoint | `databricks serving-endpoints list --profile <p>` — look for `databricks-claude-sonnet-4-5` or another pay-per-token endpoint |

## 2. Resources this app depends on

Every Databricks App is essentially a container running as a **service
principal** (SP). The SP is created the first time the bundle deploys.
For the app to do anything useful it needs permission on a handful of
external objects. Here is the complete inventory.

| Resource | Permission needed | Wired by DABs? | Notes |
|---|---|---|---|
| The MLflow experiment that traces are logged into | `CAN_MANAGE` | ✅ via `experiment` resource | The experiment itself must already exist; DABs does not create it. |
| The LLM serving endpoint (`databricks-claude-sonnet-4-5`) | `CAN_QUERY` | ✅ via `serving_endpoint` resource | Endpoint must already exist (foundation-model endpoints are usually pre-provisioned). |
| The UC **catalog** that holds the trace tables | `USE CATALOG` | ❌ — DABs `uc_securable` doesn't support `CATALOG` | Manual SQL grant once per workspace. |
| The UC **schema** under that catalog | `USE SCHEMA` | ❌ — DABs `uc_securable` doesn't support `SCHEMA` | Manual SQL grant once per workspace. |
| `<exp_id>_otel_spans` table | `SELECT` + `MODIFY` | ✅ via two `uc_securable` entries | One DABs entry per permission (`permission:` is singular). |
| `<exp_id>_otel_annotations` table | `SELECT` + `MODIFY` | ✅ via two `uc_securable` entries | Same as above. Used for assessments/feedback. |

**The DABs gap**: of the six rows above, four are fully declarative in
`databricks.yml`. The two `USE` grants on the parent catalog and schema
are not — DABs' `uc_securable` resource only accepts `CONNECTION`,
`FUNCTION`, `TABLE`, or `VOLUME` as the securable type. Those two grants
have to be applied with SQL `GRANT` statements once per workspace.

## 3. First-deploy walkthrough

### 3a. Get / create the experiment

The experiment is what shows the traces in the workspace UI. Create it
once via the workspace UI:

1. Workspace UI → **Experiments** → **Create Experiment** → choose
   **GenAI**.
2. When prompted, pick a UC **catalog + schema** for trace storage.
   (Anything you own works; the schema can be empty.)
3. Open the created experiment and copy the **Experiment ID** from the URL
   (`/ml/experiments/<id>`).
4. Note the catalog + schema you picked — you'll need both in step 3b.

> **About the trace tables**: as soon as the experiment is created, the
> platform eagerly creates **six** tables in the trace schema, all named
> with a deterministic pattern:
>
> ```
> <catalog>.<schema>.<exp_id>_otel_spans
> <catalog>.<schema>.<exp_id>_otel_annotations
> <catalog>.<schema>.<exp_id>_otel_logs
> <catalog>.<schema>.<exp_id>_otel_metrics
> <catalog>.<schema>.<exp_id>_trace_metadata
> <catalog>.<schema>.<exp_id>_trace_unified
> ```
>
> So you can fill in `databricks.yml` (step 3b) without having to "look
> up" anything — once you have the catalog, schema, and experiment id,
> every table name is determined.
>
> This minimal demo only writes to `_otel_spans` (request/response
> spans) and `_otel_annotations` (assessments), so those are the two
> tables `databricks.yml` grants on. If you extend the agent to emit
> custom metrics or structured logs you'll see `PERMISSION_DENIED` on
> `_otel_metrics` or `_otel_logs` — add matching `uc_securable` entries
> when that happens.
>
> Confirm the tables exist before you deploy with `SHOW TABLES IN
> <catalog>.<schema>` in the workspace SQL editor — you should see all
> six rows. If the schema is empty, the experiment wasn't created as a
> **GenAI** experiment (a plain ML experiment has no trace storage).

### 3b. Edit `databricks.yml` for your workspace

Open `databricks.yml` and replace four values:

```yaml
- experiment_id: "3212027249152542"
+ experiment_id: "<your-experiment-id>"

- securable_full_name: agentbuilder_serverless_stable_catalog.brand_new_fancy_schema.3212027249152542_otel_spans
+ securable_full_name: <your-catalog>.<your-schema>.<your-experiment-id>_otel_spans

- securable_full_name: agentbuilder_serverless_stable_catalog.brand_new_fancy_schema.3212027249152542_otel_annotations
+ securable_full_name: <your-catalog>.<your-schema>.<your-experiment-id>_otel_annotations

- host: https://fevm-agentbuilder-serverless-stable.cloud.databricks.com
+ host: https://<your-workspace-host>
```

(`<your-experiment-id>` appears in three places: the experiment resource
and both trace table names.)

If your workspace doesn't have `databricks-claude-sonnet-4-5`, also
change the serving endpoint name.

### 3c. Deploy and start the app

```bash
databricks bundle validate --profile <p>
databricks bundle deploy  --profile <p>
databricks bundle run app_demo --profile <p>
```

The app will boot. The `/responses` endpoint already works. **Traces will
fail to export** at this point — see step 3d.

### 3d. Grant the SP `USE CATALOG` and `USE SCHEMA`

The SP doesn't exist until the first deploy, so this grant can't happen
earlier. Grab the SP's client id from the deployed app:

```bash
SP=$(databricks apps get excellus-app --profile <p> | jq -r .service_principal_client_id)
echo "App SP: $SP"
```

Then run two GRANT statements (against any SQL warehouse you have access to):

```bash
WID=$(databricks warehouses list --profile <p> --output json | jq -r '.[] | select(.state=="RUNNING") | .id' | head -1)
HOST=$(databricks auth profiles | awk -v p=<p> '$1==p {print $2}')
TOKEN=$(databricks auth token --profile <p> | jq -r .access_token)

for SQL in \
  "GRANT USE CATALOG ON CATALOG <your-catalog> TO \`$SP\`" \
  "GRANT USE SCHEMA  ON SCHEMA  <your-catalog>.<your-schema> TO \`$SP\`"; do
  curl -sS -X POST "$HOST/api/2.0/sql/statements" \
    -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
    -d "{\"statement\":\"$SQL\",\"warehouse_id\":\"$WID\",\"wait_timeout\":\"30s\"}" \
    | jq '{state: .status.state, error: .status.error}'
done
```

Both should print `{"state": "SUCCEEDED", "error": null}`.

### 3e. Verify

Hit the app once more — traces should now land in the experiment:

```bash
curl -sS -X POST "https://<app-url>/responses" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"input":[{"role":"user","content":[{"type":"input_text","text":"Hello"}]}]}'

# Confirm rows landed in the spans table:
curl -sS -X POST "$HOST/api/2.0/sql/statements" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d "{\"statement\":\"SELECT COUNT(*) FROM <your-catalog>.<your-schema>.\\\`<exp_id>_otel_spans\\\`\",\"warehouse_id\":\"$WID\",\"wait_timeout\":\"30s\"}" \
  | jq '.result.data_array'
```

Open the experiment in the UI → **Traces** tab. Your request should be there.

## 4. Common failure modes

| Symptom (in `databricks apps logs <app>`) | What it means | Fix |
|---|---|---|
| `No dependencies file found. Skipping installation.` | `pyproject.toml` not detected — usually because the build context is wrong or the file got renamed. | Confirm `pyproject.toml` is at the bundle's `source_code_path` root. |
| `Starting app with command: [python agent.py]` (instead of `[uv run start-server]`) | Bundle didn't apply `config.command` — usually because the app was created out-of-band before the bundle. | Delete the app (`databricks apps delete <name>`), then `databricks bundle deploy`. |
| `Failed to create app <name>. An app with the same name already exists.` | Same as above. | Delete the conflicting app first. |
| `WARNING mlflow.tracing.export.mlflow_v3: ... PERMISSION_DENIED: ... USE CATALOG ...` | Step 3d wasn't run. | Run the two GRANT statements. |
| `WARNING mlflow.tracing.export.mlflow_v3: ... table permissions: SELECT, MODIFY` after step 3d | The annotations table or one of the perms wasn't granted. | `databricks bundle deploy` again to apply the four `uc_securable` entries. |
| App boots but `/responses` returns 401/403 | You're hitting the app without an OAuth token. | `Authorization: Bearer $(databricks auth token --profile <p> \| jq -r .access_token)` |

## 5. Iterating on the agent

```bash
# Edit agent_server/agent.py, then:
databricks bundle deploy --profile <p>      # uploads new source, doesn't restart
databricks bundle run    app_demo --profile <p>   # restarts the app with the new source

# Stream logs while iterating:
databricks apps logs excellus-app --profile <p>
```

You only need to repeat step 3d (`USE CATALOG` / `USE SCHEMA` SQL grants)
if you delete the app and let DABs recreate it with a new SP, or if you
change the trace storage catalog/schema.

## 6. Tearing it down

```bash
databricks bundle destroy --profile <p>
```

Removes the app, its SP, and all bundle-managed resource grants. The
manual `USE CATALOG` / `USE SCHEMA` grants on the SP outlive the SP and
are cleaned up automatically when the SP is deleted; you don't need to
revoke them by hand.
