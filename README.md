---
title: MedDataOps
emoji: "🩺"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
  - rl-environment
  - healthcare
  - data-engineering
---

<!-- markdownlint-disable MD022 MD025 MD060 -->

# MedDataOps
Clinical data engineering RL environment for training agents to clean hospital data and repair production SQL under realistic constraints.

## 1. Motivation
Clinical data pipelines are not abstract spreadsheet problems. They drive triage views, medication safety dashboards, and ICU capacity planning. When these pipelines fail, clinicians make decisions on bad information.

MedDataOps exists to train and evaluate agents on the exact failure modes that appear in real hospital analytics:

- messy, heterogeneous, and partially corrupted tabular data
- broken SQL logic in operational reporting queries
- pressure to produce correct answers with limited steps and auditability

The objective is simple: build agents that can safely recover high-integrity analytics from noisy clinical data.

## 2. Environment Overview
MedDataOps is an episodic, session-scoped environment exposed through an OpenEnv-style HTTP API.

Story setup:

- You are an on-call clinical data engineer.
- A hospital analytics report is wrong.
- You must clean the working dataset and fix the SQL query.
- You submit only when both are correct.

Core characteristics:

- deterministic task reset via optional seed
- explicit action space (clean_data / run_query / fix_query / submit)
- structured observation payloads for agent planning
- reward decomposition for cleaning quality, query quality, efficiency, and step discipline

## 3. Action Space
The API accepts action payloads via `POST /step`.

| action_type | parameters | description | example |
|---|---|---|---|
| `clean_data` | `{"operations": [...]}` | Apply cleaning transforms to the working dataset (normalize strings, type fixes, null handling, dedupe). | `{"action_type":"clean_data","parameters":{"operations":[{"operation":"normalize_strings","columns":["drug_name"],"case":"lower"}]}}` |
| `run_query` | `{"query":"SELECT ..."}` | Execute SQL against current episode tables to inspect correctness. | `{"action_type":"run_query","parameters":{"query":"SELECT ward, COUNT(*) AS n FROM patients GROUP BY ward"}}` |
| `fix_query` | `{"query":"SELECT ..."}` | Replace broken SQL with corrected SQL candidate. | `{"action_type":"fix_query","parameters":{"query":"WITH x AS (...) SELECT ..."}}` |
| `submit` | `{}` | Finalize episode scoring using current cleaned data and current SQL. | `{"action_type":"submit","parameters":{}}` |

Notes:

- unsupported actions return validation errors
- `/step` requires an active session initialized by `/reset`

## 4. Observation Space
Observation payload returned by `/reset` and included in `/step` response:

| field | type | description |
|---|---|---|
| `current_dataset_state` | `list[object]` | Current snapshot of episode working rows visible to the agent. |
| `current_sql_query` | `string` | Current SQL query under repair/evaluation. |
| `error_messages` | `list[string]` | Validation or execution errors from previous action/query attempt. |
| `task_description` | `string` | Natural-language objective for the current task. |
| `step_number` | `int` | Zero-based step index in the current episode. |

## 5. Tasks
Three benchmark tasks represent increasing operational complexity.

| task_id | difficulty | data_challenge | sql_challenge | max_score |
|---|---|---|---|---|
| `triage_report` | easy | Ward casing drift, duplicate patients, mixed date formats, null/`N/A` age values. | Fix malformed triage aggregation query and return accurate ward counts. | 1.0 |
| `medication_summary` | medium | Drug-name normalization, dosage parsing, orphan prescription rows, mixed timestamp formats. | Replace invalid cross-join pattern with correct patient-medication join and grouped counts. | 1.0 |
| `icu_capacity` | hard | Merge heterogeneous schemas across hospital systems, map ward codes, deduplicate occupancy events. | Rewrite expensive correlated-subquery design into set-based CTE aggregation for capacity metrics. | 1.0 |

## 6. Reward Function
Per-step reward is decomposed and explicitly reported.

Formula:

$$
R_{total} = (0.4 \cdot S_{clean}) + (0.4 \cdot S_{sql}) + (0.2 \cdot B_{eff}) + P_{step}
$$

Where:

- $S_{clean} \in [0,1]$: cell-level cleaning correctness against expected cleaned rows
- $S_{sql} \in [0,1]$: query result correctness against expected SQL output (exact + partial matching)
- $B_{eff} \in \{0, 0.1\}$: efficiency bonus when query plan cost is below threshold
- $P_{step} = -0.02 \times N_{unnecessary}$: penalty for unnecessary action churn

Reward object fields:

- `data_clean_score`
- `query_correct_score`
- `efficiency_bonus`
- `step_penalty`
- `total`

## 7. Quick Start

### Python local setup

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Run with Docker (recommended)

```bash
docker build -t meddataops .
docker run --rm -p 7860:7860 meddataops
```

Then open:

- root UI: `http://localhost:7860/`
- health: `http://localhost:7860/health`

### Run baseline inference client
`inference.py` expects a chat-completions compatible endpoint.

```powershell
$env:API_BASE_URL = "https://api-inference.huggingface.co/v1"
$env:MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
$env:HF_TOKEN = "<your_hf_token>"
python inference.py
```

### Quick smoke test

```bash
curl -s http://localhost:7860/health
curl -s http://localhost:7860/tasks
```

## 8. API Reference

### `GET /`
Serves the landing page (`index.html`) with task cards, demo runner, and endpoint examples.

### `GET /health`
Response:

```json
{
  "status": "ok",
  "version": "1.0.0"
}
```

### `GET /tasks`
Response:

```json
{
  "tasks": [
    {
      "id": "triage_report",
      "name": "Morning Triage Report",
      "difficulty": "easy",
      "description": "The morning triage report is broken...",
      "hints": ["Standardize ward strings before grouping."],
      "dirty_row_count": 500,
      "has_expected_sql": true
    }
  ]
}
```

### `POST /reset`
Request:

```json
{
  "task_id": "triage_report",
  "seed": 101
}
```

Response headers:

- `X-Session-Id: <uuid>`
- `Set-Cookie: session_id=<uuid>; HttpOnly; SameSite=Lax`

Response body (observation):

```json
{
  "current_dataset_state": [{"patient_id": "P100001", "ward": "icu"}],
  "current_sql_query": "SELECT ward, COUNT(*) as patient_count FROM patients ...",
  "error_messages": [],
  "task_description": "The morning triage report is broken...",
  "step_number": 0
}
```

### `POST /step`
Request:

```json
{
  "action_type": "fix_query",
  "parameters": {
    "query": "SELECT ward, COUNT(*) AS patient_count FROM patients GROUP BY ward"
  }
}
```

Header required (or cookie from `/reset`):

- `X-Session-Id: <uuid>`

Response:

```json
{
  "observation": {
    "current_dataset_state": [{"patient_id": "P100001", "ward": "ICU"}],
    "current_sql_query": "SELECT ward, COUNT(*) AS patient_count FROM patients GROUP BY ward",
    "error_messages": [],
    "task_description": "...",
    "step_number": 1
  },
  "reward": {"value": 0.0},
  "done": false,
  "info": {"action_type": "fix_query", "query_valid": true}
}
```

### `GET /state`
Header required:

- `X-Session-Id: <uuid>`

Response:

```json
{
  "done": false,
  "step_number": 1,
  "max_steps": 20,
  "task": {
    "id": "triage_report",
    "name": "Morning Triage Report",
    "difficulty": "easy",
    "description": "...",
    "hints": ["..."]
  },
  "observation": {"current_dataset_state": [], "current_sql_query": "..."},
  "latest_reward": {"value": 0.0},
  "last_info": {}
}
```

## 9. Baseline Scores
Illustrative placeholder baselines (Phase 3 review table format):

| task | model | score | steps |
|---|---|---:|---:|
| triage_report | GPT-4o | 0.82 | 6 |
| triage_report | Llama-3.1-8B-Instruct | 0.61 | 10 |
| medication_summary | GPT-4o | 0.78 | 9 |
| medication_summary | Llama-3.1-8B-Instruct | 0.57 | 13 |
| icu_capacity | GPT-4o | 0.72 | 12 |
| icu_capacity | Llama-3.1-8B-Instruct | 0.49 | 16 |

## 10. Project Structure

```text
MedDataOps/
├─ Dockerfile
├─ docker-compose.yml
├─ entrypoint.sh
├─ README.md
├─ index.html
├─ inference.py
├─ openenv.yaml
├─ requirements.txt
├─ scripts/
│  ├─ api_server.py
│  ├─ run_env.py
│  └─ seed_db.py
└─ src/
   └─ meddataops/
      ├─ __init__.py
      ├─ config.py
      ├─ data_cleaning.py
      ├─ db.py
      ├─ env.py
      ├─ models.py
      ├─ scoring.py
      ├─ sql_query.py
      ├─ sql/
      │  ├─ schema.sql
      │  └─ seed.sql
      └─ tasks/
         ├─ __init__.py
         ├─ triage_report.py
         ├─ medication_summary.py
         ├─ icu_capacity.py
         ├─ easy.py
         ├─ medium.py
         └─ hard.py
```

## 11. Contributing
We welcome contributions that improve realism, reliability, and evaluation rigor.

Recommended workflow:

1. Open an issue describing the clinical/technical gap.
2. Implement with tests and deterministic seed behavior.
3. Run local API smoke checks (`/health`, `/tasks`, `/reset`, `/step`, `/state`).
4. Submit a PR with before/after behavior and benchmark impact.

Contribution priorities:

- richer clinical noise patterns and schema drift cases
- stronger reward calibration and error attribution
- additional benchmark tasks and baseline trajectories

## 12. License
This project is released under the MIT License. See `LICENSE` for full terms.
