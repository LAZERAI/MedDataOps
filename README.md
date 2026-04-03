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

## MedDataOps

MedDataOps is an OpenEnv-compatible reinforcement learning environment for clinical data engineering.
Agents must clean noisy hospital data and repair broken SQL across three tasks:

- `triage_report` (easy)
- `medication_summary` (medium)
- `icu_capacity` (hard)

The Docker Space serves:

- Root landing page at `/`
- HTTP environment API (`/reset`, `/step`, `/state`, `/health`, `/tasks`)
- Session-scoped runtime state per client

## Local run (Docker)

```powershell
docker build -t meddataops .
docker run --rm -p 7860:7860 meddataops
```

Open `http://localhost:7860`.

## Hugging Face Space Deployment (Step by Step)

### 1. Install and authenticate the CLI

```powershell
pip install -U "huggingface_hub[cli]"
hf auth login
hf auth whoami
```

Expected output should show user `Lazerai`.

### 2. Create the Docker Space

```powershell
hf repos create Lazerai/MedDataOps --repo-type space --space-sdk docker
```

If it already exists, the command can be skipped.

### 3. Push the code

Recommended (single command, no git remote setup):

```powershell
hf upload Lazerai/MedDataOps . . --repo-type space --exclude ".venv/*" --exclude "__pycache__/*" --exclude "postgres_data/*" --exclude "tmp/*"
```

Alternative (git remote):

```powershell
git remote add hf https://huggingface.co/spaces/Lazerai/MedDataOps
git push hf main
```

If `hf` already exists, skip `git remote add` and just push.

### 4. Set Space secrets

In Space page -> Settings -> Variables and secrets, add:

- `POSTGRES_DB=meddataops`
- `POSTGRES_USER=meddataops`
- `POSTGRES_PASSWORD=<strong-password>`
- `POSTGRES_PORT=5432`

Optional:

- `LOG_LEVEL=INFO`
- `MEDDATAOPS_MAX_STEPS=16`
- `GROQ_API_KEY=<your-groq-key>` (only if your runtime needs Groq-backed inference)

Note:

- The container starts PostgreSQL internally for single-container HF Spaces compatibility.
- `POSTGRES_HOST` is handled automatically by startup logic.

### 5. Verify deployment health

After build succeeds, verify:

```powershell
curl https://lazerai-meddataops.hf.space/health
curl https://lazerai-meddataops.hf.space/tasks
curl -X POST https://lazerai-meddataops.hf.space/reset -H "Content-Type: application/json" -d "{\"task_id\":\"triage_report\"}"
```

Also open `https://lazerai-meddataops.hf.space/` in a browser and run "Run Demo Episode".

### 6. Run pre-submission validation

Run your challenge validation script against the deployed Space URL.
Example pattern:

```powershell
python pre_submission_validation.py --base-url https://lazerai-meddataops.hf.space
```

If your validator has a different filename or flags, use its documented invocation with the same base URL.

## Common deployment errors and fixes

### Space not visible in account list

- Symptom: Space was created but not visible immediately.
- Fix: refresh profile and check build status at `https://huggingface.co/spaces/Lazerai/MedDataOps`; first build can take a few minutes.

### Build fails on apt packages

- Symptom: Docker build fails while installing PostgreSQL packages.
- Fix: Re-run build; ensure Dockerfile uses Debian-based `python:3.11-slim` with `postgresql` and `postgresql-client` packages.

### `/reset` returns 500 immediately after startup

- Symptom: Startup race with PostgreSQL readiness.
- Fix: Check container logs; confirm readiness wait completed before API accepted requests.

### Validation script reports `No active session`

- Symptom: `/step` or `/state` called without `/reset` first.
- Fix: Ensure validator starts each run with `POST /reset`, and preserves `X-Session-Id` (or cookie).

### Space loads but UI is blank

- Symptom: root landing page not served.
- Fix: Confirm `index.html` is present at repo root and server starts with `uvicorn server:app --port 7860`.

### CORS issues in browser demo panel

- Symptom: frontend requests blocked.
- Fix: Keep FastAPI CORS middleware enabled with headers/methods/origins allowed.

## API quick reference

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /health`
- `GET /tasks`

See landing page at `/` for interactive demo and copy-paste curl examples.
