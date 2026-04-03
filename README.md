# MedDataOps

MedDataOps is an OpenEnv-compatible RL environment for a clinical data engineering agent. The agent must clean noisy hospital rows and fix broken PostgreSQL queries across easy, medium, and hard tasks.

## Features

- OpenEnv spec with `openenv.yaml`
- Typed environment contracts via Pydantic models
- `reset()`, `step()`, and `state()` environment API
- Three built-in tasks: easy, medium, hard
- PostgreSQL query validation backend
- OpenAI-driven policy loop in `inference.py`
- Dockerized runtime

## Quickstart

1. Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Configure environment variables:

```powershell
Copy-Item .env.example .env
```

3. Set the source directory on `PYTHONPATH`:

```powershell
$env:PYTHONPATH = "src"
```

4. Run a local demo episode:

```powershell
python scripts/run_env.py
```

5. Run OpenAI policy inference:

```powershell
python inference.py
```

## PostgreSQL setup (optional but recommended)

Start PostgreSQL and load schema/seed scripts:

```powershell
psql -h localhost -U postgres -d meddataops -f src/meddataops/sql/schema.sql
psql -h localhost -U postgres -d meddataops -f src/meddataops/sql/seed.sql
```

## Docker

```powershell
docker build -t meddataops .
docker run --rm -e PYTHONPATH=/app/src meddataops
```
