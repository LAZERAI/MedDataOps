from __future__ import annotations

import os
import random
import threading
import uuid
from dataclasses import dataclass
from typing import Any

import psycopg2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from meddataops.db import PostgresBackend
from meddataops.tasks import get_task, list_tasks


ALLOWED_ACTIONS = {"clean_data", "run_query", "fix_query", "submit"}
DEFAULT_MAX_STEPS = int(os.getenv("MEDDATAOPS_MAX_STEPS", "16"))
PREVIEW_ROW_LIMIT = int(os.getenv("MEDDATAOPS_PREVIEW_ROW_LIMIT", "80"))


class ResetRequest(BaseModel):
    task_id: str | None = None
    seed: int | None = None


class StepRequest(BaseModel):
    action_type: str
    parameters: dict[str, Any] = Field(default_factory=dict)


@dataclass
class RuntimeSession:
    episode_id: str
    task_id: str
    task_name: str
    task_difficulty: str
    task_description: str
    task_hints: list[str]
    dataset_rows: list[dict[str, Any]]
    current_sql_query: str
    step_number: int
    max_steps: int
    done: bool
    last_info: dict[str, Any]
    error_messages: list[str]


class RuntimeEngine:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._session: RuntimeSession | None = None
        self._query_backend = PostgresBackend(dsn=os.getenv("MEDDATAOPS_POSTGRES_DSN"))

    def reset(self, request: ResetRequest) -> dict[str, Any]:
        with self._lock:
            rng = random.Random(request.seed)
            available = list_tasks()
            chosen_task_id = request.task_id or rng.choice(available)

            try:
                task = get_task(chosen_task_id)
            except KeyError as exc:
                raise HTTPException(status_code=400, detail=f"Unknown task_id: {chosen_task_id}") from exc

            self._session = RuntimeSession(
                episode_id=str(uuid.uuid4()),
                task_id=task.id,
                task_name=task.name,
                task_difficulty=task.difficulty.value,
                task_description=task.description,
                task_hints=list(task.hints),
                dataset_rows=list(task.dirty_rows),
                current_sql_query=task.broken_sql,
                step_number=0,
                max_steps=DEFAULT_MAX_STEPS,
                done=False,
                last_info={"message": "Environment reset."},
                error_messages=[],
            )

            return {
                "episode_id": self._session.episode_id,
                "task": self._task_summary(),
                "observation": self._observation(),
                "done": self._session.done,
            }

    def step(self, request: StepRequest) -> dict[str, Any]:
        with self._lock:
            session = self._require_session()

            if session.done:
                return {
                    "observation": self._observation(),
                    "done": True,
                    "info": {"message": "Episode already completed. Call /reset to start again."},
                }

            action_type = request.action_type.strip().lower()
            if action_type not in ALLOWED_ACTIONS:
                raise HTTPException(status_code=400, detail=f"Unsupported action_type: {action_type}")

            info: dict[str, Any] = {"action_type": action_type}
            session.error_messages = []

            if action_type == "clean_data":
                operations = request.parameters.get("operations", [])
                applied = len(operations) if isinstance(operations, list) else 0
                info["operations_applied"] = applied
                info["message"] = "Accepted clean_data request."

            elif action_type in {"run_query", "fix_query"}:
                query = request.parameters.get("query")
                if query is None and action_type == "run_query":
                    query = session.current_sql_query

                if not isinstance(query, str) or not query.strip():
                    session.error_messages = [f"{action_type} requires a non-empty query string."]
                    info["query_valid"] = False
                else:
                    normalized_query = query.strip()
                    if action_type == "fix_query":
                        session.current_sql_query = normalized_query

                    query_result = self._query_backend.validate_query(normalized_query)
                    info["query_valid"] = bool(query_result.success)
                    info["query_check"] = query_result.model_dump()
                    if not query_result.success:
                        session.error_messages = [query_result.error or "Query validation failed."]

            elif action_type == "submit":
                session.done = True
                info["message"] = "Submission accepted."

            session.step_number += 1
            if session.step_number >= session.max_steps and not session.done:
                session.done = True
                info["termination"] = "max_steps_reached"

            session.last_info = info

            return {
                "observation": self._observation(),
                "done": session.done,
                "info": info,
            }

    def state(self) -> dict[str, Any]:
        with self._lock:
            session = self._require_session()
            return {
                "episode_id": session.episode_id,
                "step_number": session.step_number,
                "max_steps": session.max_steps,
                "done": session.done,
                "task": self._task_summary(),
                "observation": self._observation(),
                "last_info": session.last_info,
            }

    def _require_session(self) -> RuntimeSession:
        if self._session is None:
            raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")
        return self._session

    def _observation(self) -> dict[str, Any]:
        session = self._require_session()
        return {
            "current_dataset_state": list(session.dataset_rows[:PREVIEW_ROW_LIMIT]),
            "current_sql_query": session.current_sql_query,
            "error_messages": list(session.error_messages),
            "task_description": session.task_description,
            "step_number": session.step_number,
        }

    def _task_summary(self) -> dict[str, Any]:
        session = self._require_session()
        return {
            "id": session.task_id,
            "name": session.task_name,
            "difficulty": session.task_difficulty,
            "description": session.task_description,
            "hints": list(session.task_hints),
        }


def _postgres_health() -> dict[str, Any]:
    host = os.getenv("POSTGRES_HOST", "127.0.0.1")
    dbname = os.getenv("POSTGRES_DB", "meddataops")
    user = os.getenv("POSTGRES_USER", "meddataops")
    password = os.getenv("POSTGRES_PASSWORD", "meddataops")
    port = int(os.getenv("POSTGRES_PORT", "5432"))

    try:
        with psycopg2.connect(
            host=host,
            dbname=dbname,
            user=user,
            password=password,
            port=port,
            connect_timeout=3,
        ) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
    except Exception as exc:  # pragma: no cover - runtime health path
        return {"status": "degraded", "postgres": "down", "error": str(exc)}

    return {"status": "ok", "postgres": "up"}


app = FastAPI(title="MedDataOps API", version="1.0.0")
engine = RuntimeEngine()


@app.get("/health")
def health() -> dict[str, Any]:
    health_payload = _postgres_health()
    if health_payload.get("status") != "ok":
        raise HTTPException(status_code=503, detail=health_payload)
    return health_payload


@app.post("/reset")
def reset(request: ResetRequest) -> dict[str, Any]:
    return engine.reset(request)


@app.post("/step")
def step(request: StepRequest) -> dict[str, Any]:
    return engine.step(request)


@app.get("/state")
def state() -> dict[str, Any]:
    return engine.state()
