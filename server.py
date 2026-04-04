from __future__ import annotations

import logging
import os
import random
import signal
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psycopg2
from fastapi import Cookie, FastAPI, Header, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from meddataops.db import PostgresBackend, PostgresDataManager
from meddataops.tasks import get_task, list_tasks


API_VERSION = "1.0.0"
DEFAULT_MAX_STEPS = int(os.getenv("MEDDATAOPS_MAX_STEPS", "20"))
SESSION_COOKIE_NAME = "session_id"
POSTGRES_READY_TIMEOUT_SECONDS = int(os.getenv("POSTGRES_READY_TIMEOUT_SECONDS", "60"))
POSTGRES_READY_RETRY_SECONDS = float(os.getenv("POSTGRES_READY_RETRY_SECONDS", "1.0"))
ROOT_DIR = Path(__file__).resolve().parent
INDEX_HTML_PATH = ROOT_DIR / "index.html"
FAVICON_ICO_PATH = ROOT_DIR / "favicon.ico"
FAVICON_SVG_PATH = ROOT_DIR / "favicon.svg"

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Content-Security-Policy": (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com data:; "
        "img-src 'self' data: https:; "
        "connect-src 'self' https://lazerai-meddataops.hf.space; "
        "object-src 'none'; base-uri 'self'; frame-ancestors 'none'"
    ),
}

logger = logging.getLogger("meddataops.server")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)


class ResetRequestModel(BaseModel):
    task_id: str | None = None
    seed: int | None = None


class ActionRequestModel(BaseModel):
    action_type: str = Field(..., min_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict)


class ObservationResponseModel(BaseModel):
    current_dataset_state: list[dict[str, Any]] = Field(default_factory=list)
    current_sql_query: str = ""
    error_messages: list[str] = Field(default_factory=list)
    task_description: str = ""
    step_number: int = 0


class StepResponseModel(BaseModel):
    observation: dict[str, Any]
    reward: Any
    done: bool
    info: dict[str, Any]


class StateResponseModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    done: bool | None = None
    step_number: int | None = None
    max_steps: int | None = None
    observation: dict[str, Any] | None = None


class HealthResponseModel(BaseModel):
    status: str
    version: str


class TaskMetadataModel(BaseModel):
    id: str
    name: str
    difficulty: str
    description: str
    hints: list[str] = Field(default_factory=list)
    dirty_row_count: int
    has_expected_sql: bool


class TasksResponseModel(BaseModel):
    tasks: list[TaskMetadataModel]


def _to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {"value": value}


def _normalize_observation(raw: Any) -> dict[str, Any]:
    payload = _to_dict(raw)

    if "current_dataset_state" in payload:
        return {
            "current_dataset_state": payload.get("current_dataset_state", []),
            "current_sql_query": payload.get("current_sql_query", ""),
            "error_messages": payload.get("error_messages", []),
            "task_description": payload.get("task_description", ""),
            "step_number": int(payload.get("step_number", 0)),
        }

    dirty_rows = payload.get("dirty_rows", [])
    task = payload.get("task", {})
    return {
        "current_dataset_state": dirty_rows,
        "current_sql_query": payload.get("broken_sql", ""),
        "error_messages": [payload.get("last_sql_error")] if payload.get("last_sql_error") else [],
        "task_description": task.get("description", ""),
        "step_number": int(payload.get("step_index", 0)),
    }


class FallbackEnvironment:
    """Fallback runtime used when meddataops.env cannot be imported.

    Provides reset/step/state compatible behavior for HTTP contract stability.
    """

    ALLOWED_ACTIONS = {"clean_data", "run_query", "fix_query", "submit"}

    def __init__(self, *, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self._query_backend = PostgresBackend(dsn=os.getenv("MEDDATAOPS_POSTGRES_DSN"))

        self._task_id: str | None = None
        self._task_name: str = ""
        self._task_difficulty: str = ""
        self._task_description: str = ""
        self._task_hints: list[str] = []

        self._dataset_rows: list[dict[str, Any]] = []
        self._current_sql_query: str = ""
        self._step_number: int = 0
        self._done: bool = False
        self._last_reward: float = 0.0
        self._last_info: dict[str, Any] = {}
        self._error_messages: list[str] = []

    def reset(self, *, task_id: str | None = None, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self._rng.seed(seed)

        available = list_tasks()
        chosen_task_id = task_id or self._rng.choice(available)

        task = get_task(chosen_task_id)

        self._task_id = task.id
        self._task_name = task.name
        self._task_difficulty = task.difficulty.value
        self._task_description = task.description
        self._task_hints = list(task.hints)

        self._dataset_rows = list(task.dirty_rows)
        self._current_sql_query = task.broken_sql
        self._step_number = 0
        self._done = False
        self._last_reward = 0.0
        self._last_info = {"message": "Environment reset."}
        self._error_messages = []

        return self._observation()

    def step(self, action: dict[str, Any]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        if self._task_id is None:
            raise ValueError("No active session. Call reset first.")

        if self._done:
            return self._observation(), self._last_reward, True, {
                "message": "Episode already completed. Call /reset to start again."
            }

        action_type = str(action.get("action_type", "")).strip().lower()
        parameters = action.get("parameters", {})
        if not isinstance(parameters, dict):
            raise ValueError("Action parameters must be a JSON object.")

        if action_type not in self.ALLOWED_ACTIONS:
            raise ValueError(f"Unsupported action_type: {action_type}")

        info: dict[str, Any] = {"action_type": action_type}
        self._error_messages = []

        if action_type == "clean_data":
            operations = parameters.get("operations", [])
            info["operations_applied"] = len(operations) if isinstance(operations, list) else 0
            info["message"] = "Accepted clean_data request."

        elif action_type in {"run_query", "fix_query"}:
            query = parameters.get("query")
            if query is None and action_type == "run_query":
                query = self._current_sql_query

            if not isinstance(query, str) or not query.strip():
                self._error_messages = [f"{action_type} requires a non-empty query string."]
                info["query_valid"] = False
            else:
                normalized_query = query.strip()
                if action_type == "fix_query":
                    self._current_sql_query = normalized_query

                query_result = self._query_backend.validate_query(normalized_query)
                info["query_valid"] = bool(query_result.success)
                info["query_check"] = query_result.model_dump()
                if not query_result.success:
                    self._error_messages = [query_result.error or "Query validation failed."]

        elif action_type == "submit":
            self._done = True
            info["message"] = "Submission accepted."

        self._step_number += 1
        if self._step_number >= DEFAULT_MAX_STEPS and not self._done:
            self._done = True
            info["termination"] = "max_steps_reached"

        self._last_reward = 0.0
        self._last_info = info

        return self._observation(), self._last_reward, self._done, info

    def state(self) -> dict[str, Any]:
        if self._task_id is None:
            raise ValueError("No active session. Call reset first.")

        return {
            "task": {
                "id": self._task_id,
                "name": self._task_name,
                "difficulty": self._task_difficulty,
                "description": self._task_description,
                "hints": list(self._task_hints),
            },
            "observation": self._observation(),
            "latest_reward": self._last_reward,
            "done": self._done,
            "step_number": self._step_number,
            "max_steps": DEFAULT_MAX_STEPS,
            "last_info": dict(self._last_info),
        }

    def close(self) -> None:
        return

    def _observation(self) -> dict[str, Any]:
        return {
            "current_dataset_state": list(self._dataset_rows),
            "current_sql_query": self._current_sql_query,
            "error_messages": list(self._error_messages),
            "task_description": self._task_description,
            "step_number": self._step_number,
        }


class SessionEnvironmentAdapter:
    def __init__(self, *, seed: int | None = None) -> None:
        self._seed = seed
        self._env = self._create_runtime(seed)

    @staticmethod
    def _create_runtime(seed: int | None) -> Any:
        try:
            from meddataops.env import MedDataOpsEnv  # type: ignore

            return MedDataOpsEnv(seed=seed)
        except Exception as exc:  # pragma: no cover - fallback path depends on runtime state
            logger.warning("Falling back to internal runtime adapter because meddataops.env import failed: %s", exc)
            return FallbackEnvironment(seed=seed)

    def reset(self, *, task_id: str | None = None, seed: int | None = None) -> dict[str, Any]:
        attempts = []

        if seed is not None:
            attempts.extend(
                [
                    lambda: self._env.reset(task_id=task_id, seed=seed),
                    lambda: self._env.reset(task_id=task_id),
                    lambda: self._env.reset(task_id),
                ]
            )
        else:
            attempts.extend(
                [
                    lambda: self._env.reset(task_id=task_id),
                    lambda: self._env.reset(task_id),
                ]
            )

        attempts.extend(
            [
                lambda: self._env.reset(),
                lambda: self._env.reset({"task_id": task_id, "seed": seed}),
            ]
        )

        last_error: Exception | None = None
        for attempt in attempts:
            try:
                result = attempt()
                return _normalize_observation(result)
            except TypeError as exc:
                last_error = exc
                continue
            except Exception as exc:
                last_error = exc
                raise

        raise RuntimeError(f"Unable to reset environment: {last_error}")

    def step(self, action: dict[str, Any]) -> tuple[dict[str, Any], Any, bool, dict[str, Any]]:
        attempts = [
            lambda: self._env.step(action),
            lambda: self._env.step(action=action),
        ]

        last_error: Exception | None = None
        for attempt in attempts:
            try:
                raw_result = attempt()
                break
            except TypeError as exc:
                last_error = exc
                continue
        else:
            raise RuntimeError(f"Unable to execute step on environment: {last_error}")

        if isinstance(raw_result, tuple) and len(raw_result) >= 4:
            observation, reward, done, info = raw_result[0], raw_result[1], raw_result[2], raw_result[3]
            return _normalize_observation(observation), _to_dict(reward) or reward, bool(done), _to_dict(info)

        result_dict = _to_dict(raw_result)
        observation = _normalize_observation(result_dict.get("observation", result_dict))
        reward = result_dict.get("reward", 0.0)
        done = bool(result_dict.get("done", False))
        info = _to_dict(result_dict.get("info", {}))
        return observation, reward, done, info

    def state(self) -> dict[str, Any]:
        if hasattr(self._env, "state"):
            result = self._env.state()
            return _to_dict(result)
        raise RuntimeError("Environment does not expose a state() method.")

    def close(self) -> None:
        close_fn = getattr(self._env, "close", None)
        if callable(close_fn):
            close_fn()


@dataclass
class SessionContext:
    session_id: str
    adapter: SessionEnvironmentAdapter
    created_at: float
    updated_at: float
    working_tables: dict[str, str] = field(default_factory=dict)


_sessions: dict[str, SessionContext] = {}
_sessions_lock = threading.RLock()
_db_manager: PostgresDataManager | None = None
_sigterm_handler_installed = False


def _postgres_kwargs() -> dict[str, Any]:
    return {
        "host": os.getenv("POSTGRES_HOST", "127.0.0.1"),
        "dbname": os.getenv("POSTGRES_DB", "meddataops"),
        "user": os.getenv("POSTGRES_USER", "meddataops"),
        "password": os.getenv("POSTGRES_PASSWORD", "meddataops"),
        "port": int(os.getenv("POSTGRES_PORT", "5432")),
        "connect_timeout": 3,
    }


def _wait_for_postgres_ready() -> None:
    start = time.monotonic()
    last_error: Exception | None = None

    while time.monotonic() - start < POSTGRES_READY_TIMEOUT_SECONDS:
        try:
            with psycopg2.connect(**_postgres_kwargs()) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
            logger.info("PostgreSQL is ready.")
            return
        except Exception as exc:  # pragma: no cover - startup infra path
            last_error = exc
            time.sleep(POSTGRES_READY_RETRY_SECONDS)

    raise RuntimeError(f"PostgreSQL not ready after {POSTGRES_READY_TIMEOUT_SECONDS}s: {last_error}")


def _resolve_session_id(
    *,
    x_session_id: str | None,
    session_id_header: str | None,
    session_id_cookie: str | None,
) -> tuple[str, bool]:
    candidate = (x_session_id or session_id_header or session_id_cookie or "").strip()
    if candidate:
        return candidate, False
    return str(uuid.uuid4()), True


def _get_session_context(session_id: str) -> SessionContext | None:
    with _sessions_lock:
        return _sessions.get(session_id)


def _ensure_session_context(session_id: str, *, seed: int | None = None) -> SessionContext:
    with _sessions_lock:
        existing = _sessions.get(session_id)
        if existing is not None:
            existing.updated_at = time.time()
            return existing

        context = SessionContext(
            session_id=session_id,
            adapter=SessionEnvironmentAdapter(seed=seed),
            created_at=time.time(),
            updated_at=time.time(),
        )
        _sessions[session_id] = context
        return context


def _cleanup_session(session_id: str, context: SessionContext) -> None:
    try:
        context.adapter.close()
    except Exception as exc:  # pragma: no cover - cleanup path
        logger.warning("Session close failed for %s: %s", session_id, exc)

    if _db_manager is not None:
        try:
            _db_manager.cleanup_episode(session_id)
        except Exception as exc:  # pragma: no cover - cleanup path
            logger.warning("Temp-table cleanup failed for %s: %s", session_id, exc)


def _cleanup_all_sessions() -> None:
    with _sessions_lock:
        items = list(_sessions.items())
        _sessions.clear()

    for session_id, context in items:
        _cleanup_session(session_id, context)


def _drop_session_context(session_id: str) -> None:
    with _sessions_lock:
        context = _sessions.pop(session_id, None)
    if context is not None:
        _cleanup_session(session_id, context)


def _install_sigterm_handler() -> None:
    global _sigterm_handler_installed
    if _sigterm_handler_installed:
        return

    def _handle_sigterm(signum: int, _frame: Any) -> None:
        logger.info("Received signal %s. Cleaning up session resources.", signum)
        _cleanup_all_sessions()

    signal.signal(signal.SIGTERM, _handle_sigterm)
    _sigterm_handler_installed = True


app = FastAPI(title="MedDataOps HTTP Server", version=API_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_response_logging(request: Request, call_next):
    started = time.perf_counter()
    sid = request.headers.get("x-session-id") or request.headers.get("session_id") or request.cookies.get("session_id") or "-"
    try:
        response = await call_next(request)
    except Exception:
        duration_ms = (time.perf_counter() - started) * 1000.0
        logger.exception(
            "request_failed method=%s path=%s session_id=%s duration_ms=%.2f",
            request.method,
            request.url.path,
            sid,
            duration_ms,
        )
        raise

    duration_ms = (time.perf_counter() - started) * 1000.0
    logger.info(
        "request method=%s path=%s status=%s session_id=%s duration_ms=%.2f",
        request.method,
        request.url.path,
        response.status_code,
        sid,
        duration_ms,
    )

    for header_name, header_value in SECURITY_HEADERS.items():
        response.headers.setdefault(header_name, header_value)

    return response


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": exc.errors()})


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled server error: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.on_event("startup")
def on_startup() -> None:
    global _db_manager

    _wait_for_postgres_ready()
    _db_manager = PostgresDataManager(
        minconn=1,
        maxconn=8,
        host=os.getenv("POSTGRES_HOST", "127.0.0.1"),
        dbname=os.getenv("POSTGRES_DB", "meddataops"),
        user=os.getenv("POSTGRES_USER", "meddataops"),
        password=os.getenv("POSTGRES_PASSWORD", "meddataops"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
    )
    _install_sigterm_handler()
    logger.info("Startup complete.")


@app.on_event("shutdown")
def on_shutdown() -> None:
    global _db_manager

    _cleanup_all_sessions()
    if _db_manager is not None:
        try:
            _db_manager.close()
        except Exception as exc:  # pragma: no cover - shutdown path
            logger.warning("Failed to close DB manager cleanly: %s", exc)
        _db_manager = None
    logger.info("Shutdown complete.")


@app.get("/", include_in_schema=False)
def landing_page() -> Response:
    if INDEX_HTML_PATH.exists():
        return FileResponse(INDEX_HTML_PATH)
    return HTMLResponse(
        "<html><body><h1>MedDataOps</h1><p>Landing page not found. Add index.html.</p></body></html>",
        status_code=200,
    )


@app.head("/", include_in_schema=False)
def landing_page_head() -> Response:
    return Response(status_code=200, media_type="text/html")


@app.get("/favicon.svg", include_in_schema=False)
def favicon_svg() -> Response:
    if FAVICON_SVG_PATH.exists():
        return FileResponse(FAVICON_SVG_PATH, media_type="image/svg+xml")
    return Response(status_code=204)


@app.get("/favicon.ico", include_in_schema=False)
def favicon_ico() -> Response:
    if FAVICON_ICO_PATH.exists():
        return FileResponse(FAVICON_ICO_PATH, media_type="image/x-icon")
    if FAVICON_SVG_PATH.exists():
        return FileResponse(FAVICON_SVG_PATH, media_type="image/svg+xml")
    return Response(status_code=204)


@app.get("/health", response_model=HealthResponseModel)
def health() -> HealthResponseModel:
    return HealthResponseModel(status="ok", version=API_VERSION)


@app.get("/tasks", response_model=TasksResponseModel)
def tasks() -> TasksResponseModel:
    task_items: list[TaskMetadataModel] = []
    for task_id in sorted(list_tasks()):
        task = get_task(task_id)
        task_items.append(
            TaskMetadataModel(
                id=task.id,
                name=task.name,
                difficulty=task.difficulty.value,
                description=task.description,
                hints=list(task.hints),
                dirty_row_count=len(task.dirty_rows),
                has_expected_sql=bool(task.expected_sql),
            )
        )
    return TasksResponseModel(tasks=task_items)


@app.post("/reset", response_model=ObservationResponseModel)
def reset(
    payload: ResetRequestModel,
    response: Response,
    x_session_id: str | None = Header(default=None),
    session_id_header: str | None = Header(default=None, alias="session_id"),
    session_id_cookie: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME),
) -> ObservationResponseModel:
    session_id, _created = _resolve_session_id(
        x_session_id=x_session_id,
        session_id_header=session_id_header,
        session_id_cookie=session_id_cookie,
    )

    context = _ensure_session_context(session_id, seed=payload.seed)

    try:
        if _db_manager is not None:
            try:
                _db_manager.cleanup_episode(session_id)
            except Exception:
                pass
            context.working_tables = _db_manager.create_episode_working_tables(session_id=session_id, version="messy")

        observation = context.adapter.reset(task_id=payload.task_id, seed=payload.seed)
    except HTTPException:
        _drop_session_context(session_id)
        raise
    except KeyError as exc:
        _drop_session_context(session_id)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        _drop_session_context(session_id)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Reset failed for session=%s: %s", session_id, exc)
        _drop_session_context(session_id)
        raise HTTPException(status_code=500, detail="Failed to reset environment") from exc

    response.set_cookie(key=SESSION_COOKIE_NAME, value=session_id, httponly=True, samesite="lax")
    response.headers["X-Session-Id"] = session_id
    return ObservationResponseModel.model_validate(observation)


@app.post("/step", response_model=StepResponseModel)
def step(
    payload: ActionRequestModel,
    response: Response,
    x_session_id: str | None = Header(default=None),
    session_id_header: str | None = Header(default=None, alias="session_id"),
    session_id_cookie: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME),
) -> StepResponseModel:
    session_id, _ = _resolve_session_id(
        x_session_id=x_session_id,
        session_id_header=session_id_header,
        session_id_cookie=session_id_cookie,
    )

    context = _get_session_context(session_id)
    if context is None:
        raise HTTPException(status_code=400, detail="No active session. Call /reset first.")

    action_payload = {
        "action_type": payload.action_type.strip().lower(),
        "parameters": payload.parameters,
    }

    try:
        observation, reward, done, info = context.adapter.step(action_payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Step failed for session=%s: %s", session_id, exc)
        _drop_session_context(session_id)
        raise HTTPException(status_code=500, detail="Failed to execute step") from exc

    response.set_cookie(key=SESSION_COOKIE_NAME, value=session_id, httponly=True, samesite="lax")
    response.headers["X-Session-Id"] = session_id
    return StepResponseModel(observation=observation, reward=reward, done=done, info=info)


@app.get("/state", response_model=StateResponseModel)
def state(
    response: Response,
    x_session_id: str | None = Header(default=None),
    session_id_header: str | None = Header(default=None, alias="session_id"),
    session_id_cookie: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME),
) -> StateResponseModel:
    session_id, _ = _resolve_session_id(
        x_session_id=x_session_id,
        session_id_header=session_id_header,
        session_id_cookie=session_id_cookie,
    )

    context = _get_session_context(session_id)
    if context is None:
        raise HTTPException(status_code=400, detail="No active session. Call /reset first.")

    try:
        state_payload = context.adapter.state()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("State retrieval failed for session=%s: %s", session_id, exc)
        _drop_session_context(session_id)
        raise HTTPException(status_code=500, detail="Failed to fetch state") from exc

    response.set_cookie(key=SESSION_COOKIE_NAME, value=session_id, httponly=True, samesite="lax")
    response.headers["X-Session-Id"] = session_id
    return StateResponseModel.model_validate(state_payload)
