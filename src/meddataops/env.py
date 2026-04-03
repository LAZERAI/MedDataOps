from __future__ import annotations

import random
import uuid
from typing import Any

from meddataops.config import settings
from meddataops.db import PostgresBackend
from meddataops.models import (
    ActionType,
    AgentAction,
    EnvironmentState,
    import os
    ResetRequest,
    StepResult,
    TaskPublic,
    TaskSpec,
    import psycopg2
    from psycopg2 import OperationalError, sql
    from psycopg2.extensions import connection as PsycopgConnection
    from psycopg2.extras import Json, RealDictCursor
    from pydantic import ValidationError

from meddataops.tasks import get_task, list_tasks
        ActionModel,
        HistoryEntryModel,
        ObservationModel,
        OpenEnvActionType,
        RewardModel,
        StateModel,
        postgres_backend: PostgresBackend | None = None,
        max_steps: int | None = None,
        seed: int | None = None,
    from meddataops.scoring import RewardCalculator
        self._rng = random.Random(seed)
        self._db = postgres_backend or PostgresBackend(settings.postgres_dsn)
        self._max_steps = max_steps or settings.max_steps

        """OpenEnv-compatible environment for clinical data cleaning and SQL repair.

        Public interface:
        - reset() -> ObservationModel
        - step(action: ActionModel) -> tuple[ObservationModel, RewardModel, bool, dict[str, Any]]
        - state() -> StateModel
        """

        TASK_ALIASES: dict[str, str] = {
            "triage_report": "easy",
            "medication_summary": "medium",
            "icu_capacity": "hard",
        }

        TASK_TABLE_MAP: dict[str, str] = {
            "easy": "admissions",
            "medium": "lab_results",
            "hard": "icu_events",
        }

        TABLE_COLUMNS: dict[str, list[str]] = {
            "admissions": ["patient_id", "heart_rate", "ward", "encounter_id", "admit_ts"],
            "lab_results": ["encounter_id", "test_code", "result_value", "unit", "taken_at"],
            "icu_events": ["patient_id", "event_ts", "heart_rate", "spo2"],
        }

        self._episode_id: str = ""
        self._step_index: int = 0
            max_steps: int = 20,

            query_cost_threshold: float = 1000.0,
        self._task: TaskSpec | None = None
        self._candidate_rows: list[dict[str, Any]] = []
            self._max_steps = max_steps
            self._reward_calculator = RewardCalculator(query_cost_threshold=query_cost_threshold)

            self._conn: PsycopgConnection | None = None
            self._working_table_name: str = ""
        self._solved_cleaning: bool = False
        self._solved_sql: bool = False
        self._last_sql_error: str | None = None
            self._cumulative_reward: float = 0.0
    def reset(self, request: ResetRequest | None = None) -> EnvironmentState:
            self._unnecessary_actions: int = 0
        request = request or ResetRequest()

            self._candidate_clean_rows: list[dict[str, Any]] = []
            self._current_query: str = ""
            self._last_query_rows: list[dict[str, Any]] = []
            self._last_query_plan_cost: float | None = None

            self._latest_reward: RewardModel = self._zero_reward()
            self._history: list[HistoryEntryModel] = []
            self._error_messages: list[str] = []
        task_id = request.task_id or self._rng.choice(list_tasks())
            self._solved_query: bool = False

        self._episode_id = str(uuid.uuid4())
        def reset(self, task_id: str | None = None, seed: int | None = None) -> ObservationModel:
            """Start a new episode and return the initial observation.

            On reset:
            - Opens a fresh PostgreSQL session (episode isolation boundary).
            - Creates per-episode temporary tables.
            - Loads task dataset into temp working tables.
            - Initializes step counters and query state.
            """
            if seed is not None:
                self._rng.seed(seed)

            resolved_task_id = self._resolve_task_id(task_id)
            self._task = get_task(resolved_task_id)

            self._connect_new_episode()

            self._episode_id = str(uuid.uuid4())
            self._step_index = 0
            self._cumulative_reward = 0.0
            self._done = False
            self._unnecessary_actions = 0

            self._working_table_name = f"meddataops_work_{uuid.uuid4().hex[:10]}"

            self._candidate_clean_rows = list(self._task.dirty_rows)
            self._current_query = self._task.broken_sql
            self._last_query_rows = []
            self._last_query_plan_cost = None

            self._latest_reward = self._zero_reward()
            self._history = []
            self._error_messages = []
            self._solved_cleaning = False
            self._solved_query = False
            self._last_sql_error = None

            self._create_temp_tables()
            self._load_rows_for_task(self._task.dirty_rows)
            self._load_rows_into_working_table(self._task.dirty_rows)

            return self._build_observation()

        def step(self, action: ActionModel) -> tuple[ObservationModel, RewardModel, bool, dict[str, Any]]:
            """Apply one action and return (observation, reward, done, info)."""
            self._ensure_active_episode()

            if self._done:
                return self._build_observation(), self._latest_reward, True, {
                    "message": "Episode already completed. Call reset() to start a new episode."
                }

            try:
                if not isinstance(action, ActionModel):
                    action = ActionModel.model_validate(action)
            except ValidationError as exc:
                self._error_messages = [f"Invalid action payload: {exc}"]
                penalty_reward = self._reward_calculator.calculate(
                    candidate_clean_rows=self._candidate_clean_rows,
                    expected_clean_rows=self._task.expected_clean_rows if self._task else [],
                    candidate_query_rows=self._last_query_rows,
                    expected_query_rows=[],
                    sql_error="invalid_action",
                    query_plan_cost=self._last_query_plan_cost,
                    unnecessary_actions=1,
                )
                self._latest_reward = penalty_reward
                self._cumulative_reward += penalty_reward.total
                return self._build_observation(), penalty_reward, False, {"error": str(exc)}

            self._error_messages = []
            info: dict[str, Any] = {}
            reward = self._zero_reward()
            unnecessary = False

            if action.action_type == OpenEnvActionType.CLEAN_DATA:
                info, unnecessary = self._handle_clean_data(action.parameters)
            elif action.action_type == OpenEnvActionType.RUN_QUERY:
                info, unnecessary = self._handle_run_query(action.parameters)
            elif action.action_type == OpenEnvActionType.FIX_QUERY:
                info, unnecessary = self._handle_fix_query(action.parameters)
            elif action.action_type == OpenEnvActionType.SUBMIT:
                reward, info = self._handle_submit()
            else:
                unnecessary = True
                self._error_messages.append(f"Unsupported action type: {action.action_type}")
                info = {"error": f"Unsupported action type: {action.action_type}"}

            if unnecessary:
                self._unnecessary_actions += 1

            self._step_index += 1

            if self._step_index >= self._max_steps and not self._done:
                reward, forced_info = self._handle_submit(force_reason="max_steps_reached")
                info["termination"] = "max_steps_reached"
                info["forced_submit"] = forced_info

            if action.action_type != OpenEnvActionType.SUBMIT and not self._done:
                reward = self._zero_reward()

            observation = self._build_observation()
            self._history.append(
                HistoryEntryModel(
                    step_number=self._step_index,
                    action=action,
                    observation=observation,
                    reward=reward,
                )
            )
            self._latest_reward = reward
            self._cumulative_reward += reward.total

            return observation, reward, self._done, info

        def state(self) -> StateModel:
            """Return full internal environment state for the active episode."""
            self._ensure_active_episode()

            assert self._task is not None
            task_public = TaskPublic(
                id=self._task.id,
                name=self._task.name,
                difficulty=self._task.difficulty,
                description=self._task.description,
                hints=self._task.hints,
            )

            return StateModel(
                episode_id=self._episode_id,
                step_number=self._step_index,
                max_steps=self._max_steps,
                done=self._done,
                current_task=task_public,
                observation=self._build_observation(),
                latest_reward=self._latest_reward,
                cumulative_reward=self._cumulative_reward,
                solved_cleaning=self._solved_cleaning,
                solved_query=self._solved_query,
                last_sql_error=self._last_sql_error,
                history=list(self._history),
            )

        def close(self) -> None:
            """Close DB resources for the current episode."""
            if self._conn is not None:
                try:
                    self._conn.close()
                finally:
                    self._conn = None

        def __del__(self) -> None:
            self.close()

        def _resolve_task_id(self, task_id: str | None) -> str:
            if task_id is None:
                return self._rng.choice(list_tasks())

            if task_id in self.TASK_ALIASES:
                return self.TASK_ALIASES[task_id]

            if task_id in list_tasks():
                return task_id

            aliases = ", ".join(sorted(self.TASK_ALIASES.keys()))
            canonical = ", ".join(sorted(list_tasks()))
            raise ValueError(
                f"Unknown task_id '{task_id}'. Supported canonical ids: {canonical}. Supported aliases: {aliases}."
            )

        def _connection_kwargs(self) -> dict[str, Any]:
            host = os.getenv("POSTGRES_HOST")
            dbname = os.getenv("POSTGRES_DB")
            user = os.getenv("POSTGRES_USER")
            password = os.getenv("POSTGRES_PASSWORD")
            port_raw = os.getenv("POSTGRES_PORT", "5432")

            missing = [
                name
                for name, value in {
                    "POSTGRES_HOST": host,
                    "POSTGRES_DB": dbname,
                    "POSTGRES_USER": user,
                    "POSTGRES_PASSWORD": password,
                }.items()
                if not value
            ]
            if missing:
                raise RuntimeError(f"Missing required PostgreSQL environment variables: {', '.join(missing)}")

            try:
                port = int(port_raw)
            except ValueError as exc:
                raise RuntimeError("POSTGRES_PORT must be an integer.") from exc

            return {
                "host": host,
                "dbname": dbname,
                "user": user,
                "password": password,
                "port": port,
                "connect_timeout": 5,
            }

        def _connect_new_episode(self) -> None:
            self.close()

            try:
                self._conn = psycopg2.connect(**self._connection_kwargs())
                self._conn.autocommit = True
            except OperationalError as exc:
                raise RuntimeError(f"Failed to connect to PostgreSQL: {exc}") from exc

        def _ensure_active_episode(self) -> None:
            if self._task is None or self._conn is None:
                raise RuntimeError("No active episode. Call reset() before step() or state().")

        def _create_temp_tables(self) -> None:
            self._ensure_active_episode()
            assert self._conn is not None

            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TEMP TABLE admissions (
                        patient_id TEXT,
                        heart_rate TEXT,
                        ward TEXT,
                        encounter_id TEXT,
                        admit_ts TEXT
                    ) ON COMMIT PRESERVE ROWS;
                    """
                )
                cur.execute(
                    """
                    CREATE TEMP TABLE lab_results (
                        encounter_id TEXT,
                        test_code TEXT,
                        result_value TEXT,
                        unit TEXT,
                        taken_at TEXT
                    ) ON COMMIT PRESERVE ROWS;
                    """
                )
                cur.execute(
                    """
                    CREATE TEMP TABLE icu_events (
                        patient_id TEXT,
                        event_ts TEXT,
                        heart_rate TEXT,
                        spo2 TEXT
                    ) ON COMMIT PRESERVE ROWS;
                    """
                )
                cur.execute(
                    sql.SQL(
                        "CREATE TEMP TABLE {} (row_id SERIAL PRIMARY KEY, payload JSONB NOT NULL) ON COMMIT PRESERVE ROWS;"
                    ).format(sql.Identifier(self._working_table_name))
                )

        def _load_rows_for_task(self, rows: list[dict[str, Any]]) -> None:
            self._ensure_active_episode()
            assert self._task is not None
            assert self._conn is not None

            table_name = self.TASK_TABLE_MAP[self._task.id]
            columns = self.TABLE_COLUMNS[table_name]

            with self._conn.cursor() as cur:
                cur.execute(sql.SQL("TRUNCATE TABLE {}").format(sql.Identifier(table_name)))

                for row in rows:
                    values = [None if row.get(column) is None else str(row.get(column)) for column in columns]
                    cur.execute(
                        sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
                            sql.Identifier(table_name),
                            sql.SQL(", ").join(sql.Identifier(column) for column in columns),
                            sql.SQL(", ").join(sql.Placeholder() for _ in columns),
                        ),
                        values,
                    )

            if self._task.id == "medium":
                self._seed_medium_admissions(rows)

        def _seed_medium_admissions(self, source_rows: list[dict[str, Any]]) -> None:
            self._ensure_active_episode()
            assert self._conn is not None

            seen: set[str] = set()
            rows_to_insert: list[tuple[str, str]] = []
            for row in source_rows:
                encounter_id = str(row.get("encounter_id", "")).strip()
                if not encounter_id or encounter_id in seen:
                    continue

                seen.add(encounter_id)
                raw_ts = str(row.get("taken_at", "2026-03-01 08:00:00"))
                normalized_ts = raw_ts.replace("T", " ").replace("/", "-").strip()[:19]
                rows_to_insert.append((encounter_id, normalized_ts))

            with self._conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE admissions")
                for encounter_id, admit_ts in rows_to_insert:
                    cur.execute(
                        "INSERT INTO admissions (encounter_id, admit_ts) VALUES (%s, %s)",
                        (encounter_id, admit_ts),
                    )

        def _load_rows_into_working_table(self, rows: list[dict[str, Any]]) -> None:
            self._ensure_active_episode()
            assert self._conn is not None

            with self._conn.cursor() as cur:
                cur.execute(sql.SQL("TRUNCATE TABLE {}").format(sql.Identifier(self._working_table_name)))
                for row in rows:
                    cur.execute(
                        sql.SQL("INSERT INTO {} (payload) VALUES (%s)").format(sql.Identifier(self._working_table_name)),
                        (Json(row),),
                    )

        def _fetch_working_rows(self) -> list[dict[str, Any]]:
            self._ensure_active_episode()
            assert self._conn is not None

            with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    sql.SQL("SELECT payload FROM {} ORDER BY row_id ASC").format(sql.Identifier(self._working_table_name))
                )
                result = cur.fetchall()

            rows: list[dict[str, Any]] = []
            for row in result:
                payload = row.get("payload")
                if isinstance(payload, dict):
                    rows.append(payload)

            return rows

        def _validate_select_query(self, query: str) -> str:
            normalized = query.strip()
            if not normalized:
                raise ValueError("Query cannot be empty.")

            lowered = normalized.lower()
            if not lowered.startswith("select"):
                raise ValueError("Only SELECT statements are allowed.")

            if ";" in normalized.rstrip(";"):
                raise ValueError("Multiple SQL statements are not allowed.")

            return normalized

        def _execute_query(
            self,
            query: str,
            include_plan: bool = False,
        ) -> tuple[list[dict[str, Any]], str | None, float | None]:
            self._ensure_active_episode()
            assert self._conn is not None

            try:
                safe_query = self._validate_select_query(query)
            except ValueError as exc:
                return [], str(exc), None

            try:
                plan_cost: float | None = None
                with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
                    if include_plan:
                        cur.execute(f"EXPLAIN (FORMAT JSON) {safe_query}")
                        plan_row = cur.fetchone() or {}
                        plan_payload = plan_row.get("QUERY PLAN")
                        plan_cost = self._reward_calculator.query_plan_total_cost(plan_payload)

                    cur.execute(safe_query)
                    rows = cur.fetchall() if cur.description is not None else []
                    result_rows = [dict(row) for row in rows]

                return result_rows, None, plan_cost
            except Exception as exc:
                return [], str(exc), None

        def _handle_clean_data(self, parameters: dict[str, Any]) -> tuple[dict[str, Any], bool]:
            rows = parameters.get("rows")
            if not isinstance(rows, list) or any(not isinstance(item, dict) for item in rows):
                message = "clean_data requires parameters.rows as list[dict]."
                self._error_messages.append(message)
                return {"error": message}, True

            try:
                self._candidate_clean_rows = rows
                self._load_rows_for_task(rows)
                self._load_rows_into_working_table(rows)
            except Exception as exc:
                message = f"Failed to load cleaned rows: {exc}"
                self._error_messages.append(message)
                return {"error": message}, True

            assert self._task is not None
            clean_score = self._reward_calculator.data_clean_score(rows, self._task.expected_clean_rows)
            self._solved_cleaning = clean_score >= 0.999

            return {
                "status": "cleaned",
                "rows_loaded": len(rows),
                "clean_score": clean_score,
                "solved_cleaning": self._solved_cleaning,
            }, False

        def _handle_fix_query(self, parameters: dict[str, Any]) -> tuple[dict[str, Any], bool]:
            query = parameters.get("query")
            if not isinstance(query, str) or not query.strip():
                message = "fix_query requires a non-empty parameters.query string."
                self._error_messages.append(message)
                return {"error": message}, True

            self._current_query = query
            self._last_sql_error = None
            return {"status": "query_updated"}, False

        def _handle_run_query(self, parameters: dict[str, Any]) -> tuple[dict[str, Any], bool]:
            query_override = parameters.get("query")
            if query_override is not None:
                if not isinstance(query_override, str) or not query_override.strip():
                    message = "run_query parameters.query must be a non-empty string when provided."
                    self._error_messages.append(message)
                    return {"error": message}, True
                self._current_query = query_override

            rows, sql_error, plan_cost = self._execute_query(self._current_query, include_plan=True)
            self._last_query_rows = rows
            self._last_sql_error = sql_error
            self._last_query_plan_cost = plan_cost

            if sql_error:
                self._error_messages.append(sql_error)
                self._solved_query = False
                return {"error": sql_error}, True

            assert self._task is not None
            expected_rows, expected_error, _ = self._execute_query(self._task.expected_sql, include_plan=False)
            query_score = self._reward_calculator.sql_correctness_score(rows, expected_rows, sql_error=expected_error)
            self._solved_query = query_score >= 0.999

            info: dict[str, Any] = {
                "status": "query_ran",
                "row_count": len(rows),
                "query_score": query_score,
                "efficiency_cost": plan_cost,
                "solved_query": self._solved_query,
            }
            if expected_error:
                info["expected_query_error"] = expected_error

            return info, False

        def _handle_submit(self, force_reason: str | None = None) -> tuple[RewardModel, dict[str, Any]]:
            assert self._task is not None

            current_rows, current_error, plan_cost = self._execute_query(self._current_query, include_plan=True)
            self._last_query_rows = current_rows
            self._last_sql_error = current_error
            self._last_query_plan_cost = plan_cost

            expected_rows, expected_error, _ = self._execute_query(self._task.expected_sql, include_plan=False)
            grader_sql_error = current_error or expected_error

            reward = self._reward_calculator.calculate(
                candidate_clean_rows=self._candidate_clean_rows,
                expected_clean_rows=self._task.expected_clean_rows,
                candidate_query_rows=current_rows,
                expected_query_rows=expected_rows,
                sql_error=grader_sql_error,
                query_plan_cost=plan_cost,
                unnecessary_actions=self._unnecessary_actions,
            )

            self._solved_cleaning = reward.data_clean_score >= 0.999
            self._solved_query = reward.query_correct_score >= 0.999
            self._done = True

            info: dict[str, Any] = {
                "submitted": True,
                "reward": reward.model_dump(),
                "solved_cleaning": self._solved_cleaning,
                "solved_query": self._solved_query,
                "unnecessary_actions": self._unnecessary_actions,
            }
            if force_reason:
                info["force_reason"] = force_reason
            if grader_sql_error:
                info["grader_sql_error"] = grader_sql_error

            return reward, info

        def _build_observation(self) -> ObservationModel:
            dataset_state: list[dict[str, Any]] = []
            try:
                if self._conn is not None and self._working_table_name:
                    dataset_state = self._fetch_working_rows()
            except Exception as exc:
                self._error_messages.append(f"Failed to fetch working rows: {exc}")

            task_description = self._task.description if self._task is not None else ""

            return ObservationModel(
                current_dataset_state=dataset_state,
                current_sql_query=self._current_query,
                error_messages=list(self._error_messages),
                task_description=task_description,
                step_number=self._step_index,
            )

        @staticmethod
        def _zero_reward() -> RewardModel:
            return RewardModel(
                data_clean_score=0.0,
                query_correct_score=0.0,
                efficiency_bonus=0.0,
                step_penalty=0.0,
                total=0.0,
            )
