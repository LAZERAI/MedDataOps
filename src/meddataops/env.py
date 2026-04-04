from __future__ import annotations

import copy
import random
import uuid
from typing import Any

import pandas as pd
from pydantic import ValidationError

from meddataops.config import settings
from meddataops.db import PostgresBackend
from meddataops.models import (
    ActionModel,
    ActionType,
    AgentAction,
    EnvironmentState,
    OpenEnvActionType,
    ResetRequest,
    StepResult,
    TaskPublic,
)
from meddataops.scoring import normalize_sql
from meddataops.tasks import get_task, list_tasks
from meddataops.tasks import icu_capacity, medication_summary, triage_report


class MedDataOpsEnv:
    """OpenEnv-compatible environment for MedDataOps.

    Required public methods:
    - reset(...) -> EnvironmentState
    - step(...) -> StepResult
    - state() -> EnvironmentState
    """

    SUPPORTED_ACTIONS = {
        "clean_data",
        "run_query",
        "fix_query",
        "fix_sql",
        "submit",
        "noop",
    }

    def __init__(
        self,
        postgres_backend: PostgresBackend | None = None,
        max_steps: int | None = None,
        seed: int | None = None,
    ) -> None:
        self._rng = random.Random(seed)
        self._query_backend = postgres_backend or PostgresBackend(dsn=settings.postgres_dsn)
        self._max_steps = max_steps or settings.max_steps

        self._episode_id: str = ""
        self._step_index: int = 0
        self._done: bool = False
        self._reward: float = 0.0

        self._task = None
        self._candidate_rows: list[dict[str, Any]] = []
        self._current_query: str = ""
        self._last_query_rows: list[dict[str, Any]] = []
        self._last_sql_error: str | None = None

        self._solved_cleaning: bool = False
        self._solved_sql: bool = False

    def reset(
        self,
        request: ResetRequest | dict[str, Any] | str | None = None,
        task_id: str | None = None,
        seed: int | None = None,
    ) -> EnvironmentState:
        resolved_task_id, resolved_seed = self._coerce_reset_inputs(request=request, task_id=task_id, seed=seed)

        if resolved_seed is not None:
            self._rng.seed(resolved_seed)

        chosen_task = resolved_task_id or self._rng.choice(list_tasks())
        self._task = get_task(chosen_task)

        self._episode_id = str(uuid.uuid4())
        self._step_index = 0
        self._done = False
        self._reward = 0.0

        self._candidate_rows = self._deep_rows(self._task.dirty_rows)
        self._current_query = self._task.broken_sql
        self._last_query_rows = []
        self._last_sql_error = None

        self._solved_cleaning = False
        self._solved_sql = False

        return self._build_state()

    def step(self, action: AgentAction | ActionModel | dict[str, Any]) -> StepResult:
        self._require_active_episode()

        if self._done:
            return StepResult(
                observation=self._build_state(),
                reward=0.0,
                done=True,
                info={"message": "Episode already completed. Call reset() to start a new one."},
            )

        normalized = self._normalize_action(action)
        action_type = normalized["action_type"]
        parameters = normalized["parameters"]

        info: dict[str, Any] = {"action_type": action_type}
        self._last_sql_error = None

        if action_type == "clean_data":
            operations = parameters.get("operations", [])
            if not isinstance(operations, list):
                operations = []
            self._candidate_rows = self._apply_operations(self._candidate_rows, operations)
            info["operations_applied"] = len(operations)

        elif action_type in {"run_query", "fix_query", "fix_sql"}:
            query = parameters.get("query")
            if query is None and action_type == "run_query":
                query = self._current_query

            if not isinstance(query, str) or not query.strip():
                self._last_sql_error = f"{action_type} requires a non-empty query string."
                info["query_valid"] = False
            else:
                normalized_query = query.strip()
                if action_type in {"fix_query", "fix_sql"}:
                    self._current_query = normalized_query

                check = self._query_backend.validate_query(normalized_query)
                info["query_valid"] = bool(check.success)
                info["query_check"] = check.model_dump()
                if check.success:
                    self._last_query_rows = self._deep_rows(check.sample_rows)
                    self._last_sql_error = None
                else:
                    self._last_sql_error = check.error or "Query validation failed."

        elif action_type == "submit":
            self._reward = self._compute_submission_score()
            self._done = True
            info["message"] = "Submission accepted."

        elif action_type == "noop":
            info["message"] = "No operation performed."

        self._step_index += 1
        if self._step_index >= self._max_steps and not self._done:
            self._reward = self._compute_submission_score()
            self._done = True
            info["termination"] = "max_steps_reached"

        if action_type != "submit" and "termination" not in info:
            self._reward = 0.0

        self._solved_sql = self._is_query_correct()
        self._solved_cleaning = self._rows_equivalent(self._candidate_rows, self._task.expected_clean_rows)

        return StepResult(
            observation=self._build_state(),
            reward=float(self._reward),
            done=self._done,
            info=info,
        )

    def state(self) -> EnvironmentState:
        self._require_active_episode()
        return self._build_state()

    def close(self) -> None:
        return

    def _coerce_reset_inputs(
        self,
        *,
        request: ResetRequest | dict[str, Any] | str | None,
        task_id: str | None,
        seed: int | None,
    ) -> tuple[str | None, int | None]:
        req_task: str | None = None
        req_seed: int | None = None

        if isinstance(request, ResetRequest):
            req_task = request.task_id
            req_seed = request.seed
        elif isinstance(request, dict):
            parsed = ResetRequest.model_validate(request)
            req_task = parsed.task_id
            req_seed = parsed.seed
        elif isinstance(request, str):
            req_task = request
        elif request is not None:
            raise ValueError(f"Unsupported reset payload type: {type(request).__name__}")

        final_task = task_id if task_id is not None else req_task
        final_seed = seed if seed is not None else req_seed
        return final_task, final_seed

    def _normalize_action(self, action: AgentAction | ActionModel | dict[str, Any]) -> dict[str, Any]:
        if isinstance(action, ActionModel):
            return {
                "action_type": action.action_type.value,
                "parameters": dict(action.parameters),
            }

        if isinstance(action, AgentAction):
            mapped = {
                ActionType.CLEAN_DATA.value: "clean_data",
                ActionType.FIX_SQL.value: "fix_query",
                ActionType.SUBMIT.value: "submit",
                ActionType.NOOP.value: "noop",
            }
            return {
                "action_type": mapped.get(action.action_type.value, "noop"),
                "parameters": dict(action.payload),
            }

        try:
            payload = dict(action)
        except Exception as exc:  # pragma: no cover
            raise ValueError(f"Action payload must be dict-like: {exc}") from exc

        action_type = str(payload.get("action_type", "")).strip().lower()
        parameters = payload.get("parameters", payload.get("payload", {}))

        if not isinstance(parameters, dict):
            parameters = {}

        alias_map = {
            "fix_sql": "fix_query",
            "query": "run_query",
            "runquery": "run_query",
            "submit_answer": "submit",
        }
        action_type = alias_map.get(action_type, action_type)

        if action_type not in self.SUPPORTED_ACTIONS:
            raise ValueError(f"Unsupported action_type: {action_type}")

        return {"action_type": action_type, "parameters": parameters}

    def _require_active_episode(self) -> None:
        if self._task is None:
            raise RuntimeError("No active episode. Call reset() before step() or state().")

    def _build_state(self) -> EnvironmentState:
        self._require_active_episode()
        assert self._task is not None

        return EnvironmentState(
            episode_id=self._episode_id,
            step_index=self._step_index,
            max_steps=self._max_steps,
            done=self._done,
            reward=float(self._reward),
            task=TaskPublic(
                id=self._task.id,
                name=self._task.name,
                difficulty=self._task.difficulty,
                description=self._task.description,
                hints=list(self._task.hints),
            ),
            dirty_rows=self._deep_rows(self._candidate_rows),
            broken_sql=self._current_query,
            last_sql_error=self._last_sql_error,
            solved_cleaning=self._solved_cleaning,
            solved_sql=self._solved_sql,
        )

    def _apply_operations(self, rows: list[dict[str, Any]], operations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        updated = self._deep_rows(rows)

        for operation in operations:
            if not isinstance(operation, dict):
                continue
            op = str(operation.get("operation", "")).strip().lower()

            if op == "normalize_strings":
                columns = operation.get("columns", [])
                case_mode = str(operation.get("case", "lower")).strip().lower()
                for row in updated:
                    for column in columns:
                        value = row.get(column)
                        if value is None:
                            continue
                        text = str(value).strip()
                        if case_mode == "upper":
                            row[column] = text.upper()
                        elif case_mode == "title":
                            row[column] = text.title()
                        else:
                            row[column] = text.lower()

            elif op == "fix_nulls":
                columns = operation.get("columns", [])
                strategy = str(operation.get("strategy", "mode")).strip().lower()
                for column in columns:
                    values = [row.get(column) for row in updated if row.get(column) not in (None, "", "N/A", "n/a")]
                    replacement: Any = None
                    if values:
                        if strategy == "mean":
                            nums = []
                            for value in values:
                                try:
                                    nums.append(float(value))
                                except (TypeError, ValueError):
                                    continue
                            if nums:
                                replacement = sum(nums) / len(nums)
                        elif strategy == "mode":
                            replacement = max(set(values), key=values.count)
                        elif strategy == "forward_fill":
                            replacement = values[-1]

                    for row in updated:
                        if row.get(column) in (None, "", "N/A", "n/a"):
                            if strategy == "drop":
                                row[column] = None
                            elif replacement is not None:
                                row[column] = replacement

                if strategy == "drop":
                    updated = [
                        row
                        for row in updated
                        if all(row.get(column) not in (None, "", "N/A", "n/a") for column in columns)
                    ]

            elif op == "fix_dtypes":
                col_map = operation.get("columns", {})
                if not isinstance(col_map, dict):
                    continue
                for row in updated:
                    for column, dtype in col_map.items():
                        value = row.get(column)
                        if value is None:
                            continue
                        try:
                            if dtype == "int":
                                row[column] = int(float(value))
                            elif dtype == "float":
                                row[column] = float(value)
                            elif dtype == "string":
                                row[column] = str(value)
                            elif dtype == "date":
                                parsed = pd.to_datetime(value, errors="coerce", utc=True)
                                if not pd.isna(parsed):
                                    row[column] = parsed.strftime("%Y-%m-%d")
                        except (TypeError, ValueError):
                            continue

            elif op == "remove_duplicates":
                columns = operation.get("columns", [])
                if not columns:
                    continue
                deduped: dict[tuple[Any, ...], dict[str, Any]] = {}
                for row in updated:
                    signature = tuple(row.get(column) for column in columns)
                    deduped.setdefault(signature, row)
                updated = list(deduped.values())

        return updated

    def _compute_submission_score(self) -> float:
        assert self._task is not None

        task_id = self._task.id
        query_rows = self._candidate_query_rows(task_id)

        if task_id == "triage_report":
            score = triage_report.score_triage_report(self._candidate_rows, query_rows)
        elif task_id == "medication_summary":
            score = medication_summary.score_medication_summary(self._candidate_rows, query_rows)
        elif task_id == "icu_capacity":
            score = icu_capacity.score_icu_capacity(
                self._candidate_rows,
                query_rows,
                agent_query=self._current_query,
            )
        else:
            score = 0.0

        return float(max(0.0, min(1.0, score)))

    def _candidate_query_rows(self, task_id: str) -> list[dict[str, Any]]:
        if self._is_query_correct():
            if task_id == "triage_report":
                return self._deep_rows(getattr(triage_report, "_TRIAGE_EXPECTED_QUERY_RESULT", []))
            if task_id == "medication_summary":
                return self._deep_rows(medication_summary.MEDICATION_SUMMARY_GROUND_TRUTH_EXPECTED_RESULT)
            if task_id == "icu_capacity":
                return self._deep_rows(icu_capacity.ICU_CAPACITY_GROUND_TRUTH_EXPECTED_RESULT)
        return self._deep_rows(self._last_query_rows)

    def _is_query_correct(self) -> bool:
        assert self._task is not None
        return normalize_sql(self._current_query) == normalize_sql(self._task.expected_sql)

    @staticmethod
    def _deep_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return copy.deepcopy(rows)

    @staticmethod
    def _rows_equivalent(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> bool:
        def canonical(rows: list[dict[str, Any]]) -> list[tuple[tuple[str, str], ...]]:
            normalized = []
            for row in rows:
                row_items = []
                for key, value in sorted(row.items(), key=lambda item: item[0]):
                    row_items.append((str(key), str(value).strip().lower() if value is not None else ""))
                normalized.append(tuple(row_items))
            return sorted(normalized)

        return canonical(left) == canonical(right)


__all__ = ["MedDataOpsEnv"]
