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

        def _is_null_like(value: Any) -> bool:
            if value in (None, "", "N/A", "n/a"):
                return True
            try:
                return bool(pd.isna(value))
            except Exception:
                return False

        def _render_template(template: str, row: dict[str, Any]) -> str:
            safe_row = {
                key: "" if _is_null_like(value) else value
                for key, value in row.items()
            }

            class _SafeDict(dict[str, Any]):
                def __missing__(self, key: str) -> str:
                    return ""

            try:
                return str(template.format_map(_SafeDict(safe_row)))
            except Exception:
                return str(template)

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

            elif op == "rename_columns":
                mapping = operation.get("mapping", {})
                if not isinstance(mapping, dict):
                    continue

                for row in updated:
                    for old_name, new_name in mapping.items():
                        if not isinstance(old_name, str) or not isinstance(new_name, str):
                            continue
                        if old_name not in row:
                            continue
                        if old_name == new_name:
                            continue

                        old_value = row.get(old_name)
                        current_value = row.get(new_name)
                        if _is_null_like(current_value):
                            row[new_name] = old_value
                        elif new_name not in row:
                            row[new_name] = old_value

                        del row[old_name]

            elif op == "map_values":
                column = operation.get("column")
                mapping = operation.get("mapping", {})
                if not isinstance(column, str) or not isinstance(mapping, dict):
                    continue

                has_default = "default" in operation
                default_value = operation.get("default")

                for row in updated:
                    if column not in row:
                        continue

                    value = row.get(column)
                    mapped = None
                    if value in mapping:
                        mapped = mapping[value]
                    elif str(value) in mapping:
                        mapped = mapping[str(value)]

                    if mapped is not None:
                        row[column] = mapped
                    elif has_default:
                        row[column] = default_value

            elif op == "coalesce_columns":
                target_column = operation.get("target_column") or operation.get("target")
                source_columns = operation.get("source_columns") or operation.get("columns")
                if not isinstance(target_column, str) or not isinstance(source_columns, list) or not source_columns:
                    continue

                drop_sources = bool(operation.get("drop_sources", False))
                fill_value = operation.get("fill_value")

                for row in updated:
                    chosen_value: Any = None
                    for source_column in source_columns:
                        if not isinstance(source_column, str):
                            continue
                        value = row.get(source_column)
                        if not _is_null_like(value):
                            chosen_value = value
                            break

                    if chosen_value is None and "fill_value" in operation:
                        chosen_value = fill_value

                    row[target_column] = chosen_value

                    if drop_sources:
                        for source_column in source_columns:
                            if source_column != target_column:
                                row.pop(source_column, None)

            elif op == "copy_column":
                from_column = operation.get("from_column") or operation.get("source_column")
                to_column = operation.get("to_column") or operation.get("target_column")
                if not isinstance(from_column, str) or not isinstance(to_column, str):
                    continue

                overwrite = bool(operation.get("overwrite", True))
                for row in updated:
                    if from_column not in row:
                        continue

                    if overwrite or to_column not in row or _is_null_like(row.get(to_column)):
                        row[to_column] = row.get(from_column)

            elif op == "derive_column":
                target_column = operation.get("target_column") or operation.get("column")
                if not isinstance(target_column, str):
                    continue

                rule = str(operation.get("rule", "")).strip().lower()

                if not rule and "value" in operation:
                    for row in updated:
                        row[target_column] = operation.get("value")
                    continue

                if rule == "from_column":
                    source_column = operation.get("source_column")
                    if not isinstance(source_column, str):
                        continue
                    for row in updated:
                        row[target_column] = row.get(source_column)

                elif rule == "if_equals":
                    source_column = operation.get("column")
                    expected_value = operation.get("equals")
                    then_value = operation.get("then")
                    else_value = operation.get("else")
                    if not isinstance(source_column, str):
                        continue

                    for row in updated:
                        row[target_column] = then_value if row.get(source_column) == expected_value else else_value

                elif rule == "template":
                    template = operation.get("template")
                    if not isinstance(template, str):
                        continue
                    for row in updated:
                        row[target_column] = _render_template(template, row)

                elif rule == "extract_digits":
                    source_column = operation.get("column")
                    fallback = str(operation.get("fallback", "UNKNOWN"))
                    if not isinstance(source_column, str):
                        continue

                    for row in updated:
                        raw_value = row.get(source_column)
                        digits = "".join(ch for ch in str(raw_value or "") if ch.isdigit())
                        row[target_column] = digits if digits else fallback

                elif rule == "date_within_days":
                    source_column = operation.get("column")
                    reference_date = operation.get("reference_date")
                    if not isinstance(source_column, str) or not isinstance(reference_date, str):
                        continue

                    try:
                        days = int(operation.get("days", 0))
                    except (TypeError, ValueError):
                        days = 0

                    then_value = operation.get("then")
                    else_value = operation.get("else")

                    ref_ts = pd.to_datetime(reference_date, errors="coerce", utc=True)
                    if pd.isna(ref_ts):
                        continue

                    cutoff = ref_ts - pd.Timedelta(days=days)
                    for row in updated:
                        candidate_ts = pd.to_datetime(row.get(source_column), errors="coerce", utc=True)
                        if not pd.isna(candidate_ts) and candidate_ts >= cutoff:
                            row[target_column] = then_value
                        else:
                            row[target_column] = else_value

            elif op == "fix_unix_ms":
                columns = operation.get("columns")
                if not isinstance(columns, list) or not columns:
                    single_column = operation.get("column")
                    columns = [single_column] if isinstance(single_column, str) else []

                if not columns:
                    continue

                output = str(operation.get("output", "date")).strip().lower()

                for row in updated:
                    for column in columns:
                        if not isinstance(column, str):
                            continue

                        raw_value = row.get(column)
                        if _is_null_like(raw_value):
                            continue

                        parsed = pd.NaT
                        try:
                            if isinstance(raw_value, (int, float)):
                                epoch_seconds = float(raw_value)
                                if abs(epoch_seconds) > 100_000_000_000:
                                    epoch_seconds = epoch_seconds / 1000.0
                                parsed = pd.to_datetime(epoch_seconds, unit="s", errors="coerce", utc=True)
                            elif isinstance(raw_value, str) and raw_value.strip().lstrip("-").isdigit():
                                epoch_seconds = float(raw_value.strip())
                                if abs(epoch_seconds) > 100_000_000_000:
                                    epoch_seconds = epoch_seconds / 1000.0
                                parsed = pd.to_datetime(epoch_seconds, unit="s", errors="coerce", utc=True)
                            else:
                                parsed = pd.to_datetime(raw_value, errors="coerce", utc=True)
                        except Exception:
                            parsed = pd.NaT

                        if pd.isna(parsed):
                            continue

                        if output == "datetime":
                            row[column] = parsed.strftime("%Y-%m-%d %H:%M:%S")
                        else:
                            row[column] = parsed.strftime("%Y-%m-%d")

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
