from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd
from psycopg2 import sql
from psycopg2.extras import RealDictCursor, execute_values

from meddataops.db import PostgresDataManager
from meddataops.models import ObservationModel, RewardModel
from meddataops.scoring import RewardCalculator


@dataclass(frozen=True)
class CleaningResult:
    """Result envelope returned by DataCleaningHandler."""

    observation: ObservationModel
    reward: RewardModel
    info: dict[str, Any]


class DataCleaningHandler:
    """Apply agent-requested cleaning operations to a PostgreSQL working table.

    The handler always edits an episode-specific working table, never the original source table.
    A backup copy is created the first time a table is touched to make the cleaning flow reversible.
    """

    SUPPORTED_OPERATIONS = {
        "remove_duplicates",
        "fix_nulls",
        "fix_dtypes",
        "normalize_strings",
        "remove_outliers",
        "rename_columns",
        "map_values",
        "coalesce_columns",
        "copy_column",
        "derive_column",
        "fix_unix_ms",
    }

    def __init__(
        self,
        db_manager: PostgresDataManager,
        reward_calculator: RewardCalculator | None = None,
        sample_size: int = 5,
    ) -> None:
        self._db = db_manager
        self._reward_calculator = reward_calculator or RewardCalculator()
        self._sample_size = max(1, sample_size)

    def handle_clean_data(
        self,
        *,
        session_id: str,
        working_table: str,
        parameters: dict[str, Any],
        task_description: str,
        step_number: int,
        current_sql_query: str = "",
        ground_truth_rows: list[dict[str, Any]] | None = None,
    ) -> CleaningResult:
        """Apply one or more cleaning operations and return observation + partial reward.

        Expected parameter formats:
        - {"operations": [{"operation": "remove_duplicates", ...}, ...]}
        - {"operation": "fix_nulls", ...}

        Returns:
        - Observation with row_count, null_counts, and sample_rows in current_dataset_state.
        - Partial RewardModel focused on cleaning progress.
        - info dict with per-operation details.
        """
        if not isinstance(parameters, dict):
            return self._error_result(
                error_message="clean_data parameters must be a dictionary.",
                working_table=working_table,
                task_description=task_description,
                step_number=step_number,
                current_sql_query=current_sql_query,
            )

        try:
            operations = self._parse_operations(parameters)
            self._ensure_reversible_copy(session_id=session_id, working_table=working_table)

            dataframe = self._read_table(working_table)
            if dataframe.empty:
                # Empty tables are allowed; still produce a structured observation.
                dataframe = pd.DataFrame(columns=self._get_table_columns(working_table))

            op_reports: list[dict[str, Any]] = []
            for op in operations:
                dataframe, report = self._apply_operation(dataframe, op)
                op_reports.append(report)

            self._write_table(working_table, dataframe)

            table_state = self._build_table_state(dataframe)
            clean_score = 0.0
            if ground_truth_rows is not None:
                clean_score = self._reward_calculator.data_clean_score(
                    candidate_rows=dataframe.to_dict(orient="records"),
                    expected_rows=ground_truth_rows,
                )

            reward = RewardModel(
                data_clean_score=clean_score,
                query_correct_score=0.0,
                efficiency_bonus=0.0,
                step_penalty=0.0,
                total=clean_score,
            )

            observation = ObservationModel(
                current_dataset_state=[table_state],
                current_sql_query=current_sql_query,
                error_messages=[],
                task_description=task_description,
                step_number=step_number,
            )

            info: dict[str, Any] = {
                "status": "ok",
                "working_table": working_table,
                "operations_applied": op_reports,
                "partial_clean_score": clean_score,
            }
            return CleaningResult(observation=observation, reward=reward, info=info)

        except ValueError as exc:
            return self._error_result(
                error_message=str(exc),
                working_table=working_table,
                task_description=task_description,
                step_number=step_number,
                current_sql_query=current_sql_query,
            )
        except Exception as exc:
            return self._error_result(
                error_message=f"Unexpected clean_data failure: {exc}",
                working_table=working_table,
                task_description=task_description,
                step_number=step_number,
                current_sql_query=current_sql_query,
            )

    def _parse_operations(self, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        if "operations" in parameters:
            operations = parameters["operations"]
            if not isinstance(operations, list) or not operations:
                raise ValueError("parameters.operations must be a non-empty list.")
            if any(not isinstance(op, dict) for op in operations):
                raise ValueError("Each item in parameters.operations must be an object.")
            return operations

        if "operation" in parameters:
            return [parameters]

        raise ValueError("Missing operation spec. Provide 'operation' or 'operations'.")

    def _apply_operation(
        self,
        dataframe: pd.DataFrame,
        operation_payload: dict[str, Any],
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        operation_name = operation_payload.get("operation") or operation_payload.get("name")
        if not isinstance(operation_name, str):
            raise ValueError("Operation entry must include a string field 'operation'.")

        operation_name = operation_name.strip().lower()
        if operation_name not in self.SUPPORTED_OPERATIONS:
            supported = ", ".join(sorted(self.SUPPORTED_OPERATIONS))
            raise ValueError(f"Unsupported operation '{operation_name}'. Supported: {supported}.")

        before_rows = len(dataframe)

        if operation_name == "remove_duplicates":
            updated = self._op_remove_duplicates(dataframe, operation_payload)
        elif operation_name == "fix_nulls":
            updated = self._op_fix_nulls(dataframe, operation_payload)
        elif operation_name == "fix_dtypes":
            updated = self._op_fix_dtypes(dataframe, operation_payload)
        elif operation_name == "normalize_strings":
            updated = self._op_normalize_strings(dataframe, operation_payload)
        elif operation_name == "rename_columns":
            updated = self._op_rename_columns(dataframe, operation_payload)
        elif operation_name == "map_values":
            updated = self._op_map_values(dataframe, operation_payload)
        elif operation_name == "coalesce_columns":
            updated = self._op_coalesce_columns(dataframe, operation_payload)
        elif operation_name == "copy_column":
            updated = self._op_copy_column(dataframe, operation_payload)
        elif operation_name == "derive_column":
            updated = self._op_derive_column(dataframe, operation_payload)
        elif operation_name == "fix_unix_ms":
            updated = self._op_fix_unix_ms(dataframe, operation_payload)
        else:
            updated = self._op_remove_outliers(dataframe, operation_payload)

        report = {
            "operation": operation_name,
            "rows_before": before_rows,
            "rows_after": len(updated),
        }
        return updated, report

    def _op_remove_duplicates(self, dataframe: pd.DataFrame, payload: dict[str, Any]) -> pd.DataFrame:
        columns = payload.get("columns")
        if not isinstance(columns, list) or not columns or any(not isinstance(c, str) for c in columns):
            raise ValueError("remove_duplicates requires 'columns' as a non-empty list of column names.")

        self._validate_columns_exist(dataframe, columns)
        return dataframe.drop_duplicates(subset=columns, keep="first").reset_index(drop=True)

    def _op_fix_nulls(self, dataframe: pd.DataFrame, payload: dict[str, Any]) -> pd.DataFrame:
        strategy = payload.get("strategy")
        if not isinstance(strategy, str):
            raise ValueError("fix_nulls requires a string 'strategy'.")

        strategy = strategy.strip().lower()
        supported = {"mean", "mode", "drop", "forward_fill"}
        if strategy not in supported:
            raise ValueError(f"Unsupported fix_nulls strategy '{strategy}'. Supported: {sorted(supported)}")

        columns = payload.get("columns", list(dataframe.columns))
        if not isinstance(columns, list) or any(not isinstance(c, str) for c in columns):
            raise ValueError("fix_nulls columns must be a list of column names.")

        self._validate_columns_exist(dataframe, columns)

        updated = dataframe.copy()

        if strategy == "drop":
            return updated.dropna(subset=columns).reset_index(drop=True)

        if strategy == "forward_fill":
            updated[columns] = updated[columns].ffill()
            return updated

        if strategy == "mean":
            for column in columns:
                numeric = pd.to_numeric(updated[column], errors="coerce")
                mean_value = numeric.mean(skipna=True)
                if pd.isna(mean_value):
                    raise ValueError(
                        f"Cannot apply mean fill on column '{column}' because it has no numeric values."
                    )
                updated[column] = numeric.fillna(mean_value)
            return updated

        # mode
        for column in columns:
            mode_values = updated[column].mode(dropna=True)
            if mode_values.empty:
                raise ValueError(f"Cannot apply mode fill on column '{column}' because mode is undefined.")
            updated[column] = updated[column].fillna(mode_values.iloc[0])

        return updated

    def _op_fix_dtypes(self, dataframe: pd.DataFrame, payload: dict[str, Any]) -> pd.DataFrame:
        dtype_map = payload.get("columns") or payload.get("dtypes")
        if not isinstance(dtype_map, dict) or not dtype_map:
            raise ValueError("fix_dtypes requires a non-empty mapping in 'columns' or 'dtypes'.")

        updated = dataframe.copy()
        for column, target_type in dtype_map.items():
            if not isinstance(column, str) or not isinstance(target_type, str):
                raise ValueError("fix_dtypes mapping must be of type {str: str}.")
            if column not in updated.columns:
                raise ValueError(f"Column '{column}' does not exist in working table.")

            target = target_type.strip().lower()
            if target == "date":
                parsed = pd.to_datetime(updated[column], errors="coerce", utc=False)
                updated[column] = parsed.dt.strftime("%Y-%m-%d")
                updated.loc[parsed.isna(), column] = None
            elif target == "float":
                numeric = pd.to_numeric(updated[column], errors="coerce")
                updated[column] = numeric
            elif target == "int":
                numeric = pd.to_numeric(updated[column], errors="coerce")
                updated[column] = numeric.round().astype("Int64")
            elif target == "string":
                normalized = updated[column].astype("string")
                updated[column] = normalized.str.strip()
            else:
                raise ValueError(
                    f"Unsupported dtype target '{target_type}' for column '{column}'. "
                    "Use one of: date, float, int, string."
                )

        return updated

    def _op_normalize_strings(self, dataframe: pd.DataFrame, payload: dict[str, Any]) -> pd.DataFrame:
        columns = payload.get("columns")
        if not isinstance(columns, list) or not columns or any(not isinstance(c, str) for c in columns):
            raise ValueError("normalize_strings requires 'columns' as a non-empty list of column names.")

        self._validate_columns_exist(dataframe, columns)

        case_mode = str(payload.get("case", "lower")).strip().lower()
        if case_mode not in {"lower", "upper", "title"}:
            raise ValueError("normalize_strings case must be one of: lower, upper, title.")

        updated = dataframe.copy()
        for column in columns:
            series = updated[column].astype("string").str.strip()
            if case_mode == "lower":
                series = series.str.lower()
            elif case_mode == "upper":
                series = series.str.upper()
            else:
                series = series.str.title()
            updated[column] = series.where(series.notna(), None)

        return updated

    def _op_remove_outliers(self, dataframe: pd.DataFrame, payload: dict[str, Any]) -> pd.DataFrame:
        column = payload.get("column")
        if not isinstance(column, str) or not column.strip():
            raise ValueError("remove_outliers requires a non-empty 'column' string.")
        if column not in dataframe.columns:
            raise ValueError(f"Column '{column}' does not exist in working table.")

        n_std = payload.get("n_std", 3)
        try:
            n_std_float = float(n_std)
        except (TypeError, ValueError) as exc:
            raise ValueError("remove_outliers requires numeric 'n_std'.") from exc

        if n_std_float <= 0:
            raise ValueError("remove_outliers requires n_std > 0.")

        updated = dataframe.copy()
        numeric = pd.to_numeric(updated[column], errors="coerce")

        valid = numeric.dropna()
        if valid.empty:
            raise ValueError(f"remove_outliers cannot run on '{column}' because it has no numeric values.")

        mean = valid.mean()
        std = valid.std(ddof=0)

        if std is None or math.isnan(std) or std == 0:
            return updated.reset_index(drop=True)

        lower = mean - (n_std_float * std)
        upper = mean + (n_std_float * std)

        mask = numeric.isna() | ((numeric >= lower) & (numeric <= upper))
        return updated.loc[mask].reset_index(drop=True)

    def _op_rename_columns(self, dataframe: pd.DataFrame, payload: dict[str, Any]) -> pd.DataFrame:
        mapping = payload.get("mapping")
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("rename_columns requires a non-empty 'mapping' object.")

        updated = dataframe.copy()
        for old_name, new_name in mapping.items():
            if not isinstance(old_name, str) or not isinstance(new_name, str):
                raise ValueError("rename_columns mapping must be of type {str: str}.")
            if old_name == new_name:
                continue
            if old_name not in updated.columns:
                continue

            if new_name in updated.columns:
                updated[new_name] = updated[new_name].combine_first(updated[old_name])
                updated = updated.drop(columns=[old_name])
            else:
                updated = updated.rename(columns={old_name: new_name})

        return updated

    def _op_map_values(self, dataframe: pd.DataFrame, payload: dict[str, Any]) -> pd.DataFrame:
        column = payload.get("column")
        mapping = payload.get("mapping")
        if not isinstance(column, str) or not isinstance(mapping, dict) or not mapping:
            raise ValueError("map_values requires 'column' (str) and non-empty 'mapping' (object).")

        self._validate_columns_exist(dataframe, [column])

        has_default = "default" in payload
        default_value = payload.get("default")

        updated = dataframe.copy()

        def _mapper(value: Any) -> Any:
            if value in mapping:
                return mapping[value]
            key = str(value)
            if key in mapping:
                return mapping[key]
            if has_default:
                return default_value
            return value

        updated[column] = updated[column].map(_mapper)
        return updated

    def _op_coalesce_columns(self, dataframe: pd.DataFrame, payload: dict[str, Any]) -> pd.DataFrame:
        target_column = payload.get("target_column") or payload.get("target")
        source_columns = payload.get("source_columns") or payload.get("columns")

        if not isinstance(target_column, str):
            raise ValueError("coalesce_columns requires a string 'target_column'.")
        if not isinstance(source_columns, list) or not source_columns or any(not isinstance(c, str) for c in source_columns):
            raise ValueError("coalesce_columns requires 'source_columns' as a non-empty list of strings.")

        self._validate_columns_exist(dataframe, source_columns)

        updated = dataframe.copy()
        series = pd.Series([None] * len(updated), dtype="object", index=updated.index)
        for source_column in source_columns:
            series = series.combine_first(updated[source_column])

        if "fill_value" in payload:
            series = series.fillna(payload.get("fill_value"))

        updated[target_column] = series

        if bool(payload.get("drop_sources", False)):
            drop_list = [column for column in source_columns if column != target_column and column in updated.columns]
            if drop_list:
                updated = updated.drop(columns=drop_list)

        return updated

    def _op_copy_column(self, dataframe: pd.DataFrame, payload: dict[str, Any]) -> pd.DataFrame:
        from_column = payload.get("from_column") or payload.get("source_column")
        to_column = payload.get("to_column") or payload.get("target_column")

        if not isinstance(from_column, str) or not isinstance(to_column, str):
            raise ValueError("copy_column requires string fields 'from_column' and 'to_column'.")

        self._validate_columns_exist(dataframe, [from_column])

        overwrite = bool(payload.get("overwrite", True))
        updated = dataframe.copy()

        if overwrite or to_column not in updated.columns:
            updated[to_column] = updated[from_column]
        else:
            updated[to_column] = updated[to_column].combine_first(updated[from_column])

        return updated

    def _op_derive_column(self, dataframe: pd.DataFrame, payload: dict[str, Any]) -> pd.DataFrame:
        target_column = payload.get("target_column") or payload.get("column")
        if not isinstance(target_column, str):
            raise ValueError("derive_column requires a string 'target_column'.")

        rule = str(payload.get("rule", "")).strip().lower()
        updated = dataframe.copy()

        if not rule and "value" in payload:
            updated[target_column] = payload.get("value")
            return updated

        if rule == "from_column":
            source_column = payload.get("source_column")
            if not isinstance(source_column, str):
                raise ValueError("derive_column rule 'from_column' requires 'source_column'.")
            self._validate_columns_exist(updated, [source_column])
            updated[target_column] = updated[source_column]
            return updated

        if rule == "if_equals":
            source_column = payload.get("column")
            if not isinstance(source_column, str):
                raise ValueError("derive_column rule 'if_equals' requires 'column'.")
            self._validate_columns_exist(updated, [source_column])

            expected_value = payload.get("equals")
            then_value = payload.get("then")
            else_value = payload.get("else")
            updated[target_column] = updated[source_column].map(
                lambda value: then_value if value == expected_value else else_value
            )
            return updated

        if rule == "template":
            template = payload.get("template")
            if not isinstance(template, str):
                raise ValueError("derive_column rule 'template' requires a string 'template'.")

            class _SafeDict(dict[str, Any]):
                def __missing__(self, key: str) -> str:
                    return ""

            def _apply_template(row: pd.Series) -> str:
                mapping = {
                    key: "" if pd.isna(value) else value
                    for key, value in row.to_dict().items()
                }
                try:
                    return str(template.format_map(_SafeDict(mapping)))
                except Exception:
                    return template

            updated[target_column] = updated.apply(_apply_template, axis=1)
            return updated

        if rule == "extract_digits":
            source_column = payload.get("column")
            if not isinstance(source_column, str):
                raise ValueError("derive_column rule 'extract_digits' requires 'column'.")
            self._validate_columns_exist(updated, [source_column])
            fallback = str(payload.get("fallback", "UNKNOWN"))

            updated[target_column] = (
                updated[source_column]
                .astype("string")
                .str.extract(r"(\d+)", expand=False)
                .fillna(fallback)
            )
            return updated

        if rule == "date_within_days":
            source_column = payload.get("column")
            reference_date = payload.get("reference_date")
            if not isinstance(source_column, str) or not isinstance(reference_date, str):
                raise ValueError(
                    "derive_column rule 'date_within_days' requires 'column' and 'reference_date'."
                )
            self._validate_columns_exist(updated, [source_column])

            try:
                days = int(payload.get("days", 0))
            except (TypeError, ValueError) as exc:
                raise ValueError("derive_column rule 'date_within_days' requires integer 'days'.") from exc

            ref_ts = pd.to_datetime(reference_date, errors="coerce", utc=True)
            if pd.isna(ref_ts):
                raise ValueError("derive_column rule 'date_within_days' has invalid 'reference_date'.")

            then_value = payload.get("then")
            else_value = payload.get("else")
            cutoff = ref_ts - pd.Timedelta(days=days)

            parsed = pd.to_datetime(updated[source_column], errors="coerce", utc=True)
            mask = parsed.notna() & (parsed >= cutoff)
            updated[target_column] = else_value
            updated.loc[mask, target_column] = then_value
            return updated

        raise ValueError(
            "Unsupported derive_column rule. Use one of: from_column, if_equals, template, extract_digits, date_within_days."
        )

    def _op_fix_unix_ms(self, dataframe: pd.DataFrame, payload: dict[str, Any]) -> pd.DataFrame:
        columns = payload.get("columns")
        if not isinstance(columns, list) or not columns:
            single_column = payload.get("column")
            columns = [single_column] if isinstance(single_column, str) else []

        if not columns or any(not isinstance(c, str) for c in columns):
            raise ValueError("fix_unix_ms requires 'column' or non-empty 'columns' list of strings.")

        self._validate_columns_exist(dataframe, columns)
        output = str(payload.get("output", "date")).strip().lower()

        updated = dataframe.copy()

        for column in columns:
            converted = pd.to_datetime(updated[column], errors="coerce", utc=True)
            numeric = pd.to_numeric(updated[column], errors="coerce")

            numeric_mask = numeric.notna()
            if numeric_mask.any():
                as_seconds = numeric.copy()
                millisecond_mask = numeric.abs() > 100_000_000_000
                as_seconds.loc[millisecond_mask] = as_seconds.loc[millisecond_mask] / 1000.0
                converted.loc[numeric_mask] = pd.to_datetime(
                    as_seconds.loc[numeric_mask],
                    unit="s",
                    errors="coerce",
                    utc=True,
                )

            if output == "datetime":
                rendered = converted.dt.strftime("%Y-%m-%d %H:%M:%S")
            else:
                rendered = converted.dt.strftime("%Y-%m-%d")

            updated[column] = rendered.where(converted.notna(), None)

        return updated

    def _ensure_reversible_copy(self, *, session_id: str, working_table: str) -> None:
        backup_table = self._backup_table_name(session_id, working_table)

        with self._db._borrow_connection() as conn:  # noqa: SLF001 - internal helper used by package components
            with conn.cursor() as cur:
                cur.execute("SELECT to_regclass(%s)", (backup_table,))
                exists = cur.fetchone()[0] is not None
                if exists:
                    return

                cur.execute(
                    sql.SQL("CREATE TABLE {} AS TABLE {}").format(
                        sql.Identifier(backup_table),
                        sql.Identifier(working_table),
                    )
                )

    def _backup_table_name(self, session_id: str, working_table: str) -> str:
        safe_session = self._db._sanitize_session_id(session_id)  # noqa: SLF001
        return f"{safe_session}_{working_table}_original"

    def _read_table(self, table_name: str) -> pd.DataFrame:
        with self._db._borrow_connection() as conn:  # noqa: SLF001
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql.SQL("SELECT * FROM {} ORDER BY 1").format(sql.Identifier(table_name)))
                rows = cur.fetchall()
        return pd.DataFrame([dict(row) for row in rows])

    def _get_table_columns(self, table_name: str) -> list[str]:
        with self._db._borrow_connection() as conn:  # noqa: SLF001
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = %s
                    ORDER BY ordinal_position
                    """,
                    (table_name,),
                )
                rows = cur.fetchall()
        return [row[0] for row in rows]

    def _write_table(self, table_name: str, dataframe: pd.DataFrame) -> None:
        columns = self._get_table_columns(table_name)
        if not columns:
            raise ValueError(f"Working table '{table_name}' does not exist or has no columns.")

        with self._db._borrow_connection() as conn:  # noqa: SLF001
            with conn.cursor() as cur:
                cur.execute(sql.SQL("TRUNCATE TABLE {}").format(sql.Identifier(table_name)))

                if dataframe.empty:
                    return

                sanitized_df = dataframe.copy()
                sanitized_df = sanitized_df.reindex(columns=columns)

                records: list[tuple[Any, ...]] = []
                for _, row in sanitized_df.iterrows():
                    values: list[Any] = []
                    for value in row.tolist():
                        if pd.isna(value):
                            values.append(None)
                        elif isinstance(value, pd.Timestamp):
                            values.append(value.to_pydatetime())
                        else:
                            values.append(value)
                    records.append(tuple(values))

                insert_query = sql.SQL("INSERT INTO {} ({}) VALUES %s").format(
                    sql.Identifier(table_name),
                    sql.SQL(", ").join(sql.Identifier(column) for column in columns),
                )
                execute_values(cur, insert_query.as_string(cur.connection), records)

    def _build_table_state(self, dataframe: pd.DataFrame) -> dict[str, Any]:
        sample_rows = dataframe.head(self._sample_size).to_dict(orient="records")
        null_counts = {column: int(value) for column, value in dataframe.isna().sum().to_dict().items()}

        return {
            "row_count": int(len(dataframe)),
            "null_counts": null_counts,
            "sample_rows": sample_rows,
        }

    def _validate_columns_exist(self, dataframe: pd.DataFrame, columns: list[str]) -> None:
        missing = [column for column in columns if column not in dataframe.columns]
        if missing:
            available = ", ".join(dataframe.columns.astype(str).tolist())
            raise ValueError(
                f"Columns not found: {', '.join(missing)}. Available columns: {available}."
            )

    def _error_result(
        self,
        *,
        error_message: str,
        working_table: str,
        task_description: str,
        step_number: int,
        current_sql_query: str,
    ) -> CleaningResult:
        safe_state: dict[str, Any]
        try:
            dataframe = self._read_table(working_table)
            safe_state = self._build_table_state(dataframe)
        except Exception:
            safe_state = {
                "row_count": 0,
                "null_counts": {},
                "sample_rows": [],
            }

        observation = ObservationModel(
            current_dataset_state=[safe_state],
            current_sql_query=current_sql_query,
            error_messages=[error_message],
            task_description=task_description,
            step_number=step_number,
        )
        reward = RewardModel(
            data_clean_score=0.0,
            query_correct_score=0.0,
            efficiency_bonus=0.0,
            step_penalty=0.0,
            total=0.0,
        )
        return CleaningResult(
            observation=observation,
            reward=reward,
            info={"status": "error", "error": error_message},
        )
