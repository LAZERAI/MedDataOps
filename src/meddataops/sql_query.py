from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from psycopg2 import Error as Psycopg2Error

from meddataops.db import PostgresDataManager
from meddataops.models import RewardModel


@dataclass(frozen=True)
class SQLActionResult:
    """Envelope returned by SQLQueryHandler actions."""

    success: bool
    action: str
    query: str
    rows: list[dict[str, Any]]
    row_count: int
    total_cost: float | None
    error: str | None
    query_history: list[str]
    correctness: dict[str, Any] | None
    reward: RewardModel | None


class SQLQueryHandler:
    """Handler for run_query/fix_query/submit SQL actions in MedDataOps.

    Key guarantees:
    - Only SELECT/WITH statements are allowed.
    - Write/DDL statements are blocked.
    - Every executed query runs with EXPLAIN ANALYZE for cost visibility.
    - Query attempts are tracked per episode.
    - Submit computes row-by-row correctness versus ground truth.
    """

    DISALLOWED_KEYWORDS = {
        "insert",
        "update",
        "delete",
        "drop",
        "alter",
        "truncate",
        "grant",
        "revoke",
        "create",
    }

    def __init__(
        self,
        db_manager: PostgresDataManager,
        working_table_map: dict[str, str] | None = None,
        row_limit: int = 50,
    ) -> None:
        self._db = db_manager
        self._working_table_map = working_table_map or {}
        self._row_limit = max(1, row_limit)

        self._query_history: list[str] = []
        self._current_query: str = ""
        self._last_rows: list[dict[str, Any]] = []
        self._last_total_cost: float | None = None

    @property
    def query_history(self) -> list[str]:
        return list(self._query_history)

    def run_query(self, query: str) -> SQLActionResult:
        """Execute SQL query and return rows/errors with plan cost.

        Returns at most 50 rows for agent visibility while preserving full row count.
        """
        try:
            prepared_query = self._prepare_query(query)
        except ValueError as exc:
            return self._error_result("run_query", query, str(exc))

        rows, total_cost, error = self._execute_query(prepared_query)
        self._query_history.append(query)
        self._current_query = query

        if error is not None:
            return self._error_result("run_query", query, error)

        self._last_rows = rows
        self._last_total_cost = total_cost

        return SQLActionResult(
            success=True,
            action="run_query",
            query=query,
            rows=rows[: self._row_limit],
            row_count=len(rows),
            total_cost=total_cost,
            error=None,
            query_history=self.query_history,
            correctness=None,
            reward=None,
        )

    def fix_query(self, corrected_query: str) -> SQLActionResult:
        """Persist a corrected query candidate and validate it by execution."""
        try:
            prepared_query = self._prepare_query(corrected_query)
        except ValueError as exc:
            return self._error_result("fix_query", corrected_query, str(exc))

        rows, total_cost, error = self._execute_query(prepared_query)
        self._query_history.append(corrected_query)
        self._current_query = corrected_query

        if error is not None:
            return self._error_result("fix_query", corrected_query, error)

        self._last_rows = rows
        self._last_total_cost = total_cost

        return SQLActionResult(
            success=True,
            action="fix_query",
            query=corrected_query,
            rows=rows[: self._row_limit],
            row_count=len(rows),
            total_cost=total_cost,
            error=None,
            query_history=self.query_history,
            correctness=None,
            reward=None,
        )

    def submit(
        self,
        *,
        ground_truth_query: str | None = None,
        ground_truth_rows: list[dict[str, Any]] | None = None,
        final_query: str | None = None,
    ) -> SQLActionResult:
        """Finalize SQL query and compute row-by-row correctness against ground truth."""
        query_to_submit = final_query or self._current_query
        if not query_to_submit.strip():
            return self._error_result(
                action="submit",
                query=query_to_submit,
                message="No query available to submit. Provide final_query or run/fix a query first.",
            )

        try:
            prepared_submission_query = self._prepare_query(query_to_submit)
        except ValueError as exc:
            return self._error_result("submit", query_to_submit, str(exc))

        candidate_rows, total_cost, submit_error = self._execute_query(prepared_submission_query)
        self._query_history.append(query_to_submit)
        self._current_query = query_to_submit

        if submit_error is not None:
            return self._error_result("submit", query_to_submit, submit_error)

        expected_rows: list[dict[str, Any]]
        if ground_truth_rows is not None:
            expected_rows = list(ground_truth_rows)
        elif ground_truth_query is not None:
            try:
                prepared_truth_query = self._prepare_query(ground_truth_query)
            except ValueError as exc:
                return self._error_result("submit", query_to_submit, f"Invalid ground truth query: {exc}")

            expected_rows, _, expected_error = self._execute_query(prepared_truth_query)
            if expected_error is not None:
                return self._error_result(
                    "submit",
                    query_to_submit,
                    f"Failed to execute ground truth query: {expected_error}",
                )
        else:
            return self._error_result(
                "submit",
                query_to_submit,
                "Ground truth missing. Provide ground_truth_rows or ground_truth_query.",
            )

        correctness = self._score_correctness(candidate_rows=candidate_rows, expected_rows=expected_rows)
        query_score = float(correctness["score"])

        reward = RewardModel(
            data_clean_score=0.0,
            query_correct_score=query_score,
            efficiency_bonus=0.0,
            step_penalty=0.0,
            total=query_score,
        )

        self._last_rows = candidate_rows
        self._last_total_cost = total_cost

        return SQLActionResult(
            success=True,
            action="submit",
            query=query_to_submit,
            rows=candidate_rows[: self._row_limit],
            row_count=len(candidate_rows),
            total_cost=total_cost,
            error=None,
            query_history=self.query_history,
            correctness=correctness,
            reward=reward,
        )

    def _prepare_query(self, query: str) -> str:
        if not isinstance(query, str):
            raise ValueError("Query must be a string.")

        candidate = query.strip().rstrip(";")
        if not candidate:
            raise ValueError("Query cannot be empty.")

        lowered = candidate.lower()
        if not (lowered.startswith("select") or lowered.startswith("with")):
            raise ValueError("Only SELECT/WITH statements are allowed.")

        if ";" in candidate:
            raise ValueError("Multiple SQL statements are not allowed.")

        for keyword in self.DISALLOWED_KEYWORDS:
            if re.search(rf"\b{re.escape(keyword)}\b", lowered):
                raise ValueError(
                    f"Disallowed SQL keyword detected: '{keyword}'. Only read-only queries are permitted."
                )

        return self._rewrite_working_tables(candidate)

    def _rewrite_working_tables(self, query: str) -> str:
        rewritten = query
        for base_table, working_table in self._working_table_map.items():
            rewritten = re.sub(
                rf"\b{re.escape(base_table)}\b",
                working_table,
                rewritten,
                flags=re.IGNORECASE,
            )
        return rewritten

    def _execute_query(self, prepared_query: str) -> tuple[list[dict[str, Any]], float | None, str | None]:
        try:
            result = self._db.run_query(prepared_query, include_explain=True)
            rows = result.dataframe.to_dict(orient="records")
            return rows, result.total_cost, None
        except Exception as exc:
            return [], None, self._humanize_sql_error(exc)

    def _humanize_sql_error(self, exc: Exception) -> str:
        if isinstance(exc, Psycopg2Error):
            parts = []
            if exc.pgerror:
                parts.append(exc.pgerror.strip())
            if exc.diag is not None:
                if exc.diag.message_detail:
                    parts.append(f"detail: {exc.diag.message_detail}")
                if exc.diag.message_hint:
                    parts.append(f"hint: {exc.diag.message_hint}")
                if exc.diag.context:
                    parts.append(f"context: {exc.diag.context}")
            if parts:
                return " | ".join(parts)

        return str(exc)

    def _score_correctness(
        self,
        *,
        candidate_rows: list[dict[str, Any]],
        expected_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        candidate_cols = self._columns_for_rows(candidate_rows)
        expected_cols = self._columns_for_rows(expected_rows)

        if not expected_rows and not candidate_rows:
            return {
                "score": 1.0,
                "exact_match": True,
                "partial_match": 1.0,
                "wrong_columns": False,
                "missing_columns": [],
                "extra_columns": [],
                "row_count_expected": 0,
                "row_count_actual": 0,
                "row_count_score": 1.0,
                "exact_row_match_score": 1.0,
                "column_match_score": 1.0,
                "row_comparisons": [],
            }

        if not expected_rows and candidate_rows:
            return {
                "score": 0.0,
                "exact_match": False,
                "partial_match": 0.0,
                "wrong_columns": bool(candidate_cols),
                "missing_columns": [],
                "extra_columns": sorted(candidate_cols),
                "row_count_expected": 0,
                "row_count_actual": len(candidate_rows),
                "row_count_score": 0.0,
                "exact_row_match_score": 0.0,
                "column_match_score": 0.0,
                "row_comparisons": [],
            }

        if expected_rows and not candidate_rows:
            return {
                "score": 0.0,
                "exact_match": False,
                "partial_match": 0.0,
                "wrong_columns": bool(expected_cols),
                "missing_columns": sorted(expected_cols),
                "extra_columns": [],
                "row_count_expected": len(expected_rows),
                "row_count_actual": 0,
                "row_count_score": 0.0,
                "exact_row_match_score": 0.0,
                "column_match_score": 0.0,
                "row_comparisons": [],
            }

        shared_cols = sorted(expected_cols & candidate_cols)
        missing_cols = sorted(expected_cols - candidate_cols)
        extra_cols = sorted(candidate_cols - expected_cols)
        wrong_columns = bool(missing_cols or extra_cols)

        column_match_score = (len(shared_cols) / len(expected_cols)) if expected_cols else 1.0

        expected_signatures = Counter(self._row_signature(row, shared_cols) for row in expected_rows)
        candidate_signatures = Counter(self._row_signature(row, shared_cols) for row in candidate_rows)
        exact_matches = sum((expected_signatures & candidate_signatures).values())
        exact_row_match_score = exact_matches / max(1, len(expected_rows))

        row_comparisons: list[dict[str, Any]] = []
        partial_total = 0.0
        candidate_used: set[int] = set()
        for expected_index, expected_row in enumerate(expected_rows):
            best_idx = None
            best_similarity = 0.0
            for candidate_index, candidate_row in enumerate(candidate_rows):
                if candidate_index in candidate_used:
                    continue
                sim = self._row_similarity(expected_row, candidate_row, shared_cols)
                if sim > best_similarity:
                    best_similarity = sim
                    best_idx = candidate_index

            if best_idx is not None:
                candidate_used.add(best_idx)
            partial_total += best_similarity
            if expected_index < 20:
                row_comparisons.append(
                    {
                        "expected_index": expected_index,
                        "candidate_index": best_idx,
                        "similarity": round(best_similarity, 4),
                        "expected_row": expected_row,
                        "candidate_row": candidate_rows[best_idx] if best_idx is not None else None,
                    }
                )

        partial_match_score = partial_total / max(1, len(expected_rows))

        row_count_expected = len(expected_rows)
        row_count_actual = len(candidate_rows)
        row_count_score = max(
            0.0,
            1.0 - (abs(row_count_actual - row_count_expected) / max(1, row_count_expected)),
        )

        # Blend exactness and partial alignment while penalizing wrong schemas and row-count drift.
        score = (
            (0.55 * exact_row_match_score)
            + (0.25 * partial_match_score)
            + (0.10 * column_match_score)
            + (0.10 * row_count_score)
        )
        score = max(0.0, min(1.0, score))

        exact_match = (
            not wrong_columns
            and row_count_actual == row_count_expected
            and exact_row_match_score >= 1.0
        )

        return {
            "score": round(score, 6),
            "exact_match": exact_match,
            "partial_match": round(partial_match_score, 6),
            "wrong_columns": wrong_columns,
            "missing_columns": missing_cols,
            "extra_columns": extra_cols,
            "row_count_expected": row_count_expected,
            "row_count_actual": row_count_actual,
            "row_count_score": round(row_count_score, 6),
            "exact_row_match_score": round(exact_row_match_score, 6),
            "column_match_score": round(column_match_score, 6),
            "row_comparisons": row_comparisons,
        }

    def _columns_for_rows(self, rows: list[dict[str, Any]]) -> set[str]:
        cols: set[str] = set()
        for row in rows:
            cols.update(row.keys())
        return cols

    def _row_signature(self, row: dict[str, Any], columns: list[str]) -> tuple[Any, ...]:
        return tuple(self._normalize_value(row.get(col)) for col in columns)

    def _row_similarity(self, expected_row: dict[str, Any], candidate_row: dict[str, Any], columns: list[str]) -> float:
        if not columns:
            return 1.0
        matches = 0
        for col in columns:
            if self._normalize_value(expected_row.get(col)) == self._normalize_value(candidate_row.get(col)):
                matches += 1
        return matches / len(columns)

    def _normalize_value(self, value: Any) -> Any:
        if isinstance(value, str):
            stripped = value.strip()
            try:
                return int(stripped)
            except ValueError:
                pass
            try:
                return float(stripped)
            except ValueError:
                return stripped.lower()
        return value

    def _error_result(self, action: str, query: str, message: str) -> SQLActionResult:
        return SQLActionResult(
            success=False,
            action=action,
            query=query,
            rows=[],
            row_count=0,
            total_cost=None,
            error=message,
            query_history=self.query_history,
            correctness=None,
            reward=None,
        )
