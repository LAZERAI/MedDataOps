from __future__ import annotations

import re
from collections import Counter
from typing import Any

from meddataops.models import RewardModel


def normalize_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip().rstrip(";")).lower()


def _normalize_value(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()

        if stripped.isdigit():
            return int(stripped)

        try:
            return float(stripped)
        except ValueError:
            return stripped.lower()

    return value


def canonicalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    canonical_rows: list[dict[str, Any]] = []

    for row in rows:
        normalized = {key: _normalize_value(value) for key, value in row.items()}
        canonical_rows.append(dict(sorted(normalized.items(), key=lambda item: item[0])))

    return sorted(canonical_rows, key=lambda row: tuple(str(v) for v in row.values()))


def rows_match(candidate_rows: list[dict[str, Any]], expected_rows: list[dict[str, Any]]) -> bool:
    return canonicalize_rows(candidate_rows) == canonicalize_rows(expected_rows)


class RewardCalculator:
    """Reward decomposition for MedDataOps episodes.

    The calculator returns partial credit for both sub-tasks and applies
    an efficiency incentive plus action-level penalty to discourage brute-force behavior.
    """

    def __init__(self, query_cost_threshold: float = 1000.0, step_penalty_per_action: float = 0.02) -> None:
        self.query_cost_threshold = max(0.0, query_cost_threshold)
        self.step_penalty_per_action = max(0.0, step_penalty_per_action)

    @staticmethod
    def _clamp_01(value: float) -> float:
        return max(0.0, min(1.0, value))

    @staticmethod
    def _row_signature(row: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
        normalized = {key: _normalize_value(value) for key, value in row.items()}
        return tuple(sorted(normalized.items(), key=lambda item: item[0]))

    @staticmethod
    def _row_similarity(expected_row: dict[str, Any], candidate_row: dict[str, Any]) -> float:
        all_columns = sorted(set(expected_row.keys()) | set(candidate_row.keys()))
        if not all_columns:
            return 1.0

        matched = 0
        for column in all_columns:
            expected_value = _normalize_value(expected_row.get(column))
            candidate_value = _normalize_value(candidate_row.get(column))
            if expected_value == candidate_value:
                matched += 1

        return matched / len(all_columns)

    def data_clean_score(
        self,
        candidate_rows: list[dict[str, Any]] | None,
        expected_rows: list[dict[str, Any]] | None,
    ) -> float:
        """Compute cell-level cleaning accuracy in the [0.0, 1.0] range.

        Edge cases:
        - If either dataset is None, score is 0.0.
        - If both are empty, score is 1.0.
        - Missing/extra rows or cells reduce score proportionally.
        """
        if candidate_rows is None or expected_rows is None:
            return 0.0

        expected = canonicalize_rows(expected_rows)
        candidate = canonicalize_rows(candidate_rows)

        if not expected and not candidate:
            return 1.0
        if not expected:
            return 0.0

        all_columns = sorted(
            {
                *{key for row in expected for key in row.keys()},
                *{key for row in candidate for key in row.keys()},
            }
        )
        if not all_columns:
            return 1.0 if len(expected) == len(candidate) else 0.0

        row_count = max(len(expected), len(candidate))
        total_cells = row_count * len(all_columns)
        if total_cells == 0:
            return 1.0

        matched_cells = 0
        for idx in range(row_count):
            expected_row = expected[idx] if idx < len(expected) else {}
            candidate_row = candidate[idx] if idx < len(candidate) else {}

            for column in all_columns:
                if _normalize_value(candidate_row.get(column)) == _normalize_value(expected_row.get(column)):
                    matched_cells += 1

        return self._clamp_01(matched_cells / total_cells)

    def sql_correctness_score(
        self,
        candidate_result_rows: list[dict[str, Any]] | None,
        expected_result_rows: list[dict[str, Any]] | None,
        sql_error: str | None = None,
    ) -> float:
        """Compute SQL correctness in [0.0, 1.0] with exact + partial row matching.

        Edge cases:
        - SQL error returns 0.0.
        - Both empty results return 1.0.
        - One empty and one non-empty returns 0.0.
        """
        if sql_error:
            return 0.0
        if candidate_result_rows is None or expected_result_rows is None:
            return 0.0

        expected = canonicalize_rows(expected_result_rows)
        candidate = canonicalize_rows(candidate_result_rows)

        if not expected and not candidate:
            return 1.0
        if not expected or not candidate:
            return 0.0

        expected_counter = Counter(self._row_signature(row) for row in expected)
        candidate_counter = Counter(self._row_signature(row) for row in candidate)
        exact_matches = sum((expected_counter & candidate_counter).values())
        exact_ratio = exact_matches / len(expected)

        partial_total = 0.0
        for expected_row in expected:
            best_similarity = max(
                (self._row_similarity(expected_row, candidate_row) for candidate_row in candidate),
                default=0.0,
            )
            partial_total += best_similarity

        partial_ratio = partial_total / len(expected)

        # Favor exact result-set equality while still granting partial clinical credit
        # when the query captures only part of the required cohort/aggregation.
        weighted = (0.7 * exact_ratio) + (0.3 * partial_ratio)
        return self._clamp_01(weighted)

    def sql_efficiency_bonus(
        self,
        query_plan_cost: float | None,
        query_cost_threshold: float | None = None,
    ) -> float:
        """Return +0.1 bonus when query plan cost is below threshold.

        Clinical motivation:
        Fast, cost-efficient analytics reduce latency for downstream clinical workflows,
        helping clinicians access validated insights with lower infrastructure overhead.
        """
        if query_plan_cost is None:
            return 0.0

        threshold = self.query_cost_threshold if query_cost_threshold is None else max(0.0, query_cost_threshold)
        if threshold <= 0.0:
            return 0.0

        return 0.1 if query_plan_cost <= threshold else 0.0

    def step_penalty(self, unnecessary_actions: int) -> float:
        """Return cumulative step penalty for unnecessary actions.

        Clinical motivation:
        Penalizing trial-and-error action spam encourages deliberate, auditable decisions,
        which is safer for data quality pipelines in regulated clinical environments.
        """
        return -self.step_penalty_per_action * max(0, unnecessary_actions)

    def combined_episode_reward(self, clean_score: float, query_score: float, efficiency_bonus: float) -> float:
        """Compute weighted reward before step penalties.

        Formula (required):
            0.4 * clean_score + 0.4 * query_score + 0.2 * efficiency_bonus
        """
        return (0.4 * self._clamp_01(clean_score)) + (0.4 * self._clamp_01(query_score)) + (0.2 * efficiency_bonus)

    def query_plan_total_cost(self, explain_plan: dict[str, Any] | list[Any] | None) -> float | None:
        """Extract total cost from a PostgreSQL EXPLAIN (FORMAT JSON) payload.

        Returns None if cost cannot be identified.
        """
        if explain_plan is None:
            return None

        def _extract(node: Any) -> float | None:
            if isinstance(node, dict):
                if "Total Cost" in node:
                    try:
                        return float(node["Total Cost"])
                    except (TypeError, ValueError):
                        return None
                for value in node.values():
                    found = _extract(value)
                    if found is not None:
                        return found
                return None

            if isinstance(node, list):
                for item in node:
                    found = _extract(item)
                    if found is not None:
                        return found
                return None

            return None

        return _extract(explain_plan)

    def calculate(
        self,
        candidate_clean_rows: list[dict[str, Any]] | None,
        expected_clean_rows: list[dict[str, Any]] | None,
        candidate_query_rows: list[dict[str, Any]] | None,
        expected_query_rows: list[dict[str, Any]] | None,
        sql_error: str | None = None,
        query_plan_cost: float | None = None,
        unnecessary_actions: int = 0,
        query_cost_threshold: float | None = None,
    ) -> RewardModel:
        """Return structured reward breakdown including weighted total and penalties."""
        clean_score = self.data_clean_score(candidate_clean_rows, expected_clean_rows)
        query_score = self.sql_correctness_score(candidate_query_rows, expected_query_rows, sql_error=sql_error)
        efficiency_bonus = self.sql_efficiency_bonus(query_plan_cost, query_cost_threshold=query_cost_threshold)
        step_penalty = self.step_penalty(unnecessary_actions)

        weighted_reward = self.combined_episode_reward(clean_score, query_score, efficiency_bonus)
        total_reward = weighted_reward + step_penalty

        return RewardModel(
            data_clean_score=clean_score,
            query_correct_score=query_score,
            efficiency_bonus=efficiency_bonus,
            step_penalty=step_penalty,
            total=total_reward,
        )
