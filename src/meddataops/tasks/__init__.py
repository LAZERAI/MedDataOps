from __future__ import annotations

from typing import Any

from meddataops.models import TaskPublic, TaskSpec

from . import icu_capacity, medication_summary, triage_report
from .easy import TASK as EASY_TASK
from .hard import TASK as HARD_TASK
from .medium import TASK as MEDIUM_TASK


class TaskRegistry:
    """Registry and grading dispatcher for MedDataOps tasks.

    Responsibilities:
    - Store task definitions and aliases.
    - Provide deterministic access to task specs.
    - Route scorer calls per task and validate score range [0.0, 1.0].
    """

    def __init__(self) -> None:
        self._tasks: dict[str, TaskSpec] = {
            EASY_TASK.id: EASY_TASK,
            MEDIUM_TASK.id: MEDIUM_TASK,
            HARD_TASK.id: HARD_TASK,
        }
        self._aliases: dict[str, str] = {
            "easy": EASY_TASK.id,
            "medium": MEDIUM_TASK.id,
            "hard": HARD_TASK.id,
        }

    def _resolve_task_id(self, task_id: str) -> str:
        resolved = self._aliases.get(task_id, task_id)
        if resolved not in self._tasks:
            raise KeyError(f"Unknown task_id: {task_id}")
        return resolved

    def get_task(self, task_id: str) -> TaskSpec:
        resolved = self._resolve_task_id(task_id)
        return self._tasks[resolved].model_copy(deep=True)

    def list_tasks(self) -> list[TaskPublic]:
        return [
            TaskPublic(
                id=task.id,
                name=task.name,
                difficulty=task.difficulty,
                description=task.description,
                hints=list(task.hints),
            )
            for task in self._tasks.values()
        ]

    def list_task_ids(self) -> list[str]:
        return list(self._tasks.keys())

    def run_grader(
        self,
        task_id: str,
        agent_table_df: Any,
        agent_query_result: Any,
        *,
        agent_query: str | None = None,
        agent_query_cost: float | None = None,
    ) -> float:
        resolved = self._resolve_task_id(task_id)

        if resolved == "triage_report":
            score = triage_report.score_triage_report(agent_table_df, agent_query_result)
        elif resolved == "medication_summary":
            score = medication_summary.score_medication_summary(agent_table_df, agent_query_result)
        elif resolved == "icu_capacity":
            score = icu_capacity.score_icu_capacity(
                agent_table_df,
                agent_query_result,
                agent_query=agent_query,
                agent_query_cost=agent_query_cost,
            )
        else:  # pragma: no cover
            raise KeyError(f"No grader registered for task_id: {task_id}")

        score = float(score)
        if not (0.0 <= score <= 1.0):
            raise ValueError(f"Invalid grader score for {resolved}: {score} (must be in [0.0, 1.0])")
        return score


TASK_REGISTRY = TaskRegistry()


def list_tasks() -> list[str]:
    return TASK_REGISTRY.list_task_ids()


def get_task(task_id: str) -> TaskSpec:
    return TASK_REGISTRY.get_task(task_id)


__all__ = ["TaskRegistry", "TASK_REGISTRY", "list_tasks", "get_task"]
