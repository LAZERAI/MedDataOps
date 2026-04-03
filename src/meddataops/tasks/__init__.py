from __future__ import annotations

from meddataops.models import TaskSpec
from meddataops.tasks.easy import TASK as EASY_TASK
from meddataops.tasks.hard import TASK as HARD_TASK
from meddataops.tasks.medium import TASK as MEDIUM_TASK


TASK_REGISTRY: dict[str, TaskSpec] = {
    EASY_TASK.id: EASY_TASK,
    MEDIUM_TASK.id: MEDIUM_TASK,
    HARD_TASK.id: HARD_TASK,
}


def list_tasks() -> list[str]:
    return list(TASK_REGISTRY.keys())


def get_task(task_id: str) -> TaskSpec:
    if task_id not in TASK_REGISTRY:
        raise KeyError(f"Unknown task_id: {task_id}")
    return TASK_REGISTRY[task_id].model_copy(deep=True)
