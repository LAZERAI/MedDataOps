from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from meddataops.env import MedDataOpsEnv
from meddataops.tasks import TASK_REGISTRY, get_task
import meddataops.tasks.icu_capacity as ic
import meddataops.tasks.medication_summary as ms
import meddataops.tasks.triage_report as tr


@dataclass
class TaskCheckResult:
    task_id: str
    reset_ok: bool
    gt_score: float
    wrong_score: float
    gt_pass: bool
    wrong_pass: bool


def _ground_truth_query_rows(task_id: str, expected_clean_rows: list[dict]) -> list[dict]:
    if task_id == "triage_report":
        return tr._ward_count_rows(pd.DataFrame(expected_clean_rows))
    if task_id == "medication_summary":
        return ms.MEDICATION_SUMMARY_GROUND_TRUTH_EXPECTED_RESULT
    if task_id == "icu_capacity":
        return ic.ICU_CAPACITY_GROUND_TRUTH_EXPECTED_RESULT
    return []


def _score_ground_truth(task_id: str, expected_clean_rows: list[dict], expected_sql: str) -> float:
    query_rows = _ground_truth_query_rows(task_id, expected_clean_rows)
    if task_id == "icu_capacity":
        return TASK_REGISTRY.run_grader(
            task_id,
            expected_clean_rows,
            query_rows,
            agent_query=expected_sql,
        )
    return TASK_REGISTRY.run_grader(task_id, expected_clean_rows, query_rows)


def _score_wrong(task_id: str) -> float:
    if task_id == "icu_capacity":
        return TASK_REGISTRY.run_grader(task_id, [], [], agent_query="")
    return TASK_REGISTRY.run_grader(task_id, [], [])


def main() -> None:
    env = MedDataOpsEnv(seed=123)

    task_ids = ["triage_report", "medication_summary", "icu_capacity"]
    results: list[TaskCheckResult] = []

    for task_id in task_ids:
        reset_ok = False
        try:
            state = env.reset(task_id=task_id, seed=123)
            reset_ok = state.task.id == task_id
        except Exception:
            reset_ok = False

        task = get_task(task_id)
        gt_score = _score_ground_truth(task_id, task.expected_clean_rows, task.expected_sql)
        wrong_score = _score_wrong(task_id)

        results.append(
            TaskCheckResult(
                task_id=task_id,
                reset_ok=reset_ok,
                gt_score=float(gt_score),
                wrong_score=float(wrong_score),
                gt_pass=abs(float(gt_score) - 1.0) < 1e-9,
                wrong_pass=abs(float(wrong_score) - 0.0) < 1e-9,
            )
        )

    print("\nMedDataOps Task Validation Summary")
    print("task_id            | reset_ok | gt_score | wrong_score | gt_pass | wrong_pass")
    print("-------------------+----------+----------+-------------+---------+-----------")
    for row in results:
        print(
            f"{row.task_id:<18} | {str(row.reset_ok):<8} | {row.gt_score:<8.6f} | "
            f"{row.wrong_score:<11.6f} | {str(row.gt_pass):<7} | {str(row.wrong_pass):<9}"
        )

    all_ok = all(r.reset_ok and r.gt_pass and r.wrong_pass for r in results)
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
