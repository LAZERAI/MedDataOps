from __future__ import annotations

import argparse
import http.cookiejar
import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


DEFAULT_BASE_URL = "https://lazerai-meddataops.hf.space"


TRIAGE_SQL = (
    "SELECT ward, COUNT(*) AS patient_count "
    "FROM patients "
    "WHERE admission_date >= DATE '2024-01-01' "
    "GROUP BY ward "
    "ORDER BY patient_count DESC, ward ASC"
)

MEDICATION_SQL = (
    "SELECT p.ward, m.drug_name, COUNT(*) AS prescription_count "
    "FROM medications m "
    "INNER JOIN patients p ON m.patient_id = p.patient_id "
    "WHERE m.prescribed_date >= DATE '2024-01-01' "
    "GROUP BY p.ward, m.drug_name "
    "ORDER BY p.ward ASC, m.drug_name ASC"
)

ICU_SQL = (
    "WITH patient_agg AS ("
    "  SELECT "
    "    icu_unit, "
    "    COUNT(*) AS current_occupancy, "
    "    SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) AS active_count "
    "  FROM patients "
    "  GROUP BY icu_unit"
    "), capacity_agg AS ("
    "  SELECT "
    "    unit AS icu_unit, "
    "    COUNT(*) AS total_capacity "
    "  FROM icu_beds "
    "  GROUP BY unit"
    ") "
    "SELECT "
    "  c.icu_unit, "
    "  COALESCE(p.current_occupancy, 0) AS current_occupancy, "
    "  c.total_capacity, "
    "  COALESCE(p.active_count, 0) AS active_count "
    "FROM capacity_agg c "
    "LEFT JOIN patient_agg p "
    "  ON p.icu_unit = c.icu_unit "
    "ORDER BY c.icu_unit"
)


@dataclass(frozen=True)
class DeterministicTask:
    task_id: str
    actions: tuple[dict[str, Any], ...]


@dataclass
class TaskRunResult:
    task_id: str
    score: float
    steps: int
    status: str
    error: str = ""


TASKS: tuple[DeterministicTask, ...] = (
    DeterministicTask(
        task_id="triage_report",
        actions=(
            {"action_type": "fix_query", "parameters": {"query": TRIAGE_SQL}},
            {"action_type": "submit", "parameters": {}},
        ),
    ),
    DeterministicTask(
        task_id="medication_summary",
        actions=(
            {
                "action_type": "clean_data",
                "parameters": {
                    "operations": [
                        {
                            "operation": "normalize_strings",
                            "columns": ["drug_name"],
                            "case": "lower",
                        },
                        {
                            "operation": "fix_dtypes",
                            "columns": {
                                "prescribed_date": "date",
                                "dosage_mg": "float",
                                "patient_id": "string",
                            },
                        },
                    ]
                },
            },
            {"action_type": "fix_query", "parameters": {"query": MEDICATION_SQL}},
            {"action_type": "submit", "parameters": {}},
        ),
    ),
    DeterministicTask(
        task_id="icu_capacity",
        actions=(
            {"action_type": "fix_query", "parameters": {"query": ICU_SQL}},
            {"action_type": "submit", "parameters": {}},
        ),
    ),
)


def post_json(
    opener: urllib.request.OpenerDirector,
    base_url: str,
    path: str,
    payload: dict[str, Any] | None,
) -> dict[str, Any]:
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        f"{base_url}{path}",
        data=data,
        headers=headers,
        method="POST",
    )

    with opener.open(request, timeout=60) as response:
        raw = response.read().decode("utf-8")
        return json.loads(raw)


def get_json(opener: urllib.request.OpenerDirector, base_url: str, path: str) -> dict[str, Any]:
    request = urllib.request.Request(f"{base_url}{path}", method="GET")
    with opener.open(request, timeout=60) as response:
        raw = response.read().decode("utf-8")
        return json.loads(raw)


def extract_score(reward_payload: Any) -> float:
    if isinstance(reward_payload, (int, float)):
        return float(reward_payload)

    if isinstance(reward_payload, dict):
        if isinstance(reward_payload.get("value"), (int, float)):
            return float(reward_payload["value"])
        if isinstance(reward_payload.get("total"), (int, float)):
            return float(reward_payload["total"])

    raise ValueError(f"Unable to extract score from reward payload: {reward_payload!r}")


def run_task(base_url: str, task: DeterministicTask) -> TaskRunResult:
    cookie_jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))

    try:
        post_json(opener, base_url, "/reset", {"task_id": task.task_id})

        final_step: dict[str, Any] | None = None
        for action in task.actions:
            final_step = post_json(opener, base_url, "/step", action)

        if final_step is None:
            raise RuntimeError("No steps executed.")

        score = extract_score(final_step.get("reward"))
        observation = final_step.get("observation", {})
        step_number = int(observation.get("step_number", len(task.actions)))
        done = bool(final_step.get("done", False))

        if not done:
            return TaskRunResult(task_id=task.task_id, score=score, steps=step_number, status="not_done")

        return TaskRunResult(task_id=task.task_id, score=score, steps=step_number, status="ok")

    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8")
        except Exception:
            body = "<no body>"
        return TaskRunResult(
            task_id=task.task_id,
            score=0.0,
            steps=0,
            status="http_error",
            error=f"HTTP {exc.code}: {body}",
        )
    except Exception as exc:
        return TaskRunResult(task_id=task.task_id, score=0.0, steps=0, status="error", error=str(exc))


def render_markdown(results: list[TaskRunResult]) -> str:
    lines = [
        "| task | solver | score | steps | status |",
        "|---|---|---:|---:|---|",
    ]
    for row in results:
        lines.append(
            f"| {row.task_id} | reference_solver | {row.score:.4f} | {row.steps} | {row.status} |"
        )
    return "\n".join(lines)


def run(base_url: str, json_output: str | None) -> int:
    base_url = base_url.rstrip("/")

    health_cookie_jar = http.cookiejar.CookieJar()
    health_opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(health_cookie_jar))
    health = get_json(health_opener, base_url, "/health")
    if health.get("status") != "ok":
        print(f"Health check failed: {health}", file=sys.stderr)
        return 1

    results = [run_task(base_url, task) for task in TASKS]
    markdown_table = render_markdown(results)
    print(markdown_table)

    if json_output:
        payload = {
            "base_url": base_url,
            "health": health,
            "results": [
                {
                    "task_id": r.task_id,
                    "score": r.score,
                    "steps": r.steps,
                    "status": r.status,
                    "error": r.error,
                }
                for r in results
            ],
        }
        with open(json_output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")

    failed = [r for r in results if r.status != "ok"]
    if failed:
        return 1

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministic reference policy for MedDataOps. "
            "Runs hardcoded action sequences against a live MedDataOps endpoint."
        )
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Base URL of the MedDataOps API")
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional path to save raw run results as JSON",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run(base_url=args.base_url, json_output=args.json_output)


if __name__ == "__main__":
    raise SystemExit(main())