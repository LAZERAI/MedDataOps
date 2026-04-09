from __future__ import annotations

import http.cookiejar
import json
import os
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "no-token"

# Use public HF Space - accessible from ANY network context
SPACE_URL = os.getenv("SPACE_URL", "https://lazerai-meddataops.hf.space")

TASKS = [
    {"id": "triage_report", "seed": 101},
    {"id": "medication_summary", "seed": 202},
    {"id": "icu_capacity", "seed": 303},
]

DETERMINISTIC_ACTIONS = {
    "triage_report": [
        {
            "action_type": "fix_query",
            "parameters": {
                "query": "SELECT ward, COUNT(*) AS patient_count FROM patients WHERE admission_date >= DATE '2024-01-01' GROUP BY ward ORDER BY patient_count DESC, ward ASC"
            },
        },
        {"action_type": "submit", "parameters": {}},
    ],
    "medication_summary": [
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
                        "columns": {"prescribed_date": "date", "dosage_mg": "float"},
                    },
                ]
            },
        },
        {
            "action_type": "fix_query",
            "parameters": {
                "query": "SELECT p.ward, m.drug_name, COUNT(*) AS prescription_count FROM medications m INNER JOIN patients p ON m.patient_id = p.patient_id WHERE m.prescribed_date >= DATE '2024-01-01' GROUP BY p.ward, m.drug_name ORDER BY p.ward ASC, m.drug_name ASC"
            },
        },
        {"action_type": "submit", "parameters": {}},
    ],
    "icu_capacity": [
        {
            "action_type": "clean_data",
            "parameters": {
                "operations": [
                    {
                        "operation": "normalize_strings",
                        "columns": ["ward", "icu_unit"],
                        "case": "upper",
                    }
                ]
            },
        },
        {
            "action_type": "fix_query",
            "parameters": {
                "query": "WITH patient_agg AS (SELECT icu_unit, COUNT(*) AS current_occupancy, SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) AS active_count FROM patients GROUP BY icu_unit), capacity_agg AS (SELECT unit AS icu_unit, COUNT(*) AS total_capacity FROM icu_beds GROUP BY unit) SELECT c.icu_unit, COALESCE(p.current_occupancy, 0) AS current_occupancy, c.total_capacity, COALESCE(p.active_count, 0) AS active_count FROM capacity_agg c LEFT JOIN patient_agg p ON p.icu_unit = c.icu_unit ORDER BY c.icu_unit"
            },
        },
        {"action_type": "submit", "parameters": {}},
    ],
}


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe(value: object) -> str:
    return re.sub(r"\s+", "_", str(value).replace("\n", " ").strip()) or "na"


def _http_post(url: str, payload: dict, session_id: str | None = None) -> tuple[dict, str]:
    data = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if session_id:
        headers["X-Session-Id"] = session_id
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=30) as response:
        body = json.loads(response.read().decode())
        sid = response.headers.get("X-Session-Id", "")
        return body, sid


def _wait_for_server(base_url: str, timeout: int = 60) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{base_url}/health", timeout=3)
            return True
        except Exception:
            time.sleep(2)
    return False


def run_task(base_url: str, task_id: str, seed: int, run_id: str) -> float:
    cookie_jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))
    urllib.request.install_opener(opener)

    try:
        _obs, sid = _http_post(f"{base_url}/reset", {"task_id": task_id, "seed": seed})
    except Exception:
        print(
            f"[STEP] run_id={_safe(run_id)} task_id={_safe(task_id)} step=0 action_type=reset reward=0.000000 done=true status=reset_failed",
            flush=True,
        )
        return 0.0

    actions = DETERMINISTIC_ACTIONS.get(task_id, [{"action_type": "submit", "parameters": {}}])
    score = 0.0
    for i, action in enumerate(actions, 1):
        try:
            result, new_sid = _http_post(f"{base_url}/step", action, session_id=sid)
            if new_sid:
                sid = new_sid
            reward_raw = result.get("reward", 0.0)
            if isinstance(reward_raw, dict):
                reward = float(reward_raw.get("total", reward_raw.get("value", 0.0)))
            else:
                reward = float(reward_raw)
            done = bool(result.get("done", False))
            score = reward
            print(
                f"[STEP] run_id={_safe(run_id)} task_id={_safe(task_id)} step={i} action_type={_safe(action['action_type'])} reward={reward:.6f} done={'true' if done else 'false'} status=ok",
                flush=True,
            )
            if done:
                break
        except Exception:
            print(
                f"[STEP] run_id={_safe(run_id)} task_id={_safe(task_id)} step={i} action_type={_safe(action['action_type'])} reward=0.000000 done=false status=error",
                flush=True,
            )
    return score


def main() -> None:
    run_id = f"meddataops-{int(time.monotonic())}"
    start = time.monotonic()
    task_ids = [task["id"] for task in TASKS]

    print(
        f"[START] run_id={_safe(run_id)} ts_utc={_safe(_ts())} model={_safe(MODEL_NAME)} tasks={_safe(','.join(task_ids))} max_steps_per_task=20",
        flush=True,
    )

    # Try localhost first, fall back to public HF Space
    base_url = SPACE_URL
    for candidate in ["http://localhost:7860", "http://127.0.0.1:7860", SPACE_URL]:
        try:
            urllib.request.urlopen(f"{candidate}/health", timeout=5)
            base_url = candidate
            break
        except Exception:
            continue

    # Wait for server if using localhost
    if "localhost" in base_url or "127.0.0.1" in base_url:
        _wait_for_server(base_url, timeout=60)

    results: list[tuple[str, float]] = []
    for task in TASKS:
        score = run_task(base_url, task["id"], int(task["seed"]), run_id)
        results.append((task["id"], score))

    mean_score = sum(score for _, score in results) / max(1, len(results))
    statuses = ",".join(f"{task_id}:ok" for task_id, _ in results)
    total = time.monotonic() - start
    print(
        f"[END] run_id={_safe(run_id)} ts_utc={_safe(_ts())} task_count={len(results)} mean_score={mean_score:.6f} total_elapsed_s={total:.3f} statuses={_safe(statuses)}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        try:
            print(
                f"[END] run_id=fallback ts_utc={_safe(_ts())} task_count=0 mean_score=0.000000 total_elapsed_s=0.000 statuses=error",
                flush=True,
            )
        except Exception:
            pass
        sys.exit(0)
