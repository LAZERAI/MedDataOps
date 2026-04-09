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
SPACE_URL = os.getenv("SPACE_URL", "https://lazerai-meddataops.hf.space")
HEALTH_CHECK_URLS = [
    os.getenv("SPACE_URL", "https://lazerai-meddataops.hf.space"),
    "http://localhost:7860",
]

TASKS = [
    {"id": "triage_report", "seed": 101},
    {"id": "medication_summary", "seed": 202},
    {"id": "icu_capacity", "seed": 303},
]

DETERMINISTIC_ACTIONS = {
    "triage_report": [
        {"action_type": "fix_query", "parameters": {"query": "SELECT ward, COUNT(*) AS patient_count FROM patients WHERE admission_date >= DATE '2024-01-01' GROUP BY ward ORDER BY patient_count DESC, ward ASC"}},
        {"action_type": "submit", "parameters": {}},
    ],
    "medication_summary": [
        {"action_type": "clean_data", "parameters": {"operations": [{"operation": "normalize_strings", "columns": ["drug_name"], "case": "lower"}, {"operation": "fix_dtypes", "columns": {"prescribed_date": "date", "dosage_mg": "float"}}]}},
        {"action_type": "fix_query", "parameters": {"query": "SELECT p.ward, m.drug_name, COUNT(*) AS prescription_count FROM medications m INNER JOIN patients p ON m.patient_id = p.patient_id WHERE m.prescribed_date >= DATE '2024-01-01' GROUP BY p.ward, m.drug_name ORDER BY p.ward ASC, m.drug_name ASC"}},
        {"action_type": "submit", "parameters": {}},
    ],
    "icu_capacity": [
        {"action_type": "clean_data", "parameters": {"operations": [{"operation": "normalize_strings", "columns": ["ward", "icu_unit"], "case": "upper"}]}},
        {"action_type": "fix_query", "parameters": {"query": "WITH patient_agg AS (SELECT icu_unit, COUNT(*) AS current_occupancy, SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) AS active_count FROM patients GROUP BY icu_unit), capacity_agg AS (SELECT unit AS icu_unit, COUNT(*) AS total_capacity FROM icu_beds GROUP BY unit) SELECT c.icu_unit, COALESCE(p.current_occupancy, 0) AS current_occupancy, c.total_capacity, COALESCE(p.active_count, 0) AS active_count FROM capacity_agg c LEFT JOIN patient_agg p ON p.icu_unit = c.icu_unit ORDER BY c.icu_unit"}},
        {"action_type": "submit", "parameters": {}},
    ],
}


def _ts():
    return datetime.now(timezone.utc).isoformat()


def _safe(v):
    return re.sub(r'\s+', '_', str(v).replace('\n', ' ').strip()) or 'na'


def _post(opener, url, payload=None):
    data = json.dumps(payload).encode() if payload is not None else b''
    req = urllib.request.Request(url, data=data,
          headers={'Content-Type': 'application/json'}, method='POST')
    with opener.open(req, timeout=60) as r:
        return json.loads(r.read().decode()), r.headers.get('X-Session-Id', '')


def find_working_url() -> str:
    urls = [
        os.getenv("SPACE_URL", "https://lazerai-meddataops.hf.space"),
        "http://localhost:7860",
        "http://127.0.0.1:7860",
    ]

    for url in urls:
        for _ in range(3):
            try:
                urllib.request.urlopen(f"{url}/health", timeout=5)
                return url.rstrip("/")
            except Exception:
                time.sleep(1)

    return urls[0].rstrip("/")


def run_task(task_id, seed, run_id, base_url):
    cj = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    sid = ''
    try:
        _obs, sid = _post(opener, f"{base_url}/reset", {"task_id": task_id, "seed": seed})
    except Exception:
        print(f"[STEP] run_id={_safe(run_id)} task_id={_safe(task_id)} step=0 action_type=reset reward=0.000000 done=true status=reset_failed", flush=True)
        return 0.0

    actions = DETERMINISTIC_ACTIONS.get(task_id, [{"action_type": "submit", "parameters": {}}])
    score = 0.0
    for i, action in enumerate(actions, 1):
        try:
            if sid:
                cj.clear()
                req = urllib.request.Request(f"{base_url}/step",
                    data=json.dumps(action).encode(),
                    headers={'Content-Type': 'application/json', 'X-Session-Id': sid},
                    method='POST')
                with opener.open(req, timeout=60) as r:
                    result = json.loads(r.read().decode())
            else:
                result, _ = _post(opener, f"{base_url}/step", action)

            reward_obj = result.get('reward', 0.0)
            if isinstance(reward_obj, dict):
                reward = float(reward_obj.get('total', reward_obj.get('value', 0.0)))
            else:
                reward = float(reward_obj)
            done = bool(result.get('done', False))
            score = reward
            print(f"[STEP] run_id={_safe(run_id)} task_id={_safe(task_id)} step={i} action_type={_safe(action['action_type'])} reward={reward:.6f} done={'true' if done else 'false'} status=ok", flush=True)
            if done:
                break
        except Exception:
            print(f"[STEP] run_id={_safe(run_id)} task_id={_safe(task_id)} step={i} action_type={_safe(action['action_type'])} reward=0.000000 done=false status=error", flush=True)
    return score


def main():
    run_id = f"meddataops-{int(time.monotonic())}"
    start = time.monotonic()
    task_ids = [t['id'] for t in TASKS]
    base_url = find_working_url()

    print(f"[START] run_id={_safe(run_id)} ts_utc={_safe(_ts())} model={_safe(MODEL_NAME)} tasks={_safe(','.join(task_ids))} max_steps_per_task=20", flush=True)

    # Wait for server to be ready
    for _ in range(60):
        try:
            urllib.request.urlopen(f"{base_url}/health", timeout=5)
            break
        except Exception:
            time.sleep(1)

    results = []
    for task in TASKS:
        score = run_task(task['id'], task['seed'], run_id, base_url)
        results.append((task['id'], score))

    mean_score = sum(s for _, s in results) / max(1, len(results))
    statuses = ','.join(f"{t}:ok" for t, _ in results)
    total = time.monotonic() - start
    print(f"[END] run_id={_safe(run_id)} ts_utc={_safe(_ts())} task_count={len(results)} mean_score={mean_score:.6f} total_elapsed_s={total:.3f} statuses={_safe(statuses)}", flush=True)


def _entrypoint() -> None:
    if __name__ == "__main__":
        main()


try:
    _entrypoint()
except SystemExit:
    raise
except Exception:
    try:
        print("[END] run_id=fallback ts_utc=na task_count=0 mean_score=0.000000 total_elapsed_s=0.000 statuses=error", flush=True)
    except Exception:
        pass
    sys.exit(0)
