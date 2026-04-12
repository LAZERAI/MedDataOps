"""
MedDataOps inference.py
Mandatory env vars: API_BASE_URL, MODEL_NAME, HF_TOKEN
"""
import json
import os
import re
import sys
import time
from typing import Any, List, Optional

import requests
from openai import OpenAI

# ── Config (checklist-compliant) ──────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")          # NO default — mandatory
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")  # optional

ENV_BASE_URL = "http://localhost:7860"
MAX_STEPS    = 12
TASKS = [
    {"id": "triage_report",      "seed": 101},
    {"id": "medication_summary", "seed": 202},
    {"id": "icu_capacity",       "seed": 303},
]

SYSTEM_PROMPT = (
    "You are a clinical data engineer. Clean the hospital dataset and fix the broken SQL query.\n"
    "Output ONE JSON action only. No markdown. No explanation. Just the JSON.\n\n"
    "Actions:\n"
    '  {"action_type":"clean_data","parameters":{"operations":[{"operation":"normalize_strings","columns":["ward"],"case":"lower"},{"operation":"remove_duplicates","columns":["patient_id"]}]}}\n'
    '  {"action_type":"fix_query","parameters":{"query":"SELECT ..."}}\n'
    '  {"action_type":"run_query","parameters":{"query":"SELECT ..."}}\n'
    '  {"action_type":"submit","parameters":{}}\n'
)

# ── Logging (exact validator format) ─────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err = error.replace("\n", " ").replace("\r", "")[:120] if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    r = ",".join(f"{x:.2f}" for x in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={r}", flush=True)

# ── HTTP helpers ──────────────────────────────────────────────────────────────

def wait_for_server(retries: int = 15, delay: int = 4) -> bool:
    for i in range(retries):
        try:
            r = requests.get(f"{ENV_BASE_URL}/health", timeout=5)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(delay)
    return False

def env_reset(task_id: str, seed: int):
    for attempt in range(4):
        try:
            r = requests.post(f"{ENV_BASE_URL}/reset",
                              json={"task_id": task_id, "seed": seed}, timeout=30)
            r.raise_for_status()
            sid = r.headers.get("X-Session-Id") or r.cookies.get("session_id")
            return r.json(), sid
        except Exception as e:
            if attempt == 3:
                raise RuntimeError(f"reset failed: {e}")
            time.sleep(2)

def env_step(action: dict, session_id: Optional[str]) -> dict:
    headers = {"Content-Type": "application/json"}
    cookies = {}
    if session_id:
        headers["X-Session-Id"] = session_id
        cookies["session_id"] = session_id
    for attempt in range(4):
        try:
            r = requests.post(f"{ENV_BASE_URL}/step",
                              json=action, headers=headers, cookies=cookies, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == 3:
                raise RuntimeError(f"step failed: {e}")
            time.sleep(2)

# ── LLM ──────────────────────────────────────────────────────────────────────

def call_llm(client: OpenAI, obs_text: str) -> str:
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": obs_text},
            ],
            temperature=0.0,
            max_tokens=512,
        )
        return (resp.choices[0].message.content or "{}").strip()
    except Exception as e:
        print(f"[warn] LLM failed: {e}", file=sys.stderr)
        return "{}"

def parse_action(content: str) -> dict:
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    # try direct
    try:
        obj = json.loads(content)
        if isinstance(obj, dict) and "action_type" in obj:
            return obj
    except Exception:
        pass
    # balanced braces
    depth = 0; start = -1
    for i, ch in enumerate(content):
        if ch == "{":
            if depth == 0: start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    obj = json.loads(content[start:i+1])
                    if isinstance(obj, dict) and "action_type" in obj:
                        return obj
                except Exception:
                    pass
                start = -1
    return {"action_type": "submit", "parameters": {}}

def extract_reward(result: dict) -> float:
    r = result.get("reward", 0.0)
    if isinstance(r, (int, float)): return float(r)
    if isinstance(r, dict):
        for k in ("total", "value", "reward"):
            if k in r:
                try: return float(r[k])
                except Exception: pass
    return 0.0

def build_obs_text(task_id: str, step: int, obs: dict) -> str:
    dataset = obs.get("current_dataset_state", [])
    sample  = dataset[:3] if isinstance(dataset, list) else []
    rows    = len(dataset) if isinstance(dataset, list) else 0
    sql     = str(obs.get("current_sql_query", ""))[:600]
    errors  = obs.get("error_messages", [])
    desc    = str(obs.get("task_description", ""))[:400]
    return (f"Task:{task_id} Step:{step}\nDesc:{desc}\nRows:{rows} Sample:{json.dumps(sample)[:300]}\n"
            f"SQL:\n{sql}\nErrors:{json.dumps(errors)[:200]}")

# ── Episode ───────────────────────────────────────────────────────────────────

def run_task(client: OpenAI, task_id: str, seed: int) -> float:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0

    log_start(task=task_id, env="meddataops", model=MODEL_NAME)

    try:
        reset_data, session_id = env_reset(task_id, seed)
    except Exception as e:
        log_step(1, "reset_failed", 0.0, True, str(e)[:80])
        log_end(False, 0, 0.0, [])
        return 0.0

    obs  = reset_data if "current_dataset_state" in reset_data else reset_data.get("observation", reset_data)
    done = bool(reset_data.get("done", False))

    for step in range(1, MAX_STEPS + 1):
        if done:
            break
        obs_text   = build_obs_text(task_id, step, obs if isinstance(obs, dict) else {})
        llm_out    = call_llm(client, obs_text)
        action     = parse_action(llm_out)
        action_str = action.get("action_type", "submit")
        error_str: Optional[str] = None
        reward = 0.0

        try:
            result  = env_step(action, session_id)
            nxt     = result.get("observation", {})
            if isinstance(nxt, dict) and nxt:
                obs = nxt
            reward  = extract_reward(result)
            done    = bool(result.get("done", False))
            errs    = obs.get("error_messages", []) if isinstance(obs, dict) else []
            error_str = errs[0] if errs else None
            info    = result.get("info", {})
            if isinstance(info, dict):
                for k in ("final_score", "score", "task_score"):
                    if k in info:
                        try: score = max(score, float(info[k]))
                        except Exception: pass
        except Exception as e:
            error_str = str(e)[:80]
            done = True

        rewards.append(reward)
        steps_taken = step
        log_step(step, action_str, reward, done, error_str)
        if done:
            break

    if score == 0.0 and rewards:
        score = max(rewards)
    score   = min(max(score, 0.0), 1.0)
    log_end(score >= 0.5, steps_taken, score, rewards)
    return score

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy")

        server_ok = wait_for_server()
        if not server_ok:
            # server never came up — emit clean output anyway
            for t in TASKS:
                log_start(task=t["id"], env="meddataops", model=MODEL_NAME)
                log_step(1, "server_unavailable", 0.0, True, "server_not_ready")
                log_end(False, 1, 0.0, [0.0])
            return

        for t in TASKS:
            try:
                run_task(client, t["id"], t["seed"])
            except Exception as e:
                print(f"[error] task {t['id']}: {e}", file=sys.stderr)
                log_start(task=t["id"], env="meddataops", model=MODEL_NAME)
                log_end(False, 0, 0.0, [])
    except Exception as e:
        print(f"[fatal] {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
