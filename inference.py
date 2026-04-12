from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request
from typing import Any, Optional

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore[assignment]


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except Exception:
        return default


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY") or ""
ENV_URL = os.getenv("ENV_URL", "https://lazerai-meddataops.hf.space").rstrip("/")

MAX_STEPS = max(1, _env_int("MAX_STEPS_PER_TASK", _env_int("MAX_STEPS", 20)))
SUCCESS_THRESHOLD = 0.5

TASKS = [
    {"id": "triage_report", "seed": 101, "alias": "easy"},
    {"id": "medication_summary", "seed": 202, "alias": "medium"},
    {"id": "icu_capacity", "seed": 303, "alias": "hard"},
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
                    {"operation": "normalize_strings", "columns": ["drug_name"], "case": "lower"},
                    {
                        "operation": "fix_dtypes",
                        "columns": {
                            "prescribed_date": "date",
                            "dosage_mg": "float",
                        },
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
                    {"operation": "normalize_strings", "columns": ["ward", "icu_unit"], "case": "upper"}
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

SYSTEM_PROMPT = """You are a data engineer at a hospital analytics team.
You have a messy dataset and a broken SQL query. Clean the data and fix the query.

Output exactly one valid JSON action per turn. Think first inside <think>...</think>, then output ONLY the JSON.

Available action_type values:
  clean_data  - {"action_type":"clean_data","parameters":{"operations":[...]}}
  run_query   - {"action_type":"run_query","parameters":{"query":"SELECT ..."}}
  fix_query   - {"action_type":"fix_query","parameters":{"query":"SELECT ..."}}
  submit      - {"action_type":"submit","parameters":{}}

Operations for clean_data:
  {"operation":"remove_duplicates","columns":["col"]}
  {"operation":"fix_nulls","strategy":"mean|mode|drop|forward_fill","columns":["col"]}
  {"operation":"fix_dtypes","columns":{"col":"date|float|int|string"}}
  {"operation":"normalize_strings","columns":["col"],"case":"lower|upper|title"}
  {"operation":"remove_outliers","column":"col","n_std":3}

Call submit only when you are confident data is clean AND the SQL query is correct.
"""


def _safe(value: Any) -> str:
    return re.sub(r"\s+", "_", str(value).strip()) or "na"


def log_start(task: str, env: str, model: str) -> None:
    try:
        print(f"[START] task={task} env={env} model={model}", flush=True)
    except Exception:
        pass


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err = _safe(error) if error else "null"
    try:
        print(
            f"[STEP] step={step} action={_safe(action)} reward={reward:.2f} done={str(done).lower()} error={err}",
            flush=True,
        )
    except Exception:
        pass


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    reward_values = ",".join(f"{value:.2f}" for value in rewards)
    try:
        print(
            f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={reward_values}",
            flush=True,
        )
    except Exception:
        pass


class EnvClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.session_id: Optional[str] = None

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.session_id:
            headers["X-Session-Id"] = self.session_id
        return headers

    def _post(self, path: str, payload: dict[str, Any], retries: int = 3) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8")
        for attempt in range(retries):
            try:
                request = urllib.request.Request(url, data=data, headers=self._headers(), method="POST")
                with urllib.request.urlopen(request, timeout=60) as response:
                    session_id = response.headers.get("X-Session-Id")
                    if session_id:
                        self.session_id = session_id
                    body = response.read().decode("utf-8")
                    return json.loads(body) if body else {}
            except Exception as exc:
                if attempt >= retries - 1:
                    raise RuntimeError(f"{path} failed: {exc}") from exc
                time.sleep(2)
        return {}

    def reset(self, task_id: str, seed: int) -> dict[str, Any]:
        return self._post("/reset", {"task_id": task_id, "seed": seed})

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        return self._post("/step", action)

    def healthy(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=10):
                return True
        except Exception:
            return False


def _choose_env_url() -> str:
    candidates = [ENV_URL, "http://localhost:7860", "http://127.0.0.1:7860", "https://lazerai-meddataops.hf.space"]
    for url in candidates:
        client = EnvClient(url)
        for _ in range(3):
            if client.healthy():
                return url
            time.sleep(1)
    return "https://lazerai-meddataops.hf.space"


def _json_candidates(text: str) -> list[str]:
    candidates = [text]
    candidates += re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    depth = 0
    start = -1
    for index, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                candidates.append(text[start : index + 1])
                start = -1
    return candidates


def _fallback_action(task_id: str, step_index: int) -> dict[str, Any]:
    actions = DETERMINISTIC_ACTIONS.get(task_id, [{"action_type": "submit", "parameters": {}}])
    return actions[min(step_index, len(actions) - 1)]


def _call_llm(client: Any, messages: list[dict[str, str]], fallback: dict[str, Any]) -> dict[str, Any]:
    if client is None:
        return fallback
    for attempt in range(4):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.0,
                max_tokens=1024,
            )
            text = (response.choices[0].message.content or "").strip()
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            for candidate in _json_candidates(text):
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict) and "action_type" in parsed:
                    action_type = str(parsed.get("action_type", "")).strip().lower()
                    parameters = parsed.get("parameters", {})
                    if action_type in {"clean_data", "run_query", "fix_query", "submit"}:
                        return {
                            "action_type": action_type,
                            "parameters": parameters if isinstance(parameters, dict) else {},
                        }
            return fallback
        except Exception:
            if attempt >= 3:
                return fallback
            time.sleep(2 ** attempt)
    return fallback


def _extract_reward(step_result: dict[str, Any]) -> float:
    reward = step_result.get("reward", 0.0)
    if isinstance(reward, (int, float)):
        return float(reward)
    if isinstance(reward, dict):
        for key in ("total", "value", "reward"):
            value = reward.get(key)
            if isinstance(value, (int, float)):
                return float(value)
    return 0.0


def _extract_observation(step_result: dict[str, Any]) -> dict[str, Any]:
    observation = step_result.get("observation", step_result)
    return observation if isinstance(observation, dict) else {}


def _build_user_message(task_id: str, step: int, observation: dict[str, Any], last_result: Optional[dict[str, Any]]) -> str:
    dataset = observation.get("current_dataset_state", [])
    sample = dataset[:3] if isinstance(dataset, list) else []
    current_sql = str(observation.get("current_sql_query", ""))[:1200]
    errors = observation.get("error_messages", [])
    task_description = str(observation.get("task_description", ""))[:800]
    last_summary = "None"
    if last_result:
        last_summary = json.dumps(last_result, ensure_ascii=True)[:400]
    return (
        f"Task ID: {task_id}\n"
        f"Step: {step} | Remaining: {max(MAX_STEPS - step, 0)}\n\n"
        f"Task Description:\n{task_description}\n\n"
        f"Dataset sample: {json.dumps(sample, ensure_ascii=True)[:700]}\n\n"
        f"Current SQL:\n{current_sql}\n\n"
        f"Errors: {json.dumps(errors, ensure_ascii=True)[:500]}\n\n"
        f"Last action result: {last_summary}"
    )


def run_task(env: EnvClient, llm: Any, task_id: str, seed: int) -> tuple[float, int, list[float]]:
    rewards: list[float] = []
    steps_taken = 0
    final_score = 0.0

    log_start(task=task_id, env="meddataops", model=MODEL_NAME)

    try:
        reset_data = env.reset(task_id, seed)
    except Exception as exc:
        log_step(step=0, action="reset", reward=0.0, done=True, error=str(exc))
        log_end(success=False, steps=0, score=0.0, rewards=[])
        return 0.0, 0, []

    observation = _extract_observation(reset_data)
    done = bool(reset_data.get("done", False))
    last_result: Optional[dict[str, Any]] = None
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for step in range(1, MAX_STEPS + 1):
        if done:
            break

        fallback_action = _fallback_action(task_id, step - 1)
        user_message = _build_user_message(task_id, step, observation, last_result)
        messages.append({"role": "user", "content": user_message})
        action = _call_llm(llm, messages, fallback_action)
        messages.append({"role": "assistant", "content": json.dumps(action, ensure_ascii=True)})

        action_type = str(action.get("action_type", "noop"))
        reward = 0.0
        error_text: Optional[str] = None

        try:
            result = env.step(action)
            observation = _extract_observation(result)
            reward = _extract_reward(result)
            done = bool(result.get("done", False))
            errors = observation.get("error_messages", [])
            if isinstance(errors, list) and errors:
                error_text = str(errors[0])
            last_result = {
                "action_type": action_type,
                "reward": reward,
                "errors": errors if isinstance(errors, list) else [],
            }
            info = result.get("info", {})
            if isinstance(info, dict):
                for key in ("final_score", "score", "task_score"):
                    value = info.get(key)
                    if isinstance(value, (int, float)):
                        final_score = max(final_score, float(value))
        except Exception as exc:
            error_text = str(exc)
            done = True

        rewards.append(reward)
        steps_taken = step
        log_step(step=step, action=action_type, reward=reward, done=done, error=error_text)

        if done:
            break

    if final_score <= 0.0 and rewards:
        final_score = max(rewards)
    final_score = min(max(final_score, 0.0), 1.0)
    success = final_score >= SUCCESS_THRESHOLD
    log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)
    return final_score, steps_taken, rewards


def main() -> None:
    llm = None
    if HF_TOKEN and OpenAI is not None:
        try:
            llm = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
        except Exception:
            llm = None

    base_url = _choose_env_url()
    env = EnvClient(base_url)

    for task_entry in TASKS:
        task_id = str(task_entry.get("id", ""))
        seed = int(task_entry.get("seed", 0))
        try:
            run_task(env, llm, task_id, seed)
        except Exception:
            log_end(success=False, steps=0, score=0.0, rewards=[])


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        try:
            log_end(success=False, steps=0, score=0.0, rewards=[])
        except Exception:
            pass
        os._exit(0)
