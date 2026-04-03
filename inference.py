from __future__ import annotations

import importlib
import inspect
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import APIConnectionError, APIError, APITimeoutError, OpenAI, RateLimitError


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


TOTAL_RUNTIME_BUDGET_SECONDS = 19 * 60
REQUEST_TIMEOUT_SECONDS = 45
MAX_API_RETRIES = 4
MAX_STEPS_PER_TASK = int(os.getenv("MAX_STEPS_PER_TASK", "16"))
GLOBAL_RANDOM_SEED = int(os.getenv("GLOBAL_RANDOM_SEED", "42"))

TASK_RUN_ORDER = [
    {"id": "triage_report", "aliases": ["easy"], "seed": 101},
    {"id": "medication_summary", "aliases": ["medium"], "seed": 202},
    {"id": "icu_capacity", "aliases": ["hard"], "seed": 303},
]

ALLOWED_ACTIONS = {"clean_data", "run_query", "fix_query", "submit"}
LEGACY_ACTION_MAP = {
    "clean_data": "clean_data",
    "run_query": "noop",
    "fix_query": "fix_sql",
    "submit": "submit",
}

SYSTEM_PROMPT = """You are a data engineer at a hospital analytics team. You have been given a messy dataset and a broken SQL query. Your job is to clean the data and fix the query so the report is accurate.

You are interacting with an environment that expects one action per step.

Output contract:
1) Think first in a private reasoning block.
2) Then output exactly one valid JSON action object.
3) Do not output markdown, code fences, or extra commentary.

Required response format:
<think>
- What is wrong with the current data or query?
- What is the most impactful action to take next?
</think>
{"action_type":"...","parameters":{...}}

Outside the <think> block, output ONLY the action JSON.
The action JSON must always contain exactly these top-level keys:
- "action_type"
- "parameters"

Available action_type values and parameter rules:

1) clean_data
Purpose: apply one or more cleaning operations to the current working table.
JSON shape:
{"action_type":"clean_data","parameters":{"operations":[...]}}

Supported operation objects inside parameters.operations:
- remove_duplicates:
    {"operation":"remove_duplicates","columns":["col1","col2"]}
- fix_nulls:
    {"operation":"fix_nulls","strategy":"mean|mode|drop|forward_fill","columns":["col1","col2"]}
- fix_dtypes:
    {"operation":"fix_dtypes","columns":{"col1":"date","col2":"float","col3":"int","col4":"string"}}
- normalize_strings:
    {"operation":"normalize_strings","columns":["col1","col2"],"case":"lower|upper|title"}
- remove_outliers:
    {"operation":"remove_outliers","column":"numeric_col","n_std":3}

Example:
{"action_type":"clean_data","parameters":{"operations":[{"operation":"normalize_strings","columns":["drug_name"],"case":"lower"},{"operation":"fix_dtypes","columns":{"prescribed_date":"date","dosage_mg":"float"}}]}}

2) run_query
Purpose: execute SQL and inspect current results.
JSON shape:
{"action_type":"run_query","parameters":{"query":"SELECT ..."}}
Example:
{"action_type":"run_query","parameters":{"query":"SELECT icu_unit, COUNT(*) AS current_occupancy FROM patients GROUP BY icu_unit"}}

3) fix_query
Purpose: provide a corrected SQL query candidate.
JSON shape:
{"action_type":"fix_query","parameters":{"query":"SELECT ..."}}
Example:
{"action_type":"fix_query","parameters":{"query":"WITH x AS (...) SELECT ..."}}

4) submit
Purpose: finalize answer for the task.
JSON shape:
{"action_type":"submit","parameters":{}}

Error recovery rule:
If the previous step contains an error message, treat it as invalid action/query/parameters.
Do not repeat the same failing action. Choose a different approach that directly addresses the error.

Submission rule:
Call submit only when you are confident BOTH are correct:
- data cleaning state
- SQL query logic and output

Strategy guidance:
- Prioritize actions that produce the largest reduction in data/query defects.
- Use run_query/fix_query iteratively after meaningful cleaning.
- Avoid unnecessary steps.
"""


@dataclass
class TaskRunResult:
    task_id: str
    task_name: str
    steps_taken: int
    final_score: float
    status: str
    elapsed_sec: float


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value.strip()


def _clamp_score(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


def _to_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if hasattr(obj, "__dict__"):
        return dict(vars(obj))
    return {"value": obj}


def _compact_rows(rows: Any, max_rows: int = 6) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    compact: list[dict[str, Any]] = []
    for row in rows[:max_rows]:
        if isinstance(row, dict):
            compact.append(row)
    return compact


def _normalize_observation(raw_observation: Any, task_id: str, step_number: int) -> dict[str, Any]:
    payload = _to_dict(raw_observation)

    if "current_dataset_state" in payload and "current_sql_query" in payload:
        rows = payload.get("current_dataset_state")
        return {
            "task_id": task_id,
            "step_number": int(payload.get("step_number", step_number)),
            "task_description": str(payload.get("task_description", "")),
            "current_sql_query": str(payload.get("current_sql_query", "")),
            "error_messages": payload.get("error_messages", []),
            "dataset_preview": _compact_rows(rows),
            "dataset_preview_count": len(rows) if isinstance(rows, list) else 0,
        }

    dirty_rows = payload.get("dirty_rows", [])
    task_info = payload.get("task", {})
    return {
        "task_id": task_id,
        "step_number": int(payload.get("step_index", step_number)),
        "task_description": str(task_info.get("description", "")),
        "current_sql_query": str(payload.get("broken_sql", "")),
        "error_messages": [payload.get("last_sql_error")] if payload.get("last_sql_error") else [],
        "dataset_preview": _compact_rows(dirty_rows),
        "dataset_preview_count": len(dirty_rows) if isinstance(dirty_rows, list) else 0,
    }


def _extract_json_block(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}

    try:
        loaded = json.loads(text)
        if isinstance(loaded, dict):
            return loaded
    except json.JSONDecodeError:
        pass

    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx >= 0 and end_idx > start_idx:
        candidate = text[start_idx : end_idx + 1]
        try:
            loaded = json.loads(candidate)
            if isinstance(loaded, dict):
                return loaded
        except json.JSONDecodeError:
            pass

    return {}


def _parse_action(content: str) -> dict[str, Any]:
    data = _extract_json_block(content)

    action_obj = data.get("action", data)
    if not isinstance(action_obj, dict):
        action_obj = {}

    action_type = str(action_obj.get("action_type", "submit")).strip().lower()
    if action_type not in ALLOWED_ACTIONS:
        action_type = "submit"

    parameters = action_obj.get("parameters", {})
    if not isinstance(parameters, dict):
        parameters = {}

    return {
        "action_type": action_type,
        "parameters": parameters,
    }


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class EnvBridge:
    def __init__(self, seed: int) -> None:
        env_module = importlib.import_module("meddataops.env")
        env_cls = getattr(env_module, "MedDataOpsEnv")
        self._env = env_cls(seed=seed)
        self._models_module = importlib.import_module("meddataops.models")

    def _build_action_candidates(self, action: dict[str, Any]) -> list[Any]:
        action_type = action["action_type"]
        parameters = action["parameters"]

        candidates: list[Any] = []

        # Modern OpenEnv-style dict payload.
        candidates.append({"action_type": action_type, "parameters": parameters})

        # Legacy dict payload for AgentAction.
        candidates.append({"action_type": LEGACY_ACTION_MAP.get(action_type, "noop"), "payload": parameters})

        # Try ActionModel if available.
        action_model_cls = getattr(self._models_module, "ActionModel", None)
        openenv_enum = getattr(self._models_module, "OpenEnvActionType", None)
        if action_model_cls is not None and openenv_enum is not None:
            try:
                enum_value = openenv_enum(action_type)
                candidates.append(action_model_cls(action_type=enum_value, parameters=parameters))
            except Exception:
                pass

        # Try AgentAction if available.
        legacy_agent_action_cls = getattr(self._models_module, "AgentAction", None)
        legacy_enum = getattr(self._models_module, "ActionType", None)
        if legacy_agent_action_cls is not None and legacy_enum is not None:
            try:
                mapped = LEGACY_ACTION_MAP.get(action_type, "noop")
                enum_value = legacy_enum(mapped)
                candidates.append(legacy_agent_action_cls(action_type=enum_value, payload=parameters))
            except Exception:
                pass

        return candidates

    def reset_task(self, task_id_candidates: list[str], seed: int) -> tuple[dict[str, Any], str]:
        reset_request_cls = getattr(self._models_module, "ResetRequest", None)

        errors: list[str] = []
        for task_id in task_id_candidates:
            attempts: list[tuple[str, Any]] = [
                ("kwargs", {"task_id": task_id, "seed": seed}),
                ("dict", {"task_id": task_id, "seed": seed}),
            ]

            if reset_request_cls is not None:
                try:
                    attempts.append(("request", reset_request_cls(task_id=task_id, seed=seed)))
                except Exception:
                    pass

            attempts.append(("legacy_task_only", {"task_id": task_id}))

            for mode, payload in attempts:
                try:
                    if mode == "kwargs":
                        raw = self._env.reset(**payload)
                    else:
                        raw = self._env.reset(payload)
                    return _normalize_observation(raw, task_id=task_id, step_number=0), task_id
                except Exception as exc:
                    errors.append(f"task={task_id} mode={mode} error={exc}")

        raise RuntimeError("Unable to reset environment for task candidates: " + " | ".join(errors[-5:]))

    def step(self, task_id: str, action: dict[str, Any], step_index: int) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        errors: list[str] = []

        for candidate in self._build_action_candidates(action):
            try:
                result = self._env.step(candidate)
                return self._parse_step_result(task_id=task_id, step_index=step_index, result=result)
            except Exception as exc:
                errors.append(str(exc))

        raise RuntimeError("All action payload formats failed in env.step(): " + " | ".join(errors[-5:]))

    def _parse_step_result(
        self,
        *,
        task_id: str,
        step_index: int,
        result: Any,
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        if isinstance(result, tuple) and len(result) >= 4:
            raw_observation, reward_obj, done, info = result[0], result[1], result[2], result[3]
        else:
            result_dict = _to_dict(result)
            raw_observation = result_dict.get("observation", result)
            reward_obj = result_dict.get("reward", 0.0)
            done = bool(result_dict.get("done", False))
            info = result_dict.get("info", {})

        reward_value = self._extract_reward_value(reward_obj)
        observation = _normalize_observation(raw_observation, task_id=task_id, step_number=step_index)
        info_dict = _to_dict(info)
        return observation, reward_value, bool(done), info_dict

    @staticmethod
    def _extract_reward_value(reward_obj: Any) -> float:
        if isinstance(reward_obj, (int, float)):
            return float(reward_obj)

        reward_payload = _to_dict(reward_obj)
        if "total" in reward_payload:
            return _safe_float(reward_payload["total"], 0.0)
        if "reward" in reward_payload:
            return _safe_float(reward_payload["reward"], 0.0)

        return 0.0


def _call_llm_with_retry(
    *,
    client: OpenAI,
    model_name: str,
    messages: list[dict[str, str]],
    rng: random.Random,
) -> str:
    last_error: Exception | None = None

    for attempt in range(1, MAX_API_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.1,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            return response.choices[0].message.content or "{}"
        except (APITimeoutError, APIConnectionError, RateLimitError, APIError) as exc:
            last_error = exc
            if attempt >= MAX_API_RETRIES:
                break
            sleep_seconds = min(8.0, (2 ** (attempt - 1)) + rng.uniform(0.1, 0.7))
            print(f"[retry] OpenAI call failed (attempt {attempt}/{MAX_API_RETRIES}): {exc}. Sleeping {sleep_seconds:.1f}s.")
            time.sleep(sleep_seconds)

    raise RuntimeError(f"OpenAI call failed after {MAX_API_RETRIES} attempts: {last_error}")


def _build_user_message(task_id: str, step_number: int, observation: dict[str, Any], steps_remaining: int) -> str:
    trimmed_observation = {
        "task_id": task_id,
        "step_number": step_number,
        "steps_remaining": steps_remaining,
        "task_description": observation.get("task_description", ""),
        "current_sql_query": observation.get("current_sql_query", ""),
        "error_messages": observation.get("error_messages", []),
        "dataset_preview_count": observation.get("dataset_preview_count", 0),
        "dataset_preview": observation.get("dataset_preview", []),
    }
    return json.dumps(trimmed_observation, ensure_ascii=True)


def _extract_final_score(observation: dict[str, Any], last_reward: float, info: dict[str, Any]) -> float:
    candidates: list[float] = []

    for key in ("final_score", "score", "task_score", "episode_score"):
        if key in info:
            candidates.append(_safe_float(info.get(key), 0.0))

    if 0.0 <= last_reward <= 1.0:
        candidates.append(last_reward)

    for key in ("reward", "cumulative_reward"):
        if key in observation:
            value = _safe_float(observation.get(key), -1.0)
            if 0.0 <= value <= 1.0:
                candidates.append(value)

    if not candidates:
        return _clamp_score(last_reward)

    return _clamp_score(max(candidates))


def _print_summary(results: list[TaskRunResult], total_elapsed: float) -> None:
    print("\n=== MedDataOps Inference Summary ===")
    print(f"Total elapsed: {total_elapsed:.1f}s")

    headers = ["task", "steps", "final_score", "status", "elapsed_s"]
    rows = [
        [
            result.task_name,
            str(result.steps_taken),
            f"{result.final_score:.4f}",
            result.status,
            f"{result.elapsed_sec:.1f}",
        ]
        for result in results
    ]

    all_rows = [headers] + rows
    widths = [max(len(row[col]) for row in all_rows) for col in range(len(headers))]

    def fmt_row(row: list[str]) -> str:
        return " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row))

    print(fmt_row(headers))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(fmt_row(row))


def _resolve_task_candidates(task_entry: dict[str, Any]) -> list[str]:
    candidates = [str(task_entry["id"])]
    for alias in task_entry.get("aliases", []):
        alias_str = str(alias)
        if alias_str not in candidates:
            candidates.append(alias_str)
    return candidates


def main() -> None:
    start_time = time.monotonic()
    deadline = start_time + TOTAL_RUNTIME_BUDGET_SECONDS

    api_base_url = _require_env("API_BASE_URL")
    model_name = _require_env("MODEL_NAME")
    hf_token = _require_env("HF_TOKEN")

    rng = random.Random(GLOBAL_RANDOM_SEED)

    client = OpenAI(
        base_url=api_base_url,
        api_key=hf_token,
        max_retries=0,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    env = EnvBridge(seed=GLOBAL_RANDOM_SEED)
    results: list[TaskRunResult] = []

    for task_entry in TASK_RUN_ORDER:
        if time.monotonic() >= deadline:
            print("[stop] Global runtime budget reached before starting next task.")
            break

        task_candidates = _resolve_task_candidates(task_entry)
        task_seed = _safe_int(task_entry.get("seed"), 0)
        task_start = time.monotonic()

        print(f"\n=== Running task {task_candidates[0]} (seed={task_seed}) ===")

        try:
            observation, resolved_task_id = env.reset_task(task_candidates, seed=task_seed)
        except Exception as exc:
            print(f"[error] reset failed for task candidates {task_candidates}: {exc}")
            results.append(
                TaskRunResult(
                    task_id=task_candidates[0],
                    task_name=task_candidates[0],
                    steps_taken=0,
                    final_score=0.0,
                    status="reset_failed",
                    elapsed_sec=time.monotonic() - task_start,
                )
            )
            continue

        done = False
        last_reward = 0.0
        last_info: dict[str, Any] = {}
        status = "ok"
        steps_taken = 0

        for step_idx in range(1, MAX_STEPS_PER_TASK + 1):
            if time.monotonic() >= deadline:
                status = "global_timeout"
                break

            steps_remaining = MAX_STEPS_PER_TASK - step_idx
            user_message = _build_user_message(
                task_id=resolved_task_id,
                step_number=step_idx,
                observation=observation,
                steps_remaining=steps_remaining,
            )

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ]

            try:
                llm_content = _call_llm_with_retry(
                    client=client,
                    model_name=model_name,
                    messages=messages,
                    rng=rng,
                )
            except Exception as exc:
                print(f"[error] LLM call failed on task {resolved_task_id}, step {step_idx}: {exc}")
                status = "llm_failed"
                break

            action = _parse_action(llm_content)

            try:
                observation, last_reward, done, last_info = env.step(
                    task_id=resolved_task_id,
                    action=action,
                    step_index=step_idx,
                )
            except Exception as exc:
                print(f"[error] env.step failed on task {resolved_task_id}, step {step_idx}: {exc}")
                status = "step_failed"
                break

            steps_taken = step_idx
            print(
                f"[{resolved_task_id}] step={step_idx:02d} action={action['action_type']} "
                f"reward={last_reward:.4f} done={done}"
            )

            if done:
                break

        if not done and status == "ok":
            # Force a final submit attempt if the loop ended by step budget.
            submit_action = {"action_type": "submit", "parameters": {}}
            try:
                observation, last_reward, done, last_info = env.step(
                    task_id=resolved_task_id,
                    action=submit_action,
                    step_index=steps_taken + 1,
                )
                steps_taken += 1
                status = "forced_submit" if done else "max_steps"
            except Exception:
                status = "max_steps"

        final_score = _extract_final_score(observation=observation, last_reward=last_reward, info=last_info)

        task_name = str(observation.get("task_id", resolved_task_id))
        elapsed = time.monotonic() - task_start
        results.append(
            TaskRunResult(
                task_id=resolved_task_id,
                task_name=task_name,
                steps_taken=steps_taken,
                final_score=final_score,
                status=status,
                elapsed_sec=elapsed,
            )
        )

    total_elapsed = time.monotonic() - start_time
    _print_summary(results=results, total_elapsed=total_elapsed)


if __name__ == "__main__":
    main()
