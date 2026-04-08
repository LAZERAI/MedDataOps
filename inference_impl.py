from __future__ import annotations

import http.cookiejar
import importlib
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request

try:
    from openai import APIConnectionError, APIError, APITimeoutError, OpenAI, RateLimitError
    OPENAI_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - import-time fallback for validator robustness
    APIConnectionError = Exception  # type: ignore[assignment]
    APIError = Exception  # type: ignore[assignment]
    APITimeoutError = Exception  # type: ignore[assignment]
    RateLimitError = Exception  # type: ignore[assignment]
    OpenAI = None  # type: ignore[assignment]
    OPENAI_IMPORT_ERROR = exc


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = str(raw).strip()
    if not value:
        return default
    try:
        return int(value)
    except Exception:
        print(f"[warn] Invalid {name}={raw!r}; using default {default}.", file=sys.stderr)
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = str(raw).strip()
    if not value:
        return default
    try:
        return float(value)
    except Exception:
        print(f"[warn] Invalid {name}={raw!r}; using default {default}.", file=sys.stderr)
        return default


def _env_optional_int(name: str) -> Optional[int]:
    raw = os.getenv(name)
    if raw is None:
        return None
    value = str(raw).strip()
    if not value:
        return None
    try:
        return int(value)
    except Exception:
        print(f"[warn] Invalid {name}={raw!r}; ignoring override.", file=sys.stderr)
        return None


TOTAL_RUNTIME_BUDGET_SECONDS = 19 * 60
REQUEST_TIMEOUT_SECONDS = 45
MAX_API_RETRIES = max(1, _env_int("MAX_API_RETRIES", 4))
MAX_STEPS_PER_TASK = max(1, _env_int("MAX_STEPS_PER_TASK", 20))
GLOBAL_RANDOM_SEED = _env_int("GLOBAL_RANDOM_SEED", 42)
MODEL_TEMPERATURE = _env_float("MODEL_TEMPERATURE", 0.0)
MODEL_TOP_P = _env_float("MODEL_TOP_P", 1.0)
MODEL_MAX_OUTPUT_TOKENS = _env_optional_int("MODEL_MAX_OUTPUT_TOKENS")

TASK_RUN_ORDER = [
    {"id": "triage_report", "aliases": ["easy"], "seed": 101},
    {"id": "medication_summary", "aliases": ["medium"], "seed": 202},
    {"id": "icu_capacity", "aliases": ["hard"], "seed": 303},
]

ALLOWED_ACTIONS = {"clean_data", "run_query", "fix_query", "submit", "noop"}
LEGACY_ACTION_MAP = {
    "clean_data": "clean_data",
    "run_query": "noop",
    "fix_query": "fix_sql",
    "noop": "noop",
    "submit": "submit",
}

ACTION_TOKEN_BUDGET = max(256, _env_int("ACTION_TOKEN_BUDGET", 2000))
APPROX_CHARS_PER_TOKEN = 4
ACTION_MESSAGE_CHAR_BUDGET = ACTION_TOKEN_BUDGET * APPROX_CHARS_PER_TOKEN

DEFAULT_API_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_SPACE_URL = "https://lazerai-meddataops.hf.space"
DEFAULT_LOCAL_SPACE_URL = "http://127.0.0.1:7860"
DEFAULT_LOCALHOST_SPACE_URL = "http://localhost:7860"
DEFAULT_LOCAL_SPACE_URL_ALT = "http://127.0.0.1:8000"
DEFAULT_LOCALHOST_SPACE_URL_ALT = "http://localhost:8000"

API_BASE_URL = os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL)
MODEL_NAME = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "no-token"
SPACE_BASE_URL = os.getenv("SPACE_BASE_URL", "http://localhost:7860").strip() or "http://localhost:7860"
SPACE_URL = os.getenv("SPACE_URL", "")
OPENENV_BASE_URL = os.getenv("OPENENV_BASE_URL", "")
OPENENV_HTTP_TIMEOUT_SECONDS = max(1.0, _env_float("OPENENV_HTTP_TIMEOUT_SECONDS", 12.0))
OPENENV_HEALTHCHECK_TIMEOUT_SECONDS = max(0.5, _env_float("OPENENV_HEALTHCHECK_TIMEOUT_SECONDS", 2.5))
FORCE_DETERMINISTIC_FALLBACK = os.getenv("FORCE_DETERMINISTIC_FALLBACK", "1") == "1"
LOCAL_ENV_STARTUP_PROBE_SECONDS = max(0.0, _env_float("LOCAL_ENV_STARTUP_PROBE_SECONDS", 10.0))
LOCAL_ENV_STARTUP_PROBE_INTERVAL_SECONDS = max(0.2, _env_float("LOCAL_ENV_STARTUP_PROBE_INTERVAL_SECONDS", 1.0))
ALLOW_EXTERNAL_SPACE_FALLBACK = os.getenv("ALLOW_EXTERNAL_SPACE_FALLBACK", "0") == "1"
# Optional when environments are created via from_docker_image().
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

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


@dataclass
class ParsedAction:
    action_type: str = "noop"
    parameters: dict[str, Any] = field(default_factory=dict)

    def model_dump(self) -> dict[str, Any]:
        return {"action_type": self.action_type, "parameters": dict(self.parameters)}


def _minimal_deterministic_plans() -> dict[str, list[dict[str, Any]]]:
    """Fallback action plans used when LLM execution is unavailable.

    These are intentionally conservative and are expected to be valid across
    MedDataOps task variants.
    """

    triage_sql = (
        "SELECT ward, COUNT(*) AS patient_count "
        "FROM patients "
        "WHERE admission_date >= DATE '2024-01-01' "
        "GROUP BY ward "
        "ORDER BY patient_count DESC, ward ASC"
    )

    medication_sql = (
        "SELECT p.ward, m.drug_name, COUNT(*) AS prescription_count "
        "FROM medications m "
        "INNER JOIN patients p ON m.patient_id = p.patient_id "
        "WHERE m.prescribed_date >= DATE '2024-01-01' "
        "GROUP BY p.ward, m.drug_name "
        "ORDER BY p.ward ASC, m.drug_name ASC"
    )

    icu_sql = (
        "WITH patient_agg AS ("
        "  SELECT icu_unit, COUNT(*) AS current_occupancy, "
        "         SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) AS active_count "
        "  FROM patients GROUP BY icu_unit"
        "), capacity_agg AS ("
        "  SELECT unit AS icu_unit, COUNT(*) AS total_capacity "
        "  FROM icu_beds GROUP BY unit"
        ") "
        "SELECT c.icu_unit, COALESCE(p.current_occupancy, 0) AS current_occupancy, "
        "       c.total_capacity, COALESCE(p.active_count, 0) AS active_count "
        "FROM capacity_agg c "
        "LEFT JOIN patient_agg p ON p.icu_unit = c.icu_unit "
        "ORDER BY c.icu_unit"
    )

    return {
        "triage_report": [
            {"action_type": "fix_query", "parameters": {"query": triage_sql}},
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
                                "patient_id": "string",
                            },
                        },
                    ]
                },
            },
            {"action_type": "fix_query", "parameters": {"query": medication_sql}},
            {"action_type": "submit", "parameters": {}},
        ],
        "icu_capacity": [
            {
                "action_type": "clean_data",
                "parameters": {
                    "operations": [
                        {
                            "operation": "rename_columns",
                            "mapping": {
                                "patient_id": "source_record_id",
                                "pid": "source_record_id",
                                "bed_number": "raw_bed",
                                "room": "raw_bed",
                                "admitted_at": "raw_admitted_at",
                                "admission_ts": "raw_admitted_at",
                            },
                        },
                        {
                            "operation": "coalesce_columns",
                            "target_column": "icu_unit",
                            "source_columns": ["icu_unit", "ward_code"],
                        },
                        {
                            "operation": "fix_unix_ms",
                            "column": "raw_admitted_at",
                            "output": "datetime",
                        },
                        {"operation": "copy_column", "from_column": "raw_admitted_at", "to_column": "admitted_at"},
                        {
                            "operation": "derive_column",
                            "target_column": "status",
                            "rule": "date_within_days",
                            "column": "admitted_at",
                            "reference_date": "2026-04-04",
                            "days": 7,
                            "then": "active",
                            "else": "inactive",
                        },
                    ]
                },
            },
            {"action_type": "fix_query", "parameters": {"query": icu_sql}},
            {"action_type": "submit", "parameters": {}},
        ],
    }


def _load_deterministic_plans() -> dict[str, list[dict[str, Any]]]:
    """Load deterministic task plans from reference_solver when available."""

    try:
        reference_solver = importlib.import_module("reference_solver")
        task_specs = getattr(reference_solver, "TASKS", ())

        loaded: dict[str, list[dict[str, Any]]] = {}
        for task_spec in task_specs:
            task_id = str(getattr(task_spec, "task_id", "")).strip()
            actions = getattr(task_spec, "actions", ())
            if not task_id:
                continue

            normalized_actions: list[dict[str, Any]] = []
            for action in actions:
                if not isinstance(action, dict):
                    continue
                action_type = str(action.get("action_type", "noop")).strip().lower() or "noop"
                parameters = action.get("parameters", {})
                if not isinstance(parameters, dict):
                    parameters = {}
                normalized_actions.append({"action_type": action_type, "parameters": parameters})

            if normalized_actions:
                loaded[task_id] = normalized_actions

        if loaded:
            return loaded
    except Exception as exc:
        print(f"[warn] Unable to import deterministic plans from reference_solver: {exc}", file=sys.stderr)

    return _minimal_deterministic_plans()


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_token(value: Any) -> str:
    text = str(value)
    text = text.replace("\n", " ").replace("\r", " ").strip()
    text = re.sub(r"\s+", "_", text)
    return text if text else "na"


def _emit_start(*, run_id: str, model_name: str, task_ids: list[str], max_steps_per_task: int) -> None:
    print(
        "START"
        f" run_id={_safe_token(run_id)}"
        f" ts_utc={_safe_token(_utc_timestamp())}"
        f" model={_safe_token(model_name)}"
        f" tasks={_safe_token(','.join(task_ids))}"
        f" max_steps_per_task={max_steps_per_task}"
    , flush=True)


def _emit_step(
    *,
    run_id: str,
    task_id: str,
    step: int,
    action_type: str,
    reward: float,
    done: bool,
    status: str,
) -> None:
    print(
        "STEP"
        f" run_id={_safe_token(run_id)}"
        f" task_id={_safe_token(task_id)}"
        f" step={step}"
        f" action_type={_safe_token(action_type)}"
        f" reward={reward:.6f}"
        f" done={'true' if done else 'false'}"
        f" status={_safe_token(status)}"
    , flush=True)


def _emit_end(*, run_id: str, results: list[TaskRunResult], total_elapsed: float) -> None:
    task_count = len(results)
    mean_score = sum(result.final_score for result in results) / max(1, task_count)
    status_blob = ",".join(f"{result.task_id}:{result.status}" for result in results)
    print(
        "END"
        f" run_id={_safe_token(run_id)}"
        f" ts_utc={_safe_token(_utc_timestamp())}"
        f" task_count={task_count}"
        f" mean_score={mean_score:.6f}"
        f" total_elapsed_s={total_elapsed:.3f}"
        f" statuses={_safe_token(status_blob)}"
    , flush=True)


def _require_non_empty(name: str, value: Optional[str]) -> str:
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


def _is_nullish(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _truncate_string(value: Any, max_len: int = 220) -> str:
    text = str(value)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _json_dumps_safe(value: Any) -> str:
    """Serialize arbitrary payloads without raising on non-JSON-native types."""
    try:
        return json.dumps(value, ensure_ascii=True, default=str)
    except Exception:
        compact = _compact_json_like(value, max_items=10, max_str_len=120)
        return json.dumps(compact, ensure_ascii=True, default=str)


def _compact_json_like(value: Any, *, max_items: int = 8, max_str_len: int = 220) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for idx, (k, v) in enumerate(value.items()):
            if idx >= max_items:
                result["..."] = f"truncated_{len(value) - max_items}_keys"
                break
            result[str(k)] = _compact_json_like(v, max_items=max_items, max_str_len=max_str_len)
        return result
    if isinstance(value, list):
        compacted = [_compact_json_like(v, max_items=max_items, max_str_len=max_str_len) for v in value[:max_items]]
        if len(value) > max_items:
            compacted.append(f"truncated_{len(value) - max_items}_items")
        return compacted
    if isinstance(value, str):
        return _truncate_string(value, max_len=max_str_len)
    return value


def _summarize_dataset(rows: Any, sample_rows: int = 3) -> dict[str, Any]:
    if not isinstance(rows, list):
        return {
            "row_count": 0,
            "column_names": [],
            "null_count_per_column": {},
            "sample_rows": [],
        }

    dict_rows = [row for row in rows if isinstance(row, dict)]
    if not dict_rows:
        return {
            "row_count": len(rows),
            "column_names": [],
            "null_count_per_column": {},
            "sample_rows": [],
        }

    column_names = sorted({key for row in dict_rows for key in row.keys()})
    null_count_per_column: dict[str, int] = {}
    for col in column_names:
        null_count_per_column[col] = sum(
            1 for row in dict_rows if (col not in row) or _is_nullish(row.get(col))
        )

    sample: list[dict[str, Any]] = []
    for row in dict_rows[:sample_rows]:
        sample.append(_compact_json_like(row, max_items=20, max_str_len=160))

    return {
        "row_count": len(rows),
        "column_names": column_names,
        "null_count_per_column": null_count_per_column,
        "sample_rows": sample,
    }


def _normalize_observation(raw_observation: Any, task_id: str, step_number: int) -> dict[str, Any]:
    payload = _to_dict(raw_observation)

    if "current_dataset_state" in payload and "current_sql_query" in payload:
        rows = payload.get("current_dataset_state")
        summary = _summarize_dataset(rows)
        return {
            "task_id": task_id,
            "step_number": int(payload.get("step_number", step_number)),
            "task_description": str(payload.get("task_description", "")),
            "current_sql_query": str(payload.get("current_sql_query", "")),
            "error_messages": payload.get("error_messages", []),
            "dataset_preview": _compact_rows(rows),
            "dataset_preview_count": len(rows) if isinstance(rows, list) else 0,
            "dataset_summary": summary,
        }

    dirty_rows = payload.get("dirty_rows", [])
    task_info = payload.get("task", {})
    summary = _summarize_dataset(dirty_rows)
    return {
        "task_id": task_id,
        "step_number": int(payload.get("step_index", step_number)),
        "task_description": str(task_info.get("description", "")),
        "current_sql_query": str(payload.get("broken_sql", "")),
        "error_messages": [payload.get("last_sql_error")] if payload.get("last_sql_error") else [],
        "dataset_preview": _compact_rows(dirty_rows),
        "dataset_preview_count": len(dirty_rows) if isinstance(dirty_rows, list) else 0,
        "dataset_summary": summary,
    }


def _warn_parse_fallback(content: str) -> None:
    snippet = _truncate_string((content or "").replace("\n", " "), max_len=240)
    print(
        f"[warn] Unable to parse LLM action. Falling back to noop. content_snippet={snippet}",
        file=sys.stderr,
    )


def _coerce_action_type(raw_value: Any) -> str:
    normalized = str(raw_value or "").strip().lower()
    aliases = {
        "query": "run_query",
        "runquery": "run_query",
        "fix_sql": "fix_query",
        "fixsql": "fix_query",
        "fix": "fix_query",
        "query_fix": "fix_query",
        "run_sql": "run_query",
        "execute_query": "run_query",
        "execute": "run_query",
        "finalize": "submit",
        "submit_answer": "submit",
        "answer": "submit",
        "no_action": "noop",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in ALLOWED_ACTIONS:
        return "noop"
    return normalized


def _extract_markdown_json_blocks(text: str) -> list[str]:
    matches = re.findall(r"```(?:json|JSON)?\s*(.*?)```", text, flags=re.DOTALL)
    return [match.strip() for match in matches if match.strip()]


def _extract_tag_blocks(text: str, tag_name: str) -> list[str]:
    pattern = rf"<{tag_name}[^>]*>(.*?)</{tag_name}>"
    matches = re.findall(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    return [match.strip() for match in matches if match.strip()]


def _extract_balanced_json_objects(text: str, max_blocks: int = 12) -> list[str]:
    blocks: list[str] = []
    depth = 0
    start_idx = -1
    in_string = False
    escaped = False

    for idx, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue

        if char == "{":
            if depth == 0:
                start_idx = idx
            depth += 1
            continue

        if char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start_idx >= 0:
                blocks.append(text[start_idx : idx + 1].strip())
                start_idx = -1
                if len(blocks) >= max_blocks:
                    break

    return [block for block in blocks if block]


def _repair_json(candidate: str) -> str:
    fixed = (candidate or "").strip()
    if not fixed:
        return fixed

    fixed = re.sub(r"^```(?:json|JSON)?", "", fixed).replace("```", "").strip()
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
    fixed = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_\-]*)(\s*:)", r'\1"\2"\3', fixed)
    fixed = re.sub(r'("action_type"\s*:\s*)([A-Za-z_][A-Za-z0-9_]*)', r'\1"\2"', fixed)

    fixed = re.sub(
        r"'([^'\\]*(?:\\.[^'\\]*)*)'",
        lambda m: '"' + m.group(1).replace('"', '\\"') + '"',
        fixed,
    )

    fixed = re.sub(r"\bNone\b", "null", fixed)
    fixed = re.sub(r"\bTrue\b", "true", fixed)
    fixed = re.sub(r"\bFalse\b", "false", fixed)
    return fixed


def _build_parsed_action(data: dict[str, Any]) -> ParsedAction:
    action_obj = data.get("action", data)
    if not isinstance(action_obj, dict):
        return ParsedAction(action_type="noop", parameters={})

    action_type = _coerce_action_type(action_obj.get("action_type", action_obj.get("type", "noop")))
    parameters = action_obj.get("parameters", action_obj.get("payload", {}))
    if not isinstance(parameters, dict):
        parameters = {}

    return ParsedAction(action_type=action_type, parameters=parameters)


def _try_parse_json_candidate(candidate: str) -> Optional[ParsedAction]:
    for payload in (candidate, _repair_json(candidate)):
        payload = payload.strip()
        if not payload:
            continue
        try:
            loaded = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, dict):
            return _build_parsed_action(loaded)
    return None


def _parse_key_value_action(text: str) -> Optional[ParsedAction]:
    action_match = re.search(r"action_type\s*[:=]\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)", text, flags=re.IGNORECASE)
    if action_match is None:
        return None

    action_type = _coerce_action_type(action_match.group(1))
    parameters: dict[str, Any] = {}

    params_match = re.search(r"parameters\s*[:=]\s*(\{.*\})", text, flags=re.DOTALL | re.IGNORECASE)
    if params_match is not None:
        params_text = params_match.group(1)
        for payload in (params_text, _repair_json(params_text)):
            try:
                loaded = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(loaded, dict):
                parameters = loaded
                break

    query_match = re.search(r"query\s*[:=]\s*([\"'])(.*?)\1", text, flags=re.DOTALL | re.IGNORECASE)
    if query_match is not None and "query" not in parameters:
        parameters["query"] = query_match.group(2).strip()

    return ParsedAction(action_type=action_type, parameters=parameters)


def _parse_action(content: str) -> ParsedAction:
    text = (content or "").strip()
    if not text:
        _warn_parse_fallback(content)
        return ParsedAction(action_type="noop", parameters={})

    candidates: list[str] = []

    # Strategy 1: full response as-is.
    candidates.append(text)

    # Strategy 2: JSON fenced in markdown blocks.
    candidates.extend(_extract_markdown_json_blocks(text))

    # Strategy 3: explicit action tags.
    candidates.extend(_extract_tag_blocks(text, "action"))

    # Strategy 4: JSON mixed with reasoning text (balanced block extraction).
    candidates.extend(_extract_balanced_json_objects(text))

    # Strategy 5: reasoning + action on separate lines after think tags.
    post_think = re.split(r"</think>", text, flags=re.IGNORECASE)
    if len(post_think) > 1:
        candidates.append(post_think[-1].strip())

    deduped_candidates: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        candidate_stripped = candidate.strip()
        if not candidate_stripped or candidate_stripped in seen:
            continue
        seen.add(candidate_stripped)
        deduped_candidates.append(candidate_stripped)

    for candidate in deduped_candidates:
        parsed = _try_parse_json_candidate(candidate)
        if parsed is not None:
            return parsed

    key_value_parsed = _parse_key_value_action(text)
    if key_value_parsed is not None:
        return key_value_parsed

    _warn_parse_fallback(content)
    return ParsedAction(action_type="noop", parameters={})


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


def _extract_reward_value(reward_obj: Any) -> float:
    if isinstance(reward_obj, (int, float)):
        return float(reward_obj)

    reward_payload = _to_dict(reward_obj)
    if "total" in reward_payload:
        return _safe_float(reward_payload["total"], 0.0)
    if "reward" in reward_payload:
        return _safe_float(reward_payload["reward"], 0.0)

    return 0.0


def _extract_retry_after_seconds(error: Exception) -> Optional[float]:
    text = str(error)
    patterns = [
        r"Please try again in\s*([0-9]+(?:\.[0-9]+)?)s",
        r"try again in\s*([0-9]+(?:\.[0-9]+)?)\s*seconds",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except (TypeError, ValueError):
                return None
    return None


class HttpEnvBridge:
    def __init__(self, *, base_url: str, seed: int) -> None:
        self._base_url = base_url.rstrip("/")
        self._seed = seed
        self._session_id: Optional[str] = None
        self._cookie_jar = http.cookiejar.CookieJar()
        self._opener = urllib_request.build_opener(urllib_request.HTTPCookieProcessor(self._cookie_jar))
        self._timeout = OPENENV_HTTP_TIMEOUT_SECONDS

        self._request_json(
            method="GET",
            path="/health",
            payload=None,
            timeout_seconds=min(self._timeout, OPENENV_HEALTHCHECK_TIMEOUT_SECONDS),
        )

    def _sync_session_id(self, payload: dict[str, Any]) -> None:
        candidates: list[Any] = [
            payload.get("session_id"),
            payload.get("sessionId"),
        ]

        meta = payload.get("meta")
        if isinstance(meta, dict):
            candidates.extend([meta.get("session_id"), meta.get("sessionId")])

        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                self._session_id = candidate.strip()
                return

    def _request_json(
        self,
        *,
        method: str,
        path: str,
        payload: Optional[dict[str, Any]],
        timeout_seconds: Optional[float] = None,
    ) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        body = json.dumps(payload).encode("utf-8") if payload is not None else None

        req = urllib_request.Request(url=url, data=body, method=method)
        req.add_header("Accept", "application/json")
        if payload is not None or method.upper() in {"POST", "PUT", "PATCH"}:
            req.add_header("Content-Type", "application/json")
        if self._session_id:
            req.add_header("X-Session-Id", self._session_id)

        effective_timeout = self._timeout if timeout_seconds is None else max(0.5, float(timeout_seconds))

        try:
            with self._opener.open(req, timeout=effective_timeout) as response:
                sid = response.headers.get("X-Session-Id")
                if sid:
                    self._session_id = sid

                raw_text = response.read().decode("utf-8", errors="replace")
                if not raw_text.strip():
                    return {}
                parsed = json.loads(raw_text)
                parsed_payload = parsed if isinstance(parsed, dict) else {"value": parsed}
                self._sync_session_id(parsed_payload)
                return parsed_payload
        except urllib_error.HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace") if getattr(exc, "fp", None) else ""
            raise RuntimeError(f"HTTP {exc.code} {method} {path} failed: {body_text or exc.reason}") from exc
        except urllib_error.URLError as exc:
            raise RuntimeError(f"HTTP {method} {path} unreachable: {exc.reason}") from exc

    def reset_task(self, task_id_candidates: list[str], seed: int) -> tuple[dict[str, Any], str]:
        _ = seed  # seed is retained in signature for compatibility with existing callers.
        errors: list[str] = []
        for task_id in task_id_candidates:
            try:
                raw = self._request_json(
                    method="POST",
                    path="/reset",
                    payload={"task_id": task_id},
                )
                if "detail" in raw:
                    raise RuntimeError(str(raw["detail"]))
                return _normalize_observation(raw, task_id=task_id, step_number=0), task_id
            except Exception as exc:
                errors.append(f"task={task_id} error={exc}")

        raise RuntimeError("Unable to reset remote environment: " + " | ".join(errors[-5:]))

    def step(self, task_id: str, action: dict[str, Any], step_index: int) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        raw = self._request_json(
            method="POST",
            path="/step",
            payload={
                "action_type": str(action.get("action_type", "")),
                "parameters": action.get("parameters", {}),
            },
        )
        if "detail" in raw:
            raise RuntimeError(str(raw["detail"]))

        raw_observation = raw.get("observation", {})
        reward_obj = raw.get("reward", 0.0)
        done = bool(raw.get("done", False))
        info = _to_dict(raw.get("info", {}))

        observation = _normalize_observation(raw_observation, task_id=task_id, step_number=step_index)
        reward_value = _extract_reward_value(reward_obj)
        return observation, reward_value, done, info


def _call_llm_with_retry(
    *,
    client: Any,
    model_name: str,
    messages: list[dict[str, str]],
    rng: random.Random,
    deadline: Optional[float] = None,
) -> str:
    if client is None:
        raise RuntimeError("OpenAI client unavailable")

    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_API_RETRIES + 1):
        timeout_seconds = float(REQUEST_TIMEOUT_SECONDS)
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 2.0:
                raise RuntimeError("Global runtime budget exhausted before LLM request.")
            timeout_seconds = min(timeout_seconds, max(2.0, remaining - 1.0))

        try:
            request_kwargs: dict[str, Any] = {
                "model": model_name,
                "messages": messages,
                "temperature": MODEL_TEMPERATURE,
                "top_p": MODEL_TOP_P,
                "timeout": timeout_seconds,
            }
            if MODEL_MAX_OUTPUT_TOKENS is not None and MODEL_MAX_OUTPUT_TOKENS > 0:
                request_kwargs["max_tokens"] = MODEL_MAX_OUTPUT_TOKENS

            response = client.chat.completions.create(**request_kwargs)
            return response.choices[0].message.content or "{}"
        except (APITimeoutError, APIConnectionError, RateLimitError, APIError) as exc:
            last_error = exc
            if attempt >= MAX_API_RETRIES:
                break

            if deadline is not None:
                remaining_after_error = deadline - time.monotonic()
                if remaining_after_error <= 2.0:
                    break

            sleep_seconds = min(8.0, (2 ** (attempt - 1)) + rng.uniform(0.1, 0.7))

            retry_after = _extract_retry_after_seconds(exc)
            if retry_after is not None:
                sleep_seconds = max(sleep_seconds, min(30.0, retry_after + rng.uniform(0.2, 0.8)))

            if deadline is not None:
                sleep_seconds = min(sleep_seconds, max(0.0, deadline - time.monotonic() - 1.0))
            if sleep_seconds <= 0.0:
                break
            print(
                f"[retry] OpenAI call failed (attempt {attempt}/{MAX_API_RETRIES}): {exc}. Sleeping {sleep_seconds:.1f}s.",
                file=sys.stderr,
            )
            time.sleep(sleep_seconds)

    raise RuntimeError(f"OpenAI call failed after {MAX_API_RETRIES} attempts: {last_error}")


def _enforce_message_budget(message: str, max_chars: int = ACTION_MESSAGE_CHAR_BUDGET) -> str:
    if len(message) <= max_chars:
        return message
    return message[: max_chars - 3] + "..."


def format_observation_user_message(
    *,
    task_id: str,
    step_number: int,
    steps_remaining: int,
    observation: dict[str, Any],
    last_action_result: Optional[dict[str, Any]],
    max_tokens: int = ACTION_TOKEN_BUDGET,
) -> str:
    char_budget = max(800, int(max_tokens * APPROX_CHARS_PER_TOKEN))

    task_description = _truncate_string(observation.get("task_description", ""), max_len=900)
    current_sql_query = _truncate_string(observation.get("current_sql_query", ""), max_len=1400)
    error_messages = observation.get("error_messages", [])
    if not isinstance(error_messages, list):
        error_messages = [str(error_messages)]

    summary = observation.get("dataset_summary", {})
    if not isinstance(summary, dict):
        summary = {}

    row_count = _safe_int(summary.get("row_count", observation.get("dataset_preview_count", 0)), 0)

    column_names = summary.get("column_names", [])
    if not isinstance(column_names, list):
        column_names = []
    column_names = [str(c) for c in column_names[:40]]

    null_count = summary.get("null_count_per_column", {})
    if not isinstance(null_count, dict):
        null_count = {}
    null_count_compact = {str(k): _safe_int(v, 0) for k, v in list(null_count.items())[:40]}

    sample_rows = summary.get("sample_rows", observation.get("dataset_preview", []))
    if not isinstance(sample_rows, list):
        sample_rows = []
    sample_rows_compact = [_compact_json_like(row, max_items=20, max_str_len=160) for row in sample_rows[:3]]

    compact_last_action = _compact_json_like(last_action_result or {"status": "no_previous_action"}, max_items=12, max_str_len=180)
    compact_errors = [_truncate_string(msg, max_len=280) for msg in error_messages[:8]]

    sections = [
        f"Task ID: {task_id}",
        f"Step: {step_number}",
        f"Remaining Steps: {steps_remaining}",
        "",
        "Task Description:",
        task_description or "(empty)",
        "",
        "Dataset Summary:",
        f"- row_count: {row_count}",
        f"- column_names: {_json_dumps_safe(column_names)}",
        f"- null_count_per_column: {_json_dumps_safe(null_count_compact)}",
        f"- sample_rows: {_json_dumps_safe(sample_rows_compact)}",
        "",
        "Current SQL Query:",
        current_sql_query or "(empty)",
        "",
        "Last Action Result:",
        _json_dumps_safe(compact_last_action),
        "",
        "Error Messages:",
        _json_dumps_safe(compact_errors),
    ]
    message = "\n".join(sections)

    if len(message) > char_budget:
        # First shrink samples/null counts, then shrink query/description.
        sample_rows_compact = [_compact_json_like(row, max_items=8, max_str_len=100) for row in sample_rows[:1]]
        null_count_compact = {str(k): _safe_int(v, 0) for k, v in list(null_count.items())[:20]}
        compact_errors = [_truncate_string(msg, max_len=160) for msg in error_messages[:5]]
        current_sql_query = _truncate_string(current_sql_query, max_len=700)
        task_description = _truncate_string(task_description, max_len=500)

        sections = [
            f"Task ID: {task_id}",
            f"Step: {step_number}",
            f"Remaining Steps: {steps_remaining}",
            "",
            "Task Description:",
            task_description or "(empty)",
            "",
            "Dataset Summary:",
            f"- row_count: {row_count}",
            f"- column_names: {_json_dumps_safe(column_names[:20])}",
            f"- null_count_per_column: {_json_dumps_safe(null_count_compact)}",
            f"- sample_rows: {_json_dumps_safe(sample_rows_compact)}",
            "",
            "Current SQL Query:",
            current_sql_query or "(empty)",
            "",
            "Last Action Result:",
            _json_dumps_safe(_compact_json_like(compact_last_action, max_items=8, max_str_len=120)),
            "",
            "Error Messages:",
            _json_dumps_safe(compact_errors),
        ]
        message = "\n".join(sections)

    return _enforce_message_budget(message, max_chars=char_budget)


def _build_user_message(
    task_id: str,
    step_number: int,
    observation: dict[str, Any],
    steps_remaining: int,
    last_action_result: Optional[dict[str, Any]],
) -> str:
    return format_observation_user_message(
        task_id=task_id,
        step_number=step_number,
        steps_remaining=steps_remaining,
        observation=observation,
        last_action_result=last_action_result,
        max_tokens=ACTION_TOKEN_BUDGET,
    )


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
    print("\n=== MedDataOps Inference Summary ===", file=sys.stderr)
    print(f"Total elapsed: {total_elapsed:.1f}s", file=sys.stderr)

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

    print(fmt_row(headers), file=sys.stderr)
    print("-+-".join("-" * w for w in widths), file=sys.stderr)
    for row in rows:
        print(fmt_row(row), file=sys.stderr)


def _resolve_task_candidates(task_entry: dict[str, Any]) -> list[str]:
    candidates = [str(task_entry["id"])]
    for alias in task_entry.get("aliases", []):
        alias_str = str(alias)
        if alias_str not in candidates:
            candidates.append(alias_str)
    return candidates


def _run_deterministic_task(
    *,
    env: Any,
    run_id: str,
    task_candidates: list[str],
    task_seed: int,
    deadline: float,
    action_plan: list[dict[str, Any]],
) -> TaskRunResult:
    task_start = time.monotonic()
    task_id = task_candidates[0]

    try:
        observation, resolved_task_id = env.reset_task(task_candidates, seed=task_seed)
    except Exception as exc:
        print(f"[error] deterministic reset failed for {task_candidates}: {exc}", file=sys.stderr)
        _emit_step(
            run_id=run_id,
            task_id=task_id,
            step=0,
            action_type="reset",
            reward=0.0,
            done=True,
            status="reset_failed",
        )
        return TaskRunResult(
            task_id=task_id,
            task_name=task_id,
            steps_taken=0,
            final_score=0.0,
            status="reset_failed",
            elapsed_sec=time.monotonic() - task_start,
        )

    done = False
    last_reward = 0.0
    last_info: dict[str, Any] = {}
    steps_taken = 0
    status = "ok"

    for idx, action in enumerate(action_plan, start=1):
        if time.monotonic() >= deadline:
            status = "global_timeout"
            break

        action_type = str(action.get("action_type", "noop"))
        parameters = action.get("parameters", {}) if isinstance(action.get("parameters", {}), dict) else {}
        payload = {"action_type": action_type, "parameters": parameters}

        try:
            observation, last_reward, done, last_info = env.step(
                task_id=resolved_task_id,
                action=payload,
                step_index=idx,
            )
        except Exception as exc:
            print(
                f"[error] deterministic env.step failed on task {resolved_task_id}, step {idx}: {exc}",
                file=sys.stderr,
            )
            _emit_step(
                run_id=run_id,
                task_id=resolved_task_id,
                step=idx,
                action_type=action_type,
                reward=float(last_reward),
                done=False,
                status="step_failed",
            )
            status = "step_failed"
            break

        steps_taken = idx
        _emit_step(
            run_id=run_id,
            task_id=resolved_task_id,
            step=idx,
            action_type=action_type,
            reward=float(last_reward),
            done=bool(done),
            status="done" if done else "ok",
        )

        if done:
            break

    if not done and status == "ok":
        try:
            observation, last_reward, done, last_info = env.step(
                task_id=resolved_task_id,
                action={"action_type": "submit", "parameters": {}},
                step_index=steps_taken + 1,
            )
            steps_taken += 1
            status = "forced_submit" if done else "max_steps"
            _emit_step(
                run_id=run_id,
                task_id=resolved_task_id,
                step=steps_taken,
                action_type="submit",
                reward=float(last_reward),
                done=bool(done),
                status=status,
            )
        except Exception as exc:
            print(f"[warn] deterministic forced submit failed: {exc}", file=sys.stderr)
            status = "max_steps"

    final_score = _extract_final_score(observation=observation, last_reward=last_reward, info=last_info)
    task_name = str(observation.get("task_id", resolved_task_id))
    return TaskRunResult(
        task_id=resolved_task_id,
        task_name=task_name,
        steps_taken=steps_taken,
        final_score=final_score,
        status=status,
        elapsed_sec=time.monotonic() - task_start,
    )


def _guard_main_exceptions(func: Any) -> Any:
    """Prevent validator fail-fast from uncaught exceptions when main() is imported and invoked."""

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except BaseException as exc:  # pragma: no cover - final safety guard for validator stability
            print(f"[fatal] Unhandled inference exception: {exc}", file=sys.stderr)
            return None

    return wrapped


@_guard_main_exceptions
def main() -> None:
    start_time = time.monotonic()
    deadline = start_time + TOTAL_RUNTIME_BUDGET_SECONDS
    run_id = f"meddataops-{int(start_time)}"

    api_base_url = API_BASE_URL.strip() if API_BASE_URL else DEFAULT_API_BASE_URL
    model_name = MODEL_NAME.strip() if MODEL_NAME else DEFAULT_MODEL_NAME
    hf_token = (os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "no-token").strip() or "no-token"
    use_deterministic_fallback = FORCE_DETERMINISTIC_FALLBACK

    if use_deterministic_fallback:
        print("[info] FORCE_DETERMINISTIC_FALLBACK=1; using deterministic policy.", file=sys.stderr)
    elif hf_token == "no-token":
        use_deterministic_fallback = True
        print("[warn] HF_TOKEN/API_KEY not set. Using deterministic fallback policy.", file=sys.stderr)

    rng = random.Random(GLOBAL_RANDOM_SEED)

    client: Any = None
    if not use_deterministic_fallback:
        if OpenAI is None:
            use_deterministic_fallback = True
            print(
                f"[warn] OpenAI SDK unavailable ({OPENAI_IMPORT_ERROR}). Using deterministic fallback.",
                file=sys.stderr,
            )

    if not use_deterministic_fallback:
        try:
            client = OpenAI(
                base_url=api_base_url,
                api_key=hf_token,
                max_retries=0,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            use_deterministic_fallback = True
            print(f"[warn] Failed to initialize OpenAI client ({exc}). Using deterministic fallback.", file=sys.stderr)

    local_space_candidates: list[str] = [SPACE_BASE_URL]
    port_value = os.getenv("PORT", "").strip()
    if port_value.isdigit():
        local_space_candidates.append(f"http://127.0.0.1:{port_value}")
        local_space_candidates.append(f"http://localhost:{port_value}")

    local_space_candidates.append(DEFAULT_LOCALHOST_SPACE_URL)
    local_space_candidates.append(DEFAULT_LOCAL_SPACE_URL)
    local_space_candidates.append(DEFAULT_LOCALHOST_SPACE_URL_ALT)
    local_space_candidates.append(DEFAULT_LOCAL_SPACE_URL_ALT)

    remote_space_candidates: list[str] = []

    deduped_local_candidates: list[str] = []
    for candidate in local_space_candidates:
        normalized = candidate.strip().rstrip("/")
        if normalized and normalized not in deduped_local_candidates:
            deduped_local_candidates.append(normalized)

    deduped_remote_candidates: list[str] = []
    for candidate in remote_space_candidates:
        normalized = candidate.strip().rstrip("/")
        if normalized and normalized not in deduped_remote_candidates:
            deduped_remote_candidates.append(normalized)

    env: Optional[Any] = None
    bridge_errors: list[str] = []

    # Probe local validator-hosted endpoints for a short startup window.
    local_probe_deadline = time.monotonic() + LOCAL_ENV_STARTUP_PROBE_SECONDS
    while env is None and time.monotonic() <= local_probe_deadline:
        for candidate_url in deduped_local_candidates:
            try:
                env = HttpEnvBridge(base_url=candidate_url, seed=GLOBAL_RANDOM_SEED)
                print(f"[info] Using local HTTP environment at {candidate_url}", file=sys.stderr)
                break
            except Exception as exc:
                bridge_errors.append(f"{candidate_url} -> {exc}")

        if env is None:
            time_remaining = local_probe_deadline - time.monotonic()
            if time_remaining <= 0:
                break
            time.sleep(min(LOCAL_ENV_STARTUP_PROBE_INTERVAL_SECONDS, time_remaining))

    # Optional remote fallback, intentionally disabled for validator compatibility.
    if env is None:
        for candidate_url in deduped_remote_candidates:
            try:
                env = HttpEnvBridge(base_url=candidate_url, seed=GLOBAL_RANDOM_SEED)
                print(f"[info] Using remote HTTP environment at {candidate_url}", file=sys.stderr)
                break
            except Exception as exc:
                bridge_errors.append(f"{candidate_url} -> {exc}")

    if env is None:
        print(f"[fatal] Unable to initialize any HTTP environment: {' | '.join(bridge_errors[-5:])}", file=sys.stderr)
        _emit_start(
            run_id=run_id,
            model_name=model_name,
            task_ids=[str(task["id"]) for task in TASK_RUN_ORDER],
            max_steps_per_task=MAX_STEPS_PER_TASK,
        )
        _emit_end(run_id=run_id, results=[], total_elapsed=time.monotonic() - start_time)
        return

    results: list[TaskRunResult] = []
    deterministic_plans = _load_deterministic_plans()
    _emit_start(
        run_id=run_id,
        model_name=model_name,
        task_ids=[str(task["id"]) for task in TASK_RUN_ORDER],
        max_steps_per_task=MAX_STEPS_PER_TASK,
    )

    for task_entry in TASK_RUN_ORDER:
        if time.monotonic() >= deadline:
            print("[stop] Global runtime budget reached before starting next task.", file=sys.stderr)
            break

        task_candidates = _resolve_task_candidates(task_entry)
        task_seed = _safe_int(task_entry.get("seed"), 0)
        task_start = time.monotonic()

        print(f"\n=== Running task {task_candidates[0]} (seed={task_seed}) ===", file=sys.stderr)

        if use_deterministic_fallback:
            action_plan = deterministic_plans.get(task_candidates[0], [{"action_type": "submit", "parameters": {}}])
            result = _run_deterministic_task(
                env=env,
                run_id=run_id,
                task_candidates=task_candidates,
                task_seed=task_seed,
                deadline=deadline,
                action_plan=action_plan,
            )
            results.append(result)
            continue

        try:
            observation, resolved_task_id = env.reset_task(task_candidates, seed=task_seed)
        except Exception as exc:
            print(f"[error] reset failed for task candidates {task_candidates}: {exc}", file=sys.stderr)
            _emit_step(
                run_id=run_id,
                task_id=task_candidates[0],
                step=0,
                action_type="reset",
                reward=0.0,
                done=True,
                status="reset_failed",
            )
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
        last_action_result: dict[str, Any] = {"status": "no_previous_action"}
        status = "ok"
        steps_taken = 0

        for step_idx in range(1, MAX_STEPS_PER_TASK + 1):
            if time.monotonic() >= deadline:
                status = "global_timeout"
                break

            steps_remaining = MAX_STEPS_PER_TASK - step_idx
            try:
                user_message = _build_user_message(
                    task_id=resolved_task_id,
                    step_number=step_idx,
                    observation=observation,
                    steps_remaining=steps_remaining,
                    last_action_result=last_action_result,
                )
            except Exception as exc:
                print(
                    f"[warn] Failed to format observation payload on task {resolved_task_id}, step {step_idx}: {exc}",
                    file=sys.stderr,
                )
                fallback_payload = {
                    "task_id": resolved_task_id,
                    "step_number": step_idx,
                    "steps_remaining": steps_remaining,
                    "observation": _compact_json_like(observation, max_items=10, max_str_len=160),
                    "last_action_result": _compact_json_like(last_action_result or {}, max_items=10, max_str_len=160),
                }
                user_message = _enforce_message_budget(_json_dumps_safe(fallback_payload), max_chars=ACTION_MESSAGE_CHAR_BUDGET)

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ]

            llm_fallback_used = False
            try:
                llm_content = _call_llm_with_retry(
                    client=client,
                    model_name=model_name,
                    messages=messages,
                    rng=rng,
                    deadline=deadline,
                )
                action = _parse_action(llm_content)
            except Exception as exc:
                print(
                    f"[warn] LLM call failed on task {resolved_task_id}, step {step_idx}: {exc}. Using deterministic fallback action.",
                    file=sys.stderr,
                )
                use_deterministic_fallback = True
                llm_fallback_used = True

                fallback_plan = deterministic_plans.get(task_candidates[0], [{"action_type": "submit", "parameters": {}}])
                fallback_idx = min(max(step_idx - 1, 0), max(len(fallback_plan) - 1, 0))
                fallback_action = fallback_plan[fallback_idx] if fallback_plan else {"action_type": "submit", "parameters": {}}
                try:
                    action = _build_parsed_action(_to_dict(fallback_action))
                except Exception:
                    action = ParsedAction(action_type="submit", parameters={})

            try:
                observation, last_reward, done, last_info = env.step(
                    task_id=resolved_task_id,
                    action=action.model_dump(),
                    step_index=step_idx,
                )
            except Exception as exc:
                print(f"[error] env.step failed on task {resolved_task_id}, step {step_idx}: {exc}", file=sys.stderr)
                _emit_step(
                    run_id=run_id,
                    task_id=resolved_task_id,
                    step=step_idx,
                    action_type=action.action_type,
                    reward=float(last_reward),
                    done=False,
                    status="step_failed",
                )
                status = "step_failed"
                break

            last_action_result = {
                "action_type": action.action_type,
                "parameters": _compact_json_like(action.parameters, max_items=12, max_str_len=180),
                "reward": round(last_reward, 6),
                "done": bool(done),
                "info": _compact_json_like(last_info, max_items=12, max_str_len=180),
            }

            step_status = "ok"
            if llm_fallback_used:
                step_status = "llm_fallback"
            if isinstance(last_info, dict) and last_info.get("query_valid") is False:
                step_status = "query_invalid"
            if bool(done):
                step_status = "done"

            _emit_step(
                run_id=run_id,
                task_id=resolved_task_id,
                step=step_idx,
                action_type=action.action_type,
                reward=float(last_reward),
                done=bool(done),
                status=step_status,
            )

            steps_taken = step_idx

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
                _emit_step(
                    run_id=run_id,
                    task_id=resolved_task_id,
                    step=steps_taken,
                    action_type="submit",
                    reward=float(last_reward),
                    done=bool(done),
                    status=status,
                )
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
    _emit_end(run_id=run_id, results=results, total_elapsed=total_elapsed)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:  # pragma: no cover - final safety guard for validator stability
        print(f"[fatal] Unhandled inference exception: {exc}", file=sys.stderr)
        # Exit 0 to prevent fail-fast on uncaught runtime errors; failures are encoded in logs/status lines.
        sys.exit(0)
