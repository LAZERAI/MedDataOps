#!/usr/bin/env bash

set -u
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

SPACE_URL="${SPACE_URL:-https://lazerai-meddataops.hf.space}"
LOCAL_URL="${LOCAL_URL:-http://127.0.0.1:7860}"
IMAGE_TAG="${IMAGE_TAG:-meddataops:presub}"
CONTAINER_NAME="${CONTAINER_NAME:-meddataops_presub_check}"

BUILD_REASONABLE_SECONDS="${BUILD_REASONABLE_SECONDS:-900}"
INFERENCE_MAX_SECONDS="${INFERENCE_MAX_SECONDS:-1200}"
MEMORY_LIMIT_BYTES=$((8 * 1024 * 1024 * 1024))

ARTIFACT_DIR="${ARTIFACT_DIR:-.check_artifacts}"
mkdir -p "$ARTIFACT_DIR"

declare -A STATUS
declare -A DETAILS

BUILD_OK=0
CONTAINER_OK=0
INFERENCE_OK=0

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

resolve_python_cmd() {
  if [[ -n "${PYTHON_CMD:-}" ]]; then
    echo "$PYTHON_CMD"
    return
  fi

  if [[ -x "$ROOT_DIR/.venv/Scripts/python.exe" ]]; then
    echo "$ROOT_DIR/.venv/Scripts/python.exe"
    return
  fi

  if have_cmd python3; then
    echo "python3"
    return
  fi

  if have_cmd python; then
    echo "python"
    return
  fi

  echo ""
}

PYTHON_CMD="$(resolve_python_cmd)"

have_python() {
  [[ -n "$PYTHON_CMD" ]]
}

py_exec() {
  "$PYTHON_CMD" "$@"
}

set_result() {
  local idx="$1"
  local status="$2"
  local message="$3"
  STATUS["$idx"]="$status"
  DETAILS["$idx"]="$message"
  printf "[%s] %s\n" "$status" "$message"
}

cleanup() {
  if have_cmd docker; then
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT

check_1_hf_reset() {
  if ! have_cmd curl; then
    set_result 1 "FAILED" "1) curl not found; cannot test HF /reset"
    return
  fi

  local body_file="$ARTIFACT_DIR/hf_reset_body.json"
  local code
  code="$(curl -sS -o "$body_file" -w "%{http_code}" -X POST "${SPACE_URL}/reset" -H "Content-Type: application/json" --data '{"task_id":"triage_report","seed":101}' || true)"

  if [[ "$code" == "200" ]]; then
    set_result 1 "PASSED" "1) HF /reset returned 200"
  else
    set_result 1 "FAILED" "1) HF /reset expected 200, got ${code:-<no-code>} (see $body_file)"
  fi
}

check_2_openenv_validate() {
  local log="$ARTIFACT_DIR/openenv_validate.log"

  if have_cmd openenv; then
    if openenv validate . >"$log" 2>&1; then
      set_result 2 "PASSED" "2) openenv validate . passed"
    elif openenv validate openenv.yaml >"$log" 2>&1; then
      # Backward compatibility for older OpenEnv CLIs.
      set_result 2 "PASSED" "2) openenv validate openenv.yaml passed"
    else
      set_result 2 "FAILED" "2) openenv validate failed (see $log)"
    fi
    return
  fi

  if have_python && py_exec - <<'PY' >/dev/null 2>&1
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec('openenv') else 1)
PY
  then
    if py_exec -m openenv validate . >"$log" 2>&1; then
      set_result 2 "PASSED" "2) python -m openenv validate . passed"
    elif py_exec -m openenv validate openenv.yaml >"$log" 2>&1; then
      set_result 2 "PASSED" "2) python -m openenv validate openenv.yaml passed"
    else
      set_result 2 "FAILED" "2) python -m openenv validate failed (see $log)"
    fi
  else
    set_result 2 "FAILED" "2) openenv CLI/module not installed"
  fi
}

check_3_docker_build() {
  if ! have_cmd docker; then
    set_result 3 "FAILED" "3) docker not found; cannot build image"
    return
  fi

  local log="$ARTIFACT_DIR/docker_build.log"
  local start end elapsed

  start="$(date +%s)"
  if docker build -t "$IMAGE_TAG" . >"$log" 2>&1; then
    end="$(date +%s)"
    elapsed=$((end - start))
    BUILD_OK=1
    if (( elapsed <= BUILD_REASONABLE_SECONDS )); then
      set_result 3 "PASSED" "3) docker build succeeded in ${elapsed}s"
    else
      set_result 3 "FAILED" "3) docker build took ${elapsed}s (> ${BUILD_REASONABLE_SECONDS}s threshold)"
    fi
  else
    set_result 3 "FAILED" "3) docker build failed (see $log)"
  fi
}

check_4_docker_run_health() {
  if ! have_cmd docker || ! have_cmd curl; then
    set_result 4 "FAILED" "4) docker/curl missing; cannot run health check"
    return
  fi

  if (( BUILD_OK != 1 )); then
    set_result 4 "FAILED" "4) skipped because docker build did not succeed"
    return
  fi

  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

  local run_log="$ARTIFACT_DIR/docker_run_id.txt"
  local code
  if ! docker run -d --name "$CONTAINER_NAME" -p 7860:7860 "$IMAGE_TAG" >"$run_log" 2>&1; then
    set_result 4 "FAILED" "4) docker run failed (see $run_log)"
    return
  fi

  local i
  for i in $(seq 1 45); do
    code="$(curl -sS -o "$ARTIFACT_DIR/local_health_body.json" -w "%{http_code}" "${LOCAL_URL}/health" || true)"
    if [[ "$code" == "200" ]]; then
      CONTAINER_OK=1
      set_result 4 "PASSED" "4) local /health returned 200"
      return
    fi
    sleep 2
  done

  docker logs "$CONTAINER_NAME" >"$ARTIFACT_DIR/docker_container.log" 2>&1 || true
  set_result 4 "FAILED" "4) local /health did not return 200 in time (see $ARTIFACT_DIR/docker_container.log)"
}

check_9_memory_under_8gb() {
  if ! have_cmd docker || ! have_python; then
    set_result 9 "FAILED" "9) docker/python missing; cannot check memory"
    return
  fi

  if (( CONTAINER_OK != 1 )); then
    set_result 9 "FAILED" "9) skipped because container is not healthy/running"
    return
  fi

  local mem_usage raw_used mem_bytes
  mem_usage="$(docker stats --no-stream --format "{{.MemUsage}}" "$CONTAINER_NAME" 2>/dev/null | head -n 1)"
  raw_used="$(echo "$mem_usage" | awk -F'/' '{print $1}' | xargs)"

  mem_bytes="$(py_exec - "$raw_used" <<'PY'
import re, sys

s = (sys.argv[1] if len(sys.argv) > 1 else '').strip()
if not s:
    print(-1)
    raise SystemExit(0)

m = re.match(r'^([0-9]+(?:\.[0-9]+)?)\s*([A-Za-z]+)$', s)
if not m:
    print(-1)
    raise SystemExit(0)

value = float(m.group(1))
unit = m.group(2)
scale = {
    'B': 1,
    'KB': 1000,
    'MB': 1000**2,
    'GB': 1000**3,
    'TB': 1000**4,
    'KiB': 1024,
    'MiB': 1024**2,
    'GiB': 1024**3,
    'TiB': 1024**4,
}
print(int(value * scale.get(unit, -1)))
PY
)"

  if [[ -z "$mem_bytes" || "$mem_bytes" -lt 0 ]]; then
    set_result 9 "FAILED" "9) unable to parse docker memory usage: '$mem_usage'"
    return
  fi

  if (( mem_bytes < MEMORY_LIMIT_BYTES )); then
    set_result 9 "PASSED" "9) container memory usage is below 8GB (${mem_usage})"
  else
    set_result 9 "FAILED" "9) container memory usage exceeds 8GB (${mem_usage})"
  fi
}

check_5_6_8_inference() {
  if ! have_python; then
    set_result 5 "FAILED" "5) python not found; cannot run inference"
    set_result 6 "FAILED" "6) skipped because inference did not run"
    set_result 8 "FAILED" "8) skipped because inference did not run"
    return
  fi

  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "[warn] HF_TOKEN not set; inference.py will use deterministic fallback mode." >&2
  fi

  local log="$ARTIFACT_DIR/inference.log"
  local score_log="$ARTIFACT_DIR/inference_scores.json"
  local start end elapsed

  start="$(date +%s)"
  if py_exec inference.py >"$log" 2>&1; then
    INFERENCE_OK=1
    set_result 5 "PASSED" "5) inference.py completed without errors"
  else
    set_result 5 "FAILED" "5) inference.py failed (see $log)"
    set_result 6 "FAILED" "6) skipped because inference did not finish"
    set_result 8 "FAILED" "8) skipped because inference did not finish"
    return
  fi
  end="$(date +%s)"
  elapsed=$((end - start))

  if (( elapsed < INFERENCE_MAX_SECONDS )); then
    set_result 8 "PASSED" "8) inference runtime ${elapsed}s (< ${INFERENCE_MAX_SECONDS}s)"
  else
    set_result 8 "FAILED" "8) inference runtime ${elapsed}s (>= ${INFERENCE_MAX_SECONDS}s)"
  fi

  if py_exec - "$log" "$score_log" <<'PY'
import json
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
text = log_path.read_text(encoding='utf-8', errors='ignore')

pattern = re.compile(r'^\s*(triage_report|medication_summary|icu_capacity)\s*\|\s*\d+\s*\|\s*([0-9]*\.?[0-9]+)\s*\|', re.M)
scores = {task: float(score) for task, score in pattern.findall(text)}
required = {'triage_report', 'medication_summary', 'icu_capacity'}

missing = sorted(required - set(scores))
bad = {task: value for task, value in scores.items() if not (0.0 <= value <= 1.0)}

out_path.write_text(json.dumps({'scores': scores, 'missing': missing, 'out_of_range': bad}, indent=2), encoding='utf-8')

if missing or bad:
    raise SystemExit(1)
PY
  then
    set_result 6 "PASSED" "6) all 3 task scores are present and in [0.0, 1.0]"
  else
    set_result 6 "FAILED" "6) score parsing/range validation failed (see $score_log and $log)"
  fi
}

check_7_ground_truth_scores() {
  if ! have_python; then
    set_result 7 "FAILED" "7) python not found; cannot run ground-truth score check"
    return
  fi

  local log="$ARTIFACT_DIR/ground_truth_scores.json"

  if py_exec - "$log" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path('.').resolve()
SRC = ROOT / 'src'
sys.path.insert(0, str(SRC))

import meddataops.tasks.triage_report as tr
import meddataops.tasks.medication_summary as ms
import meddataops.tasks.icu_capacity as ic
from meddataops.tasks import get_task

triage_task = get_task('triage_report')
triage_query_rows = tr._ward_count_rows(pd.DataFrame(triage_task.expected_clean_rows))
triage_score = float(tr.score_triage_report(triage_task.expected_clean_rows, triage_query_rows))

med_task = get_task('medication_summary')
med_score = float(ms.score_medication_summary(med_task.expected_clean_rows, ms.MEDICATION_SUMMARY_GROUND_TRUTH_EXPECTED_RESULT))

icu_task = get_task('icu_capacity')
icu_score = float(ic.score_icu_capacity(
  icu_task.expected_clean_rows,
  ic.ICU_CAPACITY_GROUND_TRUTH_EXPECTED_RESULT,
  agent_query=icu_task.expected_sql,
))

scores = {
    'triage_report': triage_score,
    'medication_summary': med_score,
    'icu_capacity': icu_score,
}

Path(sys.argv[1]).write_text(json.dumps(scores, indent=2), encoding='utf-8')

if not all(abs(v - 1.0) < 1e-9 for v in scores.values()):
    raise SystemExit(1)
PY
  then
    set_result 7 "PASSED" "7) ground-truth scoring returns 1.0 for all tasks"
  else
    set_result 7 "FAILED" "7) ground-truth scoring check failed (see $log)"
  fi
}

check_10_readme_sections() {
  if ! have_python; then
    set_result 10 "FAILED" "10) python not found; cannot validate README sections"
    return
  fi

  local log="$ARTIFACT_DIR/readme_sections.json"

  if py_exec - "$log" <<'PY'
import json
import sys
from pathlib import Path

readme = Path('README.md').read_text(encoding='utf-8', errors='ignore')
required = [
    '## 1. Motivation',
    '## 2. Environment Overview',
    '## 3. Action Space',
    '## 4. Observation Space',
    '## 5. Tasks',
    '## 6. Reward Function',
    '## 7. Quick Start',
    '## 8. API Reference',
    '## 9. Baseline Scores',
    '## 10. Project Structure',
    '## 11. Contributing',
    '## 12. License',
]
missing = [h for h in required if h not in readme]
Path(sys.argv[1]).write_text(json.dumps({'missing_sections': missing}, indent=2), encoding='utf-8')
raise SystemExit(0 if not missing else 1)
PY
  then
    set_result 10 "PASSED" "10) README contains all required sections"
  else
    set_result 10 "FAILED" "10) README section check failed (see $log)"
  fi
}

check_11_openenv_yaml_fields() {
  if ! have_python; then
    set_result 11 "FAILED" "11) python not found; cannot validate openenv.yaml"
    return
  fi

  local log="$ARTIFACT_DIR/openenv_fields.json"

  if py_exec - "$log" <<'PY'
import json
import sys
from pathlib import Path

import yaml

cfg = yaml.safe_load(Path('openenv.yaml').read_text(encoding='utf-8'))

required_paths = [
    ('api_version',),
    ('environment',), ('environment', 'id'), ('environment', 'name'), ('environment', 'version'), ('environment', 'description'), ('environment', 'entrypoint'),
    ('interfaces',),
    ('interfaces', 'reset', 'request_model'), ('interfaces', 'reset', 'response_model'),
    ('interfaces', 'step', 'request_model'), ('interfaces', 'step', 'response_model'),
    ('interfaces', 'state', 'response_model'),
    ('tasks',),
    ('backend',), ('backend', 'type'), ('backend', 'dsn_env'),
]

def has_path(data, path):
    cur = data
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return False
        cur = cur[p]
    return True

missing = ['.'.join(path) for path in required_paths if not has_path(cfg, path)]

tasks = cfg.get('tasks', []) if isinstance(cfg, dict) else []
task_ok = isinstance(tasks, list) and len(tasks) >= 3 and all(
    isinstance(t, dict) and all(k in t for k in ('id', 'difficulty', 'description'))
    for t in tasks
)

payload = {'missing_paths': missing, 'task_ok': task_ok, 'task_count': len(tasks) if isinstance(tasks, list) else 0}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2), encoding='utf-8')

raise SystemExit(0 if (not missing and task_ok) else 1)
PY
  then
    set_result 11 "PASSED" "11) openenv.yaml required fields are present"
  else
    set_result 11 "FAILED" "11) openenv.yaml field check failed (see $log)"
  fi
}

check_12_pydantic_typed() {
  if ! have_python; then
    set_result 12 "FAILED" "12) python not found; cannot validate Pydantic typing"
    return
  fi

  local log="$ARTIFACT_DIR/pydantic_typed.json"

  if py_exec - "$log" <<'PY'
import ast
import json
import sys
from pathlib import Path

root = Path('src')
py_files = sorted(root.rglob('*.py'))
violations = []
parse_errors = []

def is_basemodel_base(base):
    if isinstance(base, ast.Name):
        return base.id == 'BaseModel'
    if isinstance(base, ast.Attribute):
        return base.attr == 'BaseModel'
    return False

ALLOWED_UNANNOTATED_CLASS_FIELDS = {'model_config'}

for file in py_files:
    try:
        tree = ast.parse(file.read_text(encoding='utf-8'))
    except Exception as exc:
        parse_errors.append(f'{file}: {exc}')
        continue

    for node in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        if not any(is_basemodel_base(base) for base in node.bases):
            continue
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                field_names = [t.id for t in stmt.targets if isinstance(t, ast.Name)]
                if field_names:
                    disallowed = [name for name in field_names if name not in ALLOWED_UNANNOTATED_CLASS_FIELDS]
                    if disallowed:
                        violations.append(f"{file}:{node.name}:{','.join(disallowed)}")

payload = {'violations': violations, 'parse_errors': parse_errors}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2), encoding='utf-8')

raise SystemExit(0 if not violations and not parse_errors else 1)
PY
  then
    set_result 12 "PASSED" "12) Pydantic model fields are fully typed (no unannotated class fields)"
  else
    set_result 12 "FAILED" "12) Pydantic typing check failed (see $log)"
  fi
}

print_summary() {
  local pass_count=0
  local fail_count=0
  local unknown_count=0

  echo
  echo "================= CHECK SUMMARY ================="

  local idx
  for idx in $(seq 1 12); do
    local st="${STATUS[$idx]:-FAILED}"
    local msg="${DETAILS[$idx]:-${idx}) no result recorded}"
    printf "%2d. %-7s %s\n" "$idx" "$st" "$msg"
    case "$st" in
      PASSED) ((pass_count+=1)) ;;
      FAILED) ((fail_count+=1)) ;;
      *) ((unknown_count+=1)) ;;
    esac
  done

  echo "-------------------------------------------------"
  echo "PASSED: $pass_count"
  echo "FAILED: $fail_count"
  echo "UNKNOWN: $unknown_count"
  echo "Artifacts: $ARTIFACT_DIR"

  if (( fail_count > 0 || unknown_count > 0 )); then
    return 1
  fi
  return 0
}

echo "Running MedDataOps pre-submission checks..."
echo "SPACE_URL=$SPACE_URL"
echo "IMAGE_TAG=$IMAGE_TAG"
echo "ARTIFACT_DIR=$ARTIFACT_DIR"
echo

check_1_hf_reset
check_2_openenv_validate
check_3_docker_build
check_4_docker_run_health
check_9_memory_under_8gb
check_5_6_8_inference
check_7_ground_truth_scores
check_10_readme_sections
check_11_openenv_yaml_fields
check_12_pydantic_typed

print_summary
