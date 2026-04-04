# MedDataOps Pre-Submission Checklist

This checklist is aligned to hackathon pre-submission/disqualification risks: deployment reliability, spec validity, runtime bounds, scoring correctness, and documentation completeness.

Use this one-command runner first:

```bash
bash run_all_checks.sh
```

If you need to run checks manually, use the exact commands below.

## 1) HF Space deploys and `/reset` returns HTTP 200

- What to check:
  - Your deployed Space accepts a task reset request and responds with status 200.
- Exact command:

```bash
curl -sS -o /tmp/hf_reset_body.json -w "%{http_code}\n" \
  -X POST "${SPACE_URL:-https://lazerai-meddataops.hf.space}/reset" \
  -H "Content-Type: application/json" \
  --data '{"task_id":"triage_report","seed":101}'
```

- Passing result:
  - Printed status code is `200`.
  - Body is valid JSON observation payload.
- If it fails:
  - Open Space logs and fix startup/runtime errors.
  - Verify Docker image starts successfully and API is listening on `7860`.
  - Re-deploy, then rerun the curl check.

## 2) `openenv validate` passes locally

- What to check:
  - `openenv.yaml` passes OpenEnv schema/interface validation.
- Exact command:

```bash
openenv validate openenv.yaml
```

- Passing result:
  - Exit code `0` and validation success output.
- If it fails:
  - Fix missing/invalid fields in `openenv.yaml`.
  - Ensure model paths (`request_model` / `response_model`) are importable.
  - Re-run validation until clean.

## 3) Docker build succeeds and build time is reasonable

- What to check:
  - Image builds successfully.
  - Build duration is under your acceptable threshold (default in script: 900s).
- Exact command:

```bash
time docker build -t meddataops:presub .
```

- Passing result:
  - Build exits `0`.
  - Elapsed time is comfortably below threshold for judge reproducibility.
- If it fails:
  - Check Dockerfile logs for package/install failures.
  - Reduce build size (layer caching, apt cleanup, pinned dependencies).
  - Rebuild and retime.

## 4) Docker run starts and `/health` returns 200

- What to check:
  - Container starts from built image.
  - API health endpoint is live.
- Exact commands:

```bash
docker run --rm -d --name meddataops_presub -p 7860:7860 meddataops:presub
curl -sS -o /tmp/local_health.json -w "%{http_code}\n" http://127.0.0.1:7860/health
docker logs --tail 100 meddataops_presub
docker rm -f meddataops_presub
```

- Passing result:
  - Health request returns `200`.
  - Body includes `{"status":"ok", ...}`.
- If it fails:
  - Inspect container logs.
  - Verify DB startup and environment variables in entrypoint.
  - Fix startup race conditions and retry.

## 5) `inference.py` runs end-to-end without errors

- What to check:
  - Full inference run finishes without uncaught exception.
- Exact command:

```bash
API_BASE_URL="https://api-inference.huggingface.co/v1" \
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct" \
HF_TOKEN="<your_hf_token>" \
python inference.py
```

- Passing result:
  - Exit code `0`.
  - Summary table printed for all tasks.
- If it fails:
  - Confirm `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` are set.
  - Check traceback and fix environment/action parsing/runtime issues.

## 6) All 3 tasks produce scores in `[0.0, 1.0]`

- What to check:
  - Final scores for `triage_report`, `medication_summary`, `icu_capacity` are all bounded.
- Exact command:

```bash
API_BASE_URL="https://api-inference.huggingface.co/v1" MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct" HF_TOKEN="<your_hf_token>" \
python inference.py | tee /tmp/inference.log
python - <<'PY'
import re, sys
from pathlib import Path

text = Path('/tmp/inference.log').read_text(encoding='utf-8', errors='ignore')
pat = re.compile(r'^\s*(triage_report|medication_summary|icu_capacity)\s*\|\s*\d+\s*\|\s*([0-9]*\.?[0-9]+)\s*\|', re.M)
scores = {task: float(score) for task, score in pat.findall(text)}
required = {'triage_report', 'medication_summary', 'icu_capacity'}
missing = sorted(required - set(scores))
bad = {k: v for k, v in scores.items() if not (0.0 <= v <= 1.0)}
if missing or bad:
    print('FAIL', {'missing': missing, 'bad': bad, 'scores': scores})
    sys.exit(1)
print('PASS', scores)
PY
```

- Passing result:
  - All three tasks found and each score is in range.
- If it fails:
  - Fix reward extraction/parsing.
  - Verify scoring functions clamp and return valid ranges.

## 7) Ground-truth submission scores `1.0` on all tasks

- What to check:
  - Task score functions return perfect score when given ground-truth clean rows + expected results.
- Exact command:

```bash
python - <<'PY'
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
triage_score = tr.score_triage_report(triage_task.expected_clean_rows, triage_query_rows)

med_task = get_task('medication_summary')
med_score = ms.score_medication_summary(med_task.expected_clean_rows, ms.MEDICATION_SUMMARY_GROUND_TRUTH_EXPECTED_RESULT)

icu_task = get_task('icu_capacity')
icu_score = ic.score_icu_capacity(icu_task.expected_clean_rows, ic.ICU_CAPACITY_GROUND_TRUTH_EXPECTED_RESULT)

scores = {
    'triage_report': float(triage_score),
    'medication_summary': float(med_score),
    'icu_capacity': float(icu_score),
}

all_perfect = all(abs(v - 1.0) < 1e-9 for v in scores.values())
print(scores)
raise SystemExit(0 if all_perfect else 1)
PY
```

- Passing result:
  - Printed scores are exactly `1.0` for all three tasks.
- If it fails:
  - Fix task scoring logic or ground-truth expected artifacts.
  - Re-check normalization assumptions (case/date/type coercion).

## 8) `inference.py` runtime is under 20 minutes

- What to check:
  - Full inference runtime is below 1200 seconds.
- Exact command:

```bash
time API_BASE_URL="https://api-inference.huggingface.co/v1" MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct" HF_TOKEN="<your_hf_token>" python inference.py
```

- Passing result:
  - Total elapsed wall-clock time `< 20:00`.
- If it fails:
  - Reduce retries/timeouts and improve step efficiency.
  - Lower unnecessary actions and prompt verbosity.

## 9) Container memory usage is under 8GB

- What to check:
  - Running API container memory (`docker stats`) stays below 8GB.
- Exact commands:

```bash
docker run --rm -d --name meddataops_memcheck -p 7860:7860 meddataops:presub
docker stats --no-stream --format "{{.MemUsage}}" meddataops_memcheck
docker rm -f meddataops_memcheck
```

- Passing result:
  - Used memory value (left side of `MemUsage`) is < 8GB.
- If it fails:
  - Profile memory hotspots and reduce in-memory dataframe volume.
  - Avoid unnecessary copies and large intermediate objects.

## 10) README has all required sections

- What to check:
  - README contains the required 12 numbered sections.
- Exact command:

```bash
python - <<'PY'
import re, sys
from pathlib import Path

text = Path('README.md').read_text(encoding='utf-8', errors='ignore')
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
missing = [h for h in required if h not in text]
print('missing=', missing)
raise SystemExit(0 if not missing else 1)
PY
```

- Passing result:
  - No missing sections.
- If it fails:
  - Add missing section headings and required content.

## 11) `openenv.yaml` has all required fields

- What to check:
  - Required top-level and nested keys exist and tasks are defined.
- Exact command:

```bash
python - <<'PY'
import sys, yaml
from pathlib import Path

cfg = yaml.safe_load(Path('openenv.yaml').read_text(encoding='utf-8'))

required_paths = [
    ('api_version',),
    ('environment',), ('environment','id'), ('environment','name'), ('environment','version'), ('environment','description'), ('environment','entrypoint'),
    ('interfaces',), ('interfaces','reset','request_model'), ('interfaces','reset','response_model'),
    ('interfaces','step','request_model'), ('interfaces','step','response_model'),
    ('interfaces','state','response_model'),
    ('tasks',),
    ('backend',), ('backend','type'), ('backend','dsn_env'),
]

def has_path(data, path):
    cur = data
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return False
        cur = cur[p]
    return True

missing = ['.'.join(p) for p in required_paths if not has_path(cfg, p)]
tasks = cfg.get('tasks', []) if isinstance(cfg, dict) else []
task_ok = isinstance(tasks, list) and len(tasks) >= 3 and all(
    isinstance(t, dict) and all(k in t for k in ('id', 'difficulty', 'description')) for t in tasks
)

print({'missing_paths': missing, 'task_ok': task_ok})
raise SystemExit(0 if (not missing and task_ok) else 1)
PY
```

- Passing result:
  - `missing_paths` is empty and `task_ok` is true.
- If it fails:
  - Add missing keys and ensure each task has `id`, `difficulty`, `description`.

## 12) Pydantic models are fully typed

- What to check:
  - No untyped class attributes in `BaseModel` subclasses.
- Exact command:

```bash
python - <<'PY'
import ast, sys
from pathlib import Path

root = Path('src')
py_files = sorted(root.rglob('*.py'))
violations = []

def is_basemodel_base(base):
    if isinstance(base, ast.Name):
        return base.id == 'BaseModel'
    if isinstance(base, ast.Attribute):
        return base.attr == 'BaseModel'
    return False

for file in py_files:
    tree = ast.parse(file.read_text(encoding='utf-8'))
    for node in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        if not any(is_basemodel_base(b) for b in node.bases):
            continue
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                targets = [t.id for t in stmt.targets if isinstance(t, ast.Name)]
                if targets:
                    violations.append(f"{file}:{node.name}:{', '.join(targets)}")

print({'violations': violations})
raise SystemExit(0 if not violations else 1)
PY
```

- Passing result:
  - `violations` list is empty.
- If it fails:
  - Convert untyped assignments to annotated fields (for example `field: str = ...`).

---

## Automated Runner

`run_all_checks.sh` automates these checks and prints a final `PASSED/FAILED` summary across all 12 items:

```bash
bash run_all_checks.sh
```
