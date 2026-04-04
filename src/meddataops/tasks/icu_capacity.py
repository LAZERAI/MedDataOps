from __future__ import annotations

import random
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
from faker import Faker

from meddataops.models import Difficulty, TaskSpec


HOSPITAL_B_WARD_CODE_MAP_TABLE: list[dict[str, str]] = [
    {"ward_code": "MICU", "icu_unit": "ICU_MEDICAL"},
    {"ward_code": "M-ICU", "icu_unit": "ICU_MEDICAL"},
    {"ward_code": "SICU", "icu_unit": "ICU_SURGICAL"},
    {"ward_code": "SURG-ICU", "icu_unit": "ICU_SURGICAL"},
    {"ward_code": "CCU", "icu_unit": "ICU_CARDIAC"},
    {"ward_code": "CARD-ICU", "icu_unit": "ICU_CARDIAC"},
    {"ward_code": "NCCU", "icu_unit": "ICU_NEURO"},
    {"ward_code": "NEURO-ICU", "icu_unit": "ICU_NEURO"},
]


_HOSPITAL_B_TO_UNIT = {row["ward_code"]: row["icu_unit"] for row in HOSPITAL_B_WARD_CODE_MAP_TABLE}


ICU_UNITS = sorted({row["icu_unit"] for row in HOSPITAL_B_WARD_CODE_MAP_TABLE})


ICU_SEED_REFERENCE_TIME = datetime(2026, 4, 4, 0, 0, 0, tzinfo=timezone.utc)


UNIT_CAPACITY: dict[str, int] = {
    "ICU_CARDIAC": 24,
    "ICU_MEDICAL": 40,
    "ICU_NEURO": 20,
    "ICU_SURGICAL": 32,
}


BROKEN_QUERY = (
    "SELECT icu_unit, COUNT(*) as current_occupancy,\n"
    "  (SELECT COUNT(*) FROM icu_beds WHERE icu_beds.unit = patients.icu_unit) as total_capacity,\n"
    "  (SELECT COUNT(*) FROM patients p2 WHERE p2.icu_unit = patients.icu_unit AND p2.status='active') as active_count\n"
    "FROM patients\n"
    "GROUP BY icu_unit"
)


CORRECT_QUERY = (
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


def seed_hospital_a_patients(
    row_count: int = 300,
    seed: int = 77,
) -> list[dict[str, Any]]:
    """Seed hospital_a_patients with schema:
    - patient_id INT
    - bed_number VARCHAR
    - icu_unit VARCHAR
    - admitted_at TIMESTAMP
    """
    faker = Faker()
    faker.seed_instance(seed)
    rng = random.Random(seed)

    rows: list[dict[str, Any]] = []
    for index in range(row_count):
        admitted_at = faker.date_time_between(
            start_date=ICU_SEED_REFERENCE_TIME - timedelta(days=45),
            end_date=ICU_SEED_REFERENCE_TIME,
            tzinfo=timezone.utc,
        )
        rows.append(
            {
                "patient_id": 200000 + index,
                "bed_number": f"Bed {rng.randint(1, 60)}",
                "icu_unit": rng.choice(ICU_UNITS),
                "admitted_at": admitted_at.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

    return rows


def seed_hospital_b_patients(
    hospital_a_rows: list[dict[str, Any]],
    row_count: int = 260,
    duplicate_ratio: float = 0.20,
    seed: int = 88,
) -> list[dict[str, Any]]:
    """Seed hospital_b_patients with schema:
    - pid VARCHAR (UUID)
    - ward_code VARCHAR
    - room VARCHAR
    - admission_ts BIGINT (unix ms)

    Approximately duplicate_ratio rows mirror hospital A occupancy patterns.
    """
    faker = Faker()
    faker.seed_instance(seed)
    rng = random.Random(seed)

    if row_count <= 0:
        return []

    duplicate_count = min(row_count, max(0, int(round(row_count * duplicate_ratio))))
    unique_count = max(0, row_count - duplicate_count)

    unit_to_ward_codes: dict[str, list[str]] = {}
    for row in HOSPITAL_B_WARD_CODE_MAP_TABLE:
        unit_to_ward_codes.setdefault(row["icu_unit"], []).append(row["ward_code"])

    rows: list[dict[str, Any]] = []

    duplicate_sources = rng.sample(hospital_a_rows, k=min(len(hospital_a_rows), duplicate_count))
    while len(duplicate_sources) < duplicate_count and hospital_a_rows:
        duplicate_sources.append(rng.choice(hospital_a_rows))

    for source in duplicate_sources:
        source_dt = _parse_timestamp(source.get("admitted_at"))
        if source_dt is None:
            source_dt = faker.date_time_between(
                start_date=ICU_SEED_REFERENCE_TIME - timedelta(days=30),
                end_date=ICU_SEED_REFERENCE_TIME,
                tzinfo=timezone.utc,
            )

        jitter_seconds = rng.randint(-120, 120)
        b_dt = source_dt + timedelta(seconds=jitter_seconds)

        mapped_codes = unit_to_ward_codes.get(str(source.get("icu_unit")), ["MICU"])
        rows.append(
            {
                "pid": faker.uuid4(),
                "ward_code": rng.choice(mapped_codes),
                "room": str(source.get("bed_number", "Bed 1")),
                "admission_ts": int(b_dt.timestamp() * 1000),
            }
        )

    ward_codes = [row["ward_code"] for row in HOSPITAL_B_WARD_CODE_MAP_TABLE]
    for _ in range(unique_count):
        b_dt = faker.date_time_between(
            start_date=ICU_SEED_REFERENCE_TIME - timedelta(days=45),
            end_date=ICU_SEED_REFERENCE_TIME,
            tzinfo=timezone.utc,
        )
        rows.append(
            {
                "pid": faker.uuid4(),
                "ward_code": rng.choice(ward_codes),
                "room": f"Bed {rng.randint(1, 60)}",
                "admission_ts": int(b_dt.timestamp() * 1000),
            }
        )

    rng.shuffle(rows)
    return rows


def seed_icu_beds(capacity_map: dict[str, int] | None = None) -> list[dict[str, Any]]:
    """Seed icu_beds table with stable per-unit capacities."""
    cap = capacity_map or UNIT_CAPACITY
    rows: list[dict[str, Any]] = []
    for unit, total in sorted(cap.items()):
        for bed_idx in range(1, total + 1):
            rows.append({"unit": unit, "bed_id": f"Bed {bed_idx}"})
    return rows


def seed_icu_capacity_dataset(
    hospital_a_count: int = 300,
    hospital_b_count: int = 260,
    duplicate_ratio: float = 0.20,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Seed both merged hospital datasets and ICU bed capacities."""
    a_rows = seed_hospital_a_patients(row_count=hospital_a_count, seed=seed)
    b_rows = seed_hospital_b_patients(
        hospital_a_rows=a_rows,
        row_count=hospital_b_count,
        duplicate_ratio=duplicate_ratio,
        seed=seed + 11,
    )
    beds = seed_icu_beds()
    return {
        "hospital_a_patients": a_rows,
        "hospital_b_patients": b_rows,
        "icu_beds": beds,
    }


def _parse_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None

    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    if isinstance(value, (int, float)):
        try:
            timestamp = float(value)
            if timestamp > 10_000_000_000:
                timestamp = timestamp / 1000.0
            return datetime.fromtimestamp(timestamp, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None

    text = str(value).strip()
    if not text:
        return None

    if text.isdigit():
        return _parse_timestamp(float(text))

    parsed = pd.to_datetime(text, errors="coerce", utc=True)
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime().astimezone(timezone.utc)


def _normalize_bed_label(value: Any) -> str:
    text = str(value or "").strip()
    match = re.search(r"(\d+)", text)
    if match is None:
        return "BED-UNKNOWN"
    return f"BED-{int(match.group(1))}"


def _status_from_datetime(dt: datetime, reference: datetime) -> str:
    return "active" if dt >= reference - timedelta(days=7) else "inactive"


def _normalize_merged_patients(
    hospital_a_rows: list[dict[str, Any]],
    hospital_b_rows: list[dict[str, Any]],
    *,
    reference_time: datetime,
) -> list[dict[str, Any]]:
    """Normalize and deduplicate merged hospital rows into a unified schema."""
    seen_signatures: set[tuple[str, str, str]] = set()
    normalized: list[dict[str, Any]] = []

    def add_row(
        *,
        source_system: str,
        source_record_id: str,
        icu_unit: str,
        bed_raw: Any,
        admitted_raw: Any,
    ) -> None:
        if icu_unit not in ICU_UNITS:
            return

        admitted_dt = _parse_timestamp(admitted_raw)
        if admitted_dt is None:
            return

        admitted_rounded = admitted_dt.replace(second=0, microsecond=0)
        dedupe_time = admitted_rounded.replace(minute=(admitted_rounded.minute // 10) * 10)

        bed_core = _normalize_bed_label(bed_raw)
        signature = (icu_unit, bed_core, dedupe_time.isoformat())
        if signature in seen_signatures:
            return

        seen_signatures.add(signature)

        source_prefix = "A" if source_system == "hospital_a" else "B"
        normalized.append(
            {
                "patient_uid": f"{source_prefix}-{source_record_id}",
                "source_system": source_system,
                "source_record_id": source_record_id,
                "icu_unit": icu_unit,
                "bed_number": f"{source_prefix}-{bed_core}",
                "admitted_at": admitted_rounded.strftime("%Y-%m-%d %H:%M:%S"),
                "status": _status_from_datetime(admitted_rounded, reference_time),
            }
        )

    # Prefer hospital A rows when deduplicating ambiguous overlaps.
    for row in hospital_a_rows:
        add_row(
            source_system="hospital_a",
            source_record_id=str(row.get("patient_id", "")),
            icu_unit=str(row.get("icu_unit", "")).strip(),
            bed_raw=row.get("bed_number"),
            admitted_raw=row.get("admitted_at"),
        )

    for row in hospital_b_rows:
        ward_code = str(row.get("ward_code", "")).strip()
        add_row(
            source_system="hospital_b",
            source_record_id=str(row.get("pid", "")),
            icu_unit=_HOSPITAL_B_TO_UNIT.get(ward_code, ""),
            bed_raw=row.get("room"),
            admitted_raw=row.get("admission_ts"),
        )

    normalized.sort(key=lambda r: (r["icu_unit"], r["admitted_at"], r["bed_number"]))
    return normalized


def _build_expected_capacity_result(
    patients_rows: list[dict[str, Any]],
    icu_beds_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    occupancy = Counter(row["icu_unit"] for row in patients_rows)
    active = Counter(row["icu_unit"] for row in patients_rows if row.get("status") == "active")
    capacity = Counter(row["unit"] for row in icu_beds_rows)

    all_units = sorted(set(capacity) | set(occupancy) | set(active))
    return [
        {
            "icu_unit": unit,
            "current_occupancy": int(occupancy.get(unit, 0)),
            "total_capacity": int(capacity.get(unit, 0)),
            "active_count": int(active.get(unit, 0)),
        }
        for unit in all_units
    ]


def _flatten_dirty_rows(
    hospital_a_rows: list[dict[str, Any]],
    hospital_b_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []

    for row in hospital_a_rows:
        merged.append({"source_table": "hospital_a_patients", **row})

    for row in hospital_b_rows:
        merged.append({"source_table": "hospital_b_patients", **row})

    return merged


def _normalize_query_rows(agent_query_result: pd.DataFrame | list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(agent_query_result, pd.DataFrame):
        rows = agent_query_result.to_dict(orient="records")
    else:
        rows = list(agent_query_result)

    normalized: list[dict[str, Any]] = []
    for row in rows:
        icu_unit = str(row.get("icu_unit", "")).strip().upper()
        if not icu_unit:
            continue
        try:
            normalized.append(
                {
                    "icu_unit": icu_unit,
                    "current_occupancy": int(float(row.get("current_occupancy", 0))),
                    "total_capacity": int(float(row.get("total_capacity", 0))),
                    "active_count": int(float(row.get("active_count", 0))),
                }
            )
        except (TypeError, ValueError):
            continue

    return normalized


def _schema_normalization_score(agent_table: pd.DataFrame | list[dict[str, Any]]) -> float:
    if isinstance(agent_table, pd.DataFrame):
        df = agent_table.copy()
    else:
        df = pd.DataFrame(agent_table)

    if df.empty:
        return 0.0

    required_columns = {
        "patient_uid",
        "source_system",
        "source_record_id",
        "icu_unit",
        "bed_number",
        "admitted_at",
        "status",
    }

    present_columns = set(df.columns)
    column_score = len(required_columns & present_columns) / len(required_columns)
    if not required_columns.issubset(present_columns):
        return round(0.20 * column_score, 6)

    working = df[list(required_columns)].copy()

    source_ok = working["source_system"].astype(str).isin({"hospital_a", "hospital_b"}).mean()
    unit_ok = working["icu_unit"].astype(str).isin(ICU_UNITS).mean()

    bed_prefix_ok = working["bed_number"].astype(str).str.match(r"^[AB]-BED-(\d+|UNKNOWN)$", na=False).mean()

    admitted_ok = pd.to_datetime(working["admitted_at"], errors="coerce", utc=True).notna().mean()
    status_ok = working["status"].astype(str).isin({"active", "inactive"}).mean()

    dedupe_ratio = 1.0 - (
        working.duplicated(subset=["icu_unit", "bed_number", "admitted_at"], keep="first").sum()
        / max(1, len(working))
    )

    row_count_score = max(
        0.0,
        1.0 - (abs(len(working) - len(_ICU_NORMALIZED_ROWS)) / max(1, len(_ICU_NORMALIZED_ROWS))),
    )

    score = (
        0.12 * column_score
        + 0.12 * source_ok
        + 0.16 * unit_ok
        + 0.16 * bed_prefix_ok
        + 0.14 * admitted_ok
        + 0.10 * status_ok
        + 0.10 * dedupe_ratio
        + 0.10 * row_count_score
    )

    return float(round(max(0.0, min(1.0, score)), 6))


def _query_correctness_score(agent_query_result: pd.DataFrame | list[dict[str, Any]]) -> float:
    rows = _normalize_query_rows(agent_query_result)
    if not rows:
        return 0.0

    expected_map = {
        row["icu_unit"]: (
            int(row["current_occupancy"]),
            int(row["total_capacity"]),
            int(row["active_count"]),
        )
        for row in ICU_CAPACITY_GROUND_TRUTH_EXPECTED_RESULT
    }

    actual_map = {
        row["icu_unit"]: (
            int(row["current_occupancy"]),
            int(row["total_capacity"]),
            int(row["active_count"]),
        )
        for row in rows
    }

    all_units = sorted(set(expected_map) | set(actual_map))
    if not all_units:
        return 0.0

    per_unit_scores: list[float] = []
    for unit in all_units:
        expected_tuple = expected_map.get(unit)
        actual_tuple = actual_map.get(unit)

        if expected_tuple is None or actual_tuple is None:
            per_unit_scores.append(0.0)
            continue

        value_scores: list[float] = []
        for expected_value, actual_value in zip(expected_tuple, actual_tuple):
            value_scores.append(max(0.0, 1.0 - (abs(actual_value - expected_value) / max(1, expected_value))))

        per_unit_scores.append(sum(value_scores) / len(value_scores))

    structural_score = 1.0 - (
        abs(len(actual_map) - len(expected_map)) / max(1, len(expected_map))
    )
    structural_score = max(0.0, structural_score)

    return float(round((0.85 * (sum(per_unit_scores) / len(per_unit_scores))) + (0.15 * structural_score), 6))


def _estimate_query_cost(query: str) -> float:
    q = (query or "").strip().lower()
    if not q:
        return 2_000.0

    cost = 800.0

    subquery_count = len(re.findall(r"\(\s*select", q))
    cost += 250.0 * subquery_count

    if "with " in q:
        cost -= 220.0

    if " join " in q:
        cost -= 80.0

    if "group by" in q:
        cost += 40.0

    if " where " in q:
        cost += 20.0

    if "patients.icu_unit" in q and subquery_count >= 2:
        cost += 350.0

    if "count(*) from patients p2" in q:
        cost += 200.0

    if "status='active'" in q or "status = 'active'" in q:
        cost += 30.0

    return max(50.0, cost)


def _cost_improvement_score(
    agent_query: str | None,
    agent_query_cost: float | None,
) -> float:
    baseline_cost = _BROKEN_QUERY_ESTIMATED_COST

    if agent_query_cost is not None:
        try:
            candidate_cost = float(agent_query_cost)
        except (TypeError, ValueError):
            candidate_cost = _estimate_query_cost(agent_query or "")
    else:
        candidate_cost = _estimate_query_cost(agent_query or "")

    if candidate_cost <= 0:
        return 1.0

    improvement = (baseline_cost - candidate_cost) / baseline_cost
    if improvement <= 0:
        return 0.0

    # Reward meaningful reductions; 35%+ reduction is considered full score.
    return float(round(min(1.0, improvement / 0.35), 6))


def score_icu_capacity(
    agent_table: pd.DataFrame | list[dict[str, Any]],
    agent_query_result: pd.DataFrame | list[dict[str, Any]],
    agent_query: str | None = None,
    agent_query_cost: float | None = None,
) -> float:
    """Score ICU capacity task with schema, correctness, and performance criteria.

    Composite score:
    - 40% schema normalization quality
    - 45% query correctness
    - 15% query cost improvement
    """
    schema_score = _schema_normalization_score(agent_table)
    query_score = _query_correctness_score(agent_query_result)
    cost_score = _cost_improvement_score(agent_query=agent_query, agent_query_cost=agent_query_cost)

    final_score = (0.40 * schema_score) + (0.45 * query_score) + (0.15 * cost_score)
    return float(round(max(0.0, min(1.0, final_score)), 6))


_ICU_REFERENCE_TIME = datetime(2026, 4, 4, 0, 0, 0, tzinfo=timezone.utc)

_ICU_DATASET = seed_icu_capacity_dataset()
_HOSPITAL_A_ROWS = _ICU_DATASET["hospital_a_patients"]
_HOSPITAL_B_ROWS = _ICU_DATASET["hospital_b_patients"]
_ICU_BEDS_ROWS = _ICU_DATASET["icu_beds"]

_ICU_DIRTY_ROWS = _flatten_dirty_rows(_HOSPITAL_A_ROWS, _HOSPITAL_B_ROWS)
_ICU_NORMALIZED_ROWS = _normalize_merged_patients(
    _HOSPITAL_A_ROWS,
    _HOSPITAL_B_ROWS,
    reference_time=_ICU_REFERENCE_TIME,
)

ICU_CAPACITY_GROUND_TRUTH_EXPECTED_RESULT = _build_expected_capacity_result(
    _ICU_NORMALIZED_ROWS,
    _ICU_BEDS_ROWS,
)

_BROKEN_QUERY_ESTIMATED_COST = _estimate_query_cost(BROKEN_QUERY)
_CORRECT_QUERY_ESTIMATED_COST = _estimate_query_cost(CORRECT_QUERY)


ICU_CAPACITY_GROUND_TRUTH_CLEAN_SPEC: dict[str, Any] = {
    "source_tables": {
        "hospital_a_patients": ["patient_id", "bed_number", "icu_unit", "admitted_at"],
        "hospital_b_patients": ["pid", "ward_code", "room", "admission_ts"],
    },
    "normalized_schema": [
        "patient_uid",
        "source_system",
        "source_record_id",
        "icu_unit",
        "bed_number",
        "admitted_at",
        "status",
    ],
    "normalization_rules": {
        "hospital_b_ward_mapping": "ward_code -> icu_unit via HOSPITAL_B_WARD_CODE_MAP_TABLE",
        "admission_ts": "unix milliseconds -> UTC timestamp",
        "bed_number": "prefix with hospital source (A-/B-) and normalize to BED-{n}",
        "dedupe": "remove cross-system duplicates by (icu_unit, bed core, 10-min admitted_at bucket)",
    },
}


ICU_CAPACITY_TASK_METADATA: dict[str, Any] = {
    "id": "icu_capacity",
    "difficulty": "hard",
    "name": "ICU Capacity Merge Repair",
    "scenario": (
        "Two hospitals merged their systems. The ICU capacity report is broken - "
        "wrong results and extremely slow. Clean the merged data and fix the query."
    ),
    "seed": {
        "hospital_a_count": 300,
        "hospital_b_count": 260,
        "cross_system_duplicate_ratio": 0.20,
        "seed": 42,
    },
    "hospital_b_ward_code_map": HOSPITAL_B_WARD_CODE_MAP_TABLE,
    "broken_query": BROKEN_QUERY,
    "correct_query": CORRECT_QUERY,
    "ground_truth_expected_result": ICU_CAPACITY_GROUND_TRUTH_EXPECTED_RESULT,
    "ground_truth_clean_spec": ICU_CAPACITY_GROUND_TRUTH_CLEAN_SPEC,
    "estimated_query_cost": {
        "broken": _BROKEN_QUERY_ESTIMATED_COST,
        "correct": _CORRECT_QUERY_ESTIMATED_COST,
    },
    "icu_beds": _ICU_BEDS_ROWS,
}


TASK = TaskSpec(
    id="icu_capacity",
    name="ICU Capacity Merge Repair",
    difficulty=Difficulty.HARD,
    description=ICU_CAPACITY_TASK_METADATA["scenario"],
    hints=[
        "Normalize the merged schemas before touching SQL logic.",
        "Map hospital B ward_code values into unified ICU units.",
        "Use hospital prefixes for bed identifiers to avoid collisions.",
        "Rewrite correlated subqueries into set-based CTE aggregations.",
    ],
    dirty_rows=_ICU_DIRTY_ROWS,
    broken_sql=BROKEN_QUERY,
    expected_clean_rows=_ICU_NORMALIZED_ROWS,
    expected_sql=CORRECT_QUERY,
)
