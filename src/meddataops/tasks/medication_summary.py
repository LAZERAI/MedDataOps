from __future__ import annotations

import random
import re
from collections import Counter
from datetime import date, datetime, timezone
from typing import Any

import pandas as pd
from faker import Faker

from meddataops.models import Difficulty, TaskSpec


BROKEN_QUERY = (
    "SELECT p.ward, m.drug_name, COUNT(*) as prescription_count\n"
    "FROM medications m\n"
    "CROSS JOIN patients p\n"
    "WHERE m.prescribed_date >= '2024-01-01'\n"
    "GROUP BY p.ward, m.drug_name"
)


FIXED_QUERY = (
    "SELECT p.ward, m.drug_name, COUNT(*) AS prescription_count "
    "FROM medications m "
    "INNER JOIN patients p ON m.patient_id = p.patient_id "
    "WHERE m.prescribed_date >= DATE '2024-01-01' "
    "GROUP BY p.ward, m.drug_name "
    "ORDER BY p.ward ASC, m.drug_name ASC"
)


WARD_OPTIONS = ["ICU", "ER", "MEDSURG", "OBS", "CARD"]

DRUG_VARIANTS: dict[str, list[str]] = {
    "aspirin": ["Aspirin", "aspirin", "ASPIRIN ", "aspirin  "],
    "metformin": ["Metformin", "metformin", "METFORMIN ", "metformin  "],
    "lisinopril": ["Lisinopril", "lisinopril", "LISINOPRIL ", "lisinopril  "],
    "atorvastatin": ["Atorvastatin", "atorvastatin", "ATORVASTATIN ", "atorvastatin  "],
    "amoxicillin": ["Amoxicillin", "amoxicillin", "AMOXICILLIN ", "amoxicillin  "],
}

BASE_DOSAGE_MG: dict[str, float] = {
    "aspirin": 100.0,
    "metformin": 500.0,
    "lisinopril": 20.0,
    "atorvastatin": 40.0,
    "amoxicillin": 250.0,
}


def seed_medication_summary_dataset(
    patient_count: int = 120,
    medication_count: int = 700,
    orphan_ratio: float = 0.10,
    unix_timestamp_ratio: float = 0.20,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Generate synthetic patients + medications with realistic medium-task noise.

    Noise profile:
    - Drug-name variants in mixed case with trailing spaces.
    - dosage_mg mixed between numeric values and strings like "100mg" / "50 mg".
    - ~10% orphan medication rows with patient_id absent from patients.
    - prescribed_date mixed between ISO date strings and Unix timestamps.
    """
    faker = Faker()
    faker.seed_instance(seed)
    rng = random.Random(seed)

    patients: list[dict[str, Any]] = []
    for index in range(patient_count):
        patient_id = f"P{100000 + index}"
        patients.append(
            {
                "patient_id": patient_id,
                "patient_name": faker.name(),
                "ward": rng.choice(WARD_OPTIONS),
            }
        )

    valid_patient_ids = [row["patient_id"] for row in patients]

    medications: list[dict[str, Any]] = []
    for index in range(medication_count):
        drug_key = rng.choice(list(DRUG_VARIANTS.keys()))
        drug_name = rng.choice(DRUG_VARIANTS[drug_key])

        base_dose = BASE_DOSAGE_MG[drug_key]
        dosage_roll = rng.random()
        if dosage_roll < 0.45:
            dosage_mg: Any = base_dose
        elif dosage_roll < 0.75:
            dosage_mg = f"{int(base_dose)}mg"
        else:
            dosage_mg = f"{int(base_dose)} mg"

        if rng.random() < orphan_ratio:
            patient_id = f"X{900000 + index}"
        else:
            patient_id = rng.choice(valid_patient_ids)

        prescribed_dt = faker.date_time_between(start_date="-3y", end_date="now", tzinfo=timezone.utc)
        if rng.random() < unix_timestamp_ratio:
            prescribed_date: Any = int(prescribed_dt.timestamp())
        else:
            prescribed_date = prescribed_dt.strftime("%Y-%m-%d")

        medications.append(
            {
                "medication_id": f"RX{200000 + index}",
                "patient_id": patient_id,
                "drug_name": drug_name,
                "dosage_mg": dosage_mg,
                "prescribed_date": prescribed_date,
            }
        )

    return {"patients": patients, "medications": medications}


def _normalize_drug_name(value: Any) -> str:
    return str(value or "").strip().lower()


def _parse_dosage_mg(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)

    if value is None:
        return None

    text = str(value).strip().lower()
    if not text:
        return None

    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if match is None:
        return None

    try:
        return float(match.group(1))
    except ValueError:
        return None


def _parse_prescribed_date(value: Any) -> str | None:
    if value is None:
        return None

    if isinstance(value, datetime):
        return value.date().isoformat()

    if isinstance(value, date):
        return value.isoformat()

    if isinstance(value, (int, float)):
        try:
            timestamp = float(value)
            if timestamp > 10_000_000_000:
                timestamp = timestamp / 1000.0
            converted = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            return converted.date().isoformat()
        except (OverflowError, OSError, ValueError):
            return None

    text = str(value).strip()
    if not text:
        return None

    if re.fullmatch(r"\d{10,13}", text):
        return _parse_prescribed_date(float(text))

    parsed = pd.to_datetime(text, errors="coerce", utc=True)
    if pd.isna(parsed):
        return None

    return parsed.date().isoformat()


def _clean_medication_rows(
    rows: list[dict[str, Any]],
    valid_patient_ids: set[str],
) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []

    for row in rows:
        patient_id = str(row.get("patient_id", "")).strip()
        if not patient_id or patient_id not in valid_patient_ids:
            continue

        drug_name = _normalize_drug_name(row.get("drug_name"))
        dosage = _parse_dosage_mg(row.get("dosage_mg"))
        prescribed_date = _parse_prescribed_date(row.get("prescribed_date"))

        if not drug_name or dosage is None or prescribed_date is None:
            continue

        cleaned.append(
            {
                "medication_id": row.get("medication_id"),
                "patient_id": patient_id,
                "drug_name": drug_name,
                "dosage_mg": round(float(dosage), 2),
                "prescribed_date": prescribed_date,
            }
        )

    return cleaned


def _aggregate_expected_result(
    medication_rows: list[dict[str, Any]],
    patient_ward_map: dict[str, str],
    *,
    normalize_drug: bool,
) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str]] = Counter()

    for row in medication_rows:
        patient_id = str(row.get("patient_id", "")).strip()
        ward = patient_ward_map.get(patient_id)
        if ward is None:
            continue

        parsed_date = _parse_prescribed_date(row.get("prescribed_date"))
        if parsed_date is None or parsed_date < "2024-01-01":
            continue

        drug_value = row.get("drug_name")
        if normalize_drug:
            drug_name = _normalize_drug_name(drug_value)
        else:
            drug_name = str(drug_value if drug_value is not None else "")

        counts[(ward, drug_name)] += 1

    result = [
        {
            "ward": ward,
            "drug_name": drug_name,
            "prescription_count": int(count),
        }
        for (ward, drug_name), count in sorted(counts.items(), key=lambda item: (item[0][0], item[0][1]))
    ]
    return result


def _cross_join_total_count(medication_rows: list[dict[str, Any]], patient_count: int) -> int:
    eligible_rows = 0
    for row in medication_rows:
        parsed_date = _parse_prescribed_date(row.get("prescribed_date"))
        if parsed_date is not None and parsed_date >= "2024-01-01":
            eligible_rows += 1
    return eligible_rows * patient_count


def _normalize_query_output(
    agent_query_result: pd.DataFrame | list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if isinstance(agent_query_result, pd.DataFrame):
        records = agent_query_result.to_dict(orient="records")
    else:
        records = list(agent_query_result)

    normalized: list[dict[str, Any]] = []
    for row in records:
        ward = str(row.get("ward", "")).strip().upper()
        drug_name = _normalize_drug_name(row.get("drug_name"))
        try:
            count = int(float(row.get("prescription_count", 0)))
        except (TypeError, ValueError):
            continue

        if not ward or not drug_name:
            continue

        normalized.append(
            {
                "ward": ward,
                "drug_name": drug_name,
                "prescription_count": count,
            }
        )

    return normalized


def _rows_to_count_map(rows: list[dict[str, Any]]) -> dict[tuple[str, str], int]:
    output: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (str(row["ward"]).upper(), _normalize_drug_name(row["drug_name"]))
        output[key] = output.get(key, 0) + int(row["prescription_count"])
    return output


def _query_alignment_score(
    agent_rows: list[dict[str, Any]],
    expected_rows: list[dict[str, Any]],
) -> float:
    expected_map = _rows_to_count_map(expected_rows)
    agent_map = _rows_to_count_map(agent_rows)

    if not expected_map and not agent_map:
        return 1.0
    if not expected_map or not agent_map:
        return 0.0

    key_union = sorted(set(expected_map) | set(agent_map))
    per_key_scores: list[float] = []
    for key in key_union:
        expected = expected_map.get(key, 0)
        actual = agent_map.get(key, 0)
        per_key_scores.append(max(0.0, 1.0 - (abs(actual - expected) / max(1, expected))))

    return float(sum(per_key_scores) / max(1, len(per_key_scores)))


def _data_cleaning_score(agent_table: pd.DataFrame | list[dict[str, Any]]) -> float:
    if isinstance(agent_table, pd.DataFrame):
        rows = agent_table.to_dict(orient="records")
    else:
        rows = list(agent_table)

    if not rows:
        return 0.0

    total = len(rows)

    valid_patient_count = 0
    normalized_drug_count = 0
    numeric_dosage_count = 0
    normalized_date_count = 0

    for row in rows:
        patient_id = str(row.get("patient_id", "")).strip()
        if patient_id in _MEDICATION_VALID_PATIENT_IDS:
            valid_patient_count += 1

        drug_raw = row.get("drug_name")
        if isinstance(drug_raw, str) and drug_raw == drug_raw.strip().lower() and drug_raw.strip() != "":
            normalized_drug_count += 1

        if _parse_dosage_mg(row.get("dosage_mg")) is not None:
            numeric_dosage_count += 1

        parsed_date = _parse_prescribed_date(row.get("prescribed_date"))
        if isinstance(row.get("prescribed_date"), str) and parsed_date == row.get("prescribed_date"):
            normalized_date_count += 1

    row_count_score = max(
        0.0,
        1.0 - (abs(total - len(_MEDICATION_CLEAN_ROWS)) / max(1, len(_MEDICATION_CLEAN_ROWS))),
    )

    score = (
        0.30 * (valid_patient_count / total)
        + 0.25 * (normalized_drug_count / total)
        + 0.25 * (numeric_dosage_count / total)
        + 0.10 * (normalized_date_count / total)
        + 0.10 * row_count_score
    )

    return float(max(0.0, min(1.0, score)))


def _is_join_fixed(agent_rows: list[dict[str, Any]]) -> bool:
    if not agent_rows:
        return False

    total_count = sum(max(0, int(row.get("prescription_count", 0))) for row in agent_rows)
    inner_distance = abs(total_count - _MEDICATION_INNER_JOIN_TOTAL)
    cross_distance = abs(total_count - _MEDICATION_CROSS_JOIN_TOTAL)

    # A fixed INNER JOIN should be much closer to the inner-join baseline than the cross-join baseline.
    return inner_distance <= cross_distance


def score_medication_summary(
    agent_table: pd.DataFrame | list[dict[str, Any]],
    agent_query_result: pd.DataFrame | list[dict[str, Any]],
) -> float:
    """Score medication_summary with required partial-credit rubric.

    Required rubric:
    - JOIN fixed, data not fixed -> 0.5
    - Data fixed, JOIN not fixed -> 0.4
    - Both fixed -> 1.0
    """
    data_score = _data_cleaning_score(agent_table)
    data_fixed = data_score >= 0.95

    query_rows = _normalize_query_output(agent_query_result)
    join_fixed = _is_join_fixed(query_rows)

    query_full_score = _query_alignment_score(query_rows, MEDICATION_SUMMARY_GROUND_TRUTH_EXPECTED_RESULT)

    if data_fixed and join_fixed and query_full_score >= 0.90:
        return 1.0
    if join_fixed and not data_fixed:
        return 0.5
    if data_fixed and not join_fixed:
        return 0.4
    if data_fixed and join_fixed:
        return 1.0
    return 0.0


_MEDICATION_DATASET = seed_medication_summary_dataset()
_MEDICATION_PATIENT_ROWS = _MEDICATION_DATASET["patients"]
_MEDICATION_RAW_ROWS = _MEDICATION_DATASET["medications"]

_MEDICATION_PATIENT_WARD_MAP = {
    str(row["patient_id"]).strip(): str(row["ward"]).strip().upper() for row in _MEDICATION_PATIENT_ROWS
}
_MEDICATION_VALID_PATIENT_IDS = set(_MEDICATION_PATIENT_WARD_MAP.keys())

_MEDICATION_CLEAN_ROWS = _clean_medication_rows(_MEDICATION_RAW_ROWS, _MEDICATION_VALID_PATIENT_IDS)

MEDICATION_SUMMARY_GROUND_TRUTH_EXPECTED_RESULT = _aggregate_expected_result(
    _MEDICATION_CLEAN_ROWS,
    _MEDICATION_PATIENT_WARD_MAP,
    normalize_drug=True,
)

_MEDICATION_JOIN_ONLY_EXPECTED_RESULT = _aggregate_expected_result(
    _MEDICATION_RAW_ROWS,
    _MEDICATION_PATIENT_WARD_MAP,
    normalize_drug=False,
)

_MEDICATION_INNER_JOIN_TOTAL = sum(row["prescription_count"] for row in _MEDICATION_JOIN_ONLY_EXPECTED_RESULT)
_MEDICATION_CROSS_JOIN_TOTAL = _cross_join_total_count(_MEDICATION_RAW_ROWS, len(_MEDICATION_PATIENT_ROWS))


MEDICATION_SUMMARY_GROUND_TRUTH_CLEAN_SPEC: dict[str, Any] = {
    "source_tables": ["medications", "patients"],
    "cleaning_requirements": {
        "drug_name": "strip + lowercase",
        "dosage_mg": "extract numeric mg value from floats/strings",
        "patient_id": "remove orphan rows not present in patients",
        "prescribed_date": "convert unix timestamps and mixed date formats to YYYY-MM-DD",
    },
    "query_requirements": {
        "join_type": "INNER JOIN",
        "join_condition": "m.patient_id = p.patient_id",
        "group_by": ["p.ward", "m.drug_name"],
        "metric": "COUNT(*) AS prescription_count",
        "date_filter": "m.prescribed_date >= 2024-01-01",
    },
}


MEDICATION_SUMMARY_TASK_METADATA: dict[str, Any] = {
    "id": "medication_summary",
    "difficulty": "medium",
    "name": "Medication Summary",
    "scenario": (
        "The pharmacy dashboard is showing inflated medication counts. "
        "Clean the medications data and fix the JOIN query."
    ),
    "seed": {
        "patient_count": 120,
        "medication_count": 700,
        "orphan_ratio": 0.10,
        "unix_timestamp_ratio": 0.20,
        "random_seed": 42,
    },
    "broken_query": BROKEN_QUERY,
    "fixed_query": FIXED_QUERY,
    "patients_table": _MEDICATION_PATIENT_ROWS,
    "ground_truth_expected_result": MEDICATION_SUMMARY_GROUND_TRUTH_EXPECTED_RESULT,
    "ground_truth_clean_dataset_spec": MEDICATION_SUMMARY_GROUND_TRUTH_CLEAN_SPEC,
}


TASK = TaskSpec(
    id="medication_summary",
    name="Medication Summary",
    difficulty=Difficulty.MEDIUM,
    description=MEDICATION_SUMMARY_TASK_METADATA["scenario"],
    hints=[
        "CROSS JOIN is inflating counts across all wards.",
        "Normalize drug_name casing/whitespace before final aggregation.",
        "Remove orphaned medication rows whose patient_id is absent in patients.",
        "Convert Unix timestamps in prescribed_date to YYYY-MM-DD before filtering.",
    ],
    dirty_rows=_MEDICATION_RAW_ROWS,
    broken_sql=BROKEN_QUERY,
    expected_clean_rows=_MEDICATION_CLEAN_ROWS,
    expected_sql=FIXED_QUERY,
)
