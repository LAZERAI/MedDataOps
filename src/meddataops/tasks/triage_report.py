from __future__ import annotations

import random
from typing import Any

import pandas as pd
from faker import Faker

from meddataops.models import Difficulty, TaskSpec


BROKEN_QUERY = (
    "SELECT ward, COUNT(*) as patient_count\n"
    "FROM patients\n"
    "WHERE admission_date >= '2024-01-01\n"
    "GROUP BY ward\n"
    "ORDER BY patient_count DESC"
)


FIXED_QUERY = (
    "SELECT ward, COUNT(*) AS patient_count "
    "FROM patients "
    "WHERE admission_date >= DATE '2024-01-01' "
    "GROUP BY ward "
    "ORDER BY patient_count DESC, ward ASC"
)


WARD_VARIANTS: dict[str, list[str]] = {
    "ICU": ["ICU", "icu", "Icu"],
    "ER": ["ER", "er", "Er"],
    "MEDSURG": ["MEDSURG", "medsurg", "Medsurg"],
    "OBS": ["OBS", "obs", "Obs"],
}


def seed_triage_report_dataset(
    row_count: int = 500,
    duplicate_ratio: float = 0.15,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate a messy admissions dataset for triage_report.

    Properties:
    - ~15% duplicated patient rows.
    - admission_date stored as DD/MM/YYYY strings.
    - ward casing inconsistency.
    - age has NULLs and non-numeric values (e.g. N/A).
    """
    if row_count <= 0:
        return []

    faker = Faker()
    faker.seed_instance(seed)
    rng = random.Random(seed)

    unique_count = max(1, int(round(row_count * (1.0 - duplicate_ratio))))
    duplicate_count = max(0, row_count - unique_count)

    records: list[dict[str, Any]] = []
    for index in range(unique_count):
        canonical_ward = rng.choice(list(WARD_VARIANTS.keys()))
        ward = rng.choice(WARD_VARIANTS[canonical_ward])

        age_value: int | None | str
        age_roll = rng.random()
        if age_roll < 0.08:
            age_value = None
        elif age_roll < 0.11:
            age_value = "N/A"
        else:
            age_value = rng.randint(18, 95)

        admission_dt = faker.date_between(start_date="-2y", end_date="today")

        record = {
            "patient_id": f"P{100000 + index}",
            "full_name": faker.name(),
            "admission_date": admission_dt.strftime("%d/%m/%Y"),
            "ward": ward,
            "age": age_value,
            "triage_level": rng.choice(["critical", "urgent", "non-urgent"]),
        }
        records.append(record)

    duplicates: list[dict[str, Any]] = [records[rng.randrange(0, len(records))].copy() for _ in range(duplicate_count)]
    dataset = records + duplicates
    rng.shuffle(dataset)
    return dataset


def _normalize_agent_table(agent_table: pd.DataFrame | list[dict[str, Any]]) -> pd.DataFrame:
    if isinstance(agent_table, pd.DataFrame):
        df = agent_table.copy()
    else:
        df = pd.DataFrame(agent_table)

    if df.empty:
        return pd.DataFrame(columns=["patient_id", "admission_date", "ward", "age"])

    df = df.copy()
    df["patient_id"] = df.get("patient_id", pd.Series(dtype="string")).astype("string").str.strip()

    parsed_dates = pd.to_datetime(df.get("admission_date"), errors="coerce", dayfirst=True)
    df["admission_date"] = parsed_dates.dt.strftime("%Y-%m-%d")
    df.loc[parsed_dates.isna(), "admission_date"] = None

    ward_values = df.get("ward", pd.Series(dtype="string")).astype("string").str.strip().str.upper()
    df["ward"] = ward_values

    numeric_age = pd.to_numeric(df.get("age"), errors="coerce")
    if numeric_age.notna().any():
        median_age = float(numeric_age.median())
        numeric_age = numeric_age.fillna(median_age)
    df["age"] = numeric_age.round().astype("Int64")

    df = df.dropna(subset=["patient_id", "admission_date", "ward"])  # keep rows relevant for reporting
    df = df.drop_duplicates(subset=["patient_id"], keep="first")

    return df[["patient_id", "admission_date", "ward", "age"]].reset_index(drop=True)


def _ground_truth_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return _normalize_agent_table(rows)


def _ward_count_rows(clean_df: pd.DataFrame) -> list[dict[str, Any]]:
    if clean_df.empty:
        return []

    filtered = clean_df[clean_df["admission_date"] >= "2024-01-01"].copy()
    grouped = (
        filtered.groupby("ward", dropna=False)
        .size()
        .reset_index(name="patient_count")
        .sort_values(["patient_count", "ward"], ascending=[False, True])
    )
    grouped["patient_count"] = grouped["patient_count"].astype(int)
    return grouped.to_dict(orient="records")


def _row_sig(row: dict[str, Any], columns: list[str]) -> tuple[Any, ...]:
    signature: list[Any] = []
    for col in columns:
        value = row.get(col)
        if isinstance(value, str):
            stripped = value.strip()
            try:
                signature.append(int(stripped))
                continue
            except ValueError:
                pass
            try:
                signature.append(float(stripped))
                continue
            except ValueError:
                signature.append(stripped.lower())
        else:
            signature.append(value)
    return tuple(signature)


def score_triage_report(
    agent_table: pd.DataFrame | list[dict[str, Any]],
    agent_query_result: pd.DataFrame | list[dict[str, Any]],
) -> float:
    """Score triage_report output on a 0.0-1.0 scale.

    Blend:
    - 60% cleaning correctness (table quality).
    - 40% query-result correctness (ward counts).
    """
    cleaned_agent_df = _normalize_agent_table(agent_table)

    required_columns = ["patient_id", "admission_date", "ward", "age"]
    gt_columns = ["patient_id", "admission_date", "ward", "age"]

    if not cleaned_agent_df.empty:
        shared_ids = sorted(set(cleaned_agent_df["patient_id"]) & set(_TRIAGE_GROUND_TRUTH_DF["patient_id"]))
    else:
        shared_ids = []

    if not shared_ids:
        clean_score = 0.0
    else:
        agent_indexed = cleaned_agent_df.set_index("patient_id")
        gt_indexed = _TRIAGE_GROUND_TRUTH_DF.set_index("patient_id")

        matched_cells = 0
        total_cells = len(shared_ids) * (len(gt_columns) - 1)
        for pid in shared_ids:
            for column in ["admission_date", "ward", "age"]:
                if _row_sig(agent_indexed.loc[pid].to_dict(), [column]) == _row_sig(gt_indexed.loc[pid].to_dict(), [column]):
                    matched_cells += 1

        coverage = len(shared_ids) / max(1, len(_TRIAGE_GROUND_TRUTH_DF))
        cell_accuracy = matched_cells / max(1, total_cells)

        duplicate_penalty = max(0.0, (cleaned_agent_df.duplicated(subset=["patient_id"]).sum() / max(1, len(cleaned_agent_df))))
        clean_score = max(0.0, min(1.0, (0.7 * cell_accuracy) + (0.3 * coverage) - (0.2 * duplicate_penalty)))

    if isinstance(agent_query_result, pd.DataFrame):
        query_df = agent_query_result.copy()
    else:
        query_df = pd.DataFrame(agent_query_result)

    expected_df = pd.DataFrame(_TRIAGE_EXPECTED_QUERY_RESULT)
    if expected_df.empty and query_df.empty:
        query_score = 1.0
    elif expected_df.empty:
        query_score = 0.0
    else:
        required_q_cols = {"ward", "patient_count"}
        available_cols = set(query_df.columns)
        column_score = len(required_q_cols & available_cols) / len(required_q_cols)

        if required_q_cols.issubset(available_cols):
            normalized_query = query_df[["ward", "patient_count"]].copy()
            normalized_query["ward"] = normalized_query["ward"].astype("string").str.strip().str.upper()
            normalized_query["patient_count"] = pd.to_numeric(normalized_query["patient_count"], errors="coerce").fillna(0).astype(int)

            expected_map = {row["ward"]: int(row["patient_count"]) for row in _TRIAGE_EXPECTED_QUERY_RESULT}
            actual_map = {
                str(row["ward"]).upper(): int(row["patient_count"])
                for row in normalized_query.to_dict(orient="records")
            }

            per_ward = []
            for ward, expected_count in expected_map.items():
                actual_count = actual_map.get(ward)
                if actual_count is None:
                    per_ward.append(0.0)
                elif actual_count == expected_count:
                    per_ward.append(1.0)
                else:
                    per_ward.append(max(0.0, 1.0 - (abs(actual_count - expected_count) / max(1, expected_count))))

            match_score = sum(per_ward) / max(1, len(per_ward))
            row_count_score = max(
                0.0,
                1.0 - (abs(len(actual_map) - len(expected_map)) / max(1, len(expected_map))),
            )
            query_score = max(0.0, min(1.0, (0.75 * match_score) + (0.15 * row_count_score) + (0.10 * column_score)))
        else:
            query_score = 0.1 * column_score

    final_score = (0.6 * clean_score) + (0.4 * query_score)
    return float(round(max(0.0, min(1.0, final_score)), 6))


TRIAGE_GROUND_TRUTH_CLEAN_SPEC: dict[str, Any] = {
    "source_table": "patients",
    "dedupe_keys": ["patient_id"],
    "type_requirements": {
        "patient_id": "string",
        "admission_date": "date(YYYY-MM-DD)",
        "ward": "string_upper",
        "age": "int_nullable",
    },
    "null_handling": {
        "age": "median_impute",
        "admission_date": "drop_invalid",
    },
    "string_normalization": {
        "ward": "strip + uppercase",
    },
    "query_expectation": {
        "filter": "admission_date >= 2024-01-01",
        "group_by": ["ward"],
        "metric": "COUNT(*) as patient_count",
        "order": ["patient_count DESC", "ward ASC"],
    },
}


_TRIAGE_RAW_ROWS = seed_triage_report_dataset()
_TRIAGE_GROUND_TRUTH_DF = _ground_truth_dataframe(_TRIAGE_RAW_ROWS)
_TRIAGE_EXPECTED_QUERY_RESULT = _ward_count_rows(_TRIAGE_GROUND_TRUTH_DF)

TRIAGE_REPORT_TASK_METADATA: dict[str, Any] = {
    "id": "triage_report",
    "difficulty": "easy",
    "name": "Morning Triage Report",
    "scenario": (
        "The morning triage report is broken. Clean the patient admissions data "
        "and fix the query that counts patients by ward."
    ),
    "row_count": 500,
    "duplicate_ratio": 0.15,
    "broken_query": BROKEN_QUERY,
    "fixed_query": FIXED_QUERY,
    "ground_truth_clean_dataset_spec": TRIAGE_GROUND_TRUTH_CLEAN_SPEC,
}


TASK = TaskSpec(
    id="triage_report",
    name="Morning Triage Report",
    difficulty=Difficulty.EASY,
    description=TRIAGE_REPORT_TASK_METADATA["scenario"],
    hints=[
        "Standardize ward strings before grouping.",
        "admission_date is DD/MM/YYYY and must be parsed to date.",
        "Handle NULL/N/A age values before finalizing cleaned table.",
    ],
    dirty_rows=_TRIAGE_RAW_ROWS,
    broken_sql=BROKEN_QUERY,
    expected_clean_rows=_TRIAGE_GROUND_TRUTH_DF.to_dict(orient="records"),
    expected_sql=FIXED_QUERY,
)
