from __future__ import annotations

from meddataops.models import Difficulty, TaskSpec


TASK = TaskSpec(
    id="medium",
    name="Lab Result Join Repair",
    difficulty=Difficulty.MEDIUM,
    description="Normalize lab rows (units and timestamps) and repair a bad JOIN condition.",
    hints=[
        "Result values should be numeric where possible.",
        "Use encounter_id consistently across both tables.",
    ],
    dirty_rows=[
        {
            "encounter_id": " 5001",
            "test_code": "CRP",
            "result_value": " 12.5",
            "unit": " mg/L ",
            "taken_at": "2026/03/01 08:00",
        },
        {
            "encounter_id": "5002 ",
            "test_code": "WBC",
            "result_value": " 7.1 ",
            "unit": "10^9/L",
            "taken_at": "2026-03-01T08:05:00",
        },
    ],
    broken_sql=(
        "SELECT lr.encounter_id, lr.test_code, a.admit_ts "
        "FROM lab_results lr "
        "JOIN admissions a ON lr.encounter_id = a.encounter "
        "WHERE lr.result_value IS NOT NULL;"
    ),
    expected_clean_rows=[
        {
            "encounter_id": 5001,
            "test_code": "crp",
            "result_value": 12.5,
            "unit": "mg/l",
            "taken_at": "2026-03-01 08:00",
        },
        {
            "encounter_id": 5002,
            "test_code": "wbc",
            "result_value": 7.1,
            "unit": "10^9/l",
            "taken_at": "2026-03-01 08:05",
        },
    ],
    expected_sql=(
        "SELECT lr.encounter_id, lr.test_code, a.admit_ts "
        "FROM lab_results lr "
        "JOIN admissions a ON lr.encounter_id = a.encounter_id "
        "WHERE lr.result_value IS NOT NULL;"
    ),
)
