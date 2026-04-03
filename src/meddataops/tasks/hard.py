from __future__ import annotations

from meddataops.models import Difficulty, TaskSpec


TASK = TaskSpec(
    id="hard",
    name="ICU Rolling Average Fix",
    difficulty=Difficulty.HARD,
    description="Standardize ICU telemetry rows and repair a broken window function query.",
    hints=[
        "Partition and order columns must match real ICU schema fields.",
        "Heart rate values should be normalized numeric values.",
    ],
    dirty_rows=[
        {
            "patient_id": " 2001",
            "event_ts": "2026-03-01 08:00:00 ",
            "heart_rate": " 101",
            "spo2": "96 %",
        },
        {
            "patient_id": "2001 ",
            "event_ts": "2026-03-01T08:05:00",
            "heart_rate": "98 ",
            "spo2": " 95%",
        },
        {
            "patient_id": "2002",
            "event_ts": "2026-03-01 08:03:00",
            "heart_rate": "110",
            "spo2": "93 %",
        },
    ],
    broken_sql=(
        "SELECT patient_id, event_ts, heart_rate, "
        "AVG(heart_rate) OVER ("
        "PARTITION BY patient "
        "ORDER BY event_time "
        "ROWS BETWEEN 2 PRECEDING AND CURRENT ROW"
        ") AS rolling_avg_hr "
        "FROM icu_events;"
    ),
    expected_clean_rows=[
        {
            "patient_id": 2001,
            "event_ts": "2026-03-01 08:00:00",
            "heart_rate": 101,
            "spo2": 96,
        },
        {
            "patient_id": 2001,
            "event_ts": "2026-03-01 08:05:00",
            "heart_rate": 98,
            "spo2": 95,
        },
        {
            "patient_id": 2002,
            "event_ts": "2026-03-01 08:03:00",
            "heart_rate": 110,
            "spo2": 93,
        },
    ],
    expected_sql=(
        "SELECT patient_id, event_ts, heart_rate, "
        "AVG(heart_rate) OVER ("
        "PARTITION BY patient_id "
        "ORDER BY event_ts "
        "ROWS BETWEEN 2 PRECEDING AND CURRENT ROW"
        ") AS rolling_avg_hr "
        "FROM icu_events;"
    ),
)
