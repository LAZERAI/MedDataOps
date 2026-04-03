INSERT INTO admissions (encounter_id, patient_id, admit_ts)
VALUES
    (5001, 1001, '2026-03-01 07:30:00'),
    (5002, 1002, '2026-03-01 07:40:00'),
    (5003, 1003, '2026-03-01 07:55:00')
ON CONFLICT DO NOTHING;

INSERT INTO lab_results (encounter_id, patient_id, test_code, result_value, unit, taken_at)
VALUES
    (5001, 1001, 'CRP', 12.5, 'mg/L', '2026-03-01 08:00:00'),
    (5002, 1002, 'WBC', 7.1, '10^9/L', '2026-03-01 08:05:00')
ON CONFLICT DO NOTHING;

INSERT INTO icu_events (patient_id, event_ts, heart_rate, spo2)
VALUES
    (2001, '2026-03-01 08:00:00', 101, 96),
    (2001, '2026-03-01 08:05:00', 98, 95),
    (2002, '2026-03-01 08:03:00', 110, 93)
ON CONFLICT DO NOTHING;
