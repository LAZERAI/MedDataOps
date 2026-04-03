CREATE TABLE IF NOT EXISTS admissions (
    admission_id SERIAL PRIMARY KEY,
    encounter_id BIGINT NOT NULL,
    patient_id BIGINT NOT NULL,
    admit_ts TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS lab_results (
    lab_result_id SERIAL PRIMARY KEY,
    encounter_id BIGINT NOT NULL,
    patient_id BIGINT NOT NULL,
    test_code TEXT NOT NULL,
    result_value NUMERIC,
    unit TEXT,
    taken_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS icu_events (
    icu_event_id SERIAL PRIMARY KEY,
    patient_id BIGINT NOT NULL,
    event_ts TIMESTAMP NOT NULL,
    heart_rate NUMERIC,
    spo2 NUMERIC
);
