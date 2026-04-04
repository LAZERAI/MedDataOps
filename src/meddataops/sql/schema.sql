CREATE TABLE IF NOT EXISTS patients (
    row_id BIGSERIAL PRIMARY KEY,
    patient_id TEXT,
    mrn TEXT,
    first_name TEXT,
    last_name TEXT,
    date_of_birth_raw TEXT,
    sex TEXT,
    admit_date_raw TEXT,
    discharge_date_raw TEXT,
    age_years_raw TEXT,
    primary_diagnosis TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS medications (
    row_id BIGSERIAL PRIMARY KEY,
    patient_id TEXT,
    encounter_id TEXT,
    medication_name TEXT,
    dose_amount_raw TEXT,
    dose_unit TEXT,
    route TEXT,
    frequency TEXT,
    ordered_datetime_raw TEXT,
    start_date_raw TEXT,
    end_date_raw TEXT,
    prescribing_clinician TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS lab_results (
    row_id BIGSERIAL PRIMARY KEY,
    patient_id TEXT,
    encounter_id TEXT,
    specimen_collected_at_raw TEXT,
    analyte_name TEXT,
    result_value_raw TEXT,
    result_unit TEXT,
    reference_range TEXT,
    abnormal_flag TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS icu_beds (
    row_id BIGSERIAL PRIMARY KEY,
    unit_name TEXT,
    bed_id TEXT,
    patient_id TEXT,
    occupancy_status TEXT,
    occupied_since_raw TEXT,
    expected_discharge_raw TEXT,
    acuity_level_raw TEXT,
    ventilator_required_raw TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
