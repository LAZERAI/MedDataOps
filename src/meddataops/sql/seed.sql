INSERT INTO patients (
    patient_id,
    mrn,
    first_name,
    last_name,
    date_of_birth_raw,
    sex,
    admit_date_raw,
    discharge_date_raw,
    age_years_raw,
    primary_diagnosis
)
VALUES
    ('1001', 'MRN-0001001', 'Maria', 'Lopez', '12/31/1977', 'F', '2026/03/05', '2026-03-08', '49', 'CHF exacerbation'),
    ('1002', 'MRN-0001002', 'Noah', 'Patel', '03-07-1981', 'M', '07-03-2026', NULL, 'forty-five', 'Sepsis'),
    ('1002', 'MRN-0001002', 'Noah', 'Patel', '03-07-1981', 'M', '07-03-2026', NULL, '45', 'Sepsis'),
    (NULL, 'MRN-0001003', 'Aisha', 'Khan', '1989-11-15', 'F', '2026-03-09', '2026-03-11', '36', 'Pneumonia');

INSERT INTO medications (
    patient_id,
    encounter_id,
    medication_name,
    dose_amount_raw,
    dose_unit,
    route,
    frequency,
    ordered_datetime_raw,
    start_date_raw,
    end_date_raw,
    prescribing_clinician
)
VALUES
    ('1001', 'E-1001-A', ' metFORMIN  ', 'five hundred', 'mg', 'PO', 'BID', '03/05/2026 08:30', '2026-03-05', '2026-03-10', 'Dr. Stone'),
    ('1001', 'E-1001-A', 'Aspirin', '81mg', 'mg', 'PO', 'daily', '2026-03-05T09:00:00', '03-05-2026', NULL, 'Dr. Stone'),
    ('1001', 'E-1001-A', 'aspirin ', '81', 'MG', 'po', 'daily', '2026-03-05 09:00', '2026-03-05', NULL, 'Dr. Stone'),
    ('1002', 'E-1002-A', NULL, '2', 'g', 'IV', 'q8h', '2026-13-01 10:00', '2026/03/07', '2026/03/09', 'Dr. Xu');

INSERT INTO lab_results (
    patient_id,
    encounter_id,
    specimen_collected_at_raw,
    analyte_name,
    result_value_raw,
    result_unit,
    reference_range,
    abnormal_flag
)
VALUES
    ('1001', 'E-1001-A', '2026/03/05 07:59', 'CRP', ' 12.5', ' mg/L ', '< 5', 'H'),
    ('1002', 'E-1002-A', '03-07-2026 09:00', 'WBC', '7,2', '10^9/L', '4-11', 'N'),
    ('1002', 'E-1002-A', '03-07-2026 09:00', 'WBC', '7,2', '10^9/L', '4-11', 'N'),
    ('1003', 'E-1003-A', '2026-03-09T06:45:00', 'Lactate', 'not done', 'mmol/L', '0.5-2.0', NULL);

INSERT INTO icu_beds (
    unit_name,
    bed_id,
    patient_id,
    occupancy_status,
    occupied_since_raw,
    expected_discharge_raw,
    acuity_level_raw,
    ventilator_required_raw
)
VALUES
    ('ICU-A', 'A-01', '1001', 'occupied', '2026/03/05 10:15', '2026-03-08', 'high', 'Y'),
    ('ICU-A', 'A-02', NULL, 'VACANT', NULL, '', '0', 'no'),
    ('ICU-B', 'B-03', '1003', 'occupied', '03-09-2026 05:55', '09/03/2026', '3', 'true'),
    ('ICU-B', 'B-03', '1003', 'occupied', '03-09-2026 05:55', '09/03/2026', '3', 'true');
