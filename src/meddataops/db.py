from __future__ import annotations

import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Literal, Sequence

import pandas as pd
import psycopg2
from psycopg2 import Error as Psycopg2Error
from psycopg2 import sql
from psycopg2.extras import RealDictCursor, execute_values
from psycopg2.extensions import connection as PsycopgConnection
from psycopg2.pool import SimpleConnectionPool

from meddataops.models import QueryCheckResult


SeedVersion = Literal["messy", "clean"]


@dataclass(frozen=True)
class QueryExecutionResult:
    """Structured response for executed analytical SQL queries."""

    dataframe: pd.DataFrame
    explain_plan: list[dict[str, Any]] | None
    total_cost: float | None
    planning_time_ms: float | None
    execution_time_ms: float | None


class PostgresDataManager:
    """Database manager for MedDataOps clinical data workloads.

    Responsibilities:
    - Manage pooled PostgreSQL connections.
    - Create baseline schema for patients, medications, lab_results, and icu_beds.
    - Load messy and clean seed variants for each dataset.
    - Create per-episode isolated working tables using a session-id prefix.
    - Execute safe read-only queries and return pandas DataFrames.
    - Capture EXPLAIN ANALYZE plans and extract query cost metrics.
    - Clean up episode tables after use.
    """

    TABLE_COLUMNS: dict[str, list[str]] = {
        "patients": [
            "patient_id",
            "mrn",
            "first_name",
            "last_name",
            "date_of_birth_raw",
            "sex",
            "admit_date_raw",
            "discharge_date_raw",
            "age_years_raw",
            "primary_diagnosis",
        ],
        "medications": [
            "patient_id",
            "encounter_id",
            "medication_name",
            "dose_amount_raw",
            "dose_unit",
            "route",
            "frequency",
            "ordered_datetime_raw",
            "start_date_raw",
            "end_date_raw",
            "prescribing_clinician",
        ],
        "lab_results": [
            "patient_id",
            "encounter_id",
            "specimen_collected_at_raw",
            "analyte_name",
            "result_value_raw",
            "result_unit",
            "reference_range",
            "abnormal_flag",
        ],
        "icu_beds": [
            "unit_name",
            "bed_id",
            "patient_id",
            "occupancy_status",
            "occupied_since_raw",
            "expected_discharge_raw",
            "acuity_level_raw",
            "ventilator_required_raw",
        ],
    }

    SCHEMA_DDL: tuple[str, ...] = (
        """
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
        """,
        """
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
        """,
        """
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
        """,
        """
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
        """,
    )

    SEED_DATA: dict[str, dict[SeedVersion, list[dict[str, Any]]]] = {
        "patients": {
            "messy": [
                {
                    "patient_id": " 1001",
                    "mrn": "MRN-0001001",
                    "first_name": "Maria",
                    "last_name": "Lopez",
                    "date_of_birth_raw": "12/31/1977",
                    "sex": "F",
                    "admit_date_raw": "2026/03/05",
                    "discharge_date_raw": "2026-03-08",
                    "age_years_raw": "49",
                    "primary_diagnosis": "CHF exacerbation",
                },
                {
                    "patient_id": "1002",
                    "mrn": "MRN-0001002",
                    "first_name": "Noah",
                    "last_name": "Patel",
                    "date_of_birth_raw": "03-07-1981",
                    "sex": "M",
                    "admit_date_raw": "07-03-2026",
                    "discharge_date_raw": None,
                    "age_years_raw": "forty-five",
                    "primary_diagnosis": "Sepsis",
                },
                {
                    "patient_id": "1002",
                    "mrn": "MRN-0001002",
                    "first_name": "Noah",
                    "last_name": "Patel",
                    "date_of_birth_raw": "03-07-1981",
                    "sex": "M",
                    "admit_date_raw": "07-03-2026",
                    "discharge_date_raw": None,
                    "age_years_raw": "45",
                    "primary_diagnosis": "Sepsis",
                },
                {
                    "patient_id": None,
                    "mrn": "MRN-0001003",
                    "first_name": "Aisha",
                    "last_name": "Khan",
                    "date_of_birth_raw": "1989-11-15",
                    "sex": "F",
                    "admit_date_raw": "2026-03-09",
                    "discharge_date_raw": "2026-03-11",
                    "age_years_raw": "36",
                    "primary_diagnosis": "Pneumonia",
                },
            ],
            "clean": [
                {
                    "patient_id": "1001",
                    "mrn": "MRN-0001001",
                    "first_name": "Maria",
                    "last_name": "Lopez",
                    "date_of_birth_raw": "1977-12-31",
                    "sex": "F",
                    "admit_date_raw": "2026-03-05",
                    "discharge_date_raw": "2026-03-08",
                    "age_years_raw": "49",
                    "primary_diagnosis": "chf exacerbation",
                },
                {
                    "patient_id": "1002",
                    "mrn": "MRN-0001002",
                    "first_name": "Noah",
                    "last_name": "Patel",
                    "date_of_birth_raw": "1981-03-07",
                    "sex": "M",
                    "admit_date_raw": "2026-03-07",
                    "discharge_date_raw": None,
                    "age_years_raw": "45",
                    "primary_diagnosis": "sepsis",
                },
                {
                    "patient_id": "1003",
                    "mrn": "MRN-0001003",
                    "first_name": "Aisha",
                    "last_name": "Khan",
                    "date_of_birth_raw": "1989-11-15",
                    "sex": "F",
                    "admit_date_raw": "2026-03-09",
                    "discharge_date_raw": "2026-03-11",
                    "age_years_raw": "36",
                    "primary_diagnosis": "pneumonia",
                },
            ],
        },
        "medications": {
            "messy": [
                {
                    "patient_id": "1001",
                    "encounter_id": "E-1001-A",
                    "medication_name": " metFORMIN  ",
                    "dose_amount_raw": "five hundred",
                    "dose_unit": "mg",
                    "route": "PO",
                    "frequency": "BID",
                    "ordered_datetime_raw": "03/05/2026 08:30",
                    "start_date_raw": "2026-03-05",
                    "end_date_raw": "2026-03-10",
                    "prescribing_clinician": "Dr. Stone",
                },
                {
                    "patient_id": "1001",
                    "encounter_id": "E-1001-A",
                    "medication_name": "Aspirin",
                    "dose_amount_raw": "81mg",
                    "dose_unit": "mg",
                    "route": "PO",
                    "frequency": "daily",
                    "ordered_datetime_raw": "2026-03-05T09:00:00",
                    "start_date_raw": "03-05-2026",
                    "end_date_raw": None,
                    "prescribing_clinician": "Dr. Stone",
                },
                {
                    "patient_id": "1001",
                    "encounter_id": "E-1001-A",
                    "medication_name": "aspirin ",
                    "dose_amount_raw": "81",
                    "dose_unit": "MG",
                    "route": "po",
                    "frequency": "daily",
                    "ordered_datetime_raw": "2026-03-05 09:00",
                    "start_date_raw": "2026-03-05",
                    "end_date_raw": None,
                    "prescribing_clinician": "Dr. Stone",
                },
                {
                    "patient_id": "1002",
                    "encounter_id": "E-1002-A",
                    "medication_name": None,
                    "dose_amount_raw": "2",
                    "dose_unit": "g",
                    "route": "IV",
                    "frequency": "q8h",
                    "ordered_datetime_raw": "2026-13-01 10:00",
                    "start_date_raw": "2026/03/07",
                    "end_date_raw": "2026/03/09",
                    "prescribing_clinician": "Dr. Xu",
                },
            ],
            "clean": [
                {
                    "patient_id": "1001",
                    "encounter_id": "E-1001-A",
                    "medication_name": "metformin",
                    "dose_amount_raw": "500",
                    "dose_unit": "mg",
                    "route": "po",
                    "frequency": "bid",
                    "ordered_datetime_raw": "2026-03-05 08:30:00",
                    "start_date_raw": "2026-03-05",
                    "end_date_raw": "2026-03-10",
                    "prescribing_clinician": "Dr. Stone",
                },
                {
                    "patient_id": "1001",
                    "encounter_id": "E-1001-A",
                    "medication_name": "aspirin",
                    "dose_amount_raw": "81",
                    "dose_unit": "mg",
                    "route": "po",
                    "frequency": "daily",
                    "ordered_datetime_raw": "2026-03-05 09:00:00",
                    "start_date_raw": "2026-03-05",
                    "end_date_raw": None,
                    "prescribing_clinician": "Dr. Stone",
                },
                {
                    "patient_id": "1002",
                    "encounter_id": "E-1002-A",
                    "medication_name": "vancomycin",
                    "dose_amount_raw": "2",
                    "dose_unit": "g",
                    "route": "iv",
                    "frequency": "q8h",
                    "ordered_datetime_raw": "2026-03-07 10:00:00",
                    "start_date_raw": "2026-03-07",
                    "end_date_raw": "2026-03-09",
                    "prescribing_clinician": "Dr. Xu",
                },
            ],
        },
        "lab_results": {
            "messy": [
                {
                    "patient_id": "1001",
                    "encounter_id": "E-1001-A",
                    "specimen_collected_at_raw": "2026/03/05 07:59",
                    "analyte_name": "CRP",
                    "result_value_raw": " 12.5",
                    "result_unit": " mg/L ",
                    "reference_range": "< 5",
                    "abnormal_flag": "H",
                },
                {
                    "patient_id": "1002",
                    "encounter_id": "E-1002-A",
                    "specimen_collected_at_raw": "03-07-2026 09:00",
                    "analyte_name": "WBC",
                    "result_value_raw": "7,2",
                    "result_unit": "10^9/L",
                    "reference_range": "4-11",
                    "abnormal_flag": "N",
                },
                {
                    "patient_id": "1002",
                    "encounter_id": "E-1002-A",
                    "specimen_collected_at_raw": "03-07-2026 09:00",
                    "analyte_name": "WBC",
                    "result_value_raw": "7,2",
                    "result_unit": "10^9/L",
                    "reference_range": "4-11",
                    "abnormal_flag": "N",
                },
                {
                    "patient_id": "1003",
                    "encounter_id": "E-1003-A",
                    "specimen_collected_at_raw": "2026-03-09T06:45:00",
                    "analyte_name": "Lactate",
                    "result_value_raw": "not done",
                    "result_unit": "mmol/L",
                    "reference_range": "0.5-2.0",
                    "abnormal_flag": None,
                },
            ],
            "clean": [
                {
                    "patient_id": "1001",
                    "encounter_id": "E-1001-A",
                    "specimen_collected_at_raw": "2026-03-05 07:59:00",
                    "analyte_name": "crp",
                    "result_value_raw": "12.5",
                    "result_unit": "mg/l",
                    "reference_range": "<5",
                    "abnormal_flag": "H",
                },
                {
                    "patient_id": "1002",
                    "encounter_id": "E-1002-A",
                    "specimen_collected_at_raw": "2026-03-07 09:00:00",
                    "analyte_name": "wbc",
                    "result_value_raw": "7.2",
                    "result_unit": "10^9/l",
                    "reference_range": "4-11",
                    "abnormal_flag": "N",
                },
                {
                    "patient_id": "1003",
                    "encounter_id": "E-1003-A",
                    "specimen_collected_at_raw": "2026-03-09 06:45:00",
                    "analyte_name": "lactate",
                    "result_value_raw": None,
                    "result_unit": "mmol/l",
                    "reference_range": "0.5-2.0",
                    "abnormal_flag": "U",
                },
            ],
        },
        "icu_beds": {
            "messy": [
                {
                    "unit_name": "ICU-A",
                    "bed_id": "A-01",
                    "patient_id": "1001",
                    "occupancy_status": "occupied",
                    "occupied_since_raw": "2026/03/05 10:15",
                    "expected_discharge_raw": "2026-03-08",
                    "acuity_level_raw": "high",
                    "ventilator_required_raw": "Y",
                },
                {
                    "unit_name": "ICU-A",
                    "bed_id": "A-02",
                    "patient_id": None,
                    "occupancy_status": "VACANT",
                    "occupied_since_raw": None,
                    "expected_discharge_raw": "",
                    "acuity_level_raw": "0",
                    "ventilator_required_raw": "no",
                },
                {
                    "unit_name": "ICU-B",
                    "bed_id": "B-03",
                    "patient_id": "1003",
                    "occupancy_status": "occupied",
                    "occupied_since_raw": "03-09-2026 05:55",
                    "expected_discharge_raw": "09/03/2026",
                    "acuity_level_raw": "3",
                    "ventilator_required_raw": "true",
                },
                {
                    "unit_name": "ICU-B",
                    "bed_id": "B-03",
                    "patient_id": "1003",
                    "occupancy_status": "occupied",
                    "occupied_since_raw": "03-09-2026 05:55",
                    "expected_discharge_raw": "09/03/2026",
                    "acuity_level_raw": "3",
                    "ventilator_required_raw": "true",
                },
            ],
            "clean": [
                {
                    "unit_name": "icu-a",
                    "bed_id": "A-01",
                    "patient_id": "1001",
                    "occupancy_status": "occupied",
                    "occupied_since_raw": "2026-03-05 10:15:00",
                    "expected_discharge_raw": "2026-03-08 00:00:00",
                    "acuity_level_raw": "3",
                    "ventilator_required_raw": "true",
                },
                {
                    "unit_name": "icu-a",
                    "bed_id": "A-02",
                    "patient_id": None,
                    "occupancy_status": "vacant",
                    "occupied_since_raw": None,
                    "expected_discharge_raw": None,
                    "acuity_level_raw": "0",
                    "ventilator_required_raw": "false",
                },
                {
                    "unit_name": "icu-b",
                    "bed_id": "B-03",
                    "patient_id": "1003",
                    "occupancy_status": "occupied",
                    "occupied_since_raw": "2026-03-09 05:55:00",
                    "expected_discharge_raw": "2026-03-09 18:00:00",
                    "acuity_level_raw": "3",
                    "ventilator_required_raw": "true",
                },
            ],
        },
    }

    def __init__(
        self,
        minconn: int = 1,
        maxconn: int = 8,
        *,
        host: str | None = None,
        dbname: str | None = None,
        user: str | None = None,
        password: str | None = None,
        port: int | None = None,
    ) -> None:
        """Initialize pooled PostgreSQL access for MedDataOps."""
        kwargs = self._connection_kwargs(host, dbname, user, password, port)
        try:
            self._pool = SimpleConnectionPool(minconn=minconn, maxconn=maxconn, **kwargs)
        except Psycopg2Error as exc:
            raise RuntimeError(f"Failed to initialize PostgreSQL connection pool: {exc}") from exc

        self._episode_tables: dict[str, list[str]] = {}

    def close(self) -> None:
        """Close all pooled PostgreSQL connections."""
        self._pool.closeall()

    def setup_schema(self) -> None:
        """Create baseline clinical tables required by MedDataOps."""
        with self._borrow_connection() as conn:
            with conn.cursor() as cur:
                for ddl in self.SCHEMA_DDL:
                    cur.execute(ddl)

    def load_patients_seed(self, version: SeedVersion = "messy") -> int:
        """Load patients seed data for the requested version."""
        return self._load_seed("patients", version)

    def load_medications_seed(self, version: SeedVersion = "messy") -> int:
        """Load medications seed data for the requested version."""
        return self._load_seed("medications", version)

    def load_lab_results_seed(self, version: SeedVersion = "messy") -> int:
        """Load lab_results seed data for the requested version."""
        return self._load_seed("lab_results", version)

    def load_icu_beds_seed(self, version: SeedVersion = "messy") -> int:
        """Load icu_beds seed data for the requested version."""
        return self._load_seed("icu_beds", version)

    def load_all_seed_data(self, version: SeedVersion = "messy") -> dict[str, int]:
        """Load all dataset seeds (messy or clean) into base tables."""
        return {
            "patients": self.load_patients_seed(version),
            "medications": self.load_medications_seed(version),
            "lab_results": self.load_lab_results_seed(version),
            "icu_beds": self.load_icu_beds_seed(version),
        }

    def create_episode_working_tables(self, session_id: str, version: SeedVersion = "messy") -> dict[str, str]:
        """Create isolated per-episode working tables prefixed by session_id.

        Returns a mapping of base table name to working table name.
        """
        safe_session = self._sanitize_session_id(session_id)
        working_names = {
            table: f"{safe_session}_{table}_work"
            for table in ("patients", "medications", "lab_results", "icu_beds")
        }

        with self._borrow_connection() as conn:
            with conn.cursor() as cur:
                for base_table, working_table in working_names.items():
                    cur.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(working_table)))
                    cur.execute(
                        sql.SQL("CREATE TABLE {} (LIKE {} INCLUDING DEFAULTS)").format(
                            sql.Identifier(working_table),
                            sql.Identifier(base_table),
                        )
                    )
                    self._insert_rows(cur, working_table, self.TABLE_COLUMNS[base_table], self.SEED_DATA[base_table][version])

        self._episode_tables[safe_session] = list(working_names.values())
        return working_names

    def run_query(
        self,
        query: str,
        params: Sequence[Any] | None = None,
        *,
        include_explain: bool = True,
    ) -> QueryExecutionResult:
        """Execute a safe read-only SQL query and return pandas results + plan metrics."""
        safe_query = self._validate_read_query(query)
        explain_plan: list[dict[str, Any]] | None = None
        total_cost: float | None = None
        planning_time_ms: float | None = None
        execution_time_ms: float | None = None

        with self._borrow_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if include_explain:
                    cur.execute(f"EXPLAIN (ANALYZE, FORMAT JSON, BUFFERS) {safe_query}", params)
                    plan_row = cur.fetchone() or {}
                    raw_plan = plan_row.get("QUERY PLAN")
                    explain_plan = raw_plan if isinstance(raw_plan, list) else None
                    total_cost = self.extract_total_cost(raw_plan)
                    planning_time_ms = self.extract_metric(raw_plan, "Planning Time")
                    execution_time_ms = self.extract_metric(raw_plan, "Execution Time")

                cur.execute(safe_query, params)
                rows = cur.fetchall() if cur.description is not None else []
                dataframe = pd.DataFrame(rows)

        return QueryExecutionResult(
            dataframe=dataframe,
            explain_plan=explain_plan,
            total_cost=total_cost,
            planning_time_ms=planning_time_ms,
            execution_time_ms=execution_time_ms,
        )

    def extract_total_cost(self, explain_output: Any) -> float | None:
        """Parse EXPLAIN ANALYZE output and return total cost as float."""
        return self.extract_metric(explain_output, "Total Cost")

    def extract_metric(self, explain_output: Any, metric_name: str) -> float | None:
        """Recursively parse a numeric metric from EXPLAIN JSON or text output."""
        if explain_output is None:
            return None

        def _extract(node: Any) -> float | None:
            if isinstance(node, dict):
                if metric_name in node:
                    try:
                        return float(node[metric_name])
                    except (TypeError, ValueError):
                        return None

                for value in node.values():
                    found = _extract(value)
                    if found is not None:
                        return found
                return None

            if isinstance(node, list):
                for item in node:
                    found = _extract(item)
                    if found is not None:
                        return found
                return None

            if isinstance(node, str):
                pattern = rf"{re.escape(metric_name)}\s*=\s*([0-9]+(?:\.[0-9]+)?)"
                match = re.search(pattern, node, flags=re.IGNORECASE)
                if match:
                    try:
                        return float(match.group(1))
                    except ValueError:
                        return None
            return None

        return _extract(explain_output)

    def cleanup_episode(self, session_id: str) -> None:
        """Drop per-episode working tables created for the provided session_id."""
        safe_session = self._sanitize_session_id(session_id)
        tables = self._episode_tables.get(safe_session, [])

        with self._borrow_connection() as conn:
            with conn.cursor() as cur:
                if not tables:
                    cur.execute(
                        """
                        SELECT tablename
                        FROM pg_tables
                        WHERE schemaname = 'public' AND tablename LIKE %s
                        """,
                        (f"{safe_session}_%",),
                    )
                    tables = [row[0] for row in cur.fetchall()]

                for table_name in tables:
                    cur.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(table_name)))

        self._episode_tables.pop(safe_session, None)

    def _load_seed(self, table_name: str, version: SeedVersion) -> int:
        if table_name not in self.SEED_DATA:
            raise ValueError(f"Unsupported table for seed loading: {table_name}")

        rows = self.SEED_DATA[table_name][version]
        columns = self.TABLE_COLUMNS[table_name]

        with self._borrow_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql.SQL("TRUNCATE TABLE {}").format(sql.Identifier(table_name)))
                self._insert_rows(cur, table_name, columns, rows)

        return len(rows)

    def _insert_rows(
        self,
        cur: Any,
        table_name: str,
        columns: list[str],
        rows: list[dict[str, Any]],
    ) -> None:
        if not rows:
            return

        values: list[tuple[Any, ...]] = [tuple(row.get(column) for column in columns) for row in rows]
        query = sql.SQL("INSERT INTO {} ({}) VALUES %s").format(
            sql.Identifier(table_name),
            sql.SQL(", ").join(sql.Identifier(column) for column in columns),
        )
        execute_values(cur, query.as_string(cur.connection), values)

    def _validate_read_query(self, query: str) -> str:
        candidate = query.strip().rstrip(";")
        if not candidate:
            raise ValueError("Query cannot be empty.")

        lowered = candidate.lower()
        if not (lowered.startswith("select") or lowered.startswith("with")):
            raise ValueError("Only read-only SELECT/WITH queries are allowed.")

        disallowed = [
            " insert ",
            " update ",
            " delete ",
            " drop ",
            " alter ",
            " truncate ",
            " grant ",
            " revoke ",
            " create ",
        ]
        lowered_padded = f" {lowered} "
        if any(token in lowered_padded for token in disallowed):
            raise ValueError("Query contains disallowed write or DDL statements.")

        if ";" in candidate:
            raise ValueError("Multiple SQL statements are not allowed.")

        return candidate

    def _sanitize_session_id(self, session_id: str) -> str:
        if not session_id or not isinstance(session_id, str):
            raise ValueError("session_id must be a non-empty string.")

        safe = re.sub(r"[^a-zA-Z0-9_]", "_", session_id.strip())
        if not safe:
            raise ValueError("session_id must contain at least one alphanumeric character.")

        return safe[:48].lower()

    def _connection_kwargs(
        self,
        host: str | None,
        dbname: str | None,
        user: str | None,
        password: str | None,
        port: int | None,
    ) -> dict[str, Any]:
        resolved = {
            "host": host or os.getenv("POSTGRES_HOST"),
            "dbname": dbname or os.getenv("POSTGRES_DB"),
            "user": user or os.getenv("POSTGRES_USER"),
            "password": password or os.getenv("POSTGRES_PASSWORD"),
            "port": port or int(os.getenv("POSTGRES_PORT", "5432")),
            "connect_timeout": 5,
        }

        missing = [key for key in ("host", "dbname", "user", "password") if not resolved[key]]
        if missing:
            env_name_map = {
                "host": "POSTGRES_HOST",
                "dbname": "POSTGRES_DB",
                "user": "POSTGRES_USER",
                "password": "POSTGRES_PASSWORD",
            }
            names = ", ".join(env_name_map[name] for name in missing)
            raise RuntimeError(f"Missing required PostgreSQL connection settings: {names}")

        return resolved

    @contextmanager
    def _borrow_connection(self) -> Iterator[PsycopgConnection]:
        conn = self._pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)


class PostgresBackend:
    """Backward-compatible query validator wrapper for legacy callers."""

    def __init__(self, dsn: str | None = None) -> None:
        self._dsn = dsn

    def validate_query(self, query: str) -> QueryCheckResult:
        stripped = query.strip()
        if not stripped.lower().startswith("select"):
            return QueryCheckResult(success=False, error="Only SELECT queries are allowed.")

        try:
            if self._dsn:
                conn = psycopg2.connect(self._dsn)
            else:
                kwargs = {
                    "host": os.getenv("POSTGRES_HOST"),
                    "dbname": os.getenv("POSTGRES_DB"),
                    "user": os.getenv("POSTGRES_USER"),
                    "password": os.getenv("POSTGRES_PASSWORD"),
                    "port": int(os.getenv("POSTGRES_PORT", "5432")),
                    "connect_timeout": 5,
                }

                if not kwargs["host"] or not kwargs["dbname"] or not kwargs["user"] or not kwargs["password"]:
                    return QueryCheckResult(
                        success=False,
                        error="Missing PostgreSQL connection settings (POSTGRES_HOST/POSTGRES_DB/POSTGRES_USER/POSTGRES_PASSWORD).",
                    )

                conn = psycopg2.connect(**kwargs)

            with conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(stripped)
                    rows = cur.fetchmany(5) if cur.description is not None else []

            return QueryCheckResult(
                success=True,
                sample_row_count=len(rows),
                sample_rows=[dict(row) for row in rows],
            )
        except Exception as exc:
            return QueryCheckResult(success=False, error=str(exc))
