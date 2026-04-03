from __future__ import annotations

import json
import os

import psycopg2
from psycopg2.extras import Json

from meddataops.db import PostgresDataManager
from meddataops.tasks import get_task, list_tasks


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"Environment variable {name} must be an integer.") from exc


def main() -> None:
    host = os.getenv("POSTGRES_HOST", "127.0.0.1")
    dbname = os.getenv("POSTGRES_DB", "meddataops")
    user = os.getenv("POSTGRES_USER", "meddataops")
    password = os.getenv("POSTGRES_PASSWORD", "meddataops")
    port = _int_env("POSTGRES_PORT", 5432)
    seed_version = os.getenv("MEDDATAOPS_SEED_VERSION", "messy").strip().lower()

    if seed_version not in {"messy", "clean"}:
        raise RuntimeError("MEDDATAOPS_SEED_VERSION must be either 'messy' or 'clean'.")

    manager = PostgresDataManager(
        minconn=1,
        maxconn=4,
        host=host,
        dbname=dbname,
        user=user,
        password=password,
        port=port,
    )
    try:
        manager.setup_schema()
        seeded_counts = manager.load_all_seed_data(version=seed_version)
    finally:
        manager.close()

    task_seed_counts: dict[str, int] = {}
    with psycopg2.connect(
        host=host,
        dbname=dbname,
        user=user,
        password=password,
        port=port,
        connect_timeout=5,
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS meddataops_task_seed (
                    task_id TEXT PRIMARY KEY,
                    task_name TEXT NOT NULL,
                    dirty_rows JSONB NOT NULL,
                    broken_sql TEXT NOT NULL,
                    expected_sql TEXT NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )

            for task_id in list_tasks():
                task = get_task(task_id)
                task_seed_counts[task.id] = len(task.dirty_rows)
                cur.execute(
                    """
                    INSERT INTO meddataops_task_seed (
                        task_id,
                        task_name,
                        dirty_rows,
                        broken_sql,
                        expected_sql,
                        updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (task_id) DO UPDATE SET
                        task_name = EXCLUDED.task_name,
                        dirty_rows = EXCLUDED.dirty_rows,
                        broken_sql = EXCLUDED.broken_sql,
                        expected_sql = EXCLUDED.expected_sql,
                        updated_at = NOW();
                    """,
                    (
                        task.id,
                        task.name,
                        Json(task.dirty_rows),
                        task.broken_sql,
                        task.expected_sql,
                    ),
                )

    print("[seed] Database seeded successfully:")
    print(
        json.dumps(
            {
                "seed_version": seed_version,
                "tables": seeded_counts,
                "task_datasets": task_seed_counts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
