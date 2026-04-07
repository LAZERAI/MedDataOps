#!/usr/bin/env bash
set -Eeuo pipefail

log() {
  echo "[entrypoint] $*"
}

require_safe_identifier() {
  local label="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
    echo "[entrypoint] ERROR: ${label} must match ^[A-Za-z_][A-Za-z0-9_]*$ (got: ${value})." >&2
    exit 1
  fi
}

wait_for_postgres() {
  local host="$1"
  local port="$2"
  local user="$3"
  local dbname="$4"

  for attempt in $(seq 1 60); do
    if pg_isready -h "$host" -p "$port" -U "$user" -d "$dbname" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done

  echo "[entrypoint] ERROR: PostgreSQL did not become ready in time." >&2
  exit 1
}

discover_postgres_bin() {
  if command -v initdb >/dev/null 2>&1 && command -v pg_ctl >/dev/null 2>&1; then
    return
  fi

  local initdb_path
  initdb_path="$(find /usr/lib/postgresql -maxdepth 3 -type f -name initdb 2>/dev/null | sort | tail -n 1 || true)"
  if [[ -z "$initdb_path" ]]; then
    echo "[entrypoint] ERROR: Could not locate initdb binary under /usr/lib/postgresql." >&2
    exit 1
  fi

  export PATH="$(dirname "$initdb_path"):$PATH"
}

start_embedded_postgres() {
  mkdir -p "$PGDATA" /var/run/postgresql
  chown -R postgres:postgres "$PGDATA" /var/run/postgresql || log "WARN: chown failed (likely restricted sandbox); continuing with existing permissions"
  chmod 700 "$PGDATA"

  if [[ ! -s "$PGDATA/PG_VERSION" ]]; then
    log "Initializing PostgreSQL 15 cluster at $PGDATA"
    su -s /bin/bash postgres -c "initdb -D '$PGDATA' --encoding=UTF8 --locale=C"
  fi

  log "Starting embedded PostgreSQL"
  su -s /bin/bash postgres -c "pg_ctl -D '$PGDATA' -o \"-c listen_addresses='*' -p ${POSTGRES_PORT}\" -w start"

  POSTGRES_HOST="127.0.0.1"
  export POSTGRES_HOST
}

ensure_db_and_user() {
  require_safe_identifier "POSTGRES_USER" "$POSTGRES_USER"
  require_safe_identifier "POSTGRES_DB" "$POSTGRES_DB"

  local escaped_password
  escaped_password=${POSTGRES_PASSWORD//\'/\'\'}

  su -s /bin/bash postgres -c "psql -v ON_ERROR_STOP=1 postgres <<'SQL'
DO \$\$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '${POSTGRES_USER}') THEN
    EXECUTE format('CREATE ROLE %I LOGIN PASSWORD %L', '${POSTGRES_USER}', '${escaped_password}');
  ELSE
    EXECUTE format('ALTER ROLE %I WITH LOGIN PASSWORD %L', '${POSTGRES_USER}', '${escaped_password}');
  END IF;
END
\$\$;

SELECT format('CREATE DATABASE %I OWNER %I', '${POSTGRES_DB}', '${POSTGRES_USER}')
WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = '${POSTGRES_DB}')
\\gexec
SQL"
}

seed_database() {
  log "Seeding MedDataOps datasets"
  python /app/scripts/seed_db.py
}

start_api() {
  log "Starting FastAPI server on port 7860"
  if [[ "${UVICORN_RELOAD:-0}" == "1" ]]; then
    exec uvicorn server:app --host 0.0.0.0 --port 7860 --reload
  fi
  exec uvicorn server:app --host 0.0.0.0 --port 7860
}

export PGDATA="${PGDATA:-/var/lib/postgresql/data}"
export POSTGRES_PORT="${POSTGRES_PORT:-5432}"
export POSTGRES_DB="${POSTGRES_DB:-meddataops}"
export POSTGRES_USER="${POSTGRES_USER:-meddataops}"
export POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-meddataops}"
export POSTGRES_HOST="${POSTGRES_HOST:-127.0.0.1}"
export EMBEDDED_POSTGRES="${EMBEDDED_POSTGRES:-1}"
export PYTHONPATH="${PYTHONPATH:-/app/src}"

discover_postgres_bin

if [[ -z "${MEDDATAOPS_POSTGRES_DSN:-}" ]]; then
  export MEDDATAOPS_POSTGRES_DSN="postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB}"
fi

if [[ "$EMBEDDED_POSTGRES" == "1" ]]; then
  start_embedded_postgres
  ensure_db_and_user
  wait_for_postgres "$POSTGRES_HOST" "$POSTGRES_PORT" "$POSTGRES_USER" "$POSTGRES_DB"
else
  log "Using external PostgreSQL at ${POSTGRES_HOST}:${POSTGRES_PORT}"
  wait_for_postgres "$POSTGRES_HOST" "$POSTGRES_PORT" "$POSTGRES_USER" "$POSTGRES_DB"
fi

seed_database
start_api
