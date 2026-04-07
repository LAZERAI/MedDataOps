#!/usr/bin/env bash
set -Eeuo pipefail

log() {
  echo "[entrypoint] $*"
}

run_as_postgres_runtime() {
  local cmd="$1"
  if [[ "$(id -u)" -eq 0 ]]; then
    su -s /bin/bash postgres -c "$cmd"
  else
    bash -c "$cmd"
  fi
}

configure_nss_wrapper_for_unknown_uid() {
  if [[ "$(id -u)" -eq 0 ]]; then
    return
  fi

  if getent passwd "$(id -u)" >/dev/null 2>&1; then
    return
  fi

  local nss_wrapper_lib=""
  for candidate in /usr/lib/*/libnss_wrapper.so /usr/lib/libnss_wrapper.so; do
    if [[ -f "$candidate" ]]; then
      nss_wrapper_lib="$candidate"
      break
    fi
  done

  if [[ -z "$nss_wrapper_lib" ]]; then
    log "WARN: UID $(id -u) has no passwd entry and libnss_wrapper is unavailable; startup may fail"
    return
  fi

  export NSS_WRAPPER_PASSWD="/tmp/nss_wrapper.passwd"
  export NSS_WRAPPER_GROUP="/tmp/nss_wrapper.group"
  cp /etc/passwd "$NSS_WRAPPER_PASSWD"
  cp /etc/group "$NSS_WRAPPER_GROUP"

  echo "appuser:x:$(id -u):$(id -g):anonymous uid:/tmp:/sbin/nologin" >> "$NSS_WRAPPER_PASSWD"
  if ! getent group "$(id -g)" >/dev/null 2>&1; then
    echo "appgroup:x:$(id -g):" >> "$NSS_WRAPPER_GROUP"
  fi

  if [[ -n "${LD_PRELOAD:-}" ]]; then
    export LD_PRELOAD="$nss_wrapper_lib $LD_PRELOAD"
  else
    export LD_PRELOAD="$nss_wrapper_lib"
  fi

  log "Configured NSS wrapper for unknown UID $(id -u)"
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
  local default_socket_dir="/var/run/postgresql"
  local fallback_socket_dir="/tmp/meddataops-postgresql"
  local fallback_pgdata="/tmp/meddataops-pgdata"

  mkdir -p "$PGDATA" || log "WARN: could not create PGDATA at $PGDATA; will attempt fallback path"
  mkdir -p "$default_socket_dir" || log "WARN: could not create $default_socket_dir; will attempt fallback socket path"

  chown -R postgres:postgres "$PGDATA" "$default_socket_dir" || log "WARN: chown failed (restricted sandbox); continuing"
  chmod 700 "$PGDATA" || log "WARN: chmod failed on $PGDATA (restricted sandbox); continuing"

  if [[ ! -d "$PGDATA" || ! -w "$PGDATA" ]]; then
    log "WARN: PGDATA path not writable ($PGDATA); falling back to $fallback_pgdata"
    export PGDATA="$fallback_pgdata"
    mkdir -p "$PGDATA"
    chmod 700 "$PGDATA" || true
  fi

  if [[ -d "$default_socket_dir" && -w "$default_socket_dir" ]]; then
    export POSTGRES_SOCKET_DIR="$default_socket_dir"
  else
    export POSTGRES_SOCKET_DIR="$fallback_socket_dir"
    mkdir -p "$POSTGRES_SOCKET_DIR"
  fi

  if [[ ! -s "$PGDATA/PG_VERSION" ]]; then
    log "Initializing PostgreSQL 15 cluster at $PGDATA"
    run_as_postgres_runtime "initdb -D '$PGDATA' --encoding=UTF8 --locale=C --username=postgres"
  fi

  log "Starting embedded PostgreSQL"
  run_as_postgres_runtime "pg_ctl -D '$PGDATA' -o \"-c listen_addresses='*' -p ${POSTGRES_PORT} -k '${POSTGRES_SOCKET_DIR}'\" -w start"

  POSTGRES_HOST="127.0.0.1"
  export POSTGRES_HOST
}

ensure_db_and_user() {
  require_safe_identifier "POSTGRES_USER" "$POSTGRES_USER"
  require_safe_identifier "POSTGRES_DB" "$POSTGRES_DB"

  local escaped_password
  escaped_password=${POSTGRES_PASSWORD//\'/\'\'}

  run_as_postgres_runtime "psql -h '${POSTGRES_SOCKET_DIR}' -p ${POSTGRES_PORT} -U postgres -v ON_ERROR_STOP=1 postgres <<'SQL'
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
export POSTGRES_SOCKET_DIR="${POSTGRES_SOCKET_DIR:-/var/run/postgresql}"
export EMBEDDED_POSTGRES="${EMBEDDED_POSTGRES:-1}"
export PYTHONPATH="${PYTHONPATH:-/app/src}"

discover_postgres_bin
configure_nss_wrapper_for_unknown_uid

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
