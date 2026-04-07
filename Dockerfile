ARG PYTHON_BASE_IMAGE=public.ecr.aws/docker/library/python:3.11-slim
FROM ${PYTHON_BASE_IMAGE}

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    PGDATA=/var/lib/postgresql/data \
    POSTGRES_PORT=5432 \
    POSTGRES_DB=meddataops \
    POSTGRES_USER=meddataops \
    POSTGRES_PASSWORD=meddataops \
    POSTGRES_HOST=127.0.0.1 \
    EMBEDDED_POSTGRES=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        libnss-wrapper \
        postgresql \
        postgresql-client \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /var/lib/postgresql/data /var/run/postgresql \
    && chown -R postgres:postgres /var/lib/postgresql /var/run/postgresql \
    && chmod +x /app/entrypoint.sh

EXPOSE 7860

ENTRYPOINT ["/app/entrypoint.sh"]
