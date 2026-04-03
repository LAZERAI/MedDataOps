FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY openenv.yaml README.md inference.py .
COPY src ./src
COPY scripts ./scripts

CMD ["python", "scripts/run_env.py"]
