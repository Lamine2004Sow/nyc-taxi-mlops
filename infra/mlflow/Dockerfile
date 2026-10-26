# MLflow tracking server with the PostgreSQL backend driver. Version matches uv.lock.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MLFLOW_DISABLE_AGENT_HINT=1

RUN pip install --no-cache-dir "mlflow==3.17.2" "psycopg2-binary==2.9.10"

WORKDIR /mlflow
EXPOSE 5000
