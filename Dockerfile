# One image for the api, the monitor and the retrain Job.
# The build context must contain a `models/` directory (Makefile and CI create it).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_NO_CACHE=1 \
    PATH="/app/.venv/bin:${PATH}" \
    MODEL_DIR=/app/models/champion \
    PREDICTION_LOG_PATH=/tmp/predictions/predictions.jsonl \
    MLFLOW_DISABLE_AGENT_HINT=1

COPY --from=ghcr.io/astral-sh/uv:0.12.1 /uv /uvx /bin/

# LightGBM needs the OpenMP runtime.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

RUN groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --home-dir /app --shell /usr/sbin/nologin app

WORKDIR /app

# Dependencies first (cached layer), then the project sources.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

COPY models ./models
RUN mkdir -p /tmp/predictions /tmp/data /tmp/reports \
 && chown -R app:app /app/models /tmp/predictions /tmp/data /tmp/reports

USER app

EXPOSE 8000

CMD ["uvicorn", "nyc_taxi.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
