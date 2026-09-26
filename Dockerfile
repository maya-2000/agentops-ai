# syntax=docker/dockerfile:1.7
#
# AgentOps production images: two targets from one file.
#
#   api  FastAPI + the LangGraph agent + secured tools   (python -m app.api, port 8000)
#   ui   Streamlit web UI; it calls the API over HTTP     (python -m app.ui,  port 8501)
#        No database driver, analytics or agent code is installed in the UI image.
#
# Data is never baked into an image. The DuckDB file and its manifest are mounted read-only at run time
# (see docker-compose.yml), and the hidden evaluation ground truth (data/seeds/) never enters the build
# context (.dockerignore is an allow-list). Both images run as an unprivileged user.
#
# Build:   docker compose build     (or: docker build --target api -t agentops-api .)
# Behind a TLS-inspecting proxy, pass its CA bundle as an optional build secret:
#          docker build --secret id=pip_ca,src=/path/to/ca-bundle.crt --target api .

ARG PYTHON_IMAGE=python:3.11-slim

# ------------------------------------------------------------------------------------------ base
FROM ${PYTHON_IMAGE} AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN groupadd --system --gid 10001 agentops \
    && useradd --system --uid 10001 --gid agentops --home-dir /home/agentops --create-home \
       --shell /usr/sbin/nologin agentops

# ------------------------------------------------------------------------------------------ api
FROM base AS api-build
WORKDIR /build
COPY pyproject.toml README.md ./
# 1. The runtime dependencies plus the "api" extra (fastapi, uvicorn); no dev tools. Installed against an
#    empty package, so this layer stays cached until pyproject.toml changes.
RUN --mount=type=secret,id=pip_ca,required=false \
    if [ -f /run/secrets/pip_ca ]; then export PIP_CERT=/run/secrets/pip_ca; fi \
    && python -m venv /opt/venv \
    && mkdir app && touch app/__init__.py \
    && /opt/venv/bin/pip install ".[api]" \
    && rm -rf app build ./*.egg-info
# 2. The application itself (a code change rebuilds only this layer).
COPY app ./app
RUN --mount=type=secret,id=pip_ca,required=false \
    if [ -f /run/secrets/pip_ca ]; then export PIP_CERT=/run/secrets/pip_ca; fi \
    && /opt/venv/bin/pip install --no-deps --force-reinstall .

FROM base AS api
COPY --from=api-build /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    APP_ENV=production \
    LOG_FORMAT=json \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    DATABASE_URL=duckdb:////data/database/northwind_cloud.duckdb \
    DATASET_MANIFEST_PATH=/data/metadata/dataset_manifest.json
WORKDIR /home/agentops
USER agentops:agentops
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/v1/readiness', timeout=4).status == 200 else 1)"]
# Exec form: python is PID 1 and receives SIGTERM directly (graceful shutdown).
CMD ["python", "-m", "app.api"]

# ------------------------------------------------------------------------------------------ ui
FROM base AS ui-build
# Only what the UI imports: Streamlit, httpx, and pydantic-settings for app/config.py. The version ranges
# match pyproject.toml (tests/deploy/test_deployment.py keeps them in step).
RUN --mount=type=secret,id=pip_ca,required=false \
    if [ -f /run/secrets/pip_ca ]; then export PIP_CERT=/run/secrets/pip_ca; fi \
    && python -m venv /opt/venv \
    && /opt/venv/bin/pip install "streamlit>=1.50,<2" "httpx>=0.27" "pydantic>=2.7,<3" "pydantic-settings>=2.3,<3"

FROM base AS ui
COPY --from=ui-build /opt/venv /opt/venv
COPY app/__init__.py app/config.py /opt/agentops/app/
COPY app/ui /opt/agentops/app/ui
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/opt/agentops \
    APP_ENV=production \
    UI_HOST=0.0.0.0 \
    UI_PORT=8501 \
    UI_API_URL=http://api:8000
WORKDIR /home/agentops
USER agentops:agentops
EXPOSE 8501
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4).status == 200 else 1)"]
CMD ["python", "-m", "app.ui"]
