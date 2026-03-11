FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY requirements ./requirements
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
COPY docs ./docs
COPY skills ./skills

RUN python -m pip install --upgrade pip && \
    DS_BUILD_OPS=0 python -m pip install -r requirements/production.lock && \
    DS_BUILD_OPS=0 python -m pip install . --no-deps

ENTRYPOINT ["openclaw-moe"]
