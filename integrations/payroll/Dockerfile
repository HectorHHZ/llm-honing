# The payroll task image: upstream slm-harness-adaptation's tasks/Dockerfile, built from its checkout.
#   (cd <checkout>/software-agent-sdk && uv export --frozen --no-hashes --all-packages --no-emit-workspace --no-dev -o constraints.txt)
#   docker build -f integrations/payroll/Dockerfile --build-arg USER_UID=$(id -u) -t attendance_payroll_audit_s2l:latest <checkout>
# Differences: upstream's root uv.lock is not published, so the SDK fork and the task's Python libraries are
# installed with `uv pip`, pinned to the SDK fork's own lock; that lock does not build on Python 3.14, so the base is
# 3.13; and Playwright's Chromium, used only by browser tasks, is left out.
FROM python:3.13-slim

ARG USER_UID

ENV HOME=/workspace \
    VIRTUAL_ENV=/workspace/.venv \
    PATH="/workspace/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gcc \
    g++ \
    make \
    python3-dev \
    poppler-utils \
    jq \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.2 /uv /bin/uv

RUN useradd --create-home --home-dir /workspace --shell /bin/bash --uid $USER_UID appuser \
 && chown appuser:appuser /workspace

COPY --chown=appuser:appuser software-agent-sdk ./software-agent-sdk

USER appuser
RUN uv venv /workspace/.venv \
 && uv pip install -c ./software-agent-sdk/constraints.txt \
    ./software-agent-sdk/openhands-sdk \
    ./software-agent-sdk/openhands-tools \
    ./software-agent-sdk/openhands-workspace \
    ./software-agent-sdk/openhands-agent-server \
    "pandas>=2.3.3" "openpyxl>=3.1.5" "boto3>=1.42.3" numpy

ENV PYTHONPATH=/workspace

EXPOSE 8000

ENTRYPOINT ["python", "-m", "openhands.agent_server"]
