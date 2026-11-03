# The OpenHands environment as an image, for OpenHandsEpisodeRunner's container mode (see README.md):
#   docker build -t honing-openhands integrations/openhands
# The runner mounts the episode directory and worker.py and runs the worker as the host user, so the image needs only
# Python 3.12 and requirements.txt. A task suite whose agent needs more (a library, a command) builds on this image.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends bash ca-certificates git procps tmux \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.2 /uv /bin/uv
COPY requirements.txt /tmp/requirements.txt
RUN uv pip install --system --no-cache -r /tmp/requirements.txt && rm /tmp/requirements.txt

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OPENHANDS_SUPPRESS_BANNER=1
