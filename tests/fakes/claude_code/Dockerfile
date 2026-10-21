# The sandbox image (honing/backends/slime/sandbox.Dockerfile) plus python3, which the fake Claude Code CLI
# next to this file runs on, and which the example tasks' code needs. BASE is that image's tag.
ARG BASE
FROM ${BASE}

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 \
    && rm -rf /var/lib/apt/lists/*
