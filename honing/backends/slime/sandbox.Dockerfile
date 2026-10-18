# The smallest image slime's agent helpers run in, for LocalContainerSandbox (honing/backends/slime/sandbox.py).
#
# ensure_agent_user needs bash, useradd, and git; exec_and_wait needs setsid and tail; install_node22 needs tar,
# and the Node tarballs it unpacks are glibc builds, so the base is Debian rather than Alpine. The sandbox itself
# needs coreutils' timeout. procps is for pgrep, which the integration tests use to check that a timed-out
# command was killed. The base is pinned by its multi-arch digest, so the same file builds on arm64 and amd64.
FROM debian:bookworm-slim@sha256:7c7b2c966bc9ee8cedfeef67e0e279108992c77681fa595db4a9d65c06ccc587

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git procps \
    && rm -rf /var/lib/apt/lists/*
