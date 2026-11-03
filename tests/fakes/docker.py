#!/usr/bin/env python3
"""Stands in for the docker CLI: ``run`` records its arguments and runs the worker as the container would.

The worker gets only the variables named with ``--env`` (a bare name takes the client's value) and runs in
``--workdir``. The first argument is a directory to record into, since the runner gives the docker client
almost no environment: each ``run`` appends its argv to ``runs.jsonl``, ``kill`` appends the container's name
to ``kills.txt``, and a ``failures`` file there says how many runs exit 125 before one starts.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

record = Path(sys.argv[1])
command, arguments = sys.argv[2], sys.argv[3:]
if command == "kill":
    with (record / "kills.txt").open("a") as kills:
        kills.write(arguments[0] + "\n")
    sys.exit(0)
assert command == "run", command
with (record / "runs.jsonl").open("a") as runs:
    runs.write(json.dumps({"argv": arguments, "client_env": sorted(os.environ)}) + "\n")
runs_so_far = len((record / "runs.jsonl").read_text().splitlines())
failures = record / "failures"
if runs_so_far <= int(failures.read_text() if failures.exists() else "0"):
    print("docker: Error response from daemon: mounts denied: EOF", file=sys.stderr)
    sys.exit(125)

volumes: dict[str, str] = {}
environment: dict[str, str] = {}
workdir = None
index = 0
while arguments[index].startswith("--"):
    flag = arguments[index]
    if flag in ("--rm", "--init"):
        index += 1
        continue
    value = arguments[index + 1]
    if flag == "--volume":
        host, inside = value.split(":")[:2]
        volumes[inside] = host
    elif flag == "--env":
        name, _, given = value.partition("=")
        if given or "=" in value:
            environment[name] = given
        elif name in os.environ:
            environment[name] = os.environ[name]
    elif flag == "--workdir":
        workdir = value
    index += 2
image, python, worker, *rest = arguments[index:]
environment["PATH"] = os.environ["PATH"]
environment["FAKE_DOCKER_IMAGE"] = image
completed = subprocess.run([sys.executable, volumes[worker], *rest], cwd=workdir, env=environment, check=False)
sys.exit(completed.returncode)
