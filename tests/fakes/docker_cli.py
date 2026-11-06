"""Stands in for the docker CLI in LocalContainerSandbox's tests that need exact timing.

Each call appends its arguments and pid to ``$FAKE_DOCKER_LOG`` (one JSON line) and, for
``exec --env-file``, the file's text and mode as the CLI saw them. ``$FAKE_DOCKER``
is a JSON object keyed by subcommand (``run``, ``exec``, ``rm``) whose values set
``seconds`` to sleep, then ``stdout``, ``stderr`` and ``exit``. A ``done`` line is
logged once the sleep is over, so a test can tell a finished call from a killed one.
"""

from __future__ import annotations

import json
import os
import sys
import time


def main() -> int:
    log = os.environ["FAKE_DOCKER_LOG"]
    argv = sys.argv[1:]
    command = argv[0]
    entry: dict[str, object] = {"argv": argv, "pid": os.getpid()}
    if command == "exec" and "--env-file" in argv:
        path = argv[argv.index("--env-file") + 1]
        with open(path, encoding="utf-8") as handle:
            entry["env_file"] = handle.read()
        entry["env_mode"] = os.stat(path).st_mode & 0o777
    with open(log, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")
    behaviour = json.loads(os.environ.get("FAKE_DOCKER", "{}")).get(command, {})
    time.sleep(behaviour.get("seconds", 0))
    with open(log, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"done": command}) + "\n")
    sys.stdout.write(behaviour.get("stdout", ""))
    sys.stderr.write(behaviour.get("stderr", ""))
    return int(behaviour.get("exit", 0))


if __name__ == "__main__":
    sys.exit(main())
