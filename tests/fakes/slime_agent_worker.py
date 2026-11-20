"""Stands in for :mod:`honing.backends.slime.agent_worker` in the runner's tests, with no Slime and no Docker.

It is run as ``python -m tests.fakes.slime_agent_worker EPISODE_DIR``, as the runner
runs the real worker. It reads ``episode.json`` with the real worker's reader
(``agent_episode.read_episode``), so a runner that writes something the worker cannot read fails
here too. It copies ``workspace/`` to ``workspace_out/``, adds ``agent.txt``, and
writes ``result.json``; it also records the environment it saw and its argv in
``fake_env.json``.

``$FAKE_AGENT_WORKER`` is a JSON object that changes what it does:

``status``, ``exit_code``, ``final_output``, ``workspace``, ``failure``
    Go into ``result.json`` (defaults ``completed``, 0, ``"done"``, ``workspace_out``, null).
``worker_exit``
    The worker's own exit code (default 0, or 3 for ``infra_error``).
``result``
    ``missing`` writes no ``result.json``, ``malformed`` writes one that is not JSON.
``sleep``
    Seconds to sleep before writing anything but ``state.json``.
``child``
    Start a child that ignores SIGTERM and records its pid in ``child.pid``, to check it is killed too.
``ignore_term``
    Ignore SIGTERM, so only SIGKILL stops it.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from honing.backends.slime import agent_episode as worker


def main(argv: list[str]) -> int:
    directory = Path(argv[0])
    knobs = json.loads(os.environ.get("FAKE_AGENT_WORKER", "{}"))
    (directory / "fake_env.json").write_text(json.dumps({"env": dict(os.environ), "argv": sys.argv}))
    episode = worker.read_episode(directory / worker.EPISODE_FILE)
    if knobs.get("ignore_term"):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    (directory / worker.STATE_FILE).write_text(json.dumps({"sandbox_id": f"{episode.name_prefix}-fake"}))
    if knobs.get("child"):
        # It ignores SIGTERM, so only the runner's SIGKILL of the whole group stops it.
        child = subprocess.Popen(["sh", "-c", "trap '' TERM; exec sleep 600"])
        (directory / "child.pid").write_text(str(child.pid))
    time.sleep(float(knobs.get("sleep", 0)))
    shutil.copytree(directory / worker.WORKSPACE_DIR, directory / worker.WORKSPACE_OUT_DIR, symlinks=True)
    (directory / worker.WORKSPACE_OUT_DIR / "agent.txt").write_text("written by the fake agent\n")
    (directory / worker.TRANSCRIPT_FILE).write_text('{"type": "result", "result": "done"}\n')
    status = knobs.get("status", worker.STATUS_COMPLETED)
    result = {
        "schema_version": worker.SCHEMA_VERSION,
        "status": status,
        "exit_code": knobs.get("exit_code", 0),
        "final_output": knobs.get("final_output", "done"),
        "transcript": worker.TRANSCRIPT_FILE,
        "workspace": knobs.get("workspace", worker.WORKSPACE_OUT_DIR),
        "failure": knobs.get("failure"),
        "delivery": {path: True for placed in episode.placement for path in placed.paths},
        "error": "fake infrastructure fault" if status == worker.STATUS_INFRA_ERROR else None,
    }
    mode = knobs.get("result", "ok")
    if mode == "ok":
        (directory / worker.RESULT_FILE).write_text(json.dumps(result))
    elif mode == "malformed":
        (directory / worker.RESULT_FILE).write_text("{not json")
    default_exit = worker.EXIT_INFRA if status == worker.STATUS_INFRA_ERROR else 0
    return int(knobs.get("worker_exit", default_exit))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
