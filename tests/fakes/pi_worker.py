"""Stands in for integrations/pi/worker.mjs: same files in and out, no Node.js, no pi, no model.

It reports every file the plan routes as delivered, and answers with what it saw: the task, its working directory,
whether the API key reached it, the plan's tools, the models.json it was given, and the names of its environment
variables. FAKE_PI_SLEEP makes it hang, FAKE_PI_CHILD starts a child process that records its pid (so a test can check
the child is killed too), and FAKE_PI_STATUS (``aborted`` or ``error``, with FAKE_PI_ERROR) makes the session stop as
pi's does: with a partial answer, a status, an error, and exit code 2.
"""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

root = Path(sys.argv[1])
episode = json.loads((root / "episode.json").read_text())
plan = episode["agent"]
if os.environ.get("FAKE_PI_CHILD"):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    (root / "child.pid").write_text(str(child.pid))
loaded = {
    path: {"route": route, "sha256": hashlib.sha256((root / "harness" / path).read_bytes()).hexdigest()}
    for path, route in plan["routes"].items()
}
(root / "delivery.json").write_text(json.dumps({"files": loaded, "tools": plan["tools"]}))
time.sleep(float(os.environ.get("FAKE_PI_SLEEP", "0")))
models = json.loads((root / "agent" / "models.json").read_text())
key = models["providers"][episode["model"]["provider"]]["apiKey"]
answer = {
    "task": episode["task"]["task_id"],
    "cwd": os.getcwd(),
    "key_present": key.startswith("$") and bool(os.environ.get(key[1:])),
    "tools": plan["tools"],
    "models": models,
    "environment": sorted(os.environ),
}
status = os.environ.get("FAKE_PI_STATUS", "finished")
(root / "transcript.jsonl").write_text(json.dumps({"type": "agent_settled", "aborted": status == "aborted"}) + "\n")
result = {"final_output": json.dumps(answer), "status": status, "error": os.environ.get("FAKE_PI_ERROR")}
(root / "result.json").write_text(json.dumps(result))
sys.exit(0 if status == "finished" else 2)
