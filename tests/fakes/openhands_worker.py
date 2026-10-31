"""Stands in for integrations/openhands/worker.py: same files in and out, no OpenHands, no model.

It reports every file the plan routes as loaded, and answers with what it saw, including the names of the
environment variables it was given. FAKE_WORKER_SLEEP makes it
hang, FAKE_WORKER_EXIT sets its exit code, and FAKE_WORKER_CHILD starts a child process that records its pid,
so a test can check the child is killed too.
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
if os.environ.get("FAKE_WORKER_CHILD"):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    (root / "child.pid").write_text(str(child.pid))
time.sleep(float(os.environ.get("FAKE_WORKER_SLEEP", "0")))
loaded = {
    path: {"route": route, "sha256": hashlib.sha256((root / "harness" / path).read_bytes()).hexdigest()}
    for path, route in plan["routes"].items()
}
(root / "delivery.json").write_text(json.dumps({"files": loaded, "activated_skills": [], "tools": plan["tools"]}))
key_env = episode["policy"].get("api_key_env")
answer = {
    "task": episode["task"]["task_id"],
    "cwd": os.getcwd(),
    "key_present": bool(key_env and os.environ.get(key_env)),
    "tools": plan["tools"],
    "environment": sorted(os.environ),
}
(root / "transcript.jsonl").write_text(json.dumps({"answer": answer}) + "\n")
(root / "result.json").write_text(json.dumps({"final_output": json.dumps(answer), "status": "finished"}))
sys.exit(int(os.environ.get("FAKE_WORKER_EXIT", "0")))
