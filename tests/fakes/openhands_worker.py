"""Stands in for integrations/openhands/worker.py: same files in and out, no OpenHands, no model.

It reads episode.json and the harness, writes result.json, delivery.json (every harness file routed, with
its hash), and a one-line transcript. FAKE_WORKER_SLEEP makes it hang, FAKE_WORKER_EXIT sets its exit code,
and FAKE_WORKER_CHILD starts a child process that writes its pid, so a test can check the child is killed too.
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
if os.environ.get("FAKE_WORKER_CHILD"):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    (root / "child.pid").write_text(str(child.pid))
time.sleep(float(os.environ.get("FAKE_WORKER_SLEEP", "0")))
files = {}
for entry in episode["harness"]:
    content = (root / "harness" / entry["path"]).read_text()
    files[entry["path"]] = {
        "kind": entry["kind"],
        "route": "fake",
        "sha256": hashlib.sha256(content.encode()).hexdigest(),
    }
(root / "delivery.json").write_text(json.dumps({"files": files, "activated_skills": []}))
key_env = episode["policy"].get("api_key_env")
answer = {
    "task": episode["task"]["task_id"],
    "cwd": os.getcwd(),
    "key_present": bool(key_env and os.environ.get(key_env)),
}
(root / "transcript.jsonl").write_text(json.dumps({"answer": answer}) + "\n")
(root / "result.json").write_text(json.dumps({"final_output": json.dumps(answer), "status": "finished"}))
sys.exit(int(os.environ.get("FAKE_WORKER_EXIT", "0")))
