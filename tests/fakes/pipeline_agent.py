"""A stand-in agent process for the all-planes pipeline, run by WorkspaceEpisodeRunner in a fresh workspace.

It does what an agent framework does with a harness, minimally: read the task and the harness files in its
workspace, build a system prompt from the instructions and skills, ask the policy through its OpenAI-compatible
endpoint (HONING_POLICY_URL), and write ``transcript.jsonl`` and ``result.json``. It reads every file and records
which, so a delivery probe can tell what reached it. It imports nothing from Honing.
"""

import json
import os
import urllib.request
from pathlib import Path

workspace = Path(os.environ["HONING_WORKSPACE"])
task = json.loads((workspace / "task.json").read_text())
files = sorted(
    str(path.relative_to(workspace))
    for path in workspace.rglob("*")
    if path.is_file() and path.name not in {"task.json", "agent.log"}
)
system = "\n".join((workspace / name).read_text() for name in files)
request = {
    "model": os.environ["HONING_POLICY_MODEL"],
    "messages": [{"role": "system", "content": system}, {"role": "user", "content": task["prompt"]}],
}
call = urllib.request.Request(
    os.environ["HONING_POLICY_URL"] + "/chat/completions",
    json.dumps(request).encode(),
    {"Content-Type": "application/json"},
)
with urllib.request.urlopen(call, timeout=60) as response:
    reply = json.load(response)
answer = reply["choices"][0]["message"]["content"]
with (workspace / "transcript.jsonl").open("w") as transcript:
    transcript.write(json.dumps({"read": files}) + "\n")
    transcript.write(json.dumps({"request": request}) + "\n")
    transcript.write(json.dumps({"reply": reply}) + "\n")
(workspace / "result.json").write_text(json.dumps({"final_output": answer}))
