"""#54 on a GPU node: the question check with the real Laya tokenizer, and the real worker's refusal
of an overlong question reaching the grader as a 0 score."""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

from honing.backends.ra2 import LayaQuestionCheck, RA2Grader, load_suite, seed_harness
from honing.backends.ra2.harness import role, worst_case
from honing.core import CampaignRuleError, EpisodeOutcome, HarnessFile

PR49 = Path(os.environ["PR49_DIR"])
REPO = Path(__file__).resolve().parents[3]
INTEGRATION = REPO / "integrations/ra2"
checkpoint = Path(json.loads((PR49 / ".cache/models/typed-decisions.json").read_text())["path"])
check = LayaQuestionCheck.for_checkpoint(checkpoint, INTEGRATION)

spec = seed_harness(INTEGRATION)
check.check(spec)
questions = json.loads(spec.get("ra2/questions.json").content)
counts: dict[str, list[tuple[int, str]]] = {}
for key, text in questions.items():
    kind = role(key)
    counts.setdefault(kind, []).append((check._tokens((" " if kind == "option" else "") + worst_case(text)), key))
for kind, values in sorted(counts.items()):
    print(f"seed harness {kind:11}: {len(values)} entries, longest {max(values)[0]} tokens ({max(values)[1]})")

recruit = questions["intent.group.recruit"]
inflated = spec.with_file(
    "ra2/questions.json", HarnessFile("instructions", json.dumps({**questions, "intent.group.recruit": recruit * 10}))
)
try:
    check.check(inflated)
    print("inflated option: NOT refused")
except CampaignRuleError as error:
    print("inflated option refused:", str(error)[:160])

long_option = " ".join(["Recruit", "more", "Rhino", "tanks", "and", "flak", "troopers"] * 10)
request = {
    "id": 1,
    "parts": {"essential": ["Tick 30."]},
    "questions": {"g1": {"type": "choice", "instructions": "Next?", "criteria": {"A": "Wait", "B": long_option}}},
}
env = {**os.environ, "LAYA_GPU": os.environ.get("GPU", "2"), "LAYA_MODEL_TYPE": "typed-decisions"}
worker = INTEGRATION / "integration/laya/worker.py"
out = subprocess.run(
    [sys.executable, "-u", str(worker)],
    input=json.dumps(request) + "\n",
    env=env,
    capture_output=True,
    text=True,
    timeout=600,
)
replies = [json.loads(line) for line in out.stdout.splitlines() if line.strip()]
message = next(r["error"] for r in replies if "error" in r)
print("real worker refused:", message)

suite = load_suite(INTEGRATION / "tasks/suite.jsonl")
task = suite.tasks("val")[0]
grader = RA2Grader(suite=suite, integration_dir=INTEGRATION, audit_command=["true"])
for label, text in (("stdio worker", message), ("HTTP server", f"ValueError: {message}")):
    run = Path(os.environ.get("TMPDIR", "/tmp")) / f"q54-{label.replace(' ', '-')}"
    run.mkdir(exist_ok=True)
    (run / "error.json").write_text(json.dumps({"error": text, "tick": 30}))
    outcome = EpisodeOutcome(task.task_id, "h", "c", exit_code=1, workspace_location=str(run))
    grade = asyncio.run(grader.grade(task, outcome))
    print(f"grader on the {label}'s message: score {grade.score}; feedback: {(grade.feedback or '')[:120]}")
