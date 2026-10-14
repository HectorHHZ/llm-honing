"""Grade one python-bugfix task: call the agent's function on hidden cases and print the fraction it gets right.

Usage: ``python3 -I grade.py <cases.json>``, from this directory, with the agent's
workspace in ``HONING_WORKSPACE``. stdout holds only the grade, one JSON line; the
children's stderr is the log. It deletes the directory holding ``<cases.json>``
before any agent code runs, so run it only on a copy, as ``CommandGrader`` does.

This process never imports the agent's code. Anything imported here could print
a grade of its own, exit before the real one is printed, or replace the
functions that compute it. Each case instead runs in a child: the child gets the
module, the function, and the arguments on stdin, and nothing else (the expected
values stay here, and its environment has no ``HONING_*`` variables). It runs in
the workspace, not here, and by then the cases are read into memory and deleted
from disk, so the agent's code cannot open them, by a relative path or by
searching. It imports
the agent's module with the workspace appended to ``sys.path`` after the
standard library, calls the function, and writes one JSON line to a copy of its
stdout that it takes before the agent's code runs; anything the agent's code
prints goes to stderr. A child that crashes, exits early, writes anything but
one result line, returns something JSON cannot encode, or runs past the case
time limit fails that case.

The agent's code still runs as the same user as this script, so it could still
read them from this process (``/proc/<pid>/mem``, or a debugger where the system
allows one); only a sandbox (#16) closes that.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import signal
import subprocess
import sys

DEFAULT_CASE_TIMEOUT_SECONDS = 10.0
# Feedback quotes values, and one huge value must not push the grade line past
# what the harness keeps of stdout, which would lose the whole grade.
MAX_REPR_CHARS = 200

CHILD = r"""
import importlib, json, os, sys
request = json.loads(sys.stdin.read())
result = os.fdopen(os.dup(1), "w")
os.dup2(2, 1)
sys.stdout = sys.stderr
sys.path.append(os.getcwd())
try:
    function = getattr(importlib.import_module(request["module"]), request["function"])
    line = json.dumps({"value": function(*request["args"])})
except BaseException as error:
    line = json.dumps({"error": type(error).__name__ + ": " + str(error)[:200]})
result.write(line + "\n")
result.flush()
"""


def case_timeout_seconds() -> float:
    path = os.environ.get("HONING_TASK_METADATA")
    if not path:
        return DEFAULT_CASE_TIMEOUT_SECONDS
    with open(path, encoding="utf-8") as metadata:
        return float(json.load(metadata).get("case_timeout_seconds", DEFAULT_CASE_TIMEOUT_SECONDS))


def short(value: object) -> str:
    text = repr(value)
    return text if len(text) <= MAX_REPR_CHARS else text[: MAX_REPR_CHARS - 3] + "..."


def same(value: object, expected: object) -> bool:
    # A bool is an int, so it is matched first: True equals True, but not 1.
    if isinstance(value, bool) or isinstance(expected, bool):
        return type(value) is type(expected) and value == expected
    if isinstance(value, (int, float)) and isinstance(expected, (int, float)):
        return value == expected
    return type(value) is type(expected) and value == expected


def run_case(
    module: str, function: str, args: list, workspace: str, timeout_seconds: float
) -> tuple[object, str | None]:
    """Run one call in a child, in ``workspace``. Returns ``(value, None)``, or ``(None, why it failed)``."""
    child = subprocess.Popen(
        [sys.executable, "-I", "-c", CHILD],
        cwd=workspace,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        env={"PATH": os.environ.get("PATH", os.defpath), "LANG": "C.UTF-8"},
        start_new_session=True,
    )
    request = json.dumps({"module": module, "function": function, "args": args}).encode("utf-8")
    try:
        stdout, _ = child.communicate(request, timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        return None, f"took longer than {timeout_seconds:g} s"
    finally:
        # The child leads its own process group: stop anything the agent's code left running.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(child.pid, signal.SIGKILL)
        if child.poll() is None:
            child.communicate()
    lines = [line for line in stdout.decode("utf-8", errors="replace").split("\n") if line.strip()]
    if len(lines) != 1:
        return None, f"gave {len(lines)} result lines (exit code {child.returncode})"
    try:
        reply = json.loads(lines[0])
    except ValueError:  # JSONDecodeError, or an integer longer than Python will convert
        return None, "gave a result that is not JSON"
    if not isinstance(reply, dict):
        return None, "gave a result that is not a JSON object"
    if "error" in reply:
        return None, f"raised {reply['error']}"
    if "value" not in reply:
        return None, "gave no value"
    return reply["value"], None


def main(cases_path: str) -> None:
    # Checked before anything else: run by hand on the source tree, this script
    # would otherwise delete the suite's own cases below.
    workspace = os.environ.get("HONING_WORKSPACE")
    if not workspace:
        sys.exit("grade.py: HONING_WORKSPACE is not set. It deletes its cases directory, so run it on a copy.")
    with open(cases_path, encoding="utf-8") as file:
        spec = json.load(file)
    # The expected values live only in this process from here on: the agent's code,
    # run next, could otherwise find the files by searching the disk.
    shutil.rmtree(os.path.dirname(os.path.abspath(cases_path)))
    timeout_seconds = case_timeout_seconds()
    passed = 0
    failures = []
    for case in spec["cases"]:
        call = f"{spec['function']}({', '.join(map(short, case['args']))})"
        value, problem = run_case(spec["module"], spec["function"], case["args"], workspace, timeout_seconds)
        if problem is None and same(value, case["expected"]):
            passed += 1
        elif problem is None:
            failures.append(f"{call} returned {short(value)}, expected {short(case['expected'])}")
        else:
            failures.append(f"{call} {problem}")
    total = len(spec["cases"])
    score = passed / total if total else 0.0
    print(json.dumps({"score": score, "feedback": "\n".join([f"{passed} of {total} tests pass.", *failures])}))


if __name__ == "__main__":
    main(sys.argv[1])
