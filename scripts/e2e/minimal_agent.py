"""A minimal coding agent for the live GEPA run, started by WorkspaceEpisodeRunner in a fresh workspace.

It reads the harness the way the run lays it out: ``AGENTS.md`` is the
instructions and ``skills/*.md`` are skills, and together they are the system
prompt. The task's source files are the ``*.py`` files at the workspace root. It
asks the policy once, through the OpenAI-compatible endpoint in
``HONING_POLICY_URL``, for the corrected files, and writes back only files that
already exist at the root.

The policy is asked to mark files with ``<<<FILE name>>>`` and ``<<<END>>>``
rather than Markdown fences: GEPA's default reflection prompt wraps the agent's
output in a fenced block, and a fence inside it would end that block early.

The key is read from the variable that ``HONING_POLICY_API_KEY_ENV`` names and
goes only into the request's ``Authorization`` header; nothing the agent writes
holds a header. Rate limits, timeouts and server errors are retried up to
``HONING_AGENT_MAX_ATTEMPTS`` times. If the policy still cannot be reached, the
agent writes ``policy_error`` (a kind and a status, never a body) to
``result.json`` and exits 75, so the runner can tell an unavailable policy from
an agent that failed the task. Every call, failed or not, is appended to
``usage.jsonl``. It imports nothing from Honing.
"""

import http.client
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

POLICY_UNAVAILABLE = 75  # EX_TEMPFAIL
MAX_OUTPUT_CHARS = 8000
RETRYABLE = {408, 429}
FORMAT = (
    "Reply with the complete corrected content of every file you change, each one as\n"
    "<<<FILE name.py>>>\n<the whole file>\n<<<END>>>\n"
    "Do not use Markdown code fences. After the files, write one line that starts with SUMMARY: and says what you"
    " changed."
)
BLOCK = re.compile(r"<<<FILE (?P<name>[^>\n]+)>>>\n(?P<body>.*?)\n?<<<END>>>", re.DOTALL)


def _kind(status):
    if status in (401, 403):
        return "auth"
    if status == 429:
        return "rate_limit"
    if status == 408:
        return "timeout"
    if 500 <= status < 600:
        return "server"
    return "other"


def _record_usage(workspace, ok, status, usage):
    with (workspace / "usage.jsonl").open("a") as log:
        log.write(
            json.dumps(
                {
                    "ok": ok,
                    "status": status,
                    "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                    "completion_tokens": int(usage.get("completion_tokens") or 0),
                }
            )
            + "\n"
        )


def ask(workspace, request):
    """The reply body, or ``(None, (kind, status))`` once the policy cannot be reached."""
    key = os.environ.get(os.environ.get("HONING_POLICY_API_KEY_ENV", ""), "")
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    attempts = int(os.environ.get("HONING_AGENT_MAX_ATTEMPTS", "3"))
    timeout_seconds = float(os.environ.get("HONING_AGENT_TIMEOUT_SECONDS", "60"))
    url = os.environ["HONING_POLICY_URL"].rstrip("/") + "/chat/completions"
    failure = None
    for attempt in range(1, attempts + 1):
        call = urllib.request.Request(url, json.dumps(request).encode(), headers)
        try:
            with urllib.request.urlopen(call, timeout=timeout_seconds) as response:
                reply = json.load(response)
        except urllib.error.HTTPError as error:
            status, kind = error.code, _kind(error.code)
            retry = status in RETRYABLE or 500 <= status < 600
        except urllib.error.URLError as error:
            status, kind, retry = None, "timeout" if isinstance(error.reason, TimeoutError) else "connection", True
        except TimeoutError:  # socket.timeout is TimeoutError from 3.10
            status, kind, retry = None, "timeout", True
        except (OSError, http.client.HTTPException):  # a dropped connection, for one
            status, kind, retry = None, "connection", True
        except ValueError:  # a body that is not JSON
            status, kind, retry = None, "server", True
        else:
            _record_usage(workspace, True, 200, reply.get("usage") or {})
            return reply, None
        _record_usage(workspace, False, status, {})
        failure = (kind, status)
        if not retry or attempt == attempts:
            break
        time.sleep(min(8.0, 2.0 ** (attempt - 1)))
    return None, failure


def _unfenced(body):
    """The block's code, without a Markdown fence the model wrapped it in despite the format."""
    lines = body.strip("\n").split("\n")
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)


def main():
    workspace = Path(os.environ["HONING_WORKSPACE"])
    task = json.loads((workspace / "task.json").read_text())
    instructions = workspace / "AGENTS.md"
    system = [instructions.read_text()] if instructions.is_file() else []
    read = ["AGENTS.md"] if instructions.is_file() else []
    for skill in sorted((workspace / "skills").glob("*.md")):
        name = skill.relative_to(workspace).as_posix()
        system.append(f"## Skill: {name}\n{skill.read_text()}")
        read.append(name)
    sources = sorted(path.name for path in workspace.glob("*.py") if path.is_file())
    files = "\n\n".join(f"<<<FILE {name}>>>\n{(workspace / name).read_text()}<<<END>>>" for name in sources)
    request = {
        "model": os.environ["HONING_POLICY_MODEL"],
        "temperature": 0,
        # Caps what one reply can cost; a corrected module is far shorter.
        "max_tokens": int(os.environ.get("HONING_AGENT_MAX_TOKENS", "2048")),
        "messages": [
            {"role": "system", "content": "\n\n".join(system)},
            {"role": "user", "content": f"{task['prompt']}\n\n{files}\n\n{FORMAT}"},
        ],
    }
    reply, failure = ask(workspace, request)
    with (workspace / "transcript.jsonl").open("w") as transcript:
        transcript.write(json.dumps({"read": [*read, *sources]}) + "\n")
        transcript.write(json.dumps({"request": request}) + "\n")
        transcript.write(json.dumps({"reply": reply, "policy_error": failure}) + "\n")
    if reply is None:
        kind, status = failure
        result = {"final_output": "", "policy_error": {"kind": kind, "status": status}}
        (workspace / "result.json").write_text(json.dumps(result))
        return POLICY_UNAVAILABLE

    text = str(reply["choices"][0]["message"].get("content") or "")
    written, refused = [], []
    for match in BLOCK.finditer(text):
        name, body = match["name"].strip(), _unfenced(match["body"])
        if name in sources:
            (workspace / name).write_text(body.rstrip("\n") + "\n")
            written.append((name, body.rstrip("\n")))
        else:
            refused.append(name)
    summary = next((line for line in text.splitlines() if line.startswith("SUMMARY:")), "SUMMARY: (none given)")
    parts = [f"<<<FILE {name}>>>\n{body}\n<<<END>>>" for name, body in written]
    if not written:
        parts.append("(no file was changed)")
    if refused:
        parts.append(f"(refused to write files that are not in the task: {', '.join(sorted(refused))})")
    parts.append(summary)
    (workspace / "result.json").write_text(json.dumps({"final_output": "\n".join(parts)[:MAX_OUTPUT_CHARS]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
