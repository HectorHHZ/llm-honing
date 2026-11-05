"""The same file-based harness delivery used by the other game integrations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from honing.core import CampaignRuleError, HarnessFile, HarnessSpec

AGENT = "rea_issac_decision"
CONTROLLER_PATH = "rea-issac/controller.json"
QUESTIONS_PATH = "rea-issac/questions.json"


@dataclass(frozen=True)
class ReaIssacHarness:
    history_steps: int
    instructions: dict[str, str]
    notes: str
    files: tuple[dict[str, str], ...]


def seed_harness(integration_dir: Path) -> HarnessSpec:
    return HarnessSpec(
        agent=AGENT,
        files=(
            (CONTROLLER_PATH, HarnessFile("settings", (integration_dir / "config/controller.json").read_text())),
            (QUESTIONS_PATH, HarnessFile("instructions", (integration_dir / "config/questions.json").read_text())),
        ),
    )


def parse_harness(spec: HarnessSpec) -> ReaIssacHarness:
    """Harness changes can edit context and instructions, never the task or game rules."""
    try:
        if spec.agent != AGENT or not spec.kinds <= {"settings", "instructions"}:
            raise ValueError(f"the REA-Issac runner supports agent {AGENT} and settings/instructions files")
        controller_file, questions_file = spec.get(CONTROLLER_PATH), spec.get(QUESTIONS_PATH)
        if controller_file is None or questions_file is None:
            raise ValueError("the controller and questions files are required")
        if controller_file.kind != "settings" or questions_file.kind != "instructions":
            raise ValueError("controller must be settings and questions must be instructions")
        controller, questions = json.loads(controller_file.content), json.loads(questions_file.content)
        if not isinstance(controller, dict) or set(controller) != {"history_steps"}:
            raise ValueError("controller must define only history_steps")
        history = controller["history_steps"]
        if type(history) is not int or not 0 <= history <= 32:
            raise ValueError("history_steps must be an integer in 0..32")
        if not isinstance(questions, dict) or set(questions) != {"move", "shoot", "bomb"}:
            raise ValueError("questions must define move, shoot and bomb")
        if any(not isinstance(v, str) or not v.strip() for v in questions.values()):
            raise ValueError("action instructions must be nonempty text")
        notes = "\n\n".join(
            f"Operating notes ({path}):\n{file.content}"
            for path, file in spec.files
            if path not in (CONTROLLER_PATH, QUESTIONS_PATH)
        )
        files = tuple(
            {"path": path, "kind": file.kind, "sha256": hashlib.sha256(file.content.encode()).hexdigest()}
            for path, file in spec.files
        )
        return ReaIssacHarness(history, questions, notes, files)
    except (ValueError, TypeError) as error:
        raise CampaignRuleError("harness_readable", str(error)) from error


def load_harness(directory: Path) -> HarnessSpec:
    """Read the index written by the runner, refusing files outside its directory."""
    files = []
    for entry in json.loads((directory / "index.json").read_text()):
        path = entry["path"]
        target = (directory / path).resolve()
        if not target.is_relative_to(directory.resolve()):
            raise CampaignRuleError("harness_readable", "harness files must stay inside the harness directory")
        files.append((path, HarnessFile(entry["kind"], target.read_text())))
    spec = HarnessSpec(AGENT, tuple(files))
    parse_harness(spec)
    return spec
