"""File delivery for a Griddly agent; tasks fix rules, observations and action options."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from honing.core import CampaignRuleError, HarnessFile, HarnessSpec

AGENT = "griddly_decision"
CONTROLLER_PATH = "griddly/controller.json"
QUESTIONS_PATH = "griddly/questions.json"


@dataclass(frozen=True)
class GriddlyHarness:
    history_steps: int
    instructions: str
    notes: str
    files: tuple[dict[str, str], ...]


def seed_harness(integration_dir: Path) -> HarnessSpec:
    return HarnessSpec(
        AGENT,
        (
            (
                CONTROLLER_PATH,
                HarnessFile("settings", (integration_dir / "config/controller.json").read_text(encoding="utf-8")),
            ),
            (
                QUESTIONS_PATH,
                HarnessFile("instructions", (integration_dir / "config/questions.json").read_text(encoding="utf-8")),
            ),
        ),
    )


def parse_harness(spec: HarnessSpec) -> GriddlyHarness:
    try:
        if spec.agent != AGENT or not spec.kinds <= {"settings", "instructions"}:
            raise ValueError("Griddly accepts its own agent and settings/instructions files")
        controller, questions = spec.get(CONTROLLER_PATH), spec.get(QUESTIONS_PATH)
        if (
            controller is None
            or questions is None
            or controller.kind != "settings"
            or questions.kind != "instructions"
        ):
            raise ValueError("controller settings and question instructions are required")
        config, text = json.loads(controller.content), json.loads(questions.content)
        if not isinstance(config, dict) or set(config) != {"history_steps"}:
            raise ValueError("controller must define only history_steps")
        history = config["history_steps"]
        if type(history) is not int or not 0 <= history <= 8:
            raise ValueError("history_steps must be in 0..8")
        if (
            not isinstance(text, dict)
            or set(text) != {"action"}
            or not isinstance(text["action"], str)
            or not text["action"].strip()
        ):
            raise ValueError("questions must define nonempty action instructions")
        notes = "\n\n".join(
            f"Operating notes ({path}):\n{file.content}"
            for path, file in spec.files
            if path not in (CONTROLLER_PATH, QUESTIONS_PATH)
        )
        files = tuple(
            {"path": path, "kind": file.kind, "sha256": hashlib.sha256(file.content.encode("utf-8")).hexdigest()}
            for path, file in spec.files
        )
        return GriddlyHarness(history, text["action"], notes, files)
    except (ValueError, TypeError) as error:
        raise CampaignRuleError("harness_readable", str(error)) from error


def write_harness(spec: HarnessSpec, directory: Path) -> None:
    parse_harness(spec)
    directory.mkdir(parents=True, exist_ok=False)
    for path, file in spec.files:
        target = directory / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(file.content, encoding="utf-8")
    (directory / "index.json").write_text(
        json.dumps([{"path": p, "kind": f.kind} for p, f in spec.files]), encoding="utf-8"
    )


def load_harness(directory: Path) -> HarnessSpec:
    files = []
    for entry in json.loads((directory / "index.json").read_text(encoding="utf-8")):
        path = entry["path"]
        target = (directory / path).resolve()
        if not target.is_relative_to(directory.resolve()):
            raise CampaignRuleError("harness_readable", "harness path leaves its directory")
        files.append((path, HarnessFile(entry["kind"], target.read_text(encoding="utf-8"))))
    spec = HarnessSpec(AGENT, tuple(files))
    parse_harness(spec)
    return spec
