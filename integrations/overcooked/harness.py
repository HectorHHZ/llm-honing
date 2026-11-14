"""Integration-local file delivery using the existing Honing HarnessSpec."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from honing.core import HarnessFile, HarnessSpec
from integrations.overcooked.engine import ROOT, write_json

AGENT = "overcooked_decision"


def seed_harness() -> HarnessSpec:
    return HarnessSpec(
        AGENT,
        tuple(
            (
                f"overcooked/{name}.json",
                HarnessFile(kind, (ROOT / "config" / f"{name}.json").read_text(encoding="utf-8")),
            )
            for name, kind in (("controller", "settings"), ("questions", "instructions"))
        ),
    )


def parse_harness(spec: HarnessSpec) -> dict[str, Any]:
    if spec.agent != AGENT or not spec.kinds <= {"settings", "instructions"}:
        raise ValueError("Overcooked requires its own agent and settings/instructions files")
    controller, question = spec.get("overcooked/controller.json"), spec.get("overcooked/questions.json")
    if controller is None or question is None or controller.kind != "settings" or question.kind != "instructions":
        raise ValueError("controller and question files are required with their correct kinds")
    settings, questions = json.loads(controller.content), json.loads(question.content)
    if (
        not isinstance(settings, dict)
        or set(settings) != {"history_steps"}
        or type(settings["history_steps"]) is not int
        or not 0 <= settings["history_steps"] <= 16
    ):
        raise ValueError("controller must specify history_steps in 0..16")
    if (
        not isinstance(questions, dict)
        or set(questions) != {"action"}
        or not isinstance(questions["action"], str)
        or not questions["action"].strip()
    ):
        raise ValueError("questions must specify nonempty action instructions")
    return {
        "history_steps": settings["history_steps"],
        "instructions": questions["action"],
        "notes": "\n\n".join(
            f"{p}:\n{f.content}"
            for p, f in spec.files
            if p not in {"overcooked/controller.json", "overcooked/questions.json"}
        ),
    }


def dump_harness(spec: HarnessSpec, directory: Path) -> None:
    parse_harness(spec)
    directory.mkdir(exist_ok=False)
    write_json(directory / "files.json", [{"path": p, "kind": f.kind, "content": f.content} for p, f in spec.files])


def load_harness(directory: Path) -> HarnessSpec:
    rows = json.loads((directory / "files.json").read_text(encoding="utf-8"))
    spec = HarnessSpec(AGENT, tuple((row["path"], HarnessFile(row["kind"], row["content"])) for row in rows))
    parse_harness(spec)
    return spec
