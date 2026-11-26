"""Verify the recorded Tank War Exp 0 v0 source and release configuration."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

DEFAULT_FREEZE = Path(__file__).resolve().parents[1] / "exp0/v0/freeze.json"


def recorded_file(repo: Path, commit: str, path: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo), "show", f"{commit}:{path}"])


def check(freeze: dict[str, Any], repo: Path) -> dict[str, Any]:
    commit = freeze["game"]["recorded_code_commit"]
    object_type = subprocess.check_output(["git", "-C", str(repo), "cat-file", "-t", commit], text=True).strip()
    if object_type != "commit":
        raise ValueError(f"recorded game object is not a commit: {commit}")
    integration = "integrations/tank-war/"
    checked = 0
    for name, expected in freeze["game"]["sources_sha256"].items():
        actual = hashlib.sha256(recorded_file(repo, commit, integration + name)).hexdigest()
        if actual != expected:
            raise ValueError(f"source hash mismatch: {name}: {actual} != {expected}")
        checked += 1
    pins = json.loads(recorded_file(repo, commit, "integrations/ra2/pins.json"))
    model = freeze["model"]
    if pins["nanojevModels"]["nanojev"]["repo"] != model["repo"]:
        raise ValueError("NanoJev repository differs from freeze")
    if pins["nanojevModels"]["nanojev"]["revision"] != model["revision"]:
        raise ValueError("NanoJev revision differs from freeze")
    if pins["nanojevSource"]["revision"] != model["source_revision"]:
        raise ValueError("NanoJev source revision differs from freeze")
    settings = json.loads(recorded_file(repo, commit, integration + "config/controller.json"))
    expected_settings = {
        "controllerMode": freeze["clean_harness"]["controller_mode"],
        "contextVersion": freeze["clean_harness"]["context_version"],
        "assistEnabled": freeze["clean_harness"]["assist_enabled"],
        "assistProfile": freeze["clean_harness"]["assist_profile"],
        "fireGuardEnabled": freeze["clean_harness"]["fire_guard_enabled"],
        "strategy": freeze["clean_harness"]["strategy"],
    }
    if settings != expected_settings:
        raise ValueError("clean controller settings differ from freeze")
    return {
        "game_commit": commit,
        "source_files_verified": checked,
        "model_pins_verified": True,
        "controller_verified": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path, help="repository containing the recorded game commit")
    parser.add_argument("--freeze", default=DEFAULT_FREEZE, type=Path)
    args = parser.parse_args()
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    print(json.dumps(check(freeze, args.repo), sort_keys=True))


if __name__ == "__main__":
    main()
