"""Run the frozen base/candidate validation, selection, then heldout pilot on Windows."""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import time
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from integrations.overcooked.decisions import DecisionClient
from integrations.overcooked.engine import ROOT, digest, write_json
from integrations.overcooked.experiments import experiment
from integrations.overcooked.select_candidate import compare

PYTHON = "/opt/honing/venvs/nanojev/bin/python"
ENDPOINT = "http://127.0.0.1:8803"


def linux_path(path: Path) -> str:
    path = path.resolve()
    return f"/mnt/{path.drive[0].lower()}/" + path.as_posix()[3:]


def helper(*args: str) -> list[str]:
    return ["wsl.exe", "-d", "Ubuntu", "--exec", PYTHON, linux_path(ROOT / "model_env.py"), *args]


class Server:
    def __init__(self) -> None:
        self.process: subprocess.Popen[bytes] | None = None

    def start(self, checksum: str, manifest: str | None, label: str) -> None:
        self.stop()
        args = ["serve", "--port", "8803", "--inference", "reference"]
        if manifest:
            args.extend(["--checkpoint-manifest", manifest])
        log_path = ROOT / f".cache/pilot-{label}-server.log"
        with log_path.open("ab") as log:
            self.process = subprocess.Popen(
                helper(*args), stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW
            )
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"model server failed; inspect {log_path}")
            try:
                health = DecisionClient(ENDPOINT, timeout=2).health()
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                time.sleep(1)
                continue
            if health["metadata"].get("weightsSha256") != checksum:
                raise ValueError("server checksum differs from the frozen evaluation checkpoint")
            print(json.dumps({"server_ready": label, "checksum": checksum}), flush=True)
            return
        raise TimeoutError(f"model startup timed out; inspect {log_path}")

    def stop(self) -> None:
        subprocess.run(helper("stop", "--port", "8803"), check=True, timeout=30, capture_output=True)
        if self.process:
            self.process.wait(timeout=30)
            self.process = None


def stage(root: Path, label: str, split: str, checksum: str | None = None) -> dict[str, Any]:
    out = root / label
    protocol_path = ROOT / "experiments/pilot-v1.json"
    protocol = json.loads(protocol_path.read_text())
    identity = digest({"protocol": protocol, "suite": (ROOT / protocol["suite"]).read_text()})
    if (out / "summary.json").exists():
        saved = json.loads((out / "summary.json").read_text())
        if not saved["complete"]:
            raise ValueError(f"partial stage retained at {out}; use a new output root for a fresh attempt")
        if saved["protocol_sha256"] != identity or saved["split"] != split:
            raise ValueError("saved stage uses a different protocol")
        if checksum and saved.get("model", {}).get("weightsSha256") != checksum:
            raise ValueError("saved stage evaluated a different checkpoint")
        print(json.dumps({"reuse_completed_stage": label}), flush=True)
        return saved
    return asyncio.run(
        experiment(
            SimpleNamespace(
                protocol=protocol_path,
                split=split,
                policy="model" if checksum else "baseline",
                endpoint=ENDPOINT,
                expected_checksum=checksum,
                harness_dir=None,
                out=out,
            )
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True, help="completed training receipt.json")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/pilot-v1")
    args = parser.parse_args()
    training = json.loads(args.training.read_text())
    base_checksum = training["base"]["checkpoint"]["checksum"]
    candidate_checksum = training["snapshot"]["checkpoint"]["checksum"]
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=True)
    server = Server()
    try:
        server.start(base_checksum, None, "base")
        incumbent = stage(root, "base-val", "val", base_checksum)
        server.start(candidate_checksum, training["checkpoint_manifest"], "candidate")
        candidate = stage(root, "candidate-val", "val", candidate_checksum)
        verdict = compare(incumbent, candidate, training)
        verdict_path = root / "selection.json"
        if verdict_path.exists() and json.loads(verdict_path.read_text()) != verdict:
            raise ValueError("existing selection changed; refusing to reuse heldout")
        write_json(verdict_path, verdict)
        print(json.dumps({"selection": verdict["decision"]}), flush=True)
        if verdict["decision"]["outcome"] == "reject":
            server.start(base_checksum, None, "selected-base")
        # No heldout task is opened until the validation verdict is persisted.
        heldout = stage(root, "selected-heldout", "heldout", verdict["selected_checksum"])
        stage(root, "rule-val", "val")
        stage(root, "rule-heldout", "heldout")
        write_json(
            root / "completed.json",
            {
                "selection": verdict,
                "heldout_mean": heldout["mean_native_score"],
                "endpoint": ENDPOINT,
                "server_pid": server.process.pid,
                "candidate_frozen_before_validation": str(args.training.resolve()),
            },
        )
        print(
            json.dumps(
                {
                    "complete": True,
                    "selected_checksum": verdict["selected_checksum"],
                    "heldout_mean": heldout["mean_native_score"],
                    "endpoint": ENDPOINT,
                }
            ),
            flush=True,
        )
        # Leave the selected checkpoint available for the user's interactive GUI.
    except BaseException:
        server.stop()
        raise


if __name__ == "__main__":
    main()
