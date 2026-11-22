"""Run a small, paired real-model REA-Issac capability check for issue #184."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from honing.backends.rea_issac.harness import seed_harness
from honing.backends.rea_issac.match import run_match
from honing.backends.rea_issac.model import DecisionClient
from honing.backends.rea_issac.replay import audit_run
from honing.backends.rea_issac.scenarios import ReaIssacScenario
from honing.backends.rea_issac_agent.controller import CONTROLLER_MODES
from honing.backends.rea_issac_agent.faults import (
    CONTROLLER_SWAP_MOVE_AXES,
    HARNESS_REVERSE_MOVE,
    reverse_movement_harness,
    verify_clean_baseline,
)
from honing.backends.rea_issac_agent.match import audit_agent_run, run_agent_match

ROOT = Path(__file__).resolve().parents[3]
INTEGRATION = Path(__file__).resolve().parent
PINS = ROOT / "integrations/ra2/pins.json"


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _measures(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: result[key]
        for key in ("status", "success", "tick", "health", "rooms_visited", "rooms_cleared", "kills", "damage_taken")
    }


def run_preflight(
    server_url: str,
    output: Path,
    *,
    seeds: tuple[int, ...] = (1002, 2001),
    max_ticks: int = 900,
    decision_frames: int = 3,
    expected_revision: str | None = None,
    expected_weights_sha256: str | None = None,
    agent_controller: str | None = None,
    harness_fault: str | None = None,
    controller_fault: str | None = None,
) -> dict[str, Any]:
    """Require checkpoint identity, run matched seeds, and audit each saved game."""
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int for seed in seeds):
        raise ValueError("seeds must be distinct integers")
    if harness_fault not in (None, HARNESS_REVERSE_MOVE) or controller_fault not in (None, CONTROLLER_SWAP_MOVE_AXES):
        raise ValueError("unsupported REA-Issac fault")
    if harness_fault and controller_fault:
        raise ValueError("a preflight may inject only one component fault")
    if controller_fault and not agent_controller:
        raise ValueError("controller fault needs the model-backed agent controller")
    frozen = verify_clean_baseline() if harness_fault or controller_fault else None
    if output.exists():
        raise FileExistsError(output)
    pinned = json.loads(PINS.read_text())["nanojevModels"]["nanojev"]["revision"]
    revision = expected_revision or pinned
    health = DecisionClient(server_url).health()
    metadata = health["metadata"]
    if metadata.get("revision") != revision or not isinstance(metadata.get("weightsSha256"), str):
        raise ValueError("decision service does not attest the requested checkpoint revision and weight checksum")
    if expected_weights_sha256 is not None and metadata["weightsSha256"] != expected_weights_sha256:
        raise ValueError("decision service weights differ from the requested checksum")
    if frozen and (
        metadata["revision"] != frozen["checkpoint_sha256"] or metadata["weightsSha256"] != frozen["checkpoint_sha256"]
    ):
        raise ValueError("harness or controller fault needs the frozen clean checkpoint")
    # Validate the entire fixed task list before creating any output or making a model call.
    scenarios = [ReaIssacScenario(seed, max_ticks, decision_frames) for seed in seeds]
    output.mkdir(parents=True)
    spec = seed_harness(INTEGRATION)
    if harness_fault:
        spec = reverse_movement_harness(spec)
    summary: dict[str, Any] = {
        "purpose": "rea-issac-exp0-model-capability-preflight",
        "agent_controller": agent_controller,
        "harness_fault": harness_fault,
        "controller_fault": controller_fault,
        "revision": revision,
        "weights_sha256": metadata["weightsSha256"],
        "source_commit": metadata.get("sourceCommit"),
        "seeds": list(seeds),
        "max_ticks": max_ticks,
        "decision_frames": decision_frames,
        "games": [],
        "status": "running",
    }
    _write(output / "summary.json", summary)
    try:
        for scenario in scenarios:
            pair: dict[str, Any] = {"seed": scenario.seed}
            for arm in ("baseline", "model"):
                run_dir = output / f"s{scenario.seed}-{arm}"
                if arm == "model" and agent_controller:
                    result = run_agent_match(
                        scenario,
                        run_dir,
                        spec,
                        decision_server=server_url,
                        policy_checksum=metadata["weightsSha256"],
                        controller_mode=agent_controller,
                        controller_fault=controller_fault,
                    )
                    audit_agent_run(run_dir)
                else:
                    result = run_match(
                        scenario,
                        run_dir,
                        spec,
                        policy=arm,
                        decision_server=server_url if arm == "model" else None,
                        policy_checksum=metadata["weightsSha256"] if arm == "model" else None,
                    )
                    audit_run(run_dir)
                pair[arm] = {"run": run_dir.name, **_measures(result)}
                summary["games"].append({"seed": scenario.seed, "arm": arm, **pair[arm]})
                _write(output / "summary.json", summary)
        summary["status"] = "complete"
        _write(output / "summary.json", summary)
        return summary
    except BaseException:
        summary["status"] = "incomplete"
        _write(output / "summary.json", summary)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-server", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(1002, 2001))
    parser.add_argument("--max-ticks", type=int, default=900)
    parser.add_argument("--decision-frames", type=int, default=3)
    parser.add_argument("--expected-revision", help="default: pinned public NanoJev revision")
    parser.add_argument("--expected-weights-sha256")
    parser.add_argument("--agent-controller", choices=CONTROLLER_MODES)
    parser.add_argument("--harness-fault", choices=(HARNESS_REVERSE_MOVE,))
    parser.add_argument("--controller-fault", choices=(CONTROLLER_SWAP_MOVE_AXES,))
    args = parser.parse_args()
    result = run_preflight(
        args.decision_server,
        args.out,
        seeds=tuple(args.seeds),
        max_ticks=args.max_ticks,
        decision_frames=args.decision_frames,
        expected_revision=args.expected_revision,
        expected_weights_sha256=args.expected_weights_sha256,
        agent_controller=args.agent_controller,
        harness_fault=args.harness_fault,
        controller_fault=args.controller_fault,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
