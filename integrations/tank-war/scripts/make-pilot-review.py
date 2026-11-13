"""Build one self-contained, readable human review page from private #183 evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

NAMES = {
    "disable_assistance-v1": "关闭辅助设置",
    "reverse_prior_scalar_edit-v1": "修改模型权重",
    "wrong_legal_projection-v1": "把合法动作映射错",
}


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def build(root: Path, template: Path, out: Path) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    for arm in ("harness-disable-assistance", "weights-reverse-edit", "controller-wrong-projection"):
        for packet in _lines(root / "private" / "evidence" / arm / "samples.jsonl"):
            source = Path(packet["source_run"])
            comparisons = _lines(Path(packet["same_state_evidence"]))
            if not comparisons or not comparisons[-1]["locally_verified"]:
                raise ValueError(f"no local difference for {packet['sample_id']}")
            comparison = comparisons[-1]
            if comparison["decision_index"] != packet["decision_index"]:
                raise ValueError(f"window differs for {packet['sample_id']}")
            state = next(
                row for row in _lines(source / "states.jsonl") if row["decision_index"] == packet["decision_index"]
            )
            nearby_states = [
                row
                for row in _lines(source / "states.jsonl")
                if packet["window"][0] <= row["decision_index"] <= packet["window"][1]
            ]
            executions = {row["decision_index"]: row for row in _lines(source / "execution.jsonl")}
            if not nearby_states or any(row["decision_index"] not in executions for row in nearby_states):
                raise ValueError(f"incomplete nearby decisions for {packet['sample_id']}")
            if state["state_hash"] != packet["state_hash"]:
                raise ValueError(f"state hash differs for {packet['sample_id']}")
            manifest = _json(source / "manifest.json")
            reference = _json(Path(packet["reference_action"]))
            if reference["state_hash"] != packet["state_hash"]:
                raise ValueError(f"planner state differs for {packet['sample_id']}")
            raw = state["raw_state"]
            samples.append(
                {
                    "id": packet["sample_id"],
                    "claim": NAMES[packet["fault_instance"]],
                    "layer": packet["injected_layer"],
                    "seed": packet["seed"],
                    "seat": packet["seat"],
                    "decision": packet["decision_index"],
                    "time": state["time"],
                    "window": packet["window"],
                    "nearby": [
                        {
                            "decision": row["decision_index"],
                            "time": row["time"],
                            "selected": executions[row["decision_index"]]["selected_action"],
                            "executed": executions[row["decision_index"]]["executed_action"],
                        }
                        for row in nearby_states
                    ],
                    "stratum": packet["task_stratum"],
                    "state_hash": packet["state_hash"],
                    "scene": {
                        "walls": raw["maze"]["walls"],
                        "tanks": raw["tanks"],
                        "bullets": raw["bullets"],
                        "player": state["player"],
                        "time_left": raw["timeLeft"],
                    },
                    "faulty": comparison["faulty"],
                    "clean": comparison["clean"],
                    "changed": {
                        "choice": comparison["local_decision_changed"],
                        "command": comparison["local_command_changed"],
                        "execution": comparison["local_execution_changed"],
                    },
                    "planner": reference["acceptable_actions"],
                    "model_input": state["observation"]["state"],
                    "fault_manifest": {
                        "harness_sha256": manifest["harness_sha256"],
                        "controller_sha256": manifest["controller_sha256"],
                        "weights_sha256": manifest["model"]["weights_sha256"],
                    },
                }
            )
    if len(samples) != 15 or len({item["id"] for item in samples}) != 15:
        raise ValueError("expected 15 distinct samples")
    html = template.read_text(encoding="utf-8").replace(
        "__SAMPLE_DATA__", json.dumps(samples, ensure_ascii=False).replace("<", "\\u003c")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return {"samples": len(samples), "out": str(out), "bytes": out.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.root, Path(__file__).with_name("pilot-review-template.html"), args.out)))


if __name__ == "__main__":
    main()
