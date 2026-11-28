"""Offline pre-flight of the whole collector chain on a recorded clean v9_keys run (no game, no model, no network).

  python3 -B offline.py <recorded run_keys run> <out root> [--decisions N] [--condition C --fault-instance I --fault-form F]

collect.py runs unchanged except that the game is OfflineGame (it serves the recorded bridge outputs: visible and
debug observations, in the order the harness plays its windows) and the decision service is RecordedLaya (it answers
every request with the probabilities recorded at that decision, echoing the request's id and full state text, as
Laya does). With the recorded choices the harness asks for exactly the recorded windows, so a clean pre-flight
reproduces the recorded run. With a harness fault the model's request differs but the recorded answers keep the
same choices; with a controller fault the keys sent differ from the recording while the served observations stay the
recorded ones: these pre-flights test the plumbing and the records, never game behaviour. Everything written is
marked as a pre-flight (manifest label) and is not a dataset episode.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from replay_env import rows


class OfflineGame:
    """KeysEnv-compatible stand-in serving one recorded session."""

    def __init__(self, out_dir, scene="new", capture=False, options=None, display=None, bridge_env=None, source=None):
        self.out = Path(out_dir)
        self.session = self.out / "session"
        self.session.mkdir(parents=True, exist_ok=False)
        self.src = Path(source)
        self.opt = dict(options or {})
        self.key_filter = None
        self.auto_frames = 0
        self.windows = self.frames = self.real_frames = 0

    def start(self):
        s = self.src
        self.stream = rows(s / "visible_observations.jsonl")
        self.debug = rows(s / "observations.jsonl")
        self.records = rows(s / "windows.jsonl")
        self.bridge = rows(s / "bridge_windows.jsonl")
        if any(r.get("adapter_windows") for r in self.records) or len(self.bridge) != len(self.records):
            raise ValueError("the offline game serves runs without adapter windows only")
        shutil.copyfile(s / "bridge.log", self.session / "bridge.log")
        self.k = 0
        self.bridge_clock = int(self.bridge[0]["bridge_tick"]) if self.bridge else 0
        self._emit(0)
        self._id, self._tick = 1, 0
        self.obs = self._present(self.raw, 1, 0)
        (self.out / "launch.json").write_text(json.dumps({"game": "offline", "source": str(s)}) + "\n")
        return self.obs

    def _emit(self, p: int) -> None:
        """The bridge's outputs after p windows: the visible and the debug observation (recorded), stamped with this
        offline game's bridge clock (it runs ahead of the recording under a controller timing fault)."""
        raw = json.loads(json.dumps(self.stream[p]))
        debug = json.loads(json.dumps(self.debug[p]))
        raw["frame"]["tick"] = debug["frame"] = self.bridge_clock
        self.raw = raw
        with (self.session / "visible_observations.jsonl").open("a") as f:
            f.write(json.dumps(raw) + "\n")
        with (self.session / "observations.jsonl").open("a") as f:
            f.write(json.dumps(debug) + "\n")

    @staticmethod
    def _present(raw, ident, tick):
        obs = json.loads(json.dumps(raw))
        obs["frame"]["id"], obs["frame"]["tick"] = ident, tick
        return obs

    def step_keys(self, frames, hold=(), press=(), first=(), label="", meta=None):
        r = self.records[self.k]
        requested = {"frames": int(frames), "hold": list(hold), "press": list(press), "first": list(first)}
        logged = r.get("requested") or {
            "frames": r["frames"],
            "hold": r["hold"],
            "press": r["press"],
            "first": r["first"],
        }
        if requested != {k: logged[k] for k in requested}:
            raise ValueError(f"offline game: the harness asked for {requested}, the recording has {logged}")
        sent, change = dict(requested), None
        if self.key_filter is not None:
            sent, change = self.key_filter(json.loads(json.dumps(requested)), dict(meta or {}))
        pid, ptick = self._id, self._tick
        record = {
            "obs_id": pid,
            "tick": ptick,
            "action": "keys",
            "hold": sent["hold"],
            "press": sent["press"],
            "first": sent["first"],
            "frames": sent["frames"],
            "label": label,
            "requested": requested,
        }
        if change is not None:
            record["controller_fault"] = change
        b = self.bridge[self.k]
        # the bridge clock of this offline game advances by the frames really sent (a controller timing fault makes
        # it run ahead of the recording, whose observations are served unchanged)
        bridge_tick = self.bridge_clock
        self.bridge_clock += int(sent["frames"])
        command = {
            "bridge_id": b["bridge_id"],
            "bridge_tick": bridge_tick,
            "why": label,
            "action": "keys",
            "frames": sent["frames"],
            "hold": sent["hold"],
            "press": sent["press"],
        }
        if sent["first"]:
            command["first"] = sent["first"]
        with (self.session / "bridge_windows.jsonl").open("a") as f:
            f.write(json.dumps(command) + "\n")
        self.k += 1
        self._emit(self.k)
        record["real_frames"] = sent["frames"]
        self._id, self._tick = pid + 1, ptick + sent["frames"]
        self.obs = self._present(self.raw, self._id, self._tick)
        self.windows += 1
        self.frames += sent["frames"]
        if meta:
            record.update(meta)
        record["t_wall"] = round(time.time(), 4)
        with (self.session / "windows.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        return self.obs

    def close(self):
        return {"windows": self.windows, "frames": self.frames, "offline": True}


class RecordedLaya:
    """Client stand-in: the probabilities recorded at each decision, with the request's own id and state text."""

    def __init__(self, url, decisions=None, plan=None, role="clean"):
        self.url, self.decisions, self.plan, self.role = url, decisions, plan, role
        self.k = 0

    def health(self):
        sha = self.plan[f"{self.role}_weights_sha256"]
        return {
            "ready": True,
            "maxBranches": 16,
            "metadata": {
                "role": self.role,
                "weightsSha256": sha,
                "repo": "offline/recorded",
                "revision": "0" * 40,
                "backend": "offline-recorded",
            },
        }

    def decide(self, request, raw_path=None):
        d = self.decisions[self.k]
        self.k += 1
        state = "\n".join(t for s in ("essential", "details", "memory", "logs") for t in request["parts"].get(s, []))
        answers, audit = {}, {}
        for q, question in request["questions"].items():
            probs = d["answers"][q]["probabilities"]
            ordered = {o: probs[o] for o in question["criteria"]}
            choice = d["answers"][q]["choice"]  # Laya's own tie-break, as recorded
            answers[q] = {"type": "choice", "choice": choice, "probabilities": ordered}
            audit[q] = {
                "stateTokens": 400,
                "stateRoom": 1800,
                "inputTokens": 700,
                "options": len(ordered),
                "stateTruncated": False,
            }
        reply = {
            "answers": answers,
            "modelResult": {"model": "offline-recorded", "answers": answers},
            "stateText": state,
            "tokenAudit": audit,
            "trimmed": [],
            "latencyMs": 0.0,
            "id": request["id"],
        }
        raw = json.dumps(reply).encode()
        if raw_path is not None:
            Path(raw_path).write_bytes(raw)
        return reply, 0.0


def preflight(
    source: Path, out: Path, decisions: int, condition: str, instance: str | None, form: str, seed: int = 1
) -> Path:
    import collect

    rec = rows(source / "session" / "decisions.jsonl")
    plan = json.loads((HERE / "plan.json").read_text())
    role = "alternate" if condition == "weights" else "clean"
    argv = [
        "--out",
        str(out),
        "--task",
        "kp-smoke-d60",
        "--seed",
        str(seed),
        "--condition",
        condition,
        "--display",
        "379",
        "--label",
        f"OFFLINE PRE-FLIGHT on {source.name} (not a dataset episode)",
        "--max-decisions",
        str(decisions),
        "--fault-form",
        form,
    ]
    if instance:
        argv += ["--fault-instance", instance]
    collect.main(
        argv,
        env_factory=lambda *a, **k: OfflineGame(*a, source=source / "session", **k),
        client_factory=lambda url: RecordedLaya(url, rec, plan, role),
        display_check=False,
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--decisions", type=int, default=60)
    ap.add_argument("--condition", default="no_injection")
    ap.add_argument("--fault-instance", default=None)
    ap.add_argument("--fault-form", default="persistent")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    preflight(a.source, a.out, a.decisions, a.condition, a.fault_instance, a.fault_form, a.seed)


if __name__ == "__main__":
    main()
