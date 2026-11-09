"""Run synthetic fault injection against a real paused Taisei process."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("taisei_demo", HERE / "run.py")
assert spec is not None and spec.loader is not None
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


def verify(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=False)
    session = args.output.resolve()
    command = [str(args.game.resolve()), "--play", "--sid", "1", "--diff", "Easy", "--shotmode", "youmuA"]
    environment = {
        **os.environ,
        "DISPLAY": args.display,
        "HONING_SESSION_DIR": str(session),
        "TAISEI_STORAGE_PATH": str(session / "storage"),
        "TAISEI_CACHE_PATH": str(session / "cache"),
        "TAISEI_RES_PATH": str(args.resources.resolve()),
        "TAISEI_NOASYNC": "1",
        "TAISEI_TASKMGR_NUM_THREADS": "4",
        "LIBGL_ALWAYS_SOFTWARE": "1",
        "LP_NUM_THREADS": "4",
    }
    number = args.display.removeprefix(":").split(".")[0]
    if Path(f"/tmp/.X11-unix/X{number}").exists():
        raise ValueError("Display is occupied")
    checks = []
    with (session / "display.log").open("wb") as display_log, (session / "game.log").open("wb") as game_log:
        xvfb = subprocess.Popen(
            ["Xvfb", args.display, "-screen", "0", "800x600x24", "-nolisten", "tcp"],
            stdout=display_log,
            stderr=subprocess.STDOUT,
        )
        game = None
        try:
            time.sleep(0.5)
            if xvfb.poll() is not None:
                raise RuntimeError("Owned Xvfb failed")
            game = subprocess.Popen(command, env=environment, stdout=game_log, stderr=subprocess.STDOUT)
            native = demo.NativeClient(session, game)
            observation = native.observe()
            before = (session / "observation.json").read_bytes()
            if args.watchdog_only:
                started = time.monotonic()
                writes = 0
                while game.poll() is None and time.monotonic() - started < 255:
                    demo.atomic_json(session / "action.json", {"id": 0, "action": "right", "frames": 30})
                    writes += 1
                    time.sleep(0.1)
                    assert before == (session / "observation.json").read_bytes()
                elapsed = time.monotonic() - started
                assert game.poll() == 74, f"Watchdog failed: exit {game.poll()}, elapsed {elapsed}"
                assert 239 <= elapsed <= 250
                demo.atomic_json(
                    session / "watchdog-check.json",
                    {
                        "syntheticFaultInjection": True,
                        "staleCommandWrites": writes,
                        "wallSeconds": elapsed,
                        "nativeExit": game.returncode,
                        "pausedFrame": observation["frame"],
                        "simulationAdvanced": False,
                    },
                )
                return
            calls = []
            real_step = native.step
            native.step = lambda *values: calls.append(values)
            for fault in ("zero", "two", "boolean", "clipped", "required_missing"):
                request = demo.request_for(observation, 30)
                response = {
                    "id": observation["id"],
                    "stateText": "\n".join(request["parts"]["essential"]),
                    "trimmed": [],
                    "tokenAudit": {"action": {"stateTruncated": False, "options": 9}},
                    "answers": {
                        "action": {
                            "choice": "right",
                            "probabilities": {key: float(key == "right") for key in demo.ACTIONS},
                        }
                    },
                }
                if fault in ("zero", "two", "boolean"):
                    value = {"zero": 0, "two": 2, "boolean": True}[fault]
                    response["answers"]["action"]["probabilities"] = dict.fromkeys(demo.ACTIONS, value)
                elif fault == "clipped":
                    response["tokenAudit"]["action"]["stateTruncated"] = True
                else:
                    response["stateText"] = ""
                raw = json.dumps(response).encode()

                class FaultClient:
                    def decide(self, _request, *, raw_path, fault_bytes=raw, fault_response=response):
                        raw_path.write_bytes(fault_bytes)
                        return fault_response

                try:
                    demo.decision_step(FaultClient(), native, observation, 30)
                except ValueError:
                    pass
                else:
                    raise AssertionError("Synthetic faulty response reached native boundary")
                time.sleep(0.05)
                assert not calls and before == (session / "observation.json").read_bytes()
                (session / f"rejected-{fault}.raw").write_bytes(raw)
                checks.append({"fault": fault, "nativeCalls": len(calls), "pausedFrame": observation["frame"]})
            native.step = real_step
            for fault, raw in (
                ("stale", b'{"id":0,"action":"right","frames":30}'),
                ("nul", b'{"id":1,"action":"right","frames":30}\0extra'),
            ):
                (session / "rejected-command.json").unlink(missing_ok=True)
                (session / "action.tmp").write_bytes(raw)
                (session / "action.tmp").replace(session / "action.json")
                deadline = time.monotonic() + 5
                while not (session / "rejected-command.json").exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                receipt = json.loads((session / "rejected-command.json").read_text())
                assert receipt["frame"] == observation["frame"]
                assert (session / "rejected-command.raw").read_bytes() == raw
                assert before == (session / "observation.json").read_bytes()
                checks.append({"fault": fault, "pausedFrame": receipt["frame"]})
            native.step(observation["id"], "right", 30)
            after = native.observe()
            assert after["frame"] - observation["frame"] == 30
            native.step(after["id"], "stop", 1)
            assert game.wait(timeout=20) == 0
            demo.atomic_json(
                session / "native-checks.json",
                {
                    "syntheticFaultInjection": True,
                    "checks": checks,
                    "validStepTicks": 30,
                    "start": observation,
                    "after": after,
                    "gameExit": game.returncode,
                },
            )
        finally:
            for owned in (game, xvfb):
                if owned is not None and owned.poll() is None:
                    owned.terminate()
                    try:
                        owned.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        owned.kill()
                        owned.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--display", default=":191")
    parser.add_argument(
        "--watchdog-only", action="store_true", help="Verify the real 240s deadline under stale-command flood"
    )
    verify(parser.parse_args())


if __name__ == "__main__":
    main()
