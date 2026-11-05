"""Local preparation, launch, JSONL control and model-loop commands."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from .controller import run_episode
from .native import NativeClient
from .runtime import launch_game, prepare_game, stop_launch


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="operation", required=True)
    prepare = commands.add_parser("prepare", help="Instrument a private copy of the pinned author release")
    prepare.add_argument("--game-dir", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--mono", default=shutil.which("mono") or "mono")
    prepare.add_argument("--mcs", default=shutil.which("mcs") or "mcs")
    prepare.add_argument("--cecil", type=Path, required=True)
    prepare.add_argument("--mono-config", type=Path)
    prepare.add_argument("--mono-root", type=Path, help="Optional root for a user-local extracted Mono installation")
    launch = commands.add_parser("launch", help="Start a new isolated native player; no existing game is reused")
    launch.add_argument("--game-dir", type=Path, required=True)
    launch.add_argument("--run-dir", type=Path, required=True)
    launch.add_argument("--wine", default=shutil.which("wine") or "wine")
    launch.add_argument("--xvfb", default=shutil.which("Xvfb") or "Xvfb")
    launch.add_argument("--display")
    launch.add_argument(
        "--wineserver", help="Matching Wine server; defaults to the sibling of the resolved Wine binary"
    )
    client = commands.add_parser("client", help="Serve state/reset/step requests on stdin/stdout as JSONL")
    client.add_argument("--ipc-dir", type=Path, required=True)
    client.add_argument("--timeout-seconds", type=float, default=30)
    run = commands.add_parser("run", help="Ask an existing Honing decision server and record native action receipts")
    run.add_argument("--ipc-dir", type=Path, required=True)
    run.add_argument("--decision-server", required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--decisions", type=int, default=12)
    run.add_argument("--frames", type=int, default=50)
    run.add_argument("--reset", action="store_true")
    run.add_argument("--seed", type=int, default=7)
    run.add_argument("--run-id")
    run.add_argument("--timeout-seconds", type=float, default=90)
    stop = commands.add_parser("stop", help="Stop only processes whose recorded launch identities still match")
    stop.add_argument("--launch-receipt", type=Path, required=True)
    return result


def client_loop(client: NativeClient) -> None:
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if not isinstance(request, dict) or request.get("op") not in ("state", "reset", "step"):
                raise ValueError("request must be an object with op state/reset/step")
            operation = request["op"]
            allowed = {"state": {"op"}, "reset": {"op", "seed"}, "step": {"op", "action", "frames"}}[operation]
            if set(request) - allowed:
                raise ValueError("unknown request fields")
            fields = {key: value for key, value in request.items() if key != "op"}
            state = getattr(client, operation)(**fields)
            response = {"ok": True, "state": state}
        except (ValueError, TypeError, RuntimeError, OSError) as error:
            response = {"ok": False, "error": str(error)}
        print(json.dumps(response, ensure_ascii=False, allow_nan=False), flush=True)


def main() -> None:
    args = parser().parse_args()
    if args.operation == "prepare":
        result = prepare_game(
            args.game_dir, args.output, args.mono, args.mcs, args.cecil, args.mono_config, args.mono_root
        )
    elif args.operation == "launch":
        result = launch_game(args.game_dir, args.run_dir, args.wine, args.display, args.xvfb, args.wineserver)
    elif args.operation == "client":
        client_loop(NativeClient(args.ipc_dir, timeout_seconds=args.timeout_seconds))
        return
    elif args.operation == "run":
        result = run_episode(
            args.ipc_dir,
            args.decision_server,
            args.output,
            args.decisions,
            args.frames,
            args.reset,
            args.seed,
            args.run_id,
            args.timeout_seconds,
        )
    else:
        result = stop_launch(args.launch_receipt)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
