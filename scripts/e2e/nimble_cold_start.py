"""Check Nimble readiness with empty compiler caches and the unchanged tank request deadlines.

Run in the pinned Nimble GPU environment with Node 22 on PATH. Model weights and
source snapshots must already be cached; only compiler caches are reset.
"""

from __future__ import annotations

import argparse
import json
import os
import select
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", required=True, help="index shown by nvidia-smi inside the allocation")
    parser.add_argument("--nimble-source", type=Path, required=True)
    parser.add_argument("--nimble-kit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="new output directory")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env.update(
        TRITON_CACHE_DIR=str(out / "triton"),
        TORCHINDUCTOR_CACHE_DIR=str(out / "inductor"),
        XDG_CACHE_HOME=str(out / "cache"),
        PYTHONPATH=os.pathsep.join(filter(None, [str(REPO), env.get("PYTHONPATH")])),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TANK_SECONDS="20",
    )
    command = [
        sys.executable,
        "-m",
        "honing.backends.jev.server",
        "--family",
        "nimble",
        "--pins",
        str(REPO / "integrations/ra2/pins.json"),
        "--nimble-source",
        str(args.nimble_source.resolve()),
        "--nimble-kit",
        str(args.nimble_kit.resolve()),
        "--gpu",
        args.gpu,
    ]
    started = time.monotonic()
    with (out / "server.stderr").open("w") as stderr:
        server = subprocess.Popen(command, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=stderr, text=True)
        try:
            assert server.stdout is not None
            ready, _, _ = select.select([server.stdout], [], [], 900)
            if not ready:
                raise TimeoutError("Nimble did not report ready within 900 seconds")
            line = server.stdout.readline()
            (out / "ready.json").write_text(line)
            message = json.loads(line)
            assert message["ready"]
            ready_seconds = time.monotonic() - started
            env["HONING_URL"] = message["url"]
            rows = []
            for run in (1, 2):
                log_path = out / f"tank-{run}.jsonl"
                with log_path.open("w") as log:
                    subprocess.run(
                        ["node", "scripts/e2e/tank-war.mjs"],
                        cwd=REPO,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=240,
                    )
                modes = [json.loads(line) for line in log_path.read_text().splitlines()]
                assert len(modes) == 2 and all(row["errors"] == 0 for row in modes)
                rows.append(modes)
            summary = {"passed": True, "ready_seconds": ready_seconds, "runs": rows}
            (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps({"passed": True, "ready_seconds": ready_seconds, "out": str(out)}))
        finally:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
            if server.stdout is not None:
                server.stdout.close()


if __name__ == "__main__":
    main()
