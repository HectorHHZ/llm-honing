"""Run one frozen Space Invaders arm on UCSD GPU 4 and return verifiable evidence.

The local mode stages an isolated job over SSH. ``--remote-job`` is an internal
entry point on ds-serv2; its GPU hold is restored in a finally block.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import time
import uuid
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
CODE_NAMES = ("lane_pilot.py", "lane_controller.py", "solo_intercept.py", "lane_io.py", "random_ab_server.py")
FLAGS = (
    "--replan-steps",
    "16",
    "--model-choice-mode",
    "all",
    "--scoreless-question",
    "--balanced-option-order",
    "--solo-predictive",
    "--calibrated-geometry",
    "--fire-while-dodging",
    "--landing-priority",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(*command: str, **kwargs: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, check=True, **kwargs)


def inspect_job(run_dir: Path, arm: str, seeds: list[int], code_dir: Path, expected_sha: str) -> dict[str, object]:
    config = json.loads((run_dir / "config.json").read_text())
    environment = config["environment"]
    required = {
        "ale": "ALE/SpaceInvaders-v5",
        "frameskip": 4,
        "sticky_action_probability": 0.25,
        "replan_steps": 16,
        "model_choice_mode": "all",
        "scoreless_question": True,
        "balanced_option_order": True,
        "solo_predictive": True,
        "calibrated_geometry": True,
        "fire_while_dodging": True,
        "landing_priority": True,
    }
    if environment != required:
        raise ValueError("Game runner only implements the pinned Space Invaders environment and flags")
    if arm not in ("incumbent", "candidate", "random", "base", "code"):
        raise ValueError("Unknown arm")
    development = json.loads((run_dir / "development-result.json").read_text())
    if arm == "incumbent":
        model = config["incumbent"]
    elif arm == "candidate":
        model = development["method_result"] if development["method"] == "sft" else config["incumbent"]
    elif arm == "base":
        model = config["base_model"]
    else:
        model = None
    if (model is None and expected_sha) or (model is not None and model["weights_sha256"] != expected_sha):
        raise ValueError("Arm weight checksum differs from frozen manifest")
    for name in CODE_NAMES:
        if not (code_dir / name).is_file():
            raise ValueError(f"Missing game code {name}")
    return {
        "config": config,
        "environment": environment,
        "model": model,
        "code_sha256": {name: sha(code_dir / name) for name in CODE_NAMES},
        "seeds": seeds,
        "arm": arm,
        "weights_sha256": expected_sha or None,
    }


def local(args: argparse.Namespace) -> None:
    run_dir, code_dir = args.run.resolve(), args.code_dir.resolve()
    seeds = [int(part) for part in args.seeds.split(",")]
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if seeds != manifest["seeds"][args.split]:
        raise ValueError("Runner seeds differ from frozen split")
    job = inspect_job(run_dir, args.arm, seeds, code_dir, args.weights_sha256)
    version = json.loads((run_dir / "pipeline-version.json").read_text())["source_sha256"]
    remote = job["config"]["remote"]
    if remote["host"] != "ds-serv2.ucsd.edu":
        raise ValueError("Only the authorized UCSD host is supported")
    pilot = remote["pilot_dir"]
    scratch = f"{pilot}/runs/honing-pipeline-{uuid.uuid4().hex[:12]}"
    stage = args.summary.parent / "remote-stage"
    stage.mkdir(parents=True, exist_ok=False)
    try:
        from pipeline_bundle import make_bundle

        job["bundle_sha256"] = make_bundle(HERE.parents[1], version, stage / "honing-bundle.tar.gz")
        for name in CODE_NAMES:
            shutil.copy2(code_dir / name, stage / name)
        shutil.copy2(__file__, stage / "pipeline_game_runner.py")
        shutil.copy2(HERE / "pipeline_bundle.py", stage / "pipeline_bundle.py")
        (stage / "job.json").write_text(json.dumps({**job, "remote": remote, "scratch": scratch}, indent=2) + "\n")
        run("ssh", remote["host"], "mkdir", "-p", scratch)
        try:
            for file in stage.iterdir():
                run("scp", str(file), f"{remote['host']}:{scratch}/{file.name}")
            run(
                "ssh",
                remote["host"],
                remote["game_python"],
                f"{scratch}/pipeline_game_runner.py",
                "--remote-job",
                f"{scratch}/job.json",
            )
            run("scp", f"{remote['host']}:{scratch}/summary.json", str(args.summary))
            run("scp", f"{remote['host']}:{scratch}/trace.jsonl.gz", str(args.trace))
            run(
                "scp",
                f"{remote['host']}:{scratch}/server-health.json",
                str(args.summary.parent / "server-health.json"),
            )
            run("scp", f"{remote['host']}:{scratch}/server.log", str(args.summary.parent / "server.log"))
            run("scp", f"{remote['host']}:{scratch}/game.log", str(args.summary.parent / "game.log"))
            if job["model"] is not None:
                run(
                    "scp",
                    f"{remote['host']}:{scratch}/model-snapshot.json",
                    str(args.summary.parent / "model-snapshot.json"),
                )
        finally:
            for name in ("server.log", "game.log", "server-health.json"):
                if not (args.summary.parent / name).exists():
                    subprocess.run(
                        ["scp", f"{remote['host']}:{scratch}/{name}", str(args.summary.parent / name)],
                        text=True,
                        capture_output=True,
                    )
            run("ssh", remote["host"], "rm", "-rf", scratch)
    finally:
        shutil.rmtree(stage)


def gpu_processes() -> list[tuple[str, int]]:
    output = run(
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader", capture_output=True
    ).stdout
    answer = []
    for line in output.splitlines():
        uuid_text, pid_text = [part.strip() for part in line.split(",", 1)]
        answer.append((uuid_text, int(pid_text)))
    return answer


def verify_hold(remote: dict[str, str]) -> int:
    pid_text = run(
        "systemctl", "--user", "show", remote["hold_unit"], "-p", "MainPID", "--value", capture_output=True
    ).stdout.strip()
    pid = int(pid_text)
    if pid <= 0:
        raise RuntimeError("GPU 4 hold has no active PID")
    command = run("ps", "-p", str(pid), "-o", "args=", capture_output=True).stdout.strip()
    if command != f"{remote['runtime']} {remote['hold_script']}":
        raise RuntimeError("GPU 4 hold command does not match the authorized process")
    if gpu_processes().count((remote["gpu_uuid"], pid)) != 1:
        raise RuntimeError("GPU 4 hold PID/UUID does not match nvidia-smi")
    return pid


def restore_hold(remote: dict[str, str]) -> None:
    for _attempt in range(5):
        run("systemctl", "--user", "reset-failed", remote["hold_unit"])
        result = subprocess.run(
            [
                "systemd-run",
                "--user",
                "--unit=gpu4-new-task-hold",
                "--description=Temporary GPU4 memory hold for user new task",
                f"--setenv=CUDA_VISIBLE_DEVICES={remote['gpu_uuid']}",
                remote["runtime"],
                remote["hold_script"],
            ],
            text=True,
            capture_output=True,
        )
        if result.returncode == 0:
            break
        time.sleep(2)
    else:
        raise RuntimeError("Could not restore GPU 4 hold")
    time.sleep(2)
    verify_hold(remote)


def await_health(port: int, process: subprocess.Popen[str]) -> dict[str, object]:
    for _attempt in range(120):
        if process.poll() is not None:
            raise RuntimeError("Model server exited before health check")
        try:
            with urlopen(f"http://127.0.0.1:{port}/v1/health", timeout=1) as response:
                return json.load(response)
        except (URLError, TimeoutError, ConnectionError):
            time.sleep(1)
    raise RuntimeError("Model server did not become healthy")


def port_is_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def remote(args: argparse.Namespace) -> None:
    job_path = args.remote_job.resolve()
    job = json.loads(job_path.read_text())
    scratch = job_path.parent
    remote_config = job["remote"]
    if socket.gethostname().split(".")[0] != "ds-serv2":
        raise RuntimeError("Remote game job must run on ds-serv2")
    if str(scratch) != job["scratch"] or not str(scratch).startswith(remote_config["pilot_dir"] + "/runs/"):
        raise RuntimeError("Untrusted remote scratch path")
    for name, expected in job["code_sha256"].items():
        if sha(scratch / name) != expected:
            raise RuntimeError(f"Staged {name} has wrong hash")
    from pipeline_bundle import extract_bundle

    extract_bundle(scratch / "honing-bundle.tar.gz", scratch, job["bundle_sha256"])
    model, arm = job["model"], job["arm"]
    server: subprocess.Popen[str] | None = None
    stopped = False
    health: dict[str, object] = {"ready": True, "metadata": {"family": "none", "weightsSha256": None}}

    def interrupt(signum: int, frame: object) -> None:
        raise KeyboardInterrupt(f"Interrupted by signal {signum}")

    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        if model is not None:
            if port_is_open(remote_config["port"]):
                raise RuntimeError("Model server port is already in use")
            verify_hold(remote_config)
            run("systemctl", "--user", "stop", remote_config["hold_unit"])
            stopped = True
            for _attempt in range(20):
                if not any(uuid_text == remote_config["gpu_uuid"] for uuid_text, _ in gpu_processes()):
                    break
                time.sleep(1)
            else:
                raise RuntimeError("GPU 4 did not become free")
            command = [
                remote_config["runtime"],
                "-m",
                "honing.backends.jev.server",
                "--family",
                "nanojev",
                "--pins",
                remote_config["pins"],
                "--nanojev-source",
                remote_config["source_dir"],
                "--gpu",
                "4",
                "--port",
                str(remote_config["port"]),
            ]
            if model.get("checkpoint_manifest"):
                command.extend(["--checkpoint-manifest", model["checkpoint_manifest"]])
            environment = os.environ.copy()
            environment.update(
                {
                    "PYTHONPATH": str(scratch),
                    "HF_HOME": remote_config["cache"],
                    "HF_HUB_OFFLINE": "1",
                    "USE_TF": "0",
                    "TOKENIZERS_PARALLELISM": "false",
                }
            )
            with (scratch / "server.log").open("w") as log:
                server = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT, text=True)
            health = await_health(remote_config["port"], server)
            if (
                not health.get("ready")
                or health["metadata"]["weightsSha256"] != job["weights_sha256"]
                or health["metadata"].get("gpuUuid") != remote_config["gpu_uuid"]
            ):
                raise RuntimeError("Model server loaded the wrong checkpoint or GPU")
            checkpoint = health["metadata"]["path"]
            snapshot_command = [
                remote_config["runtime"],
                "-c",
                "import json,sys; from pathlib import Path; "
                "from honing.backends.jev.nanojev import nanojev_snapshot; "
                "s=nanojev_snapshot(Path(sys.argv[1])); "
                "a=s.architecture; l=a.layers[0]; "
                "print(json.dumps({'checkpoint':s.checkpoint.location, "
                "'weights_sha256':s.checkpoint.checksum, 'parameters':s.parameters, "
                "'architecture':{'num_layers':a.num_layers,'hidden_size':a.hidden_size,"
                "'attention_heads':l.attention_heads,'ffn_size':l.ffn_size}}))",
                checkpoint,
            ]
            snapshot = run(*snapshot_command, env=environment, capture_output=True).stdout
            (scratch / "model-snapshot.json").write_text(snapshot)
        elif arm == "random":
            if port_is_open(18842):
                raise RuntimeError("Random A/B server port is already in use")
            with (scratch / "server.log").open("w") as log:
                server = subprocess.Popen(
                    [remote_config["game_python"], str(scratch / "random_ab_server.py")],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            for _attempt in range(30):
                if server.poll() is not None:
                    raise RuntimeError("Random A/B server exited")
                try:
                    with socket.create_connection(("127.0.0.1", 18842), timeout=1):
                        break
                except OSError:
                    time.sleep(1)
            else:
                raise RuntimeError("Random A/B server did not start")
        else:
            (scratch / "server.log").write_text("Code policy does not start a server.\n")
        (scratch / "server-health.json").write_text(json.dumps(health, indent=2) + "\n")
        policy = "best" if arm == "code" else "jev"
        game = [
            remote_config["game_python"],
            str(scratch / "lane_pilot.py"),
            "--policy",
            policy,
            "--seeds",
            ",".join(map(str, job["seeds"])),
            "--max-steps",
            "100000",
            *FLAGS,
            "--record",
            str(scratch / "trace.jsonl"),
        ]
        if policy == "jev":
            port = 18842 if arm == "random" else remote_config["port"]
            game.extend(["--decision-server", f"http://127.0.0.1:{port}"])
        environment = os.environ.copy()
        environment["PYTHONPATH"] = ":".join(
            (str(scratch), remote_config["pilot_dir"] + "/python", remote_config["pilot_dir"])
        )
        with (scratch / "game.log").open("w") as log, (scratch / "game.json").open("w") as result:
            subprocess.run(game, env=environment, stdout=result, stderr=log, timeout=14400, check=True)
        data = json.loads((scratch / "game.json").read_text())
        metadata = {
            "code_sha256": job["code_sha256"],
            "weights_sha256": job["weights_sha256"],
            "environment_sha256": hashlib.sha256(
                json.dumps(job["environment"], sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "model_health_sha256": sha(scratch / "server-health.json"),
        }
        (scratch / "summary.json").write_text(json.dumps({**data, "pipeline_metadata": metadata}, indent=2) + "\n")
        with (scratch / "trace.jsonl").open("rb") as source, gzip.open(scratch / "trace.jsonl.gz", "wb") as target:
            shutil.copyfileobj(source, target)
    finally:
        try:
            if server is not None:
                server.terminate()
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
        finally:
            try:
                if stopped:
                    restore_hold(remote_config)
            finally:
                signal.signal(signal.SIGTERM, previous)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-job", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--split", choices=("development", "validation", "gate"))
    parser.add_argument("--arm", choices=("incumbent", "candidate", "random", "base", "code"))
    parser.add_argument("--seeds")
    parser.add_argument("--code-dir", type=Path)
    parser.add_argument("--weights-sha256", default="")
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--trace", type=Path)
    args = parser.parse_args()
    if args.remote_job:
        remote(args)
    else:
        required = ("run", "split", "arm", "seeds", "code_dir", "summary", "trace")
        if any(getattr(args, field.replace("-", "_")) is None for field in required):
            parser.error("Local runner requires run, split, arm, seeds, code-dir, summary and trace")
        local(args)


if __name__ == "__main__":
    main()
