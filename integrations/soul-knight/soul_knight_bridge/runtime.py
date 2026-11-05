"""Prepare and own private native players without redistributing game resources."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from .native import NativeClient

ROOT = Path(__file__).resolve().parents[1]
CHECKOUT = ROOT.parents[1]


def sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def external_output(path: Path) -> Path:
    """Keep generated game resources and traces outside the Git checkout."""
    path = path.resolve()
    if path == CHECKOUT or CHECKOUT in path.parents:
        raise ValueError("Generated resources must be outside the Honing Git checkout")
    return path


def verify_files(directory: Path, expected: dict[str, Any]) -> None:
    for name, record in expected.items():
        path = directory / name
        if not path.is_file() or path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"Pinned author-release file mismatch: {name}")


def prepare_game(
    game_dir: Path,
    output: Path,
    mono: str,
    mcs: str,
    cecil: Path,
    mono_config: Path | None = None,
    mono_root: Path | None = None,
) -> dict[str, Any]:
    """Copy a verified author release privately and compile instrumentation only."""
    game_dir, output, cecil = game_dir.resolve(), external_output(output), cecil.resolve()
    if output == game_dir or game_dir in output.parents:
        raise ValueError("Prepared output must be outside the original game directory")
    pins = json.loads((ROOT / "pins.json").read_text(encoding="utf-8"))
    verify_files(game_dir, pins["game_files"])
    if output.exists():
        raise ValueError("Prepared output must be a fresh path; existing game data is never overwritten")
    if not cecil.is_file():
        raise FileNotFoundError("Supply an installed Mono.Cecil.dll with --cecil")
    env = dict(os.environ)
    env["MONO_PATH"] = str(cecil.parent)
    if mono_root is not None:
        mono_root = mono_root.resolve()
        if mono_config is None:
            mono_config = mono_root / "etc/mono/config"
        env["MONO_PATH"] = str(mono_root / "usr/lib/mono/4.5") + os.pathsep + str(cecil.parent)
        env["MONO_CFG_DIR"] = str(mono_root / "etc")
        env["LD_LIBRARY_PATH"] = str(mono_root / "usr/lib") + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    mono_command = [mono]
    if mono_config is not None:
        mono_command += ["--config", str(mono_config.resolve())]
    compiler = [*mono_command, mcs] if mcs.lower().endswith(".exe") else [mcs]
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(game_dir, output)
    managed = output / "Soul_Knight_Data/Managed"
    build = output / "honing-build"
    build.mkdir()
    references = (
        "netstandard.dll,Assembly-CSharp.dll,UnityEngine.CoreModule.dll,UnityEngine.Physics2DModule.dll,"
        "UnityEngine.InputLegacyModule.dll,UnityEngine.JSONSerializeModule.dll,"
        "UnityEngine.ImageConversionModule.dll,UnityEngine.UIModule.dll,UnityEngine.UI.dll"
    )
    commands = [
        [
            *compiler,
            "-target:library",
            f"-out:{managed / 'HoningBridge.dll'}",
            f"-lib:{managed}",
            f"-r:{references}",
            str(ROOT / "patches/HoningBridge.cs"),
        ],
        [*compiler, f"-out:{build / 'PatchAssembly.exe'}", f"-r:{cecil}", str(ROOT / "patches/PatchAssembly.cs")],
        [
            *mono_command,
            str(build / "PatchAssembly.exe"),
            str(managed / "Assembly-CSharp.dll"),
            str(managed / "HoningBridge.dll"),
            str(managed / "Assembly-CSharp.patched.dll"),
        ],
    ]
    with (build / "build.log").open("w", encoding="utf-8") as log:
        for command in commands:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    os.replace(managed / "Assembly-CSharp.patched.dll", managed / "Assembly-CSharp.dll")
    scripting_path = output / "Soul_Knight_Data/ScriptingAssemblies.json"
    scripting = json.loads(scripting_path.read_text(encoding="utf-8"))
    if "HoningBridge.dll" in scripting["names"]:
        raise ValueError("Author release unexpectedly already contains a bridge")
    scripting["names"].append("HoningBridge.dll")
    scripting["types"].append(16)
    write_json(scripting_path, scripting)
    inputs = [ROOT / "patches/HoningBridge.cs", ROOT / "patches/PatchAssembly.cs", ROOT / "pins.json"]
    receipt = {
        "kind": "honing.soul-knight.prepared.v1",
        "game_dir": str(output),
        "source_game_dir": str(game_dir),
        "source_commit": pins["source_commit"],
        "release_tag": pins["release_tag"],
        "original_files": pins["game_files"],
        "integration_inputs": {str(path.relative_to(ROOT)): sha256(path) for path in inputs},
        "generated_files": {
            str(path.relative_to(output)): sha256(path)
            for path in (managed / "Assembly-CSharp.dll", managed / "HoningBridge.dll", scripting_path)
        },
        "commands": commands,
        "gameplay_rebuilt": False,
    }
    write_json(output / "honing-prepared.json", receipt)
    return receipt


def process_identity(pid: int) -> dict[str, Any] | None:
    try:
        status = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        return {
            "pid": pid,
            "start_ticks": status[19],
            "command": [part.decode() for part in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if part],
        }
    except FileNotFoundError:
        return None


def stop_launch(receipt_path: Path) -> dict[str, Any]:
    """Signal only processes whose PID birth and command still match this launch."""
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("kind") != "honing.soul-knight.launch.v1":
        raise ValueError("Unrecognized launch receipt")
    run_dir = Path(receipt["run_dir"]).resolve()
    if receipt_path.resolve() != run_dir / "launch.json":
        raise ValueError("Receipt must remain inside its original private run directory")
    prefix = run_dir / "wine-prefix"
    expected_prefix = receipt["wine_prefix_identity"]
    prefix_stat = prefix.stat()
    prefix_matches = (
        not prefix.is_symlink()
        and prefix_stat.st_dev == expected_prefix["device"]
        and prefix_stat.st_ino == expected_prefix["inode"]
    )
    wineserver = Path(receipt["wineserver"])
    server_matches = wineserver.is_file() and sha256(wineserver) == receipt["wineserver_sha256"]
    prefix_result: dict[str, Any] = {"identity_matches": prefix_matches, "binary_matches": server_matches}
    if prefix_matches and server_matches:
        env = dict(os.environ, WINEPREFIX=str(prefix), WINEARCH="win64")
        with (run_dir / "wineserver-cleanup.log").open("w", encoding="utf-8") as log:
            try:
                kill = subprocess.run(
                    [str(wineserver), "-k"], env=env, stdout=log, stderr=subprocess.STDOUT, timeout=10, check=False
                )
                wait = subprocess.run(
                    [str(wineserver), "-w"], env=env, stdout=log, stderr=subprocess.STDOUT, timeout=10, check=False
                )
                prefix_result.update(stop_exit_code=kill.returncode, wait_exit_code=wait.returncode)
            except subprocess.TimeoutExpired:
                prefix_result["error"] = "Scoped wineserver cleanup timed out; no global or stronger signal sent"
    results = []
    for expected in receipt["processes"]:
        actual = process_identity(expected["pid"])
        if actual is None:
            results.append({"pid": expected["pid"], "already_exited": True})
        elif actual != expected:
            results.append({"pid": expected["pid"], "stopped": False, "reason": "Process identity changed"})
        else:
            os.kill(expected["pid"], signal.SIGTERM)
            results.append({"pid": expected["pid"], "signalled": "SIGTERM", "ownership_verified": True})
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if all(process_identity(expected["pid"]) != expected for expected in receipt["processes"]):
            break
        time.sleep(0.05)
    for result, expected in zip(results, receipt["processes"], strict=True):
        result["original_identity_exited"] = process_identity(expected["pid"]) != expected
    cleanup = {
        "launch_receipt": str(receipt_path.resolve()),
        "processes": results,
        "wine_prefix": prefix_result,
        "complete": all(result["original_identity_exited"] for result in results)
        and prefix_result.get("wait_exit_code") == 0,
    }
    write_json(receipt_path.with_name("cleanup.json"), cleanup)
    return cleanup


def choose_display() -> str:
    for number in range(90, 250):
        if not Path(f"/tmp/.X{number}-lock").exists() and not Path(f"/tmp/.X11-unix/X{number}").exists():
            return f":{number}"
    raise RuntimeError("No unused local X display found")


def launch_game(
    game_dir: Path,
    run_dir: Path,
    wine: str,
    display: str | None = None,
    xvfb: str = "Xvfb",
    wineserver: str | None = None,
) -> dict[str, Any]:
    """Start a new private Wine prefix and offscreen Unity player, paused initially."""
    game_dir, run_dir = game_dir.resolve(), external_output(run_dir)
    prepared = json.loads((game_dir / "honing-prepared.json").read_text(encoding="utf-8"))
    pins = json.loads((ROOT / "pins.json").read_text(encoding="utf-8"))
    verify_files(
        game_dir,
        {name: record for name, record in pins["game_files"].items() if name not in prepared["generated_files"]},
    )
    for name, checksum in prepared["generated_files"].items():
        if sha256(game_dir / name) != checksum:
            raise ValueError(f"Prepared instrumentation changed: {name}")
    if run_dir.exists():
        raise ValueError("run_dir must be a fresh path; existing native sessions are never attached automatically")
    wine_binary = Path(shutil.which(wine) or wine).resolve()
    server_binary = Path(wineserver).resolve() if wineserver else wine_binary.with_name("wineserver")
    if not wine_binary.is_file() or not server_binary.is_file():
        raise FileNotFoundError("Supply Wine and its matching wineserver (optional --wineserver)")
    display = display or choose_display()
    if not display.startswith(":") or not display[1:].isdigit():
        raise ValueError("display must be :<integer>")
    if Path(f"/tmp/.X{display[1:]}-lock").exists() or Path(f"/tmp/.X11-unix/X{display[1:]}").exists():
        raise ValueError("Display is occupied")
    run_dir.mkdir(parents=True)
    ipc, prefix = run_dir / "ipc", run_dir / "wine-prefix"
    ipc.mkdir()
    env = dict(
        os.environ,
        DISPLAY=display,
        WINEPREFIX=str(prefix),
        WINEARCH="win64",
        HONING_SOUL_KNIGHT_IPC="Z:" + str(ipc).replace("/", "\\"),
        WINEDLLOVERRIDES="mscoree,mshtml=",
        WINEDEBUG="-all",
        LP_NUM_THREADS="4",
    )
    command = [
        str(wine_binary),
        str(game_dir / "Soul_Knight.exe"),
        "-batchmode",
        "-force-gfx-direct",
        "-no-stereo-rendering",
        "-job-worker-count",
        "4",
        "-screen-fullscreen",
        "0",
        "-screen-width",
        "960",
        "-screen-height",
        "640",
        "-logFile",
        str(run_dir / "unity.log"),
    ]
    with (run_dir / "xvfb.log").open("wb") as log:
        screen = subprocess.Popen(
            [xvfb, display, "-screen", "0", "960x640x24", "-nolisten", "tcp", "-ac"],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:
        deadline = time.monotonic() + 10
        while not Path(f"/tmp/.X11-unix/X{display[1:]}").exists():
            if screen.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("Xvfb did not become ready; see xvfb.log")
            time.sleep(0.05)
        with (run_dir / "wine.log").open("wb") as log:
            game = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        # Wine may exec the player during initialization; record final identity after readiness.
        state = NativeClient(ipc, timeout_seconds=90).state()
        processes = [process_identity(game.pid), process_identity(screen.pid)]
        if any(value is None for value in processes):
            raise RuntimeError("A launched native process exited during startup")
        receipt = {
            "kind": "honing.soul-knight.launch.v1",
            "ready": True,
            "display": display,
            "game_dir": str(game_dir),
            "run_dir": str(run_dir),
            "ipc_dir": str(ipc),
            "render_dir": str(ipc / "frames"),
            "render_fps": 50,
            "width": 960,
            "height": 640,
            "initial_scene": state["scene"],
            "processes": processes,
            "command": command,
            "wineserver": str(server_binary),
            "wineserver_sha256": sha256(server_binary),
            "wine_prefix_identity": {"device": prefix.stat().st_dev, "inode": prefix.stat().st_ino},
        }
        write_json(run_dir / "launch.json", receipt)
        return receipt
    except BaseException:
        # This prefix was created inside this invocation's fresh run directory.
        if prefix.is_dir():
            try:
                with (run_dir / "wineserver-cleanup.log").open("w", encoding="utf-8") as log:
                    subprocess.run(
                        [str(server_binary), "-k"],
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=10,
                        check=False,
                    )
                    subprocess.run(
                        [str(server_binary), "-w"],
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=10,
                        check=False,
                    )
            except subprocess.TimeoutExpired:
                pass
        if "game" in locals() and game.poll() is None:
            game.terminate()
        if screen.poll() is None:
            screen.terminate()
        raise
