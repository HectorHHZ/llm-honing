"""Assemble the Linux player this experiment ran on from operator-supplied parts.

The parts are the game data of a Mac Mono build, Unity's Linux Mono build-support module of the same Unity version
and a Mono compiler. The result is a new directory; no input is modified. Nothing is downloaded, and no game file
enters this repository. The script compiles `HoningBridge.cs` and `VisibleObservation.cs` from this repository with
the pinned game scripts into a new `Assembly-CSharp.dll`.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from types import ModuleType

INTEGRATION = Path(__file__).resolve().parents[2]
BRIDGE_SOURCES = ("HoningBridge.cs", "VisibleObservation.cs")


def load_builder() -> ModuleType:
    """Reuse the integration's source pinning and input-redirect patching."""
    spec = importlib.util.spec_from_file_location("hollow_knight_builder", INTEGRATION / "build.py")
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load the Hollow Knight builder")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    return builder


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compile_command(mono_root: Path, managed: Path, scripts: list[Path], bridge_dir: Path) -> list[str]:
    """Compile order and references are fixed: the same inputs then give the same assembly bytes."""
    references = sorted(path for path in managed.glob("*.dll") if path.name != "Assembly-CSharp.dll")
    return [
        str(mono_root / "usr/bin/mono-sgen"),
        "--config",
        str(mono_root / "etc/mono/config"),
        str(mono_root / "usr/lib/mono/4.5/mcs.exe"),
        "-noconfig",
        "-nostdlib+",
        "-target:library",
        "-optimize+",
        "-out:" + str(managed / "Assembly-CSharp.dll"),
        *["-r:" + str(path) for path in references],
        *[str(path) for path in scripts],
        *[str(bridge_dir / name) for name in BRIDGE_SOURCES],
    ]


def build(
    source: Path, mac_data: Path, linux_module: Path, mono_root: Path, output: Path, bridge_dir: Path = INTEGRATION
) -> dict[str, object]:
    """Publish the new player directory only after compilation succeeded."""
    source, mac_data, linux_module, mono_root, output = (
        path.resolve() for path in (source, mac_data, linux_module, mono_root, output)
    )
    if output.exists():
        raise ValueError("Output must be a new path")
    builder = load_builder()
    pins = json.loads((INTEGRATION / "pins.json").read_text())
    revision = builder.verify_source(source, pins["game"]["revision"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hollow-knight-linux-", dir=output.parent) as temporary:
        staged = Path(temporary) / output.name
        data = staged / "HollowKnight_Data"
        shutil.copytree(mac_data, data)
        shutil.copy2(linux_module / "LinuxPlayer", staged / "HollowKnight")
        shutil.copy2(linux_module / "UnityPlayer.so", staged / "UnityPlayer.so")
        shutil.copytree(linux_module / "Data/MonoBleedingEdge", data / "MonoBleedingEdge")
        shutil.copy2(
            linux_module / "Data/Resources/unity default resources", data / "Resources/unity default resources"
        )
        managed = data / "Managed"
        scripts = builder.prepare_sources(source / "Hollow Knight/Assets/Scripts", Path(temporary) / "source")
        environment = {
            **os.environ,
            "MONO_CFG_DIR": str(mono_root / "etc"),
            "MONO_PATH": str(mono_root / "usr/lib/mono/4.5"),
            "LD_LIBRARY_PATH": str(mono_root / "usr/lib"),
        }
        subprocess.run(compile_command(mono_root, managed, scripts, bridge_dir), check=True, env=environment)
        manifest = {
            "gameRevision": revision,
            "originalAssemblySha256": digest(mac_data / "Managed/Assembly-CSharp.dll"),
            "assemblySha256": digest(managed / "Assembly-CSharp.dll"),
            "bridgeSha256": {name: digest(bridge_dir / name) for name in BRIDGE_SOURCES},
            "linuxPlayerSha256": digest(staged / "HollowKnight"),
            "unityPlayerSha256": digest(staged / "UnityPlayer.so"),
        }
        (staged / "honing-linux-build.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staged.rename(output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Pinned Hollow-Knight-Imitation checkout")
    parser.add_argument("--mac-data", type=Path, required=True, help="Contents/Resources/Data of the Mac Mono build")
    parser.add_argument(
        "--linux-module",
        type=Path,
        required=True,
        help="Variations/linux64_withgfx_nondevelopment_mono of Unity 2019.4.2f1 Linux Mono build support",
    )
    parser.add_argument(
        "--mono-root",
        type=Path,
        required=True,
        help="Mono 6.8 root holding usr/bin/mono-sgen, usr/lib/mono/4.5/mcs.exe and etc/mono/config",
    )
    parser.add_argument("--output", type=Path, required=True, help="New player directory")
    parser.add_argument("--bridge-dir", type=Path, default=INTEGRATION, help="Directory of the two bridge sources")
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.source, args.mac_data, args.linux_module, args.mono_root, args.output, args.bridge_dir),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
