"""Compile the Unity input bridge into a new copy of a caller-supplied Mac build."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def prepare_sources(source: Path, output: Path) -> list[Path]:
    """Patch copied input reads and a menu startup callback, refusing a different controller."""
    contents = {path.relative_to(source): path.read_text(encoding="utf-8-sig") for path in source.rglob("*.cs")}
    player_paths = [path for path in contents if path.name == "PlayerController.cs"]
    menu_paths = [path for path in contents if path.name == "Menu.cs"]
    if len(player_paths) != 1:
        raise ValueError("Expected one PlayerController with the pinned input reads")
    player = player_paths[0]
    contents[player], count = re.subn(r"\bInput\.", "HoningInput.", contents[player])
    if count != 7:
        raise ValueError(f"Expected 7 input reads in the pinned controller, found {count}")
    if len(menu_paths) != 1:
        raise ValueError("Expected one Menu class")
    menu = menu_paths[0]
    contents[menu], count = re.subn(
        r"(public\s+class\s+Menu\s*:\s*MonoBehaviour\s*\{)",
        r"\1\n    private void Start() { HoningBridge.Boot(); }",
        contents[menu],
        count=1,
    )
    if count != 1:
        raise ValueError("Pinned Menu startup class was not found")
    paths = []
    for relative, text in contents.items():
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
        paths.append(destination)
    return sorted(paths)


def validate_output(source_app: Path, output: Path) -> None:
    """Never overwrite a caller's app or recursively copy into it."""
    source_app, output = source_app.resolve(), output.resolve()
    if output == source_app or output.is_relative_to(source_app) or output.exists():
        raise ValueError("Output must be a new path outside the input game app")


def verify_source(source: Path, expected_revision: str) -> str:
    """Require the pinned commit and exactly its unchanged C# inputs, including ignored files."""
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != expected_revision:
        raise ValueError("Game source must use the revision in pins.json")
    prefix = "Hollow Knight/Assets/Scripts"
    subprocess.run(
        ["git", "-C", str(source), "diff", "--exit-code", "HEAD", "--", prefix], check=True, stdout=subprocess.PIPE
    )
    tracked = subprocess.check_output(["git", "-C", str(source), "ls-files", "-z", "--", prefix], text=True)
    expected = {Path(name) for name in tracked.split("\0") if name.endswith(".cs")}
    actual = {path.relative_to(source) for path in (source / prefix).rglob("*.cs")}
    if actual != expected:
        raise ValueError("Game source contains untracked or missing C# scripts")
    return revision


def build(source: Path, game_app: Path, unity: Path, output: Path) -> dict[str, object]:
    """Use the supplied Unity Mono compiler; publish the new app only after compilation succeeds."""
    source, game_app, unity, output = (p.resolve() for p in (source, game_app, unity, output))
    validate_output(game_app, output)
    pins = json.loads((HERE / "pins.json").read_text())
    revision = verify_source(source, pins["game"]["revision"])
    mono_root = unity / "Contents/MonoBleedingEdge"
    mono, compiler = mono_root / "bin/mono", mono_root / "lib/mono/4.5/csc.exe"
    if not mono.is_file() or not compiler.is_file():
        raise ValueError("Unity Mono compiler not found; pass a Unity 2019.4.2f1 .app")
    original = game_app / "Contents/Resources/Data/Managed/Assembly-CSharp.dll"
    original_hash = hashlib.sha256(original.read_bytes()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hollow-knight-build-", dir=output.parent) as temporary:
        temporary = Path(temporary)
        scripts = prepare_sources(source / "Hollow Knight/Assets/Scripts", temporary / "source")
        staged = temporary / output.name
        # Materialize links so compiler output stays inside the staging app.
        shutil.copytree(game_app, staged, symlinks=False)
        managed = staged / "Contents/Resources/Data/Managed"
        assembly = managed / "Assembly-CSharp.dll"
        references = sorted(path for path in managed.glob("*.dll") if path.name != assembly.name)
        subprocess.run(
            [
                str(mono),
                str(compiler),
                "-nologo",
                "-noconfig",
                "-nostdlib+",
                "-target:library",
                "-optimize+",
                "-out:" + str(assembly),
                *["-r:" + str(path) for path in references],
                *[str(path) for path in scripts],
                str(HERE / "HoningBridge.cs"),
            ],
            check=True,
        )
        manifest = {
            "gameRevision": revision,
            "originalAssemblySha256": original_hash,
            "demoAssemblySha256": hashlib.sha256(assembly.read_bytes()).hexdigest(),
            "inputReadsRedirected": 7,
            "originalAppModified": False,
        }
        (staged / "honing-build.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staged.rename(output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Pinned Hollow-Knight-Imitation checkout")
    parser.add_argument("--game-app", type=Path, required=True, help="Existing Mono Mac game .app")
    parser.add_argument("--unity", type=Path, required=True, help="Unity 2019.4.2f1 editor .app")
    parser.add_argument("--output", type=Path, required=True, help="New demo .app path")
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.game_app, args.unity, args.output), indent=2))


if __name__ == "__main__":
    main()
