"""Build an isolated Celeste Classic C port with an opt-in file bridge."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

REVISION = "261d96f15af430b8111abc7a5250229246654f52"
HERE = Path(__file__).resolve().parent
SOURCES = ("celeste.c", "celeste.h", "sdl12main.c", "sdl20compat.inc.c", "tilemap.h")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(source: Path, output: Path, sdl_prefix: Path | None = None) -> dict:
    source = source.resolve()
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise ValueError("ccleste source revision does not match the pin")
    if output.exists():
        raise ValueError("output must be a new directory")
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "diff", "--name-only", "HEAD", "--", *SOURCES, "data", "gamecontrollerdb.txt"]
    )
    if dirty:
        raise ValueError("tracked game source or resources are modified")
    for name in SOURCES:
        expected = subprocess.check_output(["git", "-C", str(source), "show", f"HEAD:{name}"])
        if (source / name).read_bytes() != expected:
            raise ValueError(f"source file modified: {name}")
    output.mkdir(parents=True)
    for name in (*SOURCES, "gamecontrollerdb.txt"):
        shutil.copyfile(source / name, output / name)
    shutil.copytree(source / "data", output / "data")
    for name in ("engine.inc.c", "frontend.inc.c"):
        shutil.copyfile(HERE / name, output / name)
    engine = output / "celeste.c"
    engine.write_text(engine.read_text() + '\n#include "engine.inc.c"\n')
    frontend = output / "sdl12main.c"
    text = frontend.read_text()
    replacements = {
        "int main(int argc, char** argv) {": '#include "frontend.inc.c"\n\nint main(int argc, char** argv) {',
        'printf("ready\\n");': 'honing_start();\n\tprintf("ready\\n");',
        "static void mainLoop(void) {": "static void mainLoop(void) {\n\tif (honing_dir) { honing_loop(); return; }",
    }
    for old, new in replacements.items():
        if text.count(old) != 1:
            raise ValueError(f"expected exactly one native hook: {old}")
        text = text.replace(old, new)
    frontend.write_text(text)
    if sdl_prefix is None:
        flags = subprocess.check_output(["sdl2-config", "--cflags", "--libs"], text=True).split()
    else:
        prefix = sdl_prefix.resolve()
        libs = [prefix / "lib/x86_64-linux-gnu", prefix / "lib"]
        flags = [f"-I{prefix / 'include/SDL2'}", f"-I{prefix / 'include/x86_64-linux-gnu'}"]
        flags += [flag for lib in libs for flag in (f"-L{lib}", f"-Wl,-rpath-link,{lib}", f"-Wl,-rpath,{lib}")]
        flags += ["-lSDL2"]
    command = [
        "gcc",
        "-Wall",
        "-O2",
        "sdl12main.c",
        "celeste.c",
        "-o",
        "ccleste-honing",
        *flags,
        "-lSDL2_mixer",
        "-lm",
    ]
    subprocess.run(command, cwd=output, check=True)
    manifest = {
        "game": "Celeste Classic (2015), community ccleste C port; not 2018 Celeste",
        "sourceRevision": revision,
        "compiler": subprocess.check_output(["gcc", "--version"], text=True).splitlines()[0],
        "command": command,
        "sourceHashes": {name: digest(source / name) for name in SOURCES},
        "bridgeHashes": {name: digest(HERE / name) for name in ("engine.inc.c", "frontend.inc.c")},
        "resourceHashes": {
            str(path.relative_to(source)): digest(path)
            for path in sorted((source / "data").rglob("*"))
            if path.is_file()
        },
        "binarySha256": digest(output / "ccleste-honing"),
        "arithmetic": "upstream floating point; no fixed-point or hacked-balloon mode",
    }
    (output / "build.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sdl-prefix", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output, args.sdl_prefix), indent=2))


if __name__ == "__main__":
    main()
