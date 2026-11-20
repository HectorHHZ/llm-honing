"""Patch and compile a caller-owned pinned VVVVVV checkout; never acquire assets."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

PIN = "40e1ec3bbf2a990a5c600564fe312c2121178a50"
HERE = Path(__file__).resolve().parent


def patch(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if new in text:
        return
    if text.count(old) != 1:
        raise ValueError(f"Unsupported source layout: {path.name}")
    path.write_text(text.replace(old, new))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--cmake-prefix", type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    source = args.source.resolve()
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != PIN:
        raise ValueError(f"Expected original VVVVVV {PIN}; got {revision}")
    src = source / "desktop_version/src"
    for name in ("HoningBridge.h", "HoningBridge.inc"):
        shutil.copyfile(HERE / name, src / name)
    patch(src / "main.cpp", '#include "ButtonGlyphs.h"', '#include "HoningBridge.h"\n#include "ButtonGlyphs.h"')
    patch(src / "Input.cpp", '#include "Game.h"', '#include "Game.h"\n#include "HoningBridge.h"')
    patch(src / "KeyPoll.cpp", '#include "KeyPoll.h"', '#include "KeyPoll.h"\n#include "HoningBridge.h"')
    keypoll = src / "KeyPoll.cpp"
    text = keypoll.read_text()
    marker = "\n}\n\nbool KeyPoll::isDown(SDL_Keycode key)"
    replacement = "\n    if (honing_active()) isActive = true;\n}\n\nbool KeyPoll::isDown(SDL_Keycode key)"
    if replacement not in text:
        if text.count(marker) != 1:
            raise ValueError("Unsupported KeyPoll::Poll layout")
        keypoll.write_text(text.replace(marker, replacement))
    patch(
        src / "Input.cpp",
        "    game.press_map = false;\n    if (key.isDown(KEYBOARD_ENTER)",
        "    honing_input();\n    game.press_map = false;\n    if (key.isDown(KEYBOARD_ENTER)",
    )
    patch(
        src / "main.cpp",
        "#ifdef SDL_PLATFORM_EMSCRIPTEN\n    emscripten_set_main_loop",
        "    if (honing_active()) honing_run();\n\n#ifdef SDL_PLATFORM_EMSCRIPTEN\n    emscripten_set_main_loop",
    )
    patch(
        src / "main.cpp", "static void cleanup(void)\n{", '#include "HoningBridge.inc"\n\nstatic void cleanup(void)\n{'
    )
    command = [
        "cmake",
        "-S",
        str(source / "desktop_version"),
        "-B",
        str(args.build.resolve()),
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    if args.cmake_prefix:
        command.append(f"-DCMAKE_PREFIX_PATH={args.cmake_prefix.resolve()}")
    subprocess.run(command, check=True)
    subprocess.run(["cmake", "--build", str(args.build.resolve()), "--parallel", str(args.jobs)], check=True)


if __name__ == "__main__":
    main()
