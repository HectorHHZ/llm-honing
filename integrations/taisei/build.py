"""Patch and build a caller-supplied checkout of genuine Taisei v1.4.6."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

PIN = "5bfc9b842fe6d9bb8d1610f76f442ba07c72c4cf"
HERE = Path(__file__).resolve().parent


def patch(source: Path) -> None:
    """Fail on source drift before applying three local, opt-in stage hooks."""
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != PIN:
        raise ValueError(f"Expected Taisei {PIN}, got {revision}")
    original = subprocess.check_output(["git", "-C", str(source), "show", "HEAD:src/stage.c"], text=True)
    patched = (
        original.replace(
            "static void stage_input(StageFrameState *fstate) {",
            '#include "honing_bridge.inc.h"\n\nstatic void stage_input(StageFrameState *fstate) {\n'
            "\tif(honing_dir()) {\n"
            "\t\tplayer_event(&global.plr, NULL, &global.replay.output, EV_INFLAGS, honing_flags);\n"
            "\t\tplayer_applymovement(&global.plr);\n\t\treturn;\n\t}",
        )
        .replace(
            "static LogicFrameAction stage_logic_frame(void *arg) {",
            "static LogicFrameAction stage_logic_frame(void *arg) {\n\tif(!honing_gate()) return LFRAME_STOP;",
        )
        .replace("seed = makeseed();", "seed = honing_dir() ? 7 : makeseed();")
    )
    target = source / "src/stage.c"
    if target.read_text() not in (original, patched):
        raise ValueError("src/stage.c contains unrecognized edits; refusing to overwrite")
    target.write_text(patched)
    # GCC 11 rejects the upstream initialized VLA extension; preserve zeroing.
    for relative, before, after in (
        (
            "src/vfs/loadpacks.c",
            "char buf[strlen(str) + 1] = {};",
            "char buf[strlen(str) + 1]; memset(buf, 0, sizeof(buf));",
        ),
        (
            "src/renderer/gl33/gl33.c",
            "vec2 out[num_color_out] = {};",
            "vec2 out[num_color_out]; memset(out, 0, sizeof(out));",
        ),
        ("src/renderer/gl33/shader_program.c", "char log[len] = {};", "char log[len]; memset(log, 0, sizeof(log));"),
        ("src/renderer/gl33/shader_object.c", "char log[len] = {};", "char log[len]; memset(log, 0, sizeof(log));"),
    ):
        path = source / relative
        baseline = subprocess.check_output(["git", "-C", str(source), "show", f"HEAD:{relative}"], text=True)
        replacement = baseline.replace(before, after)
        if path.read_text() not in (baseline, replacement):
            raise ValueError(f"Refusing to overwrite changed {relative}")
        path.write_text(replacement)
    for name in ("honing_command.h", "honing_bridge.inc.h"):
        destination = source / "src" / name
        if destination.exists() and destination.read_bytes() != (HERE / name).read_bytes():
            raise ValueError(f"Refusing to overwrite different {destination}")
        shutil.copyfile(HERE / name, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.jobs <= 8:
        parser.error("jobs must be 1..8")
    patch(args.source)
    command = ["meson", "setup"]
    if (args.build / "build.ninja").exists():
        command.append("--reconfigure")
    command += [
        str(args.build),
        str(args.source),
        "-Ddeveloper=true",
        "-Ddocs=disabled",
        "-Dshader_transpiler=disabled",
        "-Dr_gles30=disabled",
        "-Dr_sdlgpu=disabled",
        "-Da_sdl=disabled",
        "-Dgamemode=disabled",
        "-Dpackage_data=disabled",
        "-Dallocator=libc",
    ]
    subprocess.run(command, check=True)
    subprocess.run(["ninja", "-C", str(args.build), f"-j{args.jobs}"], check=True)


if __name__ == "__main__":
    main()
