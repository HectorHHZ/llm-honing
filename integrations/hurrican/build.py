"""Apply local hooks to the exact external Hurrican source and build it."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

PIN = "0d91d1f2d40f38d21e7f41afc01369ee55649fb6"
HERE = Path(__file__).resolve().parent


def build(source: Path, output: Path, jobs: int, cmake_args: list[str]) -> None:
    """Refuse another version and apply the narrow source patch once."""
    actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if actual != PIN:
        raise ValueError(f"Expected Hurrican {PIN}, got {actual}")
    patch = HERE / "native.patch"
    check = subprocess.run(["git", "-C", str(source), "apply", "--check", str(patch)], capture_output=True)
    if check.returncode == 0:
        subprocess.run(["git", "-C", str(source), "apply", str(patch)], check=True)
    else:
        subprocess.run(["git", "-C", str(source), "apply", "--reverse", "--check", str(patch)], check=True)
    for name in ("HoningBridge.cpp", "HoningBridge.hpp"):
        shutil.copyfile(HERE / name, source / "Hurrican/src" / name)
    subprocess.run(
        ["cmake", "-S", str(source / "Hurrican"), "-B", str(output), "-DCMAKE_BUILD_TYPE=Release", *cmake_args],
        check=True,
    )
    subprocess.run(["cmake", "--build", str(output), f"-j{jobs}"], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, choices=range(1, 9), default=4)
    parser.add_argument("--cmake-arg", action="append", default=[])
    args = parser.parse_args()
    build(args.source.resolve(), args.build_dir.resolve(), args.jobs, args.cmake_arg)


if __name__ == "__main__":
    main()
