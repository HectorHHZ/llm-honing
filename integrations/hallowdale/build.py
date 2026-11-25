"""Fetch the pinned hallowdale source, apply the Honing bridge and build it with dune."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

PIN = "a330d3462afe5fa7b8e65573ad8a7c7c79de17e9"
UPSTREAM = "https://github.com/mega-dean/hallowdale.git"
BRIDGE_FILES = ("honing.ml", "honing_input.ml", "honing_command.ml")
HERE = Path(__file__).resolve().parent


def fetch(source: Path) -> None:
    """Clone the upstream repository unless a checkout already exists."""
    if not source.exists():
        subprocess.run(["git", "clone", UPSTREAM, str(source)], check=True)
        subprocess.run(["git", "-C", str(source), "checkout", "--quiet", PIN], check=True)


def apply_bridge(source: Path) -> None:
    """Refuse another revision, apply the narrow patch once and copy the bridge modules."""
    if "hallowdale" not in source.name:
        # Env.find_project_root in the game looks for this name above _build.
        raise ValueError(f"The checkout directory name must contain 'hallowdale': {source.name}")
    actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if actual != PIN:
        raise ValueError(f"Expected hallowdale {PIN}, got {actual}")
    patch = HERE / "native.patch"
    check = subprocess.run(["git", "-C", str(source), "apply", "--check", str(patch)], capture_output=True)
    if check.returncode == 0:
        subprocess.run(["git", "-C", str(source), "apply", str(patch)], check=True)
    else:
        subprocess.run(["git", "-C", str(source), "apply", "--reverse", "--check", str(patch)], check=True)
    for name in BRIDGE_FILES:
        shutil.copyfile(HERE / name, source / "src" / name)


def build(source: Path, dune: str) -> Path:
    """Build the game executable and return its path; the game finds its assets from that location."""
    subprocess.run([dune, "build", "--profile", "release", "bin/main.exe"], cwd=source, check=True)
    return source / "_build/default/bin/main.exe"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Checkout to use, cloned here if it does not exist")
    parser.add_argument("--dune", default="dune", help="dune from the OCaml 5.1 switch described in README.md")
    args = parser.parse_args()
    source = args.source.resolve()
    fetch(source)
    apply_bridge(source)
    print(build(source, args.dune))


if __name__ == "__main__":
    main()
