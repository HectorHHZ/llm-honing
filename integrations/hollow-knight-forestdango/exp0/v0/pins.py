"""Write pins.json: the identity of everything a full-HK Exp 0 collection under the raw-key harness runs on (game
build, bridge source, harness snapshot, controller code, decision services). Hashes only where the delivery format
needs them (pins).

  python3 -B pins.py [--plan plan.json] [--out pins.json]

The game build is the plan's `release.game.player` (empty = the default linked build with the bridge,
code/forestdango-hk-demo-work/Builds/linux64_bridge) and its bridge source directory `release.game.bridge_source`
(default: that project's Assets/Scripts/Honing), so a faster build is pinned by changing the plan only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from runtime import WORKSPACE, workspace_path

# 2026-11-26 (privacy): no absolute local path in the code or in pins.json; paths are relative to the workspace
DEFAULT_BUILD = WORKSPACE / "work" / "hollow-knight-forestdango" / "linux64_bridge"
DEFAULT_SOURCE = HERE.parents[1] / "bridge"
BUILD_FILES = (
    "HollowKnightDemo.x86_64",
    "UnityPlayer.so",
    "HollowKnightDemo_Data/Managed/Assembly-CSharp.dll",
    "HollowKnightDemo_Data/Managed/Assembly-CSharp-firstpass.dll",
    # the deterministic devmode build (2026-11-26): its launch scripts and the overlay record
    "hk_fast.sh",
    "hk_det.sh",
)
OVERLAY = WORKSPACE / "work" / "hollow-knight-forestdango" / "overlay" / "OVERLAY.txt"


def rel(path: Path) -> str:
    """A path as pins.json writes it: relative to the workspace (`<outside-workspace>/<name>` otherwise)."""
    p = Path(path)
    try:
        return p.relative_to(WORKSPACE).as_posix()
    except ValueError:
        try:
            return p.resolve().relative_to(WORKSPACE.resolve()).as_posix()
        except ValueError:
            return f"<outside-workspace>/{p.name}"


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).resolve().read_bytes()).hexdigest()


def game_pins(player: str, source: str | None) -> dict:
    build = workspace_path(player).parent if player else DEFAULT_BUILD
    src = workspace_path(source) if source else DEFAULT_SOURCE
    out = {"build_dir": rel(build), "files": {}}
    for name in BUILD_FILES:
        p = build / name
        if p.exists():
            out["files"][name] = {"sha256": sha(p), "resolved": rel(p.resolve()), "bytes": p.resolve().stat().st_size}
    out["bridge_source_dir"] = rel(src)
    out["bridge_source"] = {p.name: sha(p) for p in sorted(src.glob("*.cs"))}
    if player and Path(player).name in ("hk_fast.sh", "hk_det.sh") and OVERLAY.exists():
        out["overlay_record"] = {"path": rel(OVERLAY), "sha256": sha(OVERLAY)}
        out["player_script"] = workspace_path(player).read_text()
    text = (src / "HoningBridge.cs").read_text() if (src / "HoningBridge.cs").exists() else ""
    marker = 'public const string Version = "'
    out["bridge_version"] = text.split(marker, 1)[1].split('"', 1)[0] if marker in text else None
    return out


def service(url: str) -> dict:
    with urllib.request.urlopen(url + "/v1/health", timeout=30) as r:
        meta = json.loads(r.read()).get("metadata", {})
    return {
        k: meta.get(k)
        for k in (
            "role",
            "repo",
            "revision",
            "weightsSha256",
            "sourceCommit",
            "layaVersion",
            "device",
            "dtype",
            "threads",
            "maxLen",
            "launcher",
            "training_performed",
        )
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, default=HERE / "plan.json")
    ap.add_argument("--out", type=Path, default=HERE / "pins.json")
    a = ap.parse_args()
    sys.path.insert(0, str(HERE))
    from collect import code_pins

    plan = json.loads(a.plan.read_text())
    game = plan["release"]["game"]
    harness_dir = HERE / plan["release"]["harness"]["dir"]
    pins = {
        "written": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "game": game_pins(game.get("player", ""), game.get("bridge_source")),
        "code": code_pins(harness_dir),
        "harness_source_record": json.loads((HERE / plan["release"]["harness"]["source_record"]).read_text()),
        "services": {role: {"url": url, **service(url)} for role, url in plan["endpoints"].items()},
        "upstream_main_sha": "9d5327f1c5a53b8b134db847952ff6281267c0a6",
    }
    a.out.write_text(json.dumps(pins, indent=1) + "\n")
    print(
        json.dumps(
            {
                "bridge_version": pins["game"]["bridge_version"],
                "harness_fingerprint": pins["code"]["harness"]["fingerprint"],
                "services": {k: v["weightsSha256"] for k, v in pins["services"].items()},
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
