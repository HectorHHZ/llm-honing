"""Evaluation-only reader of the game's debug trace (`observations.jsonl`).

NOTHING in here may reach perception, skills, language or any harness decision. It is used by
  * probes (designer-side calibration of jump arcs, like a human measuring with a ruler),
  * the run loop only to (a) end an episode once the scene has changed to Level2 and (b) print progress
    for the humans watching,
  * tools/progress.py.
"""

from __future__ import annotations

import json
from pathlib import Path


class EpisodeWatch:
    def __init__(self, session: Path):
        self.path = Path(session) / "observations.jsonl"
        self.offset = 0
        self.rows: list[dict] = []

    def poll(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open("r") as handle:
            handle.seek(self.offset)
            data = handle.read()
        if not data.endswith("\n"):
            data = data[: data.rfind("\n") + 1]
        self.offset += len(data.encode())
        new = [json.loads(line) for line in data.splitlines() if line.strip()]
        self.rows.extend(new)
        return new

    @property
    def last(self) -> dict | None:
        return self.rows[-1] if self.rows else None

    def summary(self) -> dict:
        self.poll()
        rows = self.rows
        if not rows:
            return {}
        xs = [r["x"] for r in rows]
        scenes = [r["scene"] for r in rows]
        deaths = sum(1 for a, b in zip(rows, rows[1:]) if a["health"] > 0 and b["health"] == 0)  # noqa: B905, RUF007
        by_scene = {}  # added in round 2: the same numbers per scene (multi-scene runs)
        for name in sorted(set(scenes)):
            part = [(a, b) for a, b in zip(rows, rows[1:]) if b["scene"] == name]  # noqa: B905, RUF007
            own = [r for r in rows if r["scene"] == name]
            by_scene[name] = {
                "windows": len(part),
                "x_max": max(r["x"] for r in own),
                "hp_min": min(r["health"] for r in own),
                "deaths": sum(1 for a, b in part if a["scene"] == name and a["health"] > 0 and b["health"] == 0),
            }
        return {
            "windows": len(rows) - 1,
            "frames": rows[-1]["frame"],
            "x_start": xs[0],
            "x_max": max(xs),
            "x_final": xs[-1],
            "y_final": rows[-1]["y"],
            "hp_min": min(r["health"] for r in rows),
            "hp_final": rows[-1]["health"],
            "deaths": deaths,
            "scenes": sorted(set(scenes)),
            "scene_changes": sum(1 for a, b in zip(scenes, scenes[1:]) if a != b),  # noqa: B905, RUF007
            "scene_final": scenes[-1],
            "by_scene": by_scene,
        }
