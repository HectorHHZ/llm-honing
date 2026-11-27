"""Tank runs as Reef records, for rounds played before the decision server recorded them (#227).

A run directory holds ``decisions.jsonl``, ``manifest.json`` and
``result.json``. Its records are one decision record per row and the
episode's report, scored as ``TankGrader`` scores a round. Writing them to a
JSON-lines file lets ``reef-client``'s resumable importer send them.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from honing.backends.reef.records import EpisodeReport, RecordedDecision
from honing.backends.tank.grader import DRAW, LOSS, WIN, feedback_text
from honing.core import ContractError


def run_records(run: Path, tags: Mapping[str, str]) -> list[dict[str, Any]]:
    """The decision records of one tank run in play order, then its report."""
    seat = json.loads((run / "manifest.json").read_text())["seat"]
    result = json.loads((run / "result.json").read_text())
    mine = (result.get("outcome") or {}).get(seat)
    if mine not in (1, 0, -1):
        raise ContractError(f"{run}/result.json has no outcome for the {seat} seat")
    score = WIN if mine == 1 else LOSS if mine == -1 else DRAW
    rows = [json.loads(line) for line in (run / "decisions.jsonl").read_text().splitlines() if line.strip()]
    report = EpisodeReport.from_rows(
        rows,
        episode=tags["episode"],
        score=score,
        feedback=feedback_text(result, seat, score),
        task=tags.get("task"),
    )
    return [*(RecordedDecision.from_row(row, tags).to_record() for row in rows), report.to_record()]


def run_tags(run: Path, **tags: str | None) -> dict[str, str]:
    """A run's ``episode`` (its directory name) and ``task`` (the name up to the first ``--``), and ``tags``."""
    return {"episode": run.name, "task": run.name.split("--")[0], **{k: v for k, v in tags.items() if v}}


def write_records(runs: Iterable[Path], path: Path, **tags: str | None) -> int:
    """Write every run's records to the JSON-lines file ``path``, run after run; return how many."""
    count = 0
    with path.open("w", encoding="utf-8") as target:
        for run in runs:
            for record in run_records(run, run_tags(run, **tags)):
                target.write(json.dumps(record, sort_keys=True) + "\n")
                count += 1
    return count
