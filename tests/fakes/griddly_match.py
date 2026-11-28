"""The real Griddly match and replay commands on the stand-in engine (tests/fakes/griddly_engine.py); no Griddly.

``python tests/fakes/griddly_match.py ARGS`` runs ``honing.backends.griddly.match``
and ``python tests/fakes/griddly_match.py --replay ARGS`` runs its replay check, as
``integrations/griddly/scripts/match.py`` and ``verify_replay.py`` do. Environment
variables stage faults: ``FAKE_GRIDDLY_HARNESS_ERROR`` makes the match refuse its
harness, ``FAKE_GRIDDLY_CRASH`` makes the engine fail on its first step, and
``FAKE_GRIDDLY_HANG=<path>`` writes the process id to ``path`` and sleeps.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from honing.backends.griddly import match, replay
from honing.core import CampaignRuleError
from tests.fakes.griddly_engine import FakeSpidersEngine, fake_versions

for module in (match, replay):
    module.GriddlyEngine = FakeSpidersEngine  # type: ignore[attr-defined, misc]
    module.runtime_versions = fake_versions  # type: ignore[attr-defined]

if os.environ.get("FAKE_GRIDDLY_HANG"):
    # Renamed into place, so a test that sees the file reads the whole id.
    marker = Path(os.environ["FAKE_GRIDDLY_HANG"])
    marker.with_suffix(".tmp").write_text(str(os.getpid()))
    os.replace(marker.with_suffix(".tmp"), marker)
    time.sleep(60)
if os.environ.get("FAKE_GRIDDLY_HARNESS_ERROR"):

    def refuse(spec: object) -> None:
        raise CampaignRuleError("harness_readable", "stand-in refusal of the delivered harness")

    match.parse_harness = refuse  # type: ignore[assignment]
if os.environ.get("FAKE_GRIDDLY_CRASH"):

    def crash(self: FakeSpidersEngine, action: str) -> None:
        raise RuntimeError("stand-in engine crash")

    FakeSpidersEngine.step = crash  # type: ignore[method-assign, assignment]

if __name__ == "__main__":
    if sys.argv[1:2] == ["--replay"]:
        del sys.argv[1]
        replay.main()
    else:
        match.main()
