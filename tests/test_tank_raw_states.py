"""The Python side of tank raw states: one Node process per batch, its timeout and errors, and the fixture round."""

from __future__ import annotations

import gzip
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from honing.backends.tank import seed_harness
from honing.backends.tank.raw_states import TankRawStates, harness_files, read_labels, read_states
from honing.core import ContractError, HarnessFile, HarnessSpec

REPO = Path(__file__).resolve().parents[1]
INTEGRATION = REPO / "integrations/tank-war"
# The round integrations/tank-war/tests/raw_states.test.mjs checks: every file of its run directory, gzipped.
# It was recorded on one platform, so it serves rendering and labelling. Replay records its own round.
FIXTURE = INTEGRATION / "tests/fixtures/rollout-s2002-blue.json.gz"
NODE = shutil.which("node")


def _tool(tmp_path: Path, source: str, timeout_seconds: float = 10) -> TankRawStates:
    """A stand-in for raw-states.mjs, run by this Python, so the wrapper's failure handling can be checked."""
    (tmp_path / "scripts").mkdir(parents=True)
    (tmp_path / "scripts/raw-states.mjs").write_text(source)
    return TankRawStates(tmp_path, node=sys.executable, timeout_seconds=timeout_seconds)


def _state_hash(game: dict[str, object]) -> str:
    """The state_sha256 match.mjs logs for a saved state."""
    script = (
        'import { stateHash } from "./scripts/match.mjs"; let text = "";'
        " for await (const chunk of process.stdin) text += chunk; console.log(stateHash(JSON.parse(text)));"
    )
    return subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=INTEGRATION,
        input=json.dumps(game),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def run(tmp_path: Path) -> Path:
    directory = tmp_path / "run"
    directory.mkdir()
    for name, content in json.loads(gzip.decompress(FIXTURE.read_bytes())).items():
        (directory / name).write_text(content, encoding="utf-8")
    return directory


@pytest.mark.integration
def test_a_tool_that_hangs_fails_or_answers_badly_is_reported_with_its_reason(tmp_path: Path) -> None:
    with pytest.raises(ContractError, match=r"no scripts/raw-states\.mjs"):
        TankRawStates(tmp_path)
    with pytest.raises(ContractError, match=r"replay did not finish within 0\.2 s"):
        _tool(tmp_path / "a", "import time; time.sleep(5)", timeout_seconds=0.2).replay(tmp_path)
    with pytest.raises(ContractError, match="replay failed: decision 3: the context differs"):
        _tool(tmp_path / "b", "import sys; sys.exit('decision 3: the context differs')").replay(tmp_path)
    with pytest.raises(ContractError, match="printed something other than JSON"):
        _tool(tmp_path / "c", "print('ready')").label([tmp_path])
    with pytest.raises(ContractError, match=r"state 1 \(decision 7\) did not render: No motion plan x"):
        _tool(
            tmp_path / "d",
            'import sys\nsys.stdin.read()\nprint(\'{"requests": []}\')\nprint(\'{"error": "No motion plan x"}\')',
        ).render([{"decision": 6}, {"decision": 7}], [])
    with pytest.raises(ContractError, match="cannot start"):
        TankRawStates(tmp_path / "d", node=str(tmp_path / "no-node")).label([tmp_path])


@pytest.mark.unit
def test_states_of_another_format_or_none_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ContractError, match=r"has no states\.jsonl"):
        read_states(tmp_path)
    (tmp_path / "states.jsonl").write_text(json.dumps({"format": "tank-raw-states", "version": 2}) + "\n")
    with pytest.raises(ContractError, match="version 2, and this reads tank-raw-states version 1"):
        read_states(tmp_path)
    with pytest.raises(ContractError, match="Label the run first"):
        read_labels(tmp_path)


@pytest.mark.integration
@pytest.mark.skipif(NODE is None, reason="needs Node 22")
def test_a_recorded_round_re_renders_replays_and_is_labelled_from_python(run: Path) -> None:
    tool = TankRawStates(INTEGRATION)
    header, states = read_states(run)
    rows = [json.loads(line) for line in (run / "decisions.jsonl").read_text().splitlines()]

    assert tool.round_trip([run]) == [{"run": str(run), "decisions": len(states), "rows": len(rows)}]

    # The logged harness is the seed harness: rendering under it from Python gives the logged rows.
    assert harness_files(seed_harness(INTEGRATION)) == header["harness"]
    rendered = tool.render(states, seed_harness(INTEGRATION))
    assert [r for requests in rendered for r in requests] == [
        {"id": row["request_id"], "parts": row["contextParts"], "questions": row["questions"]} for row in rows
    ]
    # Another harness, and another motion for the weapon question, render differently in one process.
    questions = json.loads(seed_harness(INTEGRATION).files[1][1].content)
    probe = HarnessSpec(
        agent="tank_classic",
        files=(
            seed_harness(INTEGRATION).files[0],
            ("tank/questions.json", HarnessFile("instructions", json.dumps({**questions, "weapon": "Probe."}))),
        ),
    )
    changed = tool.render(states, probe, motions=["stop_none"] * len(states))
    assert all(requests[1]["questions"]["weapon"]["instructions"] == "Probe." for requests in changed)
    assert all("SELECTED MOTION stop_none:" in requests[1]["parts"]["essential"][0] for requests in changed)

    [summary] = tool.label([run])
    assert summary == {"run": str(run), "rows": len(rows), "questions": len(rows)}
    labels_header, labels = read_labels(run)
    assert labels_header["source"] == "planner-teacher"
    assert [line["request_id"] for line in labels] == [row["request_id"] for row in rows]


@pytest.mark.integration
@pytest.mark.skipif(NODE is None, reason="needs Node 22")
def test_a_saved_state_that_no_longer_hashes_to_its_logged_hash_does_not_render(run: Path) -> None:
    header, states = read_states(run)
    states[0]["game"]["tanks"][0]["x"] += 10
    with pytest.raises(
        ContractError,
        match=r"state 0 \(decision 0\) did not render: decision 0: the saved state hashes to [0-9a-f]{64},"
        r" not to its logged state_sha256 " + states[0]["state_sha256"],
    ):
        TankRawStates(INTEGRATION).render(states, header["harness"])


@pytest.mark.integration
@pytest.mark.skipif(NODE is None, reason="needs Node 22")
def test_a_round_replays_where_it_was_recorded_and_a_mismatch_names_the_recording_platform(tmp_path: Path) -> None:
    run = tmp_path / "teacher"
    arguments = ["--seed=2003", "--seat=red", "--teacher", "--seconds=3", f"--out={run}"]
    subprocess.run(["node", "scripts/match.mjs", *arguments], cwd=INTEGRATION, check=True, capture_output=True)
    tool = TankRawStates(INTEGRATION)
    header, states = read_states(run)
    assert set(header["platform"]) == {"os", "arch", "node"}
    assert tool.replay(run, to=2) == {"run": str(run), "checked": 3, "to": 2}
    assert tool.replay(run)["checked"] == len(states)

    # A round from another platform whose state differs in the last bit of one angle (and so in its hash).
    header["platform"] = {"os": "linux", "arch": "x64", "node": "22.23.5"}
    tank = states[2]["game"]["tanks"][1]
    tank["angle"] = math.nextafter(tank["angle"], math.inf)
    states[2]["state_sha256"] = _state_hash(states[2]["game"])
    (run / "states.jsonl").write_text("".join(json.dumps(line) + "\n" for line in [header, *states]))
    with pytest.raises(ContractError) as raised:
        tool.replay(run)
    assert "decision 2: state hash" in str(raised.value)
    assert "state.tanks.1.angle is" in str(raised.value)
    assert "Recorded on linux-x64 node 22.23.5, replayed on " in str(raised.value)
    assert str(raised.value).endswith(": floating point may differ in the last bits.")


@pytest.mark.integration
@pytest.mark.skipif(NODE is None, reason="needs Node 22")
def test_a_run_named_relative_to_the_callers_directory_is_found(run: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(run.parent)
    tool = TankRawStates(INTEGRATION)

    (report,) = tool.round_trip([Path(run.name)])

    # Re-rendering is the same on every platform, so this uses the recorded fixture (replay is not).
    assert report["run"] == str(run)
    assert tool.label([Path(run.name)])[0]["run"] == str(run)
