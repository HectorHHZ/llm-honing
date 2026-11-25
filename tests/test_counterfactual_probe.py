"""The counterfactual probe (#170) on the recorded tank rounds, against stand-in models behind the real DecisionServer.

The four rounds under ``tests/fixtures/decision_traces`` were played by the dry
run's stand-in model (``ForecastModel`` at quality 0.7), so that model,
served again, must reproduce every logged probability, to within the last bit
(the rounds were recorded on macOS, and ``math.exp`` can round differently elsewhere). It fires only
when the state says ``Fire permitted=true.``; at quality 0.5 it holds instead.
That gives a text variant and a weights variant with a known effect.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tarfile
import threading
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from honing.backends.jev.server import DecisionServer
from honing.backends.tank.traces import load_run
from honing.core import ContractError, DecisionTrace
from honing.harness.probe import (
    IDENTITY,
    DecideClient,
    IdentityError,
    SettingsVariant,
    TextEdit,
    TextVariant,
    WeightsVariant,
    decide_request,
    probe,
)
from tests.fakes.tank import ForecastModel

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests/fixtures/decision_traces"
RUNS = ("s3114-blue-destroyed", "s3138-red-opponent-self-hit", "s3104-blue-opponent-destroyed", "s3102-red-timeout")
RECORDED, WEAKER, OTHER = "a" * 64, "b" * 64, "c" * 64
PERMITTED = "Fire permitted=true."
FIRE = {"weapon": "fire"}


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    unpacked = tmp_path_factory.mktemp("runs")
    for packed in sorted(FIXTURES.glob("*.tar.gz")):
        with tarfile.open(packed) as archive:
            archive.extractall(unpacked, filter="data")
    return {name: FIXTURES / name if (FIXTURES / name).is_dir() else unpacked / name for name in RUNS}


@pytest.fixture(scope="module")
def traces(runs: dict[str, Path]) -> list[DecisionTrace]:
    return [trace for name in RUNS for trace in load_run(runs[name], policy_checksum=RECORDED)]


@pytest.fixture(scope="module")
def weapon(traces: list[DecisionTrace]) -> list[DecisionTrace]:
    return [trace for trace in traces if dict(trace.decision.questions).get("weapon")]


def _serve(model: Any) -> Iterator[DecisionServer]:
    server = DecisionServer(("127.0.0.1", 0), model)
    thread = threading.Thread(target=server.serve, daemon=True)
    thread.start()
    yield server
    server.stop()
    thread.join()
    server.server_close()


@pytest.fixture(scope="module")
def recorded() -> Iterator[DecisionServer]:
    yield from _serve(ForecastModel(0.7, RECORDED))


@pytest.fixture(scope="module")
def weaker() -> Iterator[DecisionServer]:
    yield from _serve(ForecastModel(0.5, WEAKER))


def _state(trace: DecisionTrace) -> str:
    return "\n".join(line for _, lines in trace.decision.parts for line in lines)


# ------------------------------------------------------------ the requests


@pytest.mark.unit
@pytest.mark.parametrize("name", RUNS)
def test_a_decision_is_asked_again_with_the_request_it_logged_byte_for_byte(name: str, runs: dict[str, Path]) -> None:
    rows = [json.loads(line) for line in (runs[name] / "decisions.jsonl").read_text().splitlines() if line]

    rebuilt = [decide_request(trace) for trace in load_run(runs[name])]

    assert len(rebuilt) == len(rows)
    for request, row in zip(rebuilt, rows, strict=True):
        assert json.dumps(request["parts"]) == json.dumps(row["contextParts"])
        assert json.dumps(request["questions"]) == json.dumps(row["questions"])


@pytest.mark.unit
def test_a_text_variant_edits_the_state_and_with_questions_the_wording(weapon: list[DecisionTrace]) -> None:
    trace = next(t for t in weapon if PERMITTED in _state(t))
    question = dict(trace.decision.questions)["weapon"]
    word = question.instructions.split()[0]

    state_only = TextVariant("v", (TextEdit(PERMITTED), TextEdit(word, "WORD")))
    both = TextVariant("v", (TextEdit(PERMITTED), TextEdit(word, "WORD")), questions=True)
    (edited,), (worded,) = state_only.requests([trace]), both.requests([trace])

    assert PERMITTED not in json.dumps(edited["parts"]) and PERMITTED in _state(trace)
    assert edited["questions"] == decide_request(trace)["questions"]
    assert worded["questions"]["weapon"]["instructions"].startswith("WORD")
    with pytest.raises(ContractError, match="itself"):
        TextEdit("same", "same")


# ------------------------------------------------------------ the checks before a probe


@pytest.mark.integration
def test_the_identity_probe_reproduces_every_logged_probability_in_batches(
    traces: list[DecisionTrace], recorded: DecisionServer
) -> None:
    client = DecideClient(recorded.url)

    result = probe(traces, IDENTITY, server=client)

    assert result.identity is not None and result.identity.ok
    assert (result.identity.decisions, result.identity.flips) == (len(traces), 0)
    assert result.identity.max_difference <= 1e-12
    assert result.summary()["flipped"] == 0 and result.changed == 0
    # One call carries a request and up to 16 branches, and the identity answers are reused.
    assert client.calls == math.ceil(len(traces) / 17)
    one_by_one = DecideClient(recorded.url, batch=1)
    assert one_by_one.answer([decide_request(t) for t in traces[:40]]) == client.answer(
        [decide_request(t) for t in traces[:40]]
    )
    assert one_by_one.calls == 40


@pytest.mark.integration
def test_a_server_serving_other_weights_than_the_traces_is_refused(
    traces: list[DecisionTrace], weaker: DecisionServer
) -> None:
    with pytest.raises(ContractError, match=f"serves the weights {WEAKER}, but the probe needs {RECORDED}"):
        probe(traces, IDENTITY, server=DecideClient(weaker.url))


@pytest.mark.integration
def test_a_server_that_names_no_weights_is_refused(traces: list[DecisionTrace]) -> None:
    for server in _serve(ForecastModel(0.7)):
        with pytest.raises(ContractError, match="reports no weightsSha256"):
            probe(traces[:5], IDENTITY, server=DecideClient(server.url))


@pytest.mark.integration
def test_other_weights_under_the_traces_checksum_fail_the_identity_probe(traces: list[DecisionTrace]) -> None:
    for server in _serve(ForecastModel(0.5, RECORDED)):
        with pytest.raises(IdentityError, match="identity probe") as raised:
            probe(traces, TextVariant("strip", (TextEdit(PERMITTED),)), server=DecideClient(server.url))
    assert raised.value.report.flips_beyond_ties and raised.value.report.max_difference > 0.5


@pytest.mark.integration
def test_the_traces_must_name_one_policy(runs: dict[str, Path], recorded: DecisionServer) -> None:
    client = DecideClient(recorded.url)
    unnamed = load_run(runs[RUNS[0]])
    with pytest.raises(ContractError, match="name no policy checksum"):
        probe(unnamed, IDENTITY, server=client)
    assert probe(unnamed, IDENTITY, server=client, policy_checksum=RECORDED).identity is not None
    mixed = [*load_run(runs[RUNS[0]], policy_checksum=RECORDED), *load_run(runs[RUNS[1]], policy_checksum=OTHER)]
    with pytest.raises(ContractError, match="2 policies"):
        probe(mixed, IDENTITY, server=client)
    with pytest.raises(ContractError, match=f"made by the weights {RECORDED}, not {OTHER}"):
        probe(load_run(runs[RUNS[0]], policy_checksum=RECORDED), IDENTITY, server=client, policy_checksum=OTHER)


# ------------------------------------------------------------ the variants


@pytest.mark.integration
def test_stripping_a_field_turns_exactly_the_decisions_that_read_it(
    weapon: list[DecisionTrace], recorded: DecisionServer
) -> None:
    result = probe(
        weapon, TextVariant("strip", (TextEdit(PERMITTED),)), server=DecideClient(recorded.url), target=FIRE
    )

    by_step = {(row.episode_id, row.step): row for row in result.rows}
    fired = [t for t in weapon if PERMITTED in _state(t)]
    assert fired and result.changed == len(fired)
    for trace in weapon:
        row = by_step[trace.episode_id, trace.step]
        if PERMITTED in _state(trace):
            assert (row.before, row.after, row.top_before, row.top_after) == (0.9, 0.1, "fire", "hold")
        else:
            assert not row.changed and row.shift == 0.0
    summary = result.summary()
    assert (summary["flipped"], summary["away_from_target"], summary["toward_target"]) == (len(fired), len(fired), 0)
    assert result.decisions_per_second is not None and result.decisions_per_second > 0


@pytest.mark.integration
def test_a_weights_variant_asks_the_logged_requests_of_other_weights(
    weapon: list[DecisionTrace], recorded: DecisionServer, weaker: DecisionServer
) -> None:
    base = DecideClient(recorded.url)

    result = probe(weapon, WeightsVariant("q0.5", DecideClient(weaker.url), WEAKER), server=base, target=FIRE)

    fired = sum(PERMITTED in _state(t) for t in weapon)
    assert result.kind == "weights" and result.changed == len(weapon)
    assert result.summary()["flipped"] == fired and result.summary()["away_from_target"] == fired
    with pytest.raises(ContractError, match=f"serves the weights {WEAKER}, but the probe needs {OTHER}"):
        probe(weapon, WeightsVariant("wrong", DecideClient(weaker.url), OTHER), server=base)
    with pytest.raises(ContractError, match="that is the identity probe"):
        probe(weapon, WeightsVariant("same", DecideClient(recorded.url), RECORDED), server=base)


class _PermitEverything:
    """A stand-in for a renderer that renders from raw state under settings that always permit the shot."""

    def __init__(self, extra: int = 0, options: bool = False) -> None:
        self.extra, self.options = extra, options

    def render(self, traces: Sequence[DecisionTrace]) -> list[Mapping[str, Any]]:
        out = []
        for trace in traces:
            request = decide_request(trace)
            parts = {name: list(lines) for name, lines in request["parts"].items()}
            if PERMITTED not in _state(trace):
                parts["essential"] = [*parts["essential"], PERMITTED]
            questions = request["questions"]
            if self.options:
                questions = {k: {**q, "criteria": {"fire": "Fire."}} for k, q in questions.items()}
            out.append({"parts": parts, "questions": questions})
        return out + out[: self.extra]


@pytest.mark.integration
def test_a_settings_variant_asks_what_its_renderer_rendered(
    weapon: list[DecisionTrace], recorded: DecisionServer
) -> None:
    client = DecideClient(recorded.url)

    result = probe(weapon, SettingsVariant("permit", _PermitEverything()), server=client, target=FIRE)

    held = len(weapon) - sum(PERMITTED in _state(t) for t in weapon)
    assert result.kind == "settings" and result.changed == held
    assert result.summary()["toward_target"] == held and result.summary()["flipped_to_target"] == held
    with pytest.raises(ContractError, match="rendered"):
        probe(weapon, SettingsVariant("extra", _PermitEverything(extra=1)), server=client)
    with pytest.raises(ContractError, match="other questions or options"):
        probe(weapon, SettingsVariant("options", _PermitEverything(options=True)), server=client)


@pytest.mark.integration
def test_the_target_can_be_chosen_per_decision(weapon: list[DecisionTrace], recorded: DecisionServer) -> None:
    client = DecideClient(recorded.url)

    def chosen(trace: DecisionTrace, question: str) -> str | None:
        return dict(trace.decision.chosen).get(question)

    result = probe(weapon, IDENTITY, server=client, target=chosen)
    assert [row.target for row in result.rows] == [dict(t.decision.chosen)["weapon"] for t in weapon]
    with pytest.raises(ContractError, match="no option 'shoot'"):
        probe(weapon, IDENTITY, server=client, target={"weapon": "shoot"})


# ------------------------------------------------------------ the command line


@pytest.mark.integration
def test_the_probe_command_line_passes_its_end_to_end_check() -> None:
    process = subprocess.run(
        [sys.executable, str(REPO / "scripts/e2e/check_probe.py")], capture_output=True, text=True, check=False
    )
    assert process.returncode == 0, process.stdout + process.stderr
    assert process.stdout.count("\nok ") + process.stdout.startswith("ok ") == 5
