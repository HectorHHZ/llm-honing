"""W&B logging for end-to-end runs (#100), against the stand-in in ``tests/fakes/wandb_stub``.

Nothing here talks to W&B: the stand-in records each call to a file. A config
without a ``logging`` block must log nothing and import no ``wandb``; one with
it logs one step per schedule step from an allow-list, and uploads
``summary.json`` and ``commits.jsonl`` only if no key is in the run's files.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from honing.core import ContractError
from honing.e2e.pipeline import build, main
from honing.e2e.tracking import ChainRecords, logging_settings, step_values, without_local_paths

REPO = Path(__file__).resolve().parents[1]
ALL_PLANES = REPO / "configs/e2e/all_planes.json"
STUB = REPO / "tests/fakes/wandb_stub"
WANDB = {"project": "honing-e2e", "entity": "honing", "tags": ["issue-100", "gepa", "live"]}
KEY = "wandb-fake-key-0123456789abcdef"
LOGGED_PREFIXES = ("move/", "gate/", "candidate/", "selector/", "usage/")


@pytest.fixture
def fake_wandb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    calls = tmp_path / "wandb-calls.jsonl"
    monkeypatch.setenv("HONING_FAKE_WANDB_LOG", str(calls))
    monkeypatch.setenv("WANDB_API_KEY", KEY)
    monkeypatch.syspath_prepend(str(STUB))
    monkeypatch.delitem(sys.modules, "wandb", raising=False)
    yield calls
    sys.modules.pop("wandb", None)


def read_calls(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().split("\n") if line.strip()]


def logged_config(tmp_path: Path, **wandb: Any) -> Path:
    config = {**json.loads(ALL_PLANES.read_text()), "logging": {"wandb": {**WANDB, **wandb}}}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path


def run_main(config: Path, work_dir: Path) -> int:
    return main(["--config", str(config), "--root", str(REPO), "--work-dir", str(work_dir)])


# ---------------------------------------------------------------- without a logging block


@pytest.mark.integration
def test_a_run_without_a_logging_block_logs_nothing_and_imports_no_wandb(tmp_path: Path) -> None:
    # The stand-in is importable, so only the code's own choice keeps wandb out.
    calls, work_dir = tmp_path / "calls.jsonl", tmp_path / "run"
    script = (
        "import sys\n"
        "from honing.e2e.pipeline import main\n"
        f"code = main(['--config', {str(ALL_PLANES)!r}, '--root', {str(REPO)!r}, '--work-dir', {str(work_dir)!r}])\n"
        "print('result', code, 'wandb' in sys.modules)\n"
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(STUB), str(REPO)]), "HONING_FAKE_WANDB_LOG": str(calls)}

    result = subprocess.run(
        [sys.executable, "-c", script], cwd=REPO, env=env, capture_output=True, text=True, timeout=300
    )

    assert result.stdout.strip().split("\n")[-1] == "result 0 False", result.stdout + result.stderr
    assert not calls.exists() and not (work_dir / "wandb").exists()
    assert "wandb:" not in result.stdout


# ---------------------------------------------------------------- the pipeline, logged


@pytest.mark.unit
def test_the_pipeline_logs_one_step_per_schedule_step_and_uploads_its_files(
    tmp_path: Path, fake_wandb: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    work_dir = tmp_path / "run"

    assert run_main(logged_config(tmp_path), work_dir) == 0

    calls = read_calls(fake_wandb)
    assert [call["call"] for call in calls] == ["init"] + ["log"] * 9 + ["save", "save", "finish"]
    init = calls[0]
    assert (init["project"], init["entity"], init["tags"], init["mode"]) == (
        "honing-e2e", "honing", ["issue-100", "gepa", "live"], "online"
    )  # fmt: skip
    assert (init["name"], init["dir"]) == ("run", str(work_dir))
    assert init["settings"] == {
        "console": "off",
        "save_code": False,
        "disable_code": True,
        "disable_git": True,
        "x_disable_meta": True,
        "x_disable_machine_info": True,
        "x_save_requirements": False,
    }
    assert set(init["config"]) == {"script", "git_commit", "model", "config"}
    assert init["config"]["script"] == "honing_pipeline"
    assert init["config"]["config"]["logging"]["wandb"] == WANDB

    logs = [call for call in calls if call["call"] == "log"]
    assert [call["step"] for call in logs] == list(range(1, 10))
    summary = json.loads((work_dir / "summary.json").read_text())
    for call, step in zip(logs, summary["steps"], strict=True):
        data = call["data"]
        assert all(key.startswith(LOGGED_PREFIXES) for key in data), data
        assert (data["move/kind"], data["move/result"]) == (step["move"], step["result"])
        assert step["reason"] not in data.values()  # free text is never logged
    by_step = {call["step"]: call["data"] for call in logs}
    # A published move and a rejected one carry the gate's and the selector's numbers.
    assert by_step[1]["move/published"] == 1.0 and by_step[1]["candidate/gate_candidate_mean"] == pytest.approx(0.6)
    assert by_step[3]["move/published"] == 0.0 and "move/release" not in by_step[3]
    assert by_step[3]["selector/regression"] == pytest.approx(0.02)
    assert by_step[6]["candidate/method_episodes"] == 0.0 and by_step[6]["gate/episodes"] == 6.0
    # A skip has no gate, only the episodes the method spent; a rollback has its result and the head.
    assert by_step[7] == {
        "move/kind": "harness",
        "move/result": "skipped",
        "gate/episodes": 0.0,
        "candidate/method_episodes": 0.0,
    }
    assert by_step[9] == {"move/kind": "rollback", "move/result": "restored", "move/release": "r0001"}

    # Copies are uploaded, with the run's local paths replaced; the files on disk are unchanged.
    saves = [call for call in calls if call["call"] == "save"]
    upload = work_dir / "wandb-upload"
    assert [Path(call["path"]).relative_to(upload).as_posix() for call in saves] == [
        "summary.json",
        "releases/commits.jsonl",
    ]
    assert all(call["base_path"] == str(upload) and call["policy"] == "now" for call in saves)
    assert summary["work_dir"] == str(work_dir)
    assert json.loads(saves[0]["content"]) == {**summary, "work_dir": "<work_dir>"}
    commits = (work_dir / "releases/commits.jsonl").read_text()
    assert saves[1]["content"] == commits  # it names no local path
    assert not [call for call in saves if str(work_dir) in call["content"] or str(REPO) in call["content"]]
    assert calls[-1] == {"call": "finish", "exit_code": 0}
    assert capsys.readouterr().out.count("wandb: https://wandb.invalid/") == 2


@pytest.mark.unit
def test_a_failed_run_is_finished_as_failed(tmp_path: Path, fake_wandb: Path) -> None:
    config = {**json.loads(logged_config(tmp_path).read_text()), "expect": ["skipped"] * 9}
    path = tmp_path / "failing.json"
    path.write_text(json.dumps(config))

    assert run_main(path, tmp_path / "run") == 1
    assert read_calls(fake_wandb)[-1] == {"call": "finish", "exit_code": 1}


@pytest.mark.unit
def test_a_key_written_before_the_end_uploads_nothing(
    tmp_path: Path, fake_wandb: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HONING_FAKE_WANDB_LEAK", "init")

    assert run_main(logged_config(tmp_path), tmp_path / "run") == 1

    calls = read_calls(fake_wandb)
    assert "save" not in [call["call"] for call in calls]
    assert calls[-1] == {"call": "finish", "exit_code": 1}
    err = capsys.readouterr().err
    assert "logging: wandb/leak-init.txt holds a key; nothing was uploaded" in err
    assert KEY not in err


@pytest.mark.unit
def test_a_key_written_as_the_run_ends_fails_the_run(
    tmp_path: Path, fake_wandb: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HONING_FAKE_WANDB_LEAK", "finish")

    assert run_main(logged_config(tmp_path), tmp_path / "run") == 1
    assert "logging: wandb/leak-finish.txt holds a key after the run ended" in capsys.readouterr().err


@pytest.mark.unit
def test_a_logging_start_that_fails_stops_the_run(
    tmp_path: Path, fake_wandb: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HONING_FAKE_WANDB_FAIL", "1")

    assert run_main(logged_config(tmp_path), tmp_path / "run") == 2

    assert "cannot start logging: RuntimeError: fake wandb: init failed" in capsys.readouterr().err
    assert [call["call"] for call in read_calls(fake_wandb)] == ["init"]
    assert not (tmp_path / "run/summary.json").exists()


# ---------------------------------------------------------------- the logging block


@pytest.mark.unit
@pytest.mark.parametrize(
    ("block", "message"),
    [
        ({"mlflow": {}}, 'logging must be {"wandb": {...}}'),
        ({"wandb": {**WANDB, "transcripts": True}}, "unknown logging.wandb keys ['transcripts']"),
        ({"wandb": {"entity": "e", "tags": []}}, "logging.wandb.project must be a non-empty string"),
        ({"wandb": {**WANDB, "entity": ""}}, "logging.wandb.entity, when given, must be a non-empty string"),
        ({"wandb": {**WANDB, "project": ""}}, "logging.wandb.project must be a non-empty string"),
        ({"wandb": {**WANDB, "tags": "live"}}, "logging.wandb.tags must be a list"),
        ({"wandb": {**WANDB, "mode": "disabled"}}, "logging.wandb.mode must be one of ['online', 'offline']"),
    ],
    ids=["backend", "unknown-key", "no-project", "empty-entity", "empty-project", "tags", "mode"],
)
def test_a_bad_logging_block_is_refused_before_anything_runs(
    tmp_path: Path, fake_wandb: Path, block: dict[str, Any], message: str
) -> None:
    config = {**json.loads(ALL_PLANES.read_text()), "logging": block}

    with pytest.raises(ContractError, match=message.replace("[", r"\[").replace("{", r"\{")):
        build(config, REPO, tmp_path / "run")
    assert not (tmp_path / "run").exists()


@pytest.mark.unit
def test_online_logging_needs_the_key_and_offline_does_not(
    tmp_path: Path, fake_wandb: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("WANDB_API_KEY")
    config = {**json.loads(ALL_PLANES.read_text()), "logging": {"wandb": WANDB}}

    with pytest.raises(ContractError, match="this config needs environment variable WANDB_API_KEY"):
        build(config, REPO, tmp_path / "online")
    offline = build({**config, "logging": {"wandb": {**WANDB, "mode": "offline"}}}, REPO, tmp_path / "offline")
    assert offline.tracking is not None and offline.tracking.mode == "offline"


@pytest.mark.unit
def test_logging_without_the_wandb_extra_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "wandb", raising=False)
    monkeypatch.setattr("importlib.util.find_spec", lambda name, *args: None)

    with pytest.raises(ContractError, match=r"needs the optional 'wandb' extra: pip install -e '\.\[wandb\]'"):
        logging_settings({"wandb": WANDB})


# ---------------------------------------------------------------- what a step logs


@pytest.mark.unit
def test_a_step_logs_only_named_fields_and_finite_numbers() -> None:
    step = {
        "move": "harness",
        "label": "pre_training",
        "result": "rejected",
        "release": None,
        "reason": "the candidate's output was: secret",
        "final_output": "secret",
        "episodes": 6,
    }
    record = {
        "op": "reject",
        "candidate": {
            "metrics": {"gate_tasks": 3, "method_episodes": 6.0, "flag": True, "nan": float("nan")},
            "harness": {"files": {"AGENTS.md": "secret"}},
            "reason": "secret",
        },
        "decision": {"reason": "secret", "metrics": {"gepa_served_mean": float("-inf"), "gepa_metric_calls": 12.0}},
        "method_state": {"candidates": ["secret"]},
    }
    usage = {"policy_calls": 4, "reflection_cost_usd": None, "note": "secret"}

    values = step_values(step, [record], usage)

    assert values == {
        "move/kind": "harness",
        "move/label": "pre_training",
        "move/result": "rejected",
        "gate/episodes": 6.0,
        "move/published": 0.0,
        "candidate/gate_tasks": 3.0,
        "candidate/method_episodes": 6.0,
        "selector/gepa_metric_calls": 12.0,
        "usage/policy_calls": 4.0,
    }


@pytest.mark.unit
def test_chain_records_are_read_once_and_only_when_complete(tmp_path: Path) -> None:
    path = tmp_path / "commits.jsonl"
    path.write_text('{"op": "publish"}\n')
    records = ChainRecords(path)  # what was there before is not a step's

    assert records.new() == []
    with path.open("a") as handle:
        handle.write('{"op": "reject"}\n{"op": "sk')
    assert records.new() == [{"op": "reject"}]
    with path.open("a") as handle:
        handle.write('ip"}\n')
    assert records.new() == [{"op": "skip"}]
    assert records.new() == []


@pytest.mark.unit
def test_without_an_entity_the_run_goes_to_the_keys_default(tmp_path: Path, fake_wandb: Path) -> None:
    config = {**json.loads(ALL_PLANES.read_text()), "logging": {"wandb": {"project": "llm-honing-e2e"}}}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))

    assert run_main(path, tmp_path / "run") == 0
    init = read_calls(fake_wandb)[0]
    assert (init["project"], init["entity"], init["tags"]) == ("llm-honing-e2e", None, [])


@pytest.mark.unit
def test_local_paths_are_replaced_longest_first() -> None:
    # Under the home directory, as a checkout usually is; nothing is created there.
    root = Path.home() / "honing-test-checkout-that-does-not-exist"
    work_dir = root / "runs/one"
    text = f"{work_dir}/summary.json {root}/configs/x.json {Path.home()}/.honing-keys"

    assert without_local_paths(text, work_dir, root) == "<work_dir>/summary.json <repo>/configs/x.json ~/.honing-keys"
