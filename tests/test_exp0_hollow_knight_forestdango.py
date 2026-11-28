"""The full Hollow Knight (ForestDango) Exp 0 v0 tools, offline: a recorded v12_standing run, no game or model.

The fixture is the first six decisions of a recorded run of the pinned harness against Laya (King's Pass, new
game), cut so that every window and observation those decisions used is kept. Re-rendering a request needs Laya's
tokenizer (see integrations/hollow-knight-forestdango/README.md): those tests run when the ``tokenizers`` package
is installed and ``HONING_LAYA_TOKENIZER`` names the file, and are skipped otherwise. Nothing is downloaded.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import require

REPO = Path(__file__).resolve().parents[1]
GAME = REPO / "integrations" / "hollow-knight-forestdango"
V0 = GAME / "exp0" / "v0"
COMMON = REPO / "integrations" / "_shared" / "exp0"
RUN = Path(__file__).resolve().parent / "fixtures" / "exp0_hollow_knight_forestdango" / "f1_v12_laya_new"
PACKAGE = "harness.v12_standing"


def tool(*args: str, env: dict[str, str] | None = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run one of the v0 scripts the way an operator does, so that its bare-name imports stay in its process."""
    return subprocess.run(
        [sys.executable, "-B", *args], cwd=V0, capture_output=True, text=True, check=check, timeout=600, env=env
    )


def need_tokenizer() -> None:
    """Skip unless the harness can load Laya's tokenizer from a local file (no download)."""
    path = os.environ.get("HONING_LAYA_TOKENIZER")
    if importlib.util.find_spec("tokenizers") is None or not path or not Path(path).is_file():
        require(
            "hk_tokenizer",
            "re-rendering needs the tokenizers package and HONING_LAYA_TOKENIZER "
            "(integrations/hollow-knight-forestdango/README.md)",
        )


@pytest.mark.integration
def test_strict_replay_rerenders_every_request_and_window(tmp_path: Path) -> None:
    need_tokenizer()
    report_path = tmp_path / "replay.json"
    tool("replay_check.py", str(RUN), "--package", PACKAGE, "--json", str(report_path))
    report = json.loads(report_path.read_text())
    assert report["ok"] and not report["mismatches"]
    assert report["checked"] == report["live_render_exact"] == report["snapshot_rerender_exact"] == 6
    assert report["windows_replayed"] == report["windows_logged"] == 13


@pytest.mark.integration
def test_release_faults_act_from_the_first_decision(tmp_path: Path) -> None:
    need_tokenizer()
    plan = json.loads((V0 / "plan.json").read_text())
    assert set(plan["fault_forms"]) == {"persistent"}
    report_path = tmp_path / "faults.json"
    tool("offline_faults.py", str(RUN), "--package", PACKAGE, "--json", str(report_path))
    report = json.loads(report_path.read_text())
    found = {(entry["instance"], entry["form"]): entry for kind in ("harness", "controller") for entry in report[kind]}
    harness = found[(plan["default_instances"]["harness"], "persistent")]
    assert harness["trigger"]["decision"] == 1 and harness["changed_decisions"] == harness["decisions"] == 6
    assert harness["trigger"]["state_lines_changed"]
    assert harness["trigger"]["move_options_changed"] == harness["trigger"]["attack_options_changed"] == []
    controller = found[(plan["default_instances"]["controller"], "persistent")]
    assert controller["trigger"]["decision"] == 1 and controller["changed_decisions"] == controller["decisions"]


@pytest.mark.integration
@pytest.mark.parametrize("problem", ["missing", "not the pinned file"])
def test_tools_refuse_to_render_without_the_pinned_tokenizer(tmp_path: Path, problem: str) -> None:
    tokenizer = tmp_path / "tokenizer.json"
    if problem != "missing":
        tokenizer.write_text("{}\n")
    env = {**os.environ, "HONING_LAYA_TOKENIZER": str(tokenizer), "HF_HUB_OFFLINE": "1"}
    report_path = tmp_path / "replay.json"
    process = tool("replay_check.py", str(RUN), "--package", PACKAGE, "--json", str(report_path), env=env, check=False)
    assert process.returncode != 0
    assert "tokenizer" in process.stderr
    assert not report_path.exists()


@pytest.mark.integration
def test_contract_plan_is_the_render_of_the_frozen_plan(tmp_path: Path) -> None:
    code = "import sys; from pathlib import Path; import to_contract; print(to_contract.write_frozen_plan(Path(sys.argv[1])))"
    rendered = Path(tool("-c", code, str(tmp_path)).stdout.strip())
    assert rendered.read_bytes() == (V0 / "collection.json").read_bytes()
    spec = importlib.util.spec_from_file_location("_exp0_hk_schema_lite", COMMON / "schema_lite.py")
    assert spec is not None and spec.loader is not None
    schema_lite = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(schema_lite)
    plan = json.loads(rendered.read_text(encoding="utf-8"))
    assert schema_lite.errors(plan, schema_lite.load_schema(COMMON / "schemas" / "collection.schema.json")) == []
    assert (plan["game_id"], plan["dataset_dir"], plan["per_condition"], plan["initial_groups"]) == (
        "hollow-knight-forestdango",
        "hollow-knight-forestdango/v0/",
        10,
        10,
    )


@pytest.mark.integration
def test_release_driver_snapshots_only_files_that_exist() -> None:
    code = (
        "import json, release_v0; print(json.dumps({"
        "'missing': sorted(k for k, p in release_v0.SOURCES.items() if not p.exists()), "
        "'templates': sorted(p.name for p in release_v0.TEMPLATES.glob('*.md')), "
        "'isolation': str(release_v0.ISOLATION), 'common': str(release_v0.COMMON)}))"
    )
    report = json.loads(tool("-c", code).stdout)
    assert report["missing"] == []
    assert Path(report["common"]) == COMMON and Path(report["isolation"]) == COMMON / "isolation_check.py"
    assert report["templates"] == [
        "DATASET.md",
        "LABELS.md",
        "PROTOCOL.md",
        "RELEASE_RECEIPT.md",
        "RESULTS.md",
        "RUNBOOK.md",
        "TODO.md",
    ]


@pytest.mark.unit
def test_bridge_sources_are_the_pinned_build_sources() -> None:
    pins = json.loads((V0 / "pins.json").read_text())
    sources = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(GAME.glob("bridge/*.cs"))}
    assert sources == pins["game"]["bridge_source"]


@pytest.mark.integration
def test_harness_snapshot_is_the_pinned_one() -> None:
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from contract import fingerprint_tree; print(fingerprint_tree(sys.argv[2]))"
    )
    snapshot = V0 / "harness" / "v12_standing"
    fingerprint = tool("-c", code, str(COMMON), str(snapshot)).stdout.strip()
    record = json.loads((V0 / "harness" / "v12_standing.SOURCE.json").read_text())
    pins = json.loads((V0 / "pins.json").read_text())
    assert fingerprint == record["fingerprint"] == pins["code"]["harness"]["fingerprint"]
