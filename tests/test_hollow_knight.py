"""The Unity demo refuses bad decisions and preserves caller-owned game files."""

from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from urllib.error import URLError

import pytest

from honing.backends.jev.server import DecisionServer
from tests.conftest import load_script


@pytest.fixture
def demo(repo_root: Path):
    path = repo_root / "integrations/hollow-knight/run.py"
    assert path.is_file(), "The portable Hollow Knight decision client is missing"
    return load_script("integrations/hollow-knight/run.py")


def answer(actions: dict[str, str], *, identity: int = 7) -> dict:
    return {
        "id": identity,
        "answers": {"action": {"choice": "right", "probabilities": {key: float(key == "right") for key in actions}}},
    }


@pytest.mark.unit
def test_model_argmax_becomes_the_input_without_a_policy_fallback(demo) -> None:
    assert demo.checked_action(answer(demo.ACTIONS), 7) == ("right", 1.0)


@pytest.mark.unit
@pytest.mark.parametrize("problem", ["stale", "missing", "nan", "negative", "sum", "non_argmax", "boolean"])
def test_bad_model_response_cannot_become_an_input(demo, problem: str) -> None:
    response = answer(demo.ACTIONS)
    result = response["answers"]["action"]
    if problem == "stale":
        response["id"] = 6
    elif problem == "missing":
        del result["probabilities"]["wait"]
    elif problem == "nan":
        result["probabilities"]["right"] = float("nan")
    elif problem == "negative":
        result["probabilities"]["right"] = -1
    elif problem == "sum":
        result["probabilities"]["right"] = 0.5
    elif problem == "non_argmax":
        result["choice"] = "left"
    else:
        result["probabilities"]["right"] = True
    with pytest.raises(ValueError):
        demo.checked_action(response, 7)


@pytest.mark.unit
def test_rounded_laya_distribution_is_accepted(demo) -> None:
    response = answer(demo.ACTIONS)
    probabilities = response["answers"]["action"]["probabilities"]
    probabilities.update(dict.fromkeys(probabilities, 0.0909))
    response["answers"]["action"]["choice"] = "wait"
    assert demo.checked_action(response, 7) == ("wait", 0.0909)


@pytest.mark.unit
def test_observation_and_all_choices_survive_the_wire_request(demo) -> None:
    observation = {"id": 7, "frame": 24, "scene": "Level1", "x": -2.0, "y": 4.0, "health": 5}
    request = demo.request_for(observation, 12)
    assert request["id"] == 7
    assert json.loads(request["parts"]["essential"][-1]) == observation
    assert set(request["questions"]["action"]["criteria"]) == {
        "wait",
        "left",
        "right",
        "jump_left",
        "jump_right",
        "dash_left",
        "dash_right",
        "attack_left",
        "attack_right",
        "attack_up",
        "attack_down",
    }


@pytest.mark.integration
def test_client_uses_honing_health_and_decision_routes(demo) -> None:
    class Model:
        metadata: ClassVar[dict[str, str]] = {"family": "unit-test"}

        def predict(self, parts, questions):
            return {"answers": answer(questions["action"]["criteria"])["answers"]}

    server = DecisionServer(("127.0.0.1", 0), Model())
    thread = threading.Thread(target=server.serve, daemon=True)
    thread.start()
    try:
        client = demo.DecisionClient(server.url, timeout_seconds=2)
        assert client.health()["metadata"] == {"family": "unit-test"}
        request = demo.request_for({"id": 7, "frame": 0}, 12)
        assert demo.checked_action(client.decide(request), 7) == ("right", 1.0)
    finally:
        server.stop()
        thread.join(timeout=5)
        server.server_close()


@pytest.mark.unit
def test_atomic_command_replaces_the_old_input(demo, tmp_path: Path) -> None:
    target = tmp_path / "action.json"
    target.write_text('{"id": 1, "action": "left"}')
    demo.atomic_json(target, {"id": 2, "action": "right"})
    assert json.loads(target.read_text()) == {"id": 2, "action": "right"}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["action.json"]


@pytest.mark.unit
def test_copy_patch_preserves_original_controller(repo_root: Path, tmp_path: Path) -> None:
    path = repo_root / "integrations/hollow-knight/build.py"
    assert path.is_file(), "The portable isolated game builder is missing"
    builder = load_script("integrations/hollow-knight/build.py")
    source = tmp_path / "source"
    source.mkdir()
    controller = (
        'class PlayerController { void Read() { Input.GetAxis("Horizontal"); Input.GetAxis("Horizontal"); '
        'Input.GetAxis("Vertical"); Input.GetButtonDown("Jump"); Input.GetButtonUp("Jump"); '
        "Input.GetKeyDown(KeyCode.J); Input.GetKeyDown(KeyCode.K); } }"
    )
    (source / "PlayerController.cs").write_text(controller)
    (source / "Menu.cs").write_text("public class Menu : MonoBehaviour\n{\n}")
    originals = {p.name: p.read_bytes() for p in source.iterdir()}
    output = tmp_path / "patched"
    builder.prepare_sources(source, output)
    assert {p.name: p.read_bytes() for p in source.iterdir()} == originals
    compiled_input = (output / "PlayerController.cs").read_text()
    assert compiled_input.count("HoningInput.") == 7
    assert "HoningBridge.Boot()" in (output / "Menu.cs").read_text()


@pytest.mark.unit
def test_different_controller_is_refused_before_building(repo_root: Path, tmp_path: Path) -> None:
    path = repo_root / "integrations/hollow-knight/build.py"
    assert path.is_file(), "The portable isolated game builder is missing"
    builder = load_script("integrations/hollow-knight/build.py")
    (tmp_path / "PlayerController.cs").write_text("class PlayerController {}")
    with pytest.raises(ValueError, match="input"):
        builder.prepare_sources(tmp_path, tmp_path / "patched")


@pytest.mark.unit
def test_reusing_an_input_app_as_output_is_refused(repo_root: Path, tmp_path: Path) -> None:
    path = repo_root / "integrations/hollow-knight/build.py"
    assert path.is_file(), "The portable isolated game builder is missing"
    builder = load_script("integrations/hollow-knight/build.py")
    app = tmp_path / "Game.app"
    app.mkdir()
    with pytest.raises(ValueError):
        builder.validate_output(app, app)
    with pytest.raises(ValueError):
        builder.validate_output(app, app / "Demo.app")


@pytest.mark.unit
def test_missing_render_frame_refuses_video_export(repo_root: Path, tmp_path: Path) -> None:
    path = repo_root / "integrations/hollow-knight/encode.py"
    assert path.is_file(), "The portable video exporter is missing"
    encoder = load_script("integrations/hollow-knight/encode.py")
    (tmp_path / "result.json").write_text('{"capture": true, "headless": false, "heuristicFallback": false}')
    (tmp_path / "observations.jsonl").write_text('{"frame": 0}\n{"frame": 3}\n')
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "000000.png").write_bytes(b"frame")
    (frames / "000002.png").write_bytes(b"frame")
    with pytest.raises(ValueError, match="frame"):
        encoder.validate_capture(tmp_path)


@pytest.mark.unit
def test_headless_session_cannot_be_published_as_a_recording(repo_root: Path, tmp_path: Path) -> None:
    path = repo_root / "integrations/hollow-knight/encode.py"
    assert path.is_file(), "The portable video exporter is missing"
    encoder = load_script("integrations/hollow-knight/encode.py")
    (tmp_path / "result.json").write_text('{"capture": false, "headless": true, "heuristicFallback": false}')
    with pytest.raises(ValueError, match="captur"):
        encoder.validate_capture(tmp_path)


@pytest.mark.unit
def test_linked_managed_directory_cannot_modify_the_input_app(repo_root: Path, tmp_path: Path, monkeypatch) -> None:
    builder = load_script("integrations/hollow-knight/build.py")
    source = tmp_path / "source"
    scripts = source / "Hollow Knight/Assets/Scripts"
    scripts.mkdir(parents=True)
    (scripts / "PlayerController.cs").write_text("class PlayerController {" + 'Input.GetAxis("x");' * 7 + "}")
    (scripts / "Menu.cs").write_text("public class Menu : MonoBehaviour {}")
    app = tmp_path / "Game.app"
    data = app / "Contents/Resources/Data"
    data.mkdir(parents=True)
    original_managed = tmp_path / "caller-managed"
    original_managed.mkdir()
    original = original_managed / "Assembly-CSharp.dll"
    original.write_bytes(b"caller assembly")
    (data / "Managed").symlink_to(original_managed, target_is_directory=True)
    unity = tmp_path / "Unity.app"
    mono = unity / "Contents/MonoBleedingEdge/bin/mono"
    mono.parent.mkdir(parents=True)
    mono.touch()
    compiler = unity / "Contents/MonoBleedingEdge/lib/mono/4.5/csc.exe"
    compiler.parent.mkdir(parents=True)
    compiler.touch()
    revision = json.loads((repo_root / "integrations/hollow-knight/pins.json").read_text())["game"]["revision"]

    def check_output(command, **_kwargs):
        if "ls-files" in command:
            return "Hollow Knight/Assets/Scripts/PlayerController.cs\0Hollow Knight/Assets/Scripts/Menu.cs\0"
        return revision + "\n"

    def compile_or_check(command, **_kwargs):
        if command[0] == str(mono):
            target = next(value.removeprefix("-out:") for value in command if value.startswith("-out:"))
            Path(target).write_bytes(b"demo assembly")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(builder.subprocess, "check_output", check_output)
    monkeypatch.setattr(builder.subprocess, "run", compile_or_check)
    output = tmp_path / "Demo.app"
    builder.build(source, app, unity, output)
    assert original.read_bytes() == b"caller assembly"
    assert (output / "Contents/Resources/Data/Managed/Assembly-CSharp.dll").read_bytes() == b"demo assembly"
    assert (data / "Managed").is_symlink()


@pytest.mark.integration
def test_untracked_csharp_cannot_enter_a_pinned_build(tmp_path: Path) -> None:
    builder = load_script("integrations/hollow-knight/build.py")
    scripts = tmp_path / "Hollow Knight/Assets/Scripts"
    scripts.mkdir(parents=True)
    (scripts / "PlayerController.cs").write_text("class PlayerController {}")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "-c",
            "core.hooksPath=/dev/null",
            "commit",
            "--no-gpg-sign",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    revision = subprocess.check_output(["git", "-C", str(tmp_path), "rev-parse", "HEAD"], text=True).strip()
    (scripts / "Unexpected.cs").write_text("class Unexpected {}")
    with pytest.raises(ValueError, match="untracked"):
        builder.verify_source(tmp_path, revision)


@pytest.mark.integration
@pytest.mark.parametrize("case", ["invalid_response", "http_error", "stalled_game", "success"])
def test_episode_owns_its_player_on_success_and_failures(demo, tmp_path: Path, monkeypatch, case: str) -> None:
    player = tmp_path / "player"
    player.write_text("#!/usr/bin/env python3\nstall = " + repr(case == "stalled_game") + "\n" + """
import json, os, time
from pathlib import Path
session = Path(os.environ['HONING_SESSION_DIR'])
(session / 'watchdog.txt').write_text(os.environ.get('HONING_MODEL_TIMEOUT_SECONDS', '180'))
identity = 1
while True:
    if not stall:
        observation = dict(id=identity, frame=(identity-1)*12, health=5, jumps=2, fixedSteps=(identity-1)*18,
                           scene='Level1', x=0, y=0, vx=0, vy=0, wallLeft=2, wallRight=2,
                           simulationTime=(identity-1)*0.36, grounded=True, floorLeft=True, floorRight=True,
                           enemies=[], exits=[])
        temporary = session / 'observation.tmp'
        temporary.write_text(json.dumps(observation))
        temporary.replace(session / 'observation.json')
    while True:
        action = session / 'action.json'
        if action.exists():
            command = json.loads(action.read_text())
            if command['id'] == identity:
                if command['action'] == 'stop':
                    raise SystemExit(0)
                identity += 1
                break
        time.sleep(0.01)
""")
    player.chmod(0o755)

    class Client:
        def __init__(self, *_args):
            pass

        def health(self):
            return {"ready": True, "metadata": {"family": "protocol-test"}}

        def decide(self, request):
            if case == "http_error":
                raise URLError("test server failure")
            response = answer(demo.ACTIONS, identity=request["id"])
            if case == "invalid_response":
                response["answers"]["action"]["choice"] = "left"
            return response

    launched = []
    popen = subprocess.Popen

    def launch(*args, **kwargs):
        process = popen(*args, **kwargs)
        launched.append(process)
        return process

    monkeypatch.setattr(demo, "DecisionClient", Client)
    monkeypatch.setattr(demo.subprocess, "Popen", launch)
    args = SimpleNamespace(
        game=player,
        decision_url="http://unused",
        timeout=300,
        output_dir=tmp_path / "runs",
        headless=True,
        capture=False,
        scene="Level1",
        decisions=1,
        frames=12,
        observation_timeout=0.1 if case == "stalled_game" else 2,
    )
    if case == "success":
        session = demo.run_episode(args)
        assert json.loads((session / "result.json").read_text())["decisions"] == 1
        assert float((session / "watchdog.txt").read_text()) > 300
    else:
        expected = {"invalid_response": ValueError, "http_error": URLError, "stalled_game": TimeoutError}[case]
        with pytest.raises(expected):
            demo.run_episode(args)
        session = next(args.output_dir.iterdir())
        if case != "stalled_game":
            record = json.loads((session / "decisions.jsonl").read_text())
            assert record["request"]["id"] == 1
            assert (
                "error" in record
                if case == "http_error"
                else record["response"]["answers"]["action"]["choice"] == "left"
            )
        assert not (session / "result.json").exists()
    assert len(launched) == 1 and launched[0].poll() is not None
