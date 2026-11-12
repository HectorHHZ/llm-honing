"""What the slime agent worker calls in slime, checked against slime's source at the pin, without importing it.

slime's adapter and harness import torch, which Honing's environments do not have, so
these tests parse the source with ``ast`` instead. They read a checkout of slime on
``PYTHONPATH`` (``scripts/fetch-slime-agent.sh``, or the full one from
``scripts/setup-slime-agent-env.sh``) and skip without one; the ``test-sandbox`` CI job
sets ``HONING_REQUIRE_SLIME=1``. The signatures and constants the worker relies on are
listed in the worker itself (``SLIME_SIGNATURES``, ``SLIME_CONSTANTS``), so a change on
either side fails here. ``tests/test_slime_agent_worker.py`` runs the same calls for real.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

from honing.backends.slime import agent_episode as files
from honing.backends.slime import agent_worker as worker
from tests.conftest import require

Definition = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


@pytest.fixture(scope="module")
def slime_agent() -> Path:
    spec = importlib.util.find_spec("slime")
    locations = list(spec.submodule_search_locations or []) if spec is not None else []
    if not locations or not (Path(locations[0]) / "agent" / "adapters" / "common.py").is_file():
        require("slime", "no slime checkout is on PYTHONPATH; run scripts/fetch-slime-agent.sh")
    return Path(locations[0]) / "agent"


def module(root: Path, relative: str) -> ast.Module:
    path = root / relative
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def find(tree: ast.Module, qualname: str) -> Definition:
    body: list[ast.stmt] = tree.body
    found: Definition | None = None
    for name in qualname.split("."):
        found = next(
            (node for node in body if isinstance(node, DEFINITIONS) and node.name == name),
            None,
        )
        assert found is not None, f"{qualname} is not defined"
        body = found.body
    assert found is not None
    return found


def parameters(function: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[str, ...]:
    """Parameter names after ``self``, keyword-only ones as ``*name``, and ``*args``/``**kwargs`` marked."""
    arguments = function.args
    positional = [argument.arg for argument in (*arguments.posonlyargs, *arguments.args)]
    if positional[:1] == ["self"]:
        positional = positional[1:]
    names = [*positional]
    if arguments.vararg is not None:
        names.append(f"*args:{arguments.vararg.arg}")
    names += [f"*{argument.arg}" for argument in arguments.kwonlyargs]
    if arguments.kwarg is not None:
        names.append(f"**{arguments.kwarg.arg}")
    return tuple(names)


def split(key: str) -> tuple[str, str]:
    relative, qualname = key.split(":")
    return relative, qualname


@pytest.mark.unit
@pytest.mark.parametrize("key", sorted(worker.SLIME_SIGNATURES))
def test_each_slime_function_the_worker_calls_has_the_parameters_it_passes(slime_agent: Path, key: str) -> None:
    relative, qualname = split(key)
    function = find(module(slime_agent, relative), qualname)
    assert isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)), f"{key} is not a function"
    assert parameters(function) == worker.SLIME_SIGNATURES[key]


@pytest.mark.unit
@pytest.mark.parametrize("key", sorted(worker.SLIME_CONSTANTS))
def test_each_slime_constant_the_worker_relies_on_has_its_value(slime_agent: Path, key: str) -> None:
    relative, qualname = split(key)
    owner, _, name = qualname.rpartition(".")
    tree = module(slime_agent, relative)
    body = find(tree, owner).body if owner else tree.body
    values = [
        ast.literal_eval(node.value)
        for node in body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    ]
    assert values == [worker.SLIME_CONSTANTS[key]]


def _awaited(function: Definition) -> list[ast.Call]:
    """The calls ``function`` awaits, in source order."""
    calls = [
        node.value for node in ast.walk(function) if isinstance(node, ast.Await) and isinstance(node.value, ast.Call)
    ]
    return sorted(calls, key=lambda call: (call.lineno, call.col_offset))


def _name(call: ast.Call) -> str:
    return ast.unparse(call.func)


@pytest.mark.unit
def test_slimes_steps_before_the_agents_clock_are_the_ones_the_worker_deadline_counts(slime_agent: Path) -> None:
    # The runner's deadline for the worker adds these limits (agent_episode.PRE_LAUNCH_SECONDS): what
    # ClaudeCodeHarness.run awaits in the sandbox before exec_and_wait starts polling against the budget.
    run = find(module(slime_agent, "harness/common.py"), "BaseHarness.run")
    assert [_name(call) for call in _awaited(run)] == [
        "_sandbox.ensure_agent_user",
        "self.write_config",
        "self.launch_and_wait",
    ]
    launch = find(module(slime_agent, "harness/claude_code.py"), "ClaudeCodeHarness.launch_and_wait")
    assert [_name(call) for call in _awaited(launch)] == ["run_agent"]
    run_agent = find(module(slime_agent, "harness/common.py"), "run_agent")
    assert [_name(call) for call in _awaited(run_agent)] == ["sb.exec", "exec_and_wait"]

    found: dict[str, tuple[int, ...]] = {}
    writes = 0
    for key in files.SLIME_PRE_LAUNCH_EXEC_SECONDS:
        relative, qualname = split(key)
        timeouts = []
        for call in _awaited(find(module(slime_agent, relative), qualname)):
            if _name(call) == "_await_done_marker":  # the budget starts here
                break
            if _name(call) == "sb.write_file":
                writes += 1
            elif _name(call) == "sb.exec":
                (timeout,) = [keyword.value for keyword in call.keywords if keyword.arg == "timeout"]
                timeouts.append(ast.literal_eval(timeout))
        found[key] = tuple(timeouts)
    assert found == dict(files.SLIME_PRE_LAUNCH_EXEC_SECONDS)
    assert writes == files.SLIME_PRE_LAUNCH_FILE_WRITES


@pytest.mark.unit
def test_the_adapter_calls_call_sglang_generate_by_its_module_level_name(slime_agent: Path) -> None:
    # The worker counts turns by replacing the module attribute (agent_worker._count_turns),
    # which works only while _run_turn looks the name up in the module when it runs.
    tree = module(slime_agent, "adapters/common.py")
    run_turn = find(tree, "BaseAdapter._run_turn")
    calls = [
        node
        for node in ast.walk(run_turn)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "call_sglang_generate"
    ]
    assert len(calls) == 1
    imported = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "call_sglang_generate" not in imported


@pytest.mark.unit
def test_a_turn_record_has_the_token_fields_the_worker_counts(slime_agent: Path) -> None:
    record = find(module(slime_agent, "trajectory.py"), "TurnRecord")
    fields = [
        node.target.id for node in record.body if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    ]
    assert {"prompt_ids", "output_ids"} <= set(fields)


@pytest.mark.unit
def test_slime_writes_the_transcript_where_the_worker_reads_it(slime_agent: Path) -> None:
    source = (slime_agent / "harness" / "common.py").read_text(encoding="utf-8")
    run_agent = ast.get_source_segment(source, find(ast.parse(source), "run_agent"))
    assert run_agent is not None
    assert f'meta_dir = f"{{workdir}}/{worker.SLIME_META_DIR}"' in run_agent
    assert 'out_file=f"{meta_dir}/trajectory.jsonl"' in run_agent
    assert f"{worker.WORKDIR}/{worker.SLIME_META_DIR}/trajectory.jsonl" == worker.SANDBOX_TRANSCRIPT


@pytest.mark.unit
def test_the_harness_runs_the_cli_where_the_worker_checks_for_it(slime_agent: Path) -> None:
    source = (slime_agent / "harness" / "claude_code.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for method in ("ClaudeCodeHarness.install_cli", "ClaudeCodeHarness.launch_and_wait"):
        segment = ast.get_source_segment(source, find(tree, method))
        assert segment is not None and worker.CLI_PATH in segment, method


@pytest.mark.unit
def test_the_adapter_serves_the_messages_route_and_the_packages_export_what_the_worker_imports(
    slime_agent: Path,
) -> None:
    source = (slime_agent / "adapters" / "anthropic.py").read_text(encoding="utf-8")
    routes = ast.get_source_segment(source, find(ast.parse(source), "AnthropicAdapter._register_routes"))
    assert routes is not None and 'add_post("/v1/messages"' in routes
    exported = {
        "adapters/__init__.py": "AnthropicAdapter",
        "harness/__init__.py": "ClaudeCodeHarness",
    }
    for relative, name in exported.items():
        names = {
            alias.name
            for node in ast.walk(module(slime_agent, relative))
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        assert name in names, relative
    handle = find(module(slime_agent, "aiohttp_threaded.py"), "AppHandle")
    members = {
        node.target.id for node in handle.body if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    members |= {node.name for node in handle.body if isinstance(node, ast.FunctionDef)}
    assert {"port", "loop", "stop"} <= members
