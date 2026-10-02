"""Component factories for the all-planes pipeline (``configs/e2e/all_planes.json``).

Every plane is the walking skeleton's fake, sharing one :class:`FakeWorld`,
except the harness runner: :class:`~honing.e2e.workspace_runner.WorkspaceEpisodeRunner`
runs ``tests/fakes/pipeline_agent.py`` as an agent process, which calls a
stand-in OpenAI-compatible policy server started here. The server answers as
the policy's weights would: the reply reports the weights' quality, plus a
bonus when the system prompt asks the agent to verify its work. So training,
pruning, and quantization change what the agent's model says, and the harness
changes what the model is told, as with a real agent.
"""

from __future__ import annotations

import json
import sys
import threading
from collections.abc import Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from honing.campaign import rules
from honing.core import (
    HarnessSpec,
    ModelSnapshot,
    PolicyEndpoint,
    PolicyRef,
    RunnerSupport,
    Task,
    TrainingBatch,
    TrainJobSpec,
    TrainResult,
)
from honing.e2e.components import PipelineContext
from honing.e2e.workspace_runner import WorkspaceEpisodeRunner
from tests.fakes.builder import TASKS
from tests.fakes.compression import (
    ActivationCountScorer,
    FakeQuantizer,
    FixedTelemetryStore,
    KeepTopKExpertSelector,
    UniformExpertEditor,
    skewed_expert_telemetry,
)
from tests.fakes.data import FakeRolloutProvider, ListDatasetProvider, token_batch
from tests.fakes.harness import SEED_HARNESS, AppendInstructionMethod, FakeTaskGrader
from tests.fakes.model import SLIME_LIKE, FakeModelSource, FakeTrainer, FakeWorld, small_moe

VERIFY_BONUS = 0.05


def _world(context: PipelineContext) -> FakeWorld:
    world = context.shared.setdefault("world", FakeWorld())
    if not isinstance(world, FakeWorld):
        raise TypeError("context.shared['world'] is not a FakeWorld")
    return world


def model_source(context: PipelineContext) -> FakeModelSource:
    return FakeModelSource(_world(context))


class SlimeLikeTrainer:
    """Satisfies Trainer with the BackendSupport the walking skeleton expects of Slime (``SLIME_LIKE``).

    A starting point for a real backend's stand-in: it refuses a job or batch the campaign's rules would refuse,
    as a real trainer must, then trains the fake model.
    """

    def __init__(self, world: FakeWorld, gain: float) -> None:
        self._inner = FakeTrainer(world, support=SLIME_LIKE, gain=gain)
        self.support = SLIME_LIKE

    def train(self, job: TrainJobSpec, base: ModelSnapshot, batch: TrainingBatch | None) -> TrainResult:
        rules.algorithm_requirements_met(job, self.support)
        rules.batch_meets_contract(job, batch)
        return self._inner.train(job, base, batch)


def trainer(context: PipelineContext, gain: float = 0.1) -> SlimeLikeTrainer:
    return SlimeLikeTrainer(_world(context), gain)


def data(context: PipelineContext) -> ListDatasetProvider:
    return ListDatasetProvider([token_batch()])


def rollout(context: PipelineContext) -> FakeRolloutProvider:
    return FakeRolloutProvider()


def scorer(context: PipelineContext) -> ActivationCountScorer:
    return ActivationCountScorer()


def selector(context: PipelineContext) -> KeepTopKExpertSelector:
    return KeepTopKExpertSelector()


def editor(context: PipelineContext) -> UniformExpertEditor:
    return UniformExpertEditor(_world(context))


def quantizer(context: PipelineContext) -> FakeQuantizer:
    return FakeQuantizer(_world(context))


def telemetry(context: PipelineContext) -> FixedTelemetryStore:
    return FixedTelemetryStore(skewed_expert_telemetry(small_moe()))


def grader(context: PipelineContext) -> FakeTaskGrader:
    return FakeTaskGrader()


def method(context: PipelineContext) -> AppendInstructionMethod:
    return AppendInstructionMethod()


def tasks(context: PipelineContext) -> tuple[Task, ...]:
    return TASKS


def harness_seed(context: PipelineContext) -> HarnessSpec:
    return SEED_HARNESS


# ---------------------------------------------------------------- the policy an agent calls


class _PolicyHandler(BaseHTTPRequestHandler):
    server: _PolicyServer

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_POST(self) -> None:
        # /<checksum>/v1/chat/completions, the shape an OpenAI-compatible client sends
        checksum, _, route = self.path.strip("/").partition("/")
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        if route != "v1/chat/completions" or checksum not in self.server.world.quality:
            self.send_response(404)
            self.end_headers()
            return
        system = " ".join(m.get("content", "") for m in body.get("messages", []) if m.get("role") == "system")
        verified = "verify" in system.lower()
        quality = round(min(1.0, self.server.world.quality[checksum] + (VERIFY_BONUS if verified else 0.0)), 6)
        content = f"quality={quality} verified={int(verified)}"
        reply = {
            "object": "chat.completion",
            "model": body.get("model"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": len(system.split()), "completion_tokens": 2},
        }
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class _PolicyServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, world: FakeWorld) -> None:
        super().__init__(("127.0.0.1", 0), _PolicyHandler)
        self.world = world


class FakePolicyEndpoints:
    """Satisfies PolicyEndpoints: every checkpoint of the fake world is served by one local policy server."""

    def __init__(self, world: FakeWorld) -> None:
        self._server = _PolicyServer(world)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        self.port = self._server.server_address[1]

    def resolve(self, policy: PolicyRef) -> PolicyEndpoint:
        return PolicyEndpoint(
            base_url=f"http://127.0.0.1:{self.port}/{policy.checkpoint.checksum}/v1",
            tokenizer_location="fake",
            served_model="fake-policy",
            runtime="sglang",
        )


def endpoints(context: PipelineContext) -> FakePolicyEndpoints:
    return FakePolicyEndpoints(_world(context))


def runner(
    context: PipelineContext,
    command: Sequence[str] = ("{python}", "{root}/tests/fakes/pipeline_agent.py"),
    agents: Sequence[str] = ("claude_code",),
    file_kinds: Sequence[str] = ("instructions", "skill", "command", "settings"),
    concurrency: int = 4,
) -> WorkspaceEpisodeRunner:
    """The reference workspace runner around the stand-in agent; reads the ``endpoints`` component."""
    names = {"python": sys.executable, "root": str(context.root)}
    return WorkspaceEpisodeRunner(
        command=[part.format(**names) for part in command],
        workspaces=context.work_dir / "workspaces",
        endpoints=context.component("endpoints"),
        support=RunnerSupport(
            agents=frozenset(agents),
            file_kinds=frozenset(file_kinds),  # type: ignore[arg-type]
            programmable_loop=False,
        ),
        concurrency=concurrency,
    )
