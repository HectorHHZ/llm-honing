"""SGLangPolicyEndpoints on GPUs, as a campaign uses them: a release and a checkpoint trained from it, each started on
demand, the least recently used one stopped when the GPU is needed, and the same checkpoint served as a chat model.

Usage, from an environment with torch, safetensors and transformers, with SGLANG_PYTHON naming the SGLang
environment's interpreter:

    python scripts/e2e/sglang_endpoints.py <checkpoint> <work-dir> <decision-gpu> <chat-gpu>

The "trained" checkpoint is the given one with its final norm scaled by 1.02: new weights, a new checksum, and
answers that move a little. Each step prints what it checked and how long the server took to start.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from honing.backends.sglang import SGLangChatModel, hf_snapshot  # noqa: E402
from honing.backends.sglang.endpoints import ServingSettings, SGLangPolicyEndpoints  # noqa: E402
from honing.core import ChatMessage, CheckpointRef, ContractError, PolicyRef  # noqa: E402

REQUEST = {
    "parts": {"essential": ["Tick 120. Credits 3100. Army 4."], "details": ["Enemy tanks seen to the north."]},
    "questions": {
        "build": {
            "type": "choice",
            "instructions": "Which structure should be built next?",
            "criteria": {"power": "Power plant", "refinery": "Ore refinery", "barracks": "Barracks"},
        },
        "attack": {"type": "noul", "instructions": "Should the army attack now?"},
    },
}


def trained_copy(checkpoint: Path, out: Path) -> Path:
    """The checkpoint with its final norm scaled by 1.02, written as a new Hugging Face directory."""
    import torch
    from safetensors.torch import load_file, save_file

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for path in checkpoint.iterdir():
        if not path.name.endswith(".safetensors"):
            shutil.copy(path, out / path.name)
    changed = False
    for path in sorted(checkpoint.glob("*.safetensors")):
        tensors = load_file(str(path))
        for name in list(tensors):
            if name.endswith("model.norm.weight") or name.endswith("language_model.norm.weight"):
                tensors[name] = (tensors[name].float() * 1.02).to(tensors[name].dtype)
                changed = True
        save_file(tensors, str(out / path.name), metadata={"format": "pt"})
    assert changed, "no final norm found"
    del torch
    return out


def policy(checkpoint: Path) -> PolicyRef:
    snapshot = hf_snapshot(checkpoint)
    return PolicyRef(snapshot.checkpoint, snapshot.architecture, "master")


def decide(url: str) -> dict:
    body = json.dumps(REQUEST).encode()
    request = urllib.request.Request(f"{url}/v1/decide", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        answers: dict = json.load(response)["answers"]
        return answers


def health(url: str) -> dict:
    with urllib.request.urlopen(f"{url}/v1/health", timeout=30) as response:
        metadata: dict = json.load(response)["metadata"]
        return metadata


def largest_change(a: dict, b: dict) -> float:
    """The largest probability difference between two answers to REQUEST."""
    choice = max(
        abs(a["build"]["probabilities"][k] - b["build"]["probabilities"][k]) for k in a["build"]["probabilities"]
    )
    return max(choice, abs(a["attack"]["noul"] - b["attack"]["noul"]))


def timed(endpoints: SGLangPolicyEndpoints, p: PolicyRef) -> tuple[object, float]:
    started = time.perf_counter()
    endpoint = endpoints.resolve(p)
    return endpoint, time.perf_counter() - started


def main() -> None:
    checkpoint, work, decision_gpu, chat_gpu = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
    python = Path(os.environ["SGLANG_PYTHON"])
    release, trained = policy(checkpoint), policy(trained_copy(checkpoint, work / "trained"))
    print(f"release {release.checkpoint.checksum[:12]}, trained {trained.checkpoint.checksum[:12]}", flush=True)

    settings = ServingSettings(python=python, gpus=(decision_gpu,), runtime="jev_decision", log_dir=work / "logs")
    with SGLangPolicyEndpoints(settings) as endpoints:
        first, seconds = timed(endpoints, release)
        meta = health(first.base_url)  # type: ignore[attr-defined]
        assert meta["weightsSha256"] == release.checkpoint.checksum and meta["model"]
        before = decide(first.base_url)  # type: ignore[attr-defined]
        print(f"decision server for the release started in {seconds:.0f} s: {meta['model']} {meta['revision']}")
        print("  release answers:", json.dumps(before))
        again, seconds = timed(endpoints, release)
        assert again == first and seconds < 1, "a second resolve must reuse the running server"
        print(f"resolving the release again reused its server ({seconds * 1000:.0f} ms)")

        second, seconds = timed(endpoints, trained)
        assert endpoints.serving == {trained.checkpoint.checksum: decision_gpu}, "the release should have stopped"
        after = decide(second.base_url)  # type: ignore[attr-defined]
        assert health(second.base_url)["weightsSha256"] == trained.checkpoint.checksum  # type: ignore[attr-defined]
        moved = largest_change(before, after)
        print(
            f"trained checkpoint started in {seconds:.0f} s on the same GPU after stopping the release; "
            f"answers moved by up to {moved:.4f}"
        )
        print("  trained answers:", json.dumps(after))

        third, seconds = timed(endpoints, release)
        same = decide(third.base_url)  # type: ignore[attr-defined]
        drift = largest_change(before, same)
        print(
            f"the release restarted in {seconds:.0f} s; its answers differ from before it was stopped by {drift:.2e}"
        )

        wrong = PolicyRef(
            CheckpointRef(release.checkpoint.location, "hf_safetensors", "0" * 64), release.architecture, "master"
        )
        try:
            endpoints.resolve(wrong)
        except ContractError as error:
            print("a policy naming other weights was refused:", error)
        else:
            raise AssertionError("a wrong checksum must be refused")

    chat = ServingSettings(python=python, gpus=(chat_gpu,), runtime="sglang", log_dir=work / "logs")
    with SGLangPolicyEndpoints(chat) as endpoints:
        endpoint, seconds = timed(endpoints, trained)
        model = SGLangChatModel(endpoint.base_url, endpoint.served_model)  # type: ignore[attr-defined]
        reply = model.complete(
            [ChatMessage("user", "Name one structure a base needs first. Answer in five words or fewer.")],
            max_tokens=32,
            temperature=0.0,
        )
        print(
            f"chat server for the trained checkpoint started in {seconds:.0f} s; reply {reply.text!r} "
            f"({reply.input_tokens} in, {reply.output_tokens} out)"
        )
    print("endpoints: ok")


if __name__ == "__main__":
    main()
