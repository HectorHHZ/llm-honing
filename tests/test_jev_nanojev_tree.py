"""NanoJev's token-tree inference: the tree against a brute-force reading of what each path must see, the setting and
server option that select it, the response it assembles, and, where torch and transformers are installed, the tree
forward against running each path alone. The GPU comparison with NanoJev's own DecisionPredictor is the gpu test
below and integrations/ra2/parity/nanojev_tree.py."""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from honing.backends.jev import nanojev
from honing.backends.jev.nanojev import NanoJevSettings, checked_answer
from honing.backends.jev.nanojev_tree import (
    HEAD_ROWS,
    HEAD_SLOTS,
    GraphForward,
    PrefetchedTokenizer,
    TokenTree,
    TreeScorer,
    common_prefix_length,
    nanojev_segments,
    padded_size,
    slot_layout,
)
from honing.backends.jev.server import main as server_main
from tests.test_ra2_backend import REPO

QUESTIONS = {
    "g1": {"type": "choice", "instructions": "Which action now?", "criteria": {"A": "wait", "B": "build power"}},
    "g2": {"type": "choice", "instructions": "Where?", "criteria": {"x": "north", "y": "south", "z": "east"}},
}


def path_from_root(tree: TokenTree, node: int) -> list[int]:
    """A node's path from the root, read from preorder depths alone: a node's parent is the nearest earlier node
    one level up."""
    path = [node]
    for earlier in range(node - 1, -1, -1):
        if tree.depth[earlier] == tree.depth[path[-1]] - 1:
            path.append(earlier)
    return path[::-1]


def random_paths(rng: random.Random) -> list[list[int]]:
    prefix = [rng.randrange(5) for _ in range(rng.randrange(0, 30))]
    paths = [
        prefix[: rng.randrange(0, len(prefix) + 1)] + [rng.randrange(5) for _ in range(rng.randrange(1, 8))]
        for _ in range(rng.randrange(1, 12))
    ]
    if rng.random() < 0.3:
        paths.append(list(paths[0]))  # a duplicate path
    if rng.random() < 0.3 and len(paths[0]) > 1:
        paths.append(paths[0][:-1])  # a path that is a prefix of another
    return paths


# ---------------------------------------------------------------- the tree


@pytest.mark.unit
def test_each_path_s_leaf_spells_the_path_with_depths_as_positions() -> None:
    rng = random.Random(7)
    for _ in range(300):
        paths = random_paths(rng)
        tree = TokenTree(paths)
        for path, leaf in zip(paths, tree.leaf, strict=True):
            nodes = path_from_root(tree, leaf)
            assert [tree.tokens[n] for n in nodes] == path
            assert [tree.depth[n] for n in nodes] == list(range(len(path)))


@pytest.mark.unit
def test_a_node_sees_exactly_its_ancestors_and_itself_and_padding_sees_only_itself() -> None:
    rng = random.Random(11)
    for _ in range(100):
        tree = TokenTree(random_paths(rng))
        size = tree.size + rng.randrange(0, 4)
        end = tree.subtree_end(size)
        for i in range(size):
            seen = {j for j in range(size) if j <= i < end[j]}
            assert seen == (set(path_from_root(tree, i)) if i < tree.size else {i})


@pytest.mark.unit
def test_paths_that_share_a_prefix_store_it_once() -> None:
    state = list(range(100, 1300))
    first = [[*state, 1, 2, k, 3] for k in range(9)]  # nine options of one question about the state
    second = [[*state, 4, k, 5] for k in range(2)]  # a second question about the same state
    tree = TokenTree(first + second)
    assert tree.size == 1200 + (2 + 9 * 2) + (1 + 2 * 2)
    assert padded_size(tree.size) == 1280


@pytest.mark.unit
def test_common_prefix_length_and_refusals() -> None:
    assert common_prefix_length([1, 2, 3], [1, 2, 4]) == 2
    assert common_prefix_length([1, 2], [1, 2, 3]) == 2
    assert common_prefix_length([], [1]) == 0
    with pytest.raises(ValueError, match="no empty path"):
        TokenTree([[1], []])
    with pytest.raises(ValueError, match="cannot shrink"):
        TokenTree([[1, 2]]).subtree_end(1)


# ---------------------------------------------------------------- the setting and the server option


@pytest.mark.unit
def test_the_setting_refuses_an_unknown_mode_and_a_compiled_tree_without_bf16() -> None:
    assert NanoJevSettings(Path("pins.json"), Path("src")).inference == "reference"
    with pytest.raises(ValueError, match="inference must be one of"):
        NanoJevSettings(Path("pins.json"), Path("src"), inference="fast")
    with pytest.raises(ValueError, match="bf16 autocast"):
        NanoJevSettings(Path("pins.json"), Path("src"), precision="fp32", inference="tree-compiled")
    assert NanoJevSettings(Path("pins.json"), Path("src"), precision="fp32", inference="tree").inference == "tree"


@pytest.mark.unit
def test_the_server_passes_the_inference_mode_to_the_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[NanoJevSettings] = []

    def load(settings: NanoJevSettings) -> Any:
        seen.append(settings)
        raise SystemExit(0)

    monkeypatch.setattr(nanojev.NanoJevDecisionModel, "load", load)
    with pytest.raises(SystemExit):
        server_main(
            [
                "--family",
                "nanojev",
                "--pins",
                "pins.json",
                "--nanojev-source",
                "src",
                "--nanojev-inference",
                "tree-compiled",
            ]
        )
    assert seen[-1].inference == "tree-compiled"
    with pytest.raises(SystemExit):
        server_main(["--family", "nanojev", "--pins", "pins.json", "--nanojev-source", "src"])
    assert seen[-1].inference == "reference"


# ---------------------------------------------------------------- the response (torch)


def stand_in_upstream() -> types.ModuleType:
    """The three functions TreeScorer takes from NanoJev's predictor module, reduced as in test_jev_nanojev.py."""

    def validate_request(payload: dict[str, Any]) -> list[dict[str, Any]]:
        return payload["states"]

    def prepare_examples(payload: dict[str, Any], tokenizer: Any, max_length: int) -> list[dict[str, Any]]:
        (row,) = payload["states"]
        examples = []
        for qid, q in row["questions"].items():
            prefix = [len(word) for word in f"State: {row['state']} Question: {q['instructions']}".split()]
            leaves = [[*prefix, len(k), len(v), 0] for k, v in q["criteria"].items()]
            examples.append(
                {"qid": qid, "state_id": row["id"], "type": "choice", "candidate_ids": list(q["criteria"]),
                 "leaf_tokens": leaves}
            )  # fmt: skip
        return examples

    def answer_from_probabilities(example: dict[str, Any], probabilities: list[float]) -> dict[str, Any]:
        best = max(range(len(probabilities)), key=probabilities.__getitem__)
        ids = example["candidate_ids"]
        return {"type": "choice", "probabilities": dict(zip(ids, probabilities, strict=True)), "choice": ids[best]}

    module = types.ModuleType("stand_in_predictor")
    module.__dict__.update(
        validate_request=validate_request,
        prepare_examples=prepare_examples,
        answer_from_probabilities=answer_from_probabilities,
    )
    return module


@pytest.mark.unit
def test_the_tree_scorer_answers_with_the_reference_s_fields() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    hidden = 8
    model = torch.nn.Module()
    model.norm, model.scalar, model.set_head = torch.nn.LayerNorm(hidden), torch.nn.Linear(hidden, 1), "none"
    predictor = types.SimpleNamespace(
        model=model, tokenizer=None, limit=64, device=torch.device("cpu"), precision="fp32", root=Path("ckpt"),
        run_config={"model": "Qwen/Qwen3-0.6B", "resolved_model_revision": "r", "set_head": "none"},
        inference_calls=0, disable_native_triton=False,
    )  # fmt: skip
    scorer = TreeScorer.__new__(TreeScorer)
    scorer._predictor, scorer._upstream, scorer.mode, scorer._graphs = predictor, stand_in_upstream(), "tree", None
    leaves = torch.randn(5, hidden)
    scorer._leaf_states = lambda tree: leaves
    payload = nanojev.payload("tick 90 credits 1200", QUESTIONS)
    response = scorer.predict(payload)
    assert set(response) == {"schema_version", "checkpoint", "temperature", "execution", "states"}
    execution = response["execution"]
    assert (execution["questions"], execution["candidate_paths"], execution["forward_passes"]) == (2, 5, 1)
    assert execution["prefix_sharing"] is True and execution["inference"] == "tree"
    assert execution["requests_in_forward"] == 1 and execution["heads_in_graph"] is False
    answers = response["states"][0]["answers"]
    with torch.inference_mode():
        expected = model.scalar(model.norm(leaves)).squeeze(-1)
    for qid, rows in (("g1", slice(0, 2)), ("g2", slice(2, 5))):
        checked_answer(QUESTIONS[qid]["criteria"], answers[qid]["probabilities"], answers[qid]["choice"])
        want = expected[rows].softmax(-1).tolist()
        assert list(answers[qid]["probabilities"].values()) == pytest.approx(want)


# ---------------------------------------------------------------- the tree forward (torch and transformers)


@pytest.mark.unit
def test_the_tree_forward_gives_each_path_the_state_it_has_when_run_alone() -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from honing.backends.jev.nanojev_tree import hf_leaf_states

    torch.manual_seed(0)
    config = transformers.Qwen3Config(
        vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, max_position_embeddings=256,
    )  # fmt: skip
    backbone = transformers.AutoModel.from_config(config, attn_implementation="sdpa").eval()
    rng = random.Random(3)
    state = [rng.randrange(97) for _ in range(40)]
    paths = [state + [rng.randrange(97) for _ in range(rng.randrange(1, 6))] for _ in range(7)]
    paths += [[*state[:20], 5, 6], list(paths[0])]
    with torch.inference_mode():
        merged = hf_leaf_states(backbone, torch.device("cpu"), TokenTree(paths))
        alone = torch.stack(
            [backbone(input_ids=torch.tensor([p]), use_cache=False).last_hidden_state[0, -1] for p in paths]
        )
    torch.testing.assert_close(merged, alone, atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------- the graph's heads and input (torch)


def stand_in_heads(torch: Any, hidden: int, set_head: str = "attention") -> Any:
    """A model with DecisionModel's head modules (NanoJev's scripts/train_toy_decisions.py), randomly set."""
    torch.manual_seed(5)
    model = torch.nn.Module()
    model.norm, model.scalar, model.set_head = torch.nn.LayerNorm(hidden), torch.nn.Linear(hidden, 1), set_head
    model.set_project = torch.nn.Linear(hidden + 1, 16)
    model.set_attention = torch.nn.MultiheadAttention(16, 2, batch_first=True)
    model.set_output = torch.nn.Linear(16, 1)
    return model.eval()


MIXED = [
    {"type": "choice", "candidate_ids": ["a", "b", "c"], "leaf_tokens": [[1], [2], [3]]},
    {"type": "score", "candidate_ids": ["0", "1", "2", "3"], "leaf_tokens": [[4], [5], [6], [7]]},
    {"type": "boolean", "candidate_ids": ["false", "true"], "leaf_tokens": [[8]]},
    {"type": "choice", "candidate_ids": ["x", "y"], "leaf_tokens": [[9], [10]]},
]


@pytest.mark.unit
@pytest.mark.parametrize("set_head", ["attention", "none"])
def test_the_graph_s_heads_score_a_whole_call_as_head_logits_scores_each_question(set_head: str) -> None:
    torch = pytest.importorskip("torch")
    from honing.backends.jev.nanojev_tree import _row_probabilities, graph_heads, head_logits

    model = stand_in_heads(torch, 8, set_head)
    nodes = torch.randn(20, 8)  # every tree node's final state; the call's paths end on nodes 10..19
    leaf = list(range(10, 20))
    rows = slot_layout(MIXED)
    assert rows is not None
    shape = (HEAD_ROWS, HEAD_SLOTS)
    slots, valid = torch.zeros(shape, dtype=torch.long), torch.zeros(shape, dtype=torch.bool)
    for row, paths in enumerate(rows):
        for slot, path in enumerate(paths):
            slots[row, slot], valid[row, slot] = leaf[path], True
    valid[len(rows) :, 0] = True
    choice = torch.tensor([e["type"] == "choice" for e in MIXED] + [False] * (HEAD_ROWS - len(MIXED)))
    with torch.inference_mode():
        logits = graph_heads(model, nodes, slots, valid, choice)
        probabilities = logits.softmax(-1)
        offset = 0
        for row, example in enumerate(MIXED):
            count = len(example["leaf_tokens"])
            alone = head_logits(model, [example], nodes[leaf[offset : offset + count]])[0]
            offset += count
            k = len(example["candidate_ids"])
            want = alone[:k].softmax(-1).tolist()
            assert _row_probabilities(example, logits[row], probabilities[row]) == pytest.approx(want, abs=1e-6)


@pytest.mark.unit
def test_the_graph_layout_has_room_for_sixteen_questions_of_sixteen_options() -> None:
    assert slot_layout(MIXED) == [[0, 1, 2], [3, 4, 5, 6], [7], [8, 9]]
    many = [{"leaf_tokens": [[1], [2]]}] * (HEAD_ROWS + 1)
    wide = [{"leaf_tokens": [[i] for i in range(HEAD_SLOTS + 1)]}]
    assert slot_layout(many) is None and slot_layout(wide) is None


@pytest.mark.unit
def test_the_packed_input_holds_the_padded_tree_then_the_heads_layout() -> None:
    tree = TokenTree([[5, 6], [5, 7]])  # nodes: 5, 6, 7; leaves 1 and 2
    values = GraphForward.pack(tree, [[0, 1]], [True], 4)
    cells = HEAD_ROWS * HEAD_SLOTS
    tokens, positions, end = values[:4], values[4:8], values[8:12]
    slots, valid, choice = values[12 : 12 + cells], values[12 + cells : 12 + 2 * cells], values[12 + 2 * cells :]
    assert (tokens, positions, end) == ([5, 6, 7, 0], [0, 1, 1, 0], tree.subtree_end(4))
    assert slots[:2] == [1, 2] and valid[:2] == [1, 1] and sum(valid) == 2 + (HEAD_ROWS - 1)
    assert choice == [1] + [0] * (HEAD_ROWS - 1)


# ---------------------------------------------------------------- encoding


class LetterTokenizer:
    """One id per character, plus a count of encode calls, as a stand-in for a Hugging Face tokenizer."""

    eos_token_id = 0

    def __init__(self) -> None:
        self.calls = 0

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        self.calls += 1
        return ([1] if add_special_tokens else []) + [ord(c) % 89 + 2 for c in text]

    def __call__(self, texts: list[str], add_special_tokens: bool = True) -> dict[str, list[list[int]]]:
        self.calls += 1
        return {"input_ids": [[ord(c) % 89 + 2 for c in text] for text in texts]}


@pytest.mark.unit
def test_a_prefetched_tokenizer_encodes_as_the_tokenizer_does_in_one_call() -> None:
    tokenizer = LetterTokenizer()
    prefetched = PrefetchedTokenizer(tokenizer, ["State:\nx\n", "Candidate:\na: b\nDecision:", "State:\nx\n"])
    assert tokenizer.calls == 1
    plain = LetterTokenizer()
    for text in ["State:\nx\n", "Candidate:\na: b\nDecision:"]:
        assert prefetched.encode(text, add_special_tokens=False) == plain.encode(text, add_special_tokens=False)
    assert tokenizer.calls == 1
    assert prefetched.encode("other", add_special_tokens=False) == plain.encode("other", add_special_tokens=False)
    assert prefetched.encode("State:\nx\n") == plain.encode("State:\nx\n")  # with special tokens: the original
    assert tokenizer.calls == 3 and prefetched.eos_token_id == 0


@pytest.mark.unit
def test_the_segments_are_the_strings_nanojev_encodes() -> None:
    questions = {
        "g1": {"type": "choice", "instructions": "Which?", "criteria": {"A": "wait", "B": "build"}},
        "g2": {"type": "boolean", "instructions": "Safe?", "criteria": {"true": "no enemy near"}},
        "g3": {"type": "score", "instructions": "How far?", "criteria": ["near", "far"]},
    }
    assert nanojev_segments("tick 9", questions) == [
        "State:\ntick 9\n",
        "Question type: choice\nQuestion:\nWhich?\n",
        "Candidate:\nA: wait\nDecision:",
        "Candidate:\nB: build\nDecision:",
        "Question type: boolean\nQuestion:\nSafe?\nTrue criterion: no enemy near\n",
        "Candidate:\nThe proposition is true.\nDecision:",
        "Question type: score\nQuestion:\nHow far?\n",
        "Candidate:\nnear\nDecision:",
        "Candidate:\nfar\nDecision:",
    ]


# ---------------------------------------------------------------- the tree model (torch)


def nanojev_like_upstream(calls: list[int]) -> types.ModuleType:
    """NanoJev's prepare_examples for choice questions, reduced to its encoding, with a call count."""
    module = stand_in_upstream()

    def prepare_examples(payload: dict[str, Any], tokenizer: Any, max_length: int) -> list[dict[str, Any]]:
        calls.append(1)
        (row,) = payload["states"]
        examples = []
        for qid, q in row["questions"].items():
            prefix = tokenizer.encode(f"State:\n{row['state']}\n", add_special_tokens=False) + tokenizer.encode(
                f"Question type: choice\nQuestion:\n{q['instructions']}\n", add_special_tokens=False
            )
            leaves = [
                prefix
                + tokenizer.encode(f"Candidate:\n{k}: {v}\nDecision:", add_special_tokens=False)
                + [tokenizer.eos_token_id]
                for k, v in q["criteria"].items()
            ]
            if max(map(len, leaves)) > max_length:
                raise ValueError("over max_length")
            examples.append(
                {"qid": qid, "state_id": row["id"], "type": "choice", "candidate_ids": list(q["criteria"]),
                 "leaf_tokens": leaves}
            )  # fmt: skip
        return examples

    module.__dict__["prepare_examples"] = prepare_examples
    return module


@pytest.mark.unit
def test_the_tree_model_answers_several_requests_in_one_call_as_it_answers_each() -> None:
    torch = pytest.importorskip("torch")
    from honing.backends.jev.nanojev import NanoJevTreeDecisionModel

    hidden = 8
    model = stand_in_heads(torch, hidden)
    calls: list[int] = []
    upstream = nanojev_like_upstream(calls)
    predictor = types.SimpleNamespace(
        model=model, tokenizer=LetterTokenizer(), limit=512, device=torch.device("cpu"), precision="fp32",
        root=Path("ckpt"), run_config={"model": "m", "resolved_model_revision": "r", "set_head": "attention"},
        inference_calls=0, disable_native_triton=False,
    )  # fmt: skip
    scorer = TreeScorer.__new__(TreeScorer)
    scorer._predictor, scorer._upstream, scorer.mode, scorer._graphs = predictor, upstream, "tree", None

    def leaf_states(tree: TokenTree) -> Any:
        # A leaf's state depends only on its path, as the tree forward guarantees.
        paths = [[tree.tokens[n] for n in path_from_root(tree, leaf)] for leaf in tree.leaf]
        return torch.stack([torch.sin(torch.arange(hidden) * (1 + sum(p) % 97) + len(p)) for p in paths])

    scorer._leaf_states = leaf_states
    tree_model = NanoJevTreeDecisionModel(predictor, upstream, {"family": "nanojev"}, scorer)
    requests = [
        ({"essential": ["tick 90 credits 1200"]}, QUESTIONS),
        ({"essential": ["tick 90 credits 1200", "fire after x"]}, {"g1": QUESTIONS["g1"]}),
        ({"essential": ["tick 90 credits 1200", "fire after y"]}, {"g1": QUESTIONS["g1"]}),
    ]
    together = tree_model.predict_many(requests)
    assert len(calls) == 3  # one encoding per request: the packing's measurement is what the model scores
    alone = [tree_model.predict(parts, questions) for parts, questions in requests]
    for one, other in zip(together, alone, strict=True):
        assert one["stateText"] == other["stateText"] and one["tokenAudit"] == other["tokenAudit"]
        for qid, answer in one["answers"].items():
            assert answer["choice"] == other["answers"][qid]["choice"]
            assert answer["probabilities"] == pytest.approx(other["answers"][qid]["probabilities"], abs=1e-6)
    executions = [r["modelResult"]["execution"] for r in together]
    assert {e["requests_in_forward"] for e in executions} == {3}
    assert len({e["token_tree_tokens"] for e in executions}) == 1


# ---------------------------------------------------------------- against DecisionPredictor (GPU)


@pytest.mark.gpu
@pytest.mark.slow
def test_on_a_gpu_the_tree_stays_as_close_to_fp32_as_the_reference(tmp_path: Path) -> None:
    """Runs integrations/ra2/parity/nanojev_tree.py on 8 requests. Needs NANOJEV_SOURCE_DIR (the pinned NanoJev
    checkout), the pinned model in the Hugging Face cache, and the GPU index in NANOJEV_GPU (default 0)."""
    if "NANOJEV_SOURCE_DIR" not in os.environ:
        pytest.skip("NANOJEV_SOURCE_DIR is not set")
    script = REPO / "integrations/ra2/parity/nanojev_tree.py"
    subprocess.run([sys.executable, str(script), str(tmp_path), os.environ.get("NANOJEV_GPU", "0"), "8"], check=True)
    summary = json.loads((tmp_path / "summary.json").read_text())
    reference = summary["reference bf16"]["largest_abs_probability_difference_from_fp32"]
    for mode in ("tree bf16", "tree-compiled bf16"):
        largest = summary[mode]["largest_abs_probability_difference_from_fp32"]
        assert largest < max(2 * reference, 0.005)
        # A choice may differ from fp32 only where fp32's top two options are closer than the rounding moved them.
        assert all(d["fp32_top2_gap"] <= 2 * largest for d in summary[mode]["choices_differing_from_fp32"])
