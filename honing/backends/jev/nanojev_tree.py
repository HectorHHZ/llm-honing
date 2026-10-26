"""Score NanoJev's candidate paths as one token tree, in one forward pass per model call.

NanoJev scores each option of a question on its own path: the state, the
question, then that candidate and an end token. ``DecisionPredictor`` pads the
paths into rows and runs every row, so the state and the question, which all
paths of a question share, pass through the backbone once per option. A request
with nine options of about 1,250 tokens each runs about 11,000 tokens to score
about 1,700 distinct ones.

``TokenTree`` merges the paths of a call into a trie instead. Each node is one
token; it attends to itself and its ancestors only, and its position id is its
depth. A path's last node therefore sees the same tokens at the same positions
in the same causal order as when its path runs alone, so its hidden state is the
same function of the weights; only the order of floating-point sums differs.
Paths of different questions about one state share the state's nodes too, and
one call may carry several requests (the decision server's branches), whose
states share whatever prefix they have. The scoring heads then run per question
as ``DecisionModel.forward`` runs them.

``NanoJevSettings.inference`` picks the forward over the tree:

- ``tree``: the checkpoint's own Hugging Face backbone, given the tree as a 4D
  attention mask and depth position ids, then the heads per question.
- ``tree-compiled``: a Qwen3 forward written out with the dtypes the Hugging
  Face model uses under bf16 autocast, the q/k/v and gate/up projections each as
  one matmul, block-sparse FlexAttention over the tree mask, ``torch.compile``,
  and one CUDA graph per tree size rounded up to ``BUCKET_TOKENS``. The graph
  also runs the scoring heads, for every question of the call at once
  (``graph_heads``). Loading compiles the forward and captures the graphs, which
  takes minutes the first time a compile cache is empty.

A call's CPU work (the tree, the graph's packed input) is ``TreeScorer.prepare``
and its GPU work ``TreeScorer.score``, so a server can prepare one call while
the GPU runs another. ``PrefetchedTokenizer`` lets NanoJev's own
``prepare_examples`` encode every string of a request with one tokenizer call.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

INFERENCE_MODES = ("reference", "tree", "tree-compiled")
# Trees are padded to a multiple of this many tokens, which is also the tile of FlexAttention's block mask.
BUCKET_TOKENS = 128
# FlexAttention's default tiles for head dimension 128 need more shared memory than an RTX A6000 (sm86) has.
# These fit, and read the tree mask faster; they read out of bounds on strided q/k/v, so the forward hands
# FlexAttention contiguous tensors.
FLEX_KERNEL = {"BLOCK_M": 128, "BLOCK_N": 32, "num_stages": 2, "num_warps": 4}
# The compiled graph scores up to HEAD_ROWS questions of up to HEAD_SLOTS options each; a call with more runs
# the compiled forward without a graph and the heads per question.
HEAD_ROWS, HEAD_SLOTS = 16, 16


def common_prefix_length(a: list[int], b: list[int]) -> int:
    """Length of the longest common prefix of two token lists, by binary search over slice comparisons."""
    low, high = 0, min(len(a), len(b))
    while low < high:
        middle = (low + high + 1) // 2
        if a[:middle] == b[:middle]:
            low = middle
        else:
            high = middle - 1
    return low


class TokenTree:
    """A trie over token paths, numbered in depth-first preorder.

    Paths are inserted in sorted order, so each one branches off the path before
    it at their longest common prefix. That numbers the nodes in preorder: a
    node's parent has a smaller index, and every subtree is a contiguous range,
    so node ``j`` is an ancestor-or-self of node ``i`` exactly when
    ``j <= i < subtree_end()[j]``. ``leaf[k]`` is the node of path ``k``'s last token.
    """

    def __init__(self, paths: Sequence[Sequence[int]]) -> None:
        rows = [list(path) for path in paths]
        if not rows or not all(rows):
            raise ValueError("A token tree needs at least one path and no empty path")
        self.tokens: list[int] = []
        self.depth: list[int] = []
        self.leaf: list[int] = [0] * len(rows)
        previous: list[int] = []
        previous_nodes: list[int] = []
        for index in sorted(range(len(rows)), key=rows.__getitem__):
            path = rows[index]
            shared = common_prefix_length(path, previous)
            start = len(self.tokens)
            self.tokens.extend(path[shared:])
            self.depth.extend(range(shared, len(path)))
            nodes = previous_nodes[:shared] + list(range(start, len(self.tokens)))
            self.leaf[index] = nodes[-1]
            previous, previous_nodes = path, nodes

    @property
    def size(self) -> int:
        return len(self.tokens)

    def subtree_end(self, size: int | None = None) -> list[int]:
        """One past each node's last descendant, for the tree padded to ``size`` nodes.

        In preorder a subtree ends at the first later node no deeper than its
        root. Padding nodes cover only themselves, and no real node covers them.
        """
        size = self.size if size is None else size
        if size < self.size:
            raise ValueError("Padding cannot shrink the tree")
        end = [self.size] * self.size + list(range(self.size + 1, size + 1))
        open_nodes: list[int] = []
        for node, depth in enumerate(self.depth):
            while open_nodes and self.depth[open_nodes[-1]] >= depth:
                end[open_nodes.pop()] = node
            open_nodes.append(node)
        return end


def padded_size(size: int) -> int:
    return math.ceil(size / BUCKET_TOKENS) * BUCKET_TOKENS


def tree_mask(end: torch.Tensor) -> torch.Tensor:
    """The (n, n) boolean attention mask of a tree: node ``i`` may attend to node ``j`` iff ``j <= i < end[j]``."""
    import torch

    index = torch.arange(end.shape[0], device=end.device)
    return (index[None, :] <= index[:, None]) & (index[:, None] < end[None, :])


def hf_leaf_states(backbone: Any, device: Any, tree: TokenTree) -> torch.Tensor:
    """Final hidden states of the tree's leaves from the Hugging Face backbone, given the tree as a 4D mask."""
    import torch

    tokens = torch.tensor([tree.tokens], device=device)
    positions = torch.tensor([tree.depth], device=device)
    mask = tree_mask(torch.tensor(tree.subtree_end(), device=device))[None, None]
    hidden = backbone(input_ids=tokens, attention_mask=mask, position_ids=positions, use_cache=False)
    return hidden.last_hidden_state[0, torch.tensor(tree.leaf, device=device)]


def head_logits(model: Any, examples: list[Mapping[str, Any]], leaves: torch.Tensor) -> torch.Tensor:
    """Option logits from the paths' final hidden states, as ``DecisionModel.forward`` computes them after its
    backbone call (NanoJev's ``scripts/train_toy_decisions.py``). ``leaves`` holds the examples' paths in order."""
    import torch
    from torch.nn import functional

    device = leaves.device
    kmax = max(len(ex["candidate_ids"]) for ex in examples)
    h = leaves.new_zeros((len(examples), kmax, leaves.shape[-1]))
    valid = torch.zeros((len(examples), kmax), dtype=torch.bool, device=device)
    offset = 0
    for i, ex in enumerate(examples):
        n = len(ex["leaf_tokens"])
        h[i, :n] = leaves[offset : offset + n]
        valid[i, : len(ex["candidate_ids"])] = True
        offset += n
    h = model.norm(h)
    z = model.scalar(h).squeeze(-1).float()
    choice = torch.tensor([i for i, ex in enumerate(examples) if ex["type"] == "choice"], device=device)
    if model.set_head == "attention" and len(choice):
        log_k = valid[choice].sum(-1).float().log()[:, None, None].expand(-1, kmax, 1)
        u = model.set_project(torch.cat([h[choice], log_k.to(h.dtype)], dim=-1))
        mixed, _ = model.set_attention(u, u, u, key_padding_mask=~valid[choice], need_weights=False)
        delta = model.set_output(torch.tanh(u + mixed)).squeeze(-1).float()
        z = z.index_add(0, choice, delta)
    out = []
    for i, ex in enumerate(examples):
        if ex["type"] == "boolean":
            out.append(functional.pad(torch.stack([z[i, 0] * 0, z[i, 0]]), (0, kmax - 2)))
        else:
            out.append(z[i])
    return torch.stack(out).masked_fill(~valid, -1e9)


# ---------------------------------------------------------------- the compiled forward


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Qwen3RMSNorm: normalize in fp32, round back to the input dtype, then scale by the (fp32) weight."""
    import torch

    dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return weight * x.to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    import torch

    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _block_mask(allowed: torch.Tensor, end: torch.Tensor) -> Any:
    """FlexAttention BlockMask for a tree: tiles with no allowed entry are skipped, fully allowed tiles skip the
    mask, and partial tiles test ``kv <= q < end[kv]``. Tensor ops only, so it is captured with the graph."""
    import torch
    from torch.nn.attention.flex_attention import BlockMask

    count = allowed.shape[-1] // BUCKET_TOKENS
    tiles = allowed.view(count, BUCKET_TOKENS, count, BUCKET_TOKENS)
    some, every = tiles.any(dim=3).any(dim=1), tiles.all(dim=3).all(dim=1)

    def ordered(flags: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        order = torch.argsort(flags.to(torch.int8), dim=-1, descending=True, stable=True).to(torch.int32)
        return flags.sum(-1, dtype=torch.int32)[None, None], order[None, None]

    def mask_mod(b: torch.Tensor, h: torch.Tensor, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        return (kv <= q) & (q < end[kv])

    return BlockMask.from_kv_blocks(
        *ordered(some & ~every), *ordered(every), BLOCK_SIZE=BUCKET_TOKENS, mask_mod=mask_mod
    )


class FusedQwen3:
    """A Qwen3 backbone's forward over a token tree, for ``torch.compile``; the Hugging Face module is only read.

    It takes the steps the Hugging Face model takes under bf16 autocast, in the
    same dtypes: an fp32 residual stream and fp32 RMSNorms, bf16 matmuls, q and k
    rounded to bf16 inside their RMSNorm before the fp32 weight multiplies them,
    RoPE in fp32 and attention in bf16. The q/k/v and gate/up projections each
    run as one matmul of the same bf16 operands.
    """

    def __init__(self, backbone: Any) -> None:
        import torch

        config = backbone.config
        if config.model_type != "qwen3" or getattr(backbone, "has_sliding_layers", False):
            raise ValueError("tree-compiled needs a Qwen3 backbone with full attention in every layer; use tree")
        self.heads, self.kv_heads = config.num_attention_heads, config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // self.heads
        self.eps = config.rms_norm_eps
        self.embed = backbone.embed_tokens.weight.detach()
        self.rotary = backbone.rotary_emb
        self.norm = backbone.norm.weight.detach()
        bf16 = torch.bfloat16
        self.layers = []
        for layer in backbone.layers[: config.num_hidden_layers]:
            attention, mlp = layer.self_attn, layer.mlp
            qkv = torch.cat([attention.q_proj.weight, attention.k_proj.weight, attention.v_proj.weight])
            self.layers.append(
                {
                    "input_norm": layer.input_layernorm.weight.detach(),
                    "qkv": qkv.detach().to(bf16),
                    "q_norm": attention.q_norm.weight.detach(),
                    "k_norm": attention.k_norm.weight.detach(),
                    "o": attention.o_proj.weight.detach().to(bf16),
                    "post_norm": layer.post_attention_layernorm.weight.detach(),
                    "gate_up": torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight]).detach().to(bf16),
                    "down": mlp.down_proj.weight.detach().to(bf16),
                }
            )

    def __call__(self, tokens: torch.Tensor, positions: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
        """Final normed hidden states, shape (1, n, hidden), for a tree padded to n nodes (a multiple of
        ``BUCKET_TOKENS``) with subtree ends ``end``."""
        import torch
        from torch.nn import functional
        from torch.nn.attention.flex_attention import flex_attention

        hidden = functional.embedding(tokens, self.embed)
        cos, sin = self.rotary(hidden, positions)  # fp32, as the Hugging Face model computes them
        cos, sin = cos[:, None], sin[:, None]
        n = tokens.shape[1]
        q_size, kv_size = self.heads * self.head_dim, self.kv_heads * self.head_dim
        blocks = _block_mask(tree_mask(end), end)
        for w in self.layers:
            x = _rms_norm(hidden, w["input_norm"], self.eps)
            q, k, v = functional.linear(x.to(torch.bfloat16), w["qkv"]).split((q_size, kv_size, kv_size), dim=-1)
            q = _rms_norm(q.view(1, n, self.heads, self.head_dim), w["q_norm"], self.eps).transpose(1, 2)
            k = _rms_norm(k.view(1, n, self.kv_heads, self.head_dim), w["k_norm"], self.eps).transpose(1, 2)
            v = v.view(1, n, self.kv_heads, self.head_dim).transpose(1, 2)
            q = q * cos + _rotate_half(q) * sin
            k = k * cos + _rotate_half(k) * sin
            attended = flex_attention(
                q.to(torch.bfloat16).contiguous(),
                k.to(torch.bfloat16).contiguous(),
                v.contiguous(),
                block_mask=blocks,
                scale=self.head_dim**-0.5,
                enable_gqa=True,
                kernel_options=FLEX_KERNEL,
            )
            hidden = hidden + functional.linear(attended.transpose(1, 2).reshape(1, n, q_size), w["o"])
            x = _rms_norm(hidden, w["post_norm"], self.eps)
            gate, up = functional.linear(x.to(torch.bfloat16), w["gate_up"]).chunk(2, dim=-1)
            hidden = hidden + functional.linear(functional.silu(gate) * up, w["down"])
        return _rms_norm(hidden, self.norm, self.eps)


def nanojev_segments(state: str, questions: Mapping[str, Any]) -> list[str]:
    """The strings NanoJev's ``prepare_examples`` encodes for ``questions`` about ``state``, one call each
    (``scripts/predict_toy_decisions.py``): the state segment, each question's segment, each candidate's text."""
    strings = [f"State:\n{state}\n"]
    for question in questions.values():
        kind, criteria = question.get("type"), question.get("criteria")
        segment = f"Question type: {kind}\nQuestion:\n{question.get('instructions')}\n"
        if kind == "boolean":
            texts = ["The proposition is true."]
            for key, label in (("false", "False"), ("true", "True")):
                if isinstance(criteria, Mapping) and key in criteria:
                    segment += f"{label} criterion: {criteria[key]}\n"
        elif kind == "choice" and isinstance(criteria, Mapping):
            texts = [f"{key}: {value}" for key, value in criteria.items()]
        else:
            texts = list(criteria) if isinstance(criteria, list) else []
        strings += [segment, *(f"Candidate:\n{text}\nDecision:" for text in texts)]
    return strings


class PrefetchedTokenizer:
    """A tokenizer whose ``encode(text, add_special_tokens=False)`` answers from one batch call over ``texts``.

    Encoding strings as one batch gives each the ids it gets alone, so NanoJev's
    ``prepare_examples`` builds the same paths with this tokenizer as with the
    original; a string outside ``texts``, or with special tokens, goes to the
    original. Everything else is the original's.
    """

    def __init__(self, tokenizer: Any, texts: Sequence[str]) -> None:
        unique = list(dict.fromkeys(texts))
        self._tokenizer = tokenizer
        self._ids = dict(zip(unique, tokenizer(unique, add_special_tokens=False)["input_ids"], strict=True))

    def encode(self, text: str, add_special_tokens: bool = True, **kwargs: Any) -> list[int]:
        if not add_special_tokens and not kwargs and text in self._ids:
            return list(self._ids[text])
        return list(self._tokenizer.encode(text, add_special_tokens=add_special_tokens, **kwargs))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tokenizer, name)


def slot_layout(examples: Sequence[Mapping[str, Any]]) -> list[list[int]] | None:
    """Each example's paths, by index in the call's path order, as one row of the graph's heads; None when the
    call has more than HEAD_ROWS questions or a question has more than HEAD_SLOTS paths."""
    if len(examples) > HEAD_ROWS:
        return None
    rows, offset = [], 0
    for example in examples:
        count = len(example["leaf_tokens"])
        if count > HEAD_SLOTS:
            return None
        rows.append(list(range(offset, offset + count)))
        offset += count
    return rows


def graph_heads(
    model: Any, hidden: torch.Tensor, slots: torch.Tensor, valid: torch.Tensor, choice: torch.Tensor
) -> torch.Tensor:
    """``head_logits`` for a whole call on a fixed (HEAD_ROWS, HEAD_SLOTS) batch, as the graph runs it.

    ``hidden`` holds every tree node's final state and ``slots`` the node each
    option's path ends on; ``valid`` marks the slots that hold an option and
    ``choice`` the rows that are choice questions, the only ones the set head
    adjusts. Returns fp32 logits with -1e9 where no option is. A boolean row's
    logit is its first slot's; :meth:`TreeScorer.score` pairs it with 0 as
    ``head_logits`` does.
    """
    import torch

    h = model.norm(hidden[slots])
    z = model.scalar(h).squeeze(-1).float()
    if model.set_head == "attention":
        log_k = valid.sum(-1).float().log()[:, None, None].expand(-1, slots.shape[1], 1)
        u = model.set_project(torch.cat([h, log_k.to(h.dtype)], dim=-1))
        mixed, _ = model.set_attention(u, u, u, key_padding_mask=~valid, need_weights=False)
        delta = model.set_output(torch.tanh(u + mixed)).squeeze(-1).float()
        z = torch.where(choice[:, None], z + delta, z)
    return z.masked_fill(~valid, -1e9)


class GraphForward:
    """Option logits of a token tree, replayed from a CUDA graph captured for the tree's padded size.

    The input is one int64 buffer, built by :meth:`pack` on any thread: the
    padded tree's token ids, positions and subtree ends, then the heads' layout
    (the node each option's path ends on, which slots hold an option, which rows
    are choice questions). A call copies it to the GPU in one copy; the graph
    rebuilds the mask from the subtree ends, runs the forward, scores every
    question with ``heads`` (:func:`graph_heads`) and takes the softmax over each
    row. Graphs are captured only by :meth:`precapture`, before any is
    replayed: a graph captured after others had been replayed broke them (the
    next replay of an older graph read illegal memory, torch 2.7 on an RTX
    A6000, with or without a shared memory pool). A tree of a size without a
    graph takes the same steps without one. Calls must come from one thread at
    a time.
    """

    def __init__(self, forward: Callable[..., Any], heads: Callable[..., Any], device: Any) -> None:
        self.forward, self.heads, self.device = forward, heads, device
        self.graphs: dict[int, tuple[Any, Any, Any, Any]] = {}
        self.pool: Any = None

    @staticmethod
    def pack(tree: TokenTree, rows: Sequence[Sequence[int]], choice: Sequence[bool], size: int) -> list[int]:
        """The input for ``tree`` padded to ``size`` nodes, with ``rows`` from :func:`slot_layout`."""
        slots, valid = [0] * (HEAD_ROWS * HEAD_SLOTS), [0] * (HEAD_ROWS * HEAD_SLOTS)
        for row, paths in enumerate(rows):
            for slot, path in enumerate(paths):
                slots[row * HEAD_SLOTS + slot], valid[row * HEAD_SLOTS + slot] = tree.leaf[path], 1
        for row in range(len(rows), HEAD_ROWS):
            valid[row * HEAD_SLOTS] = 1  # an unused row scores node 0, so its set attention has a key
        flags = [int(flag) for flag in choice] + [0] * (HEAD_ROWS - len(choice))
        pad = [0] * (size - tree.size)
        return tree.tokens + pad + tree.depth + pad + tree.subtree_end(size) + slots + valid + flags

    def _run(self, packed: torch.Tensor, size: int) -> torch.Tensor:
        import torch

        cells = HEAD_ROWS * HEAD_SLOTS
        tokens, positions, end, slots, valid, choice = packed.split((size, size, size, cells, cells, HEAD_ROWS))
        # A fresh tensor rather than a view into ``packed``, as the mask_mod reading it was measured with.
        hidden = self.forward(tokens[None], positions[None], end.clone())[0]
        shape = (HEAD_ROWS, HEAD_SLOTS)
        logits = self.heads(hidden, slots.view(shape), valid.view(shape).bool(), choice.bool())
        return torch.stack([logits, logits.softmax(-1)])

    def capture(self, size: int) -> None:
        import torch

        host = torch.tensor(self.pack(TokenTree([[0]]), [[0]], [False], size), dtype=torch.long).pin_memory()
        packed = host.to(self.device)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self._run(packed, size)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool):
            out = self._run(packed, size)
        self.pool = graph.pool()
        self.graphs[size] = (graph, host, packed, out)

    def precapture(self, sizes: Sequence[int]) -> None:
        import torch

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for size in sizes:
                self.capture(size)

    def __call__(self, values: torch.Tensor, size: int) -> torch.Tensor:
        """Logits and probabilities, shape (2, HEAD_ROWS, HEAD_SLOTS), for the input :meth:`pack` built for a tree
        padded to ``size`` (``values``, a CPU tensor)."""
        if size not in self.graphs:
            return self._run(values.to(self.device), size)
        graph, host, packed, out = self.graphs[size]
        host.copy_(values)
        packed.copy_(host, non_blocking=True)
        graph.replay()
        return out.clone()


# ---------------------------------------------------------------- scoring calls


@dataclass
class TreeCall:
    """One model call's CPU work, from :meth:`TreeScorer.prepare`: its examples in order, their merged tree, and
    for the compiled graph the tree's padded size and packed input (None when the heads do not fit the graph)."""

    examples: list[Mapping[str, Any]]
    tree: TokenTree
    size: int = 0
    packed: Any = None


class TreeScorer:
    """Answers NanoJev payloads as ``DecisionPredictor.predict`` does, from one forward over their paths' tree.

    ``predictor`` is the loaded ``DecisionPredictor`` (its tokenizer, model and
    settings are used as they are) and ``upstream`` its module, whose
    ``validate_request``, ``prepare_examples`` and ``answer_from_probabilities``
    encode the request and check the answers. A response has the reference's
    fields; its ``execution`` record names the tree forward. :meth:`prepare` is
    CPU work and may run on any thread; :meth:`score` uses the GPU, one call at a
    time.
    """

    def __init__(self, predictor: Any, upstream: ModuleType, mode: str) -> None:
        import torch

        self._predictor, self._upstream, self.mode = predictor, upstream, mode
        self._graphs: GraphForward | None = None
        self._leaf_states: Callable[[TokenTree], Any]
        if mode == "tree":
            self._leaf_states = functools.partial(hf_leaf_states, predictor.model.backbone, predictor.device)
        elif mode == "tree-compiled":
            if predictor.precision != "bf16":
                raise ValueError("tree-compiled runs under bf16 autocast; use precision bf16, or inference tree")
            self._forward = torch.compile(FusedQwen3(predictor.model.backbone), dynamic=True)
            heads = functools.partial(graph_heads, predictor.model)
            # Most calls fit the sizes captured here, a path limit plus room for the other paths' tails; a larger
            # tree runs the compiled forward without a graph.
            self._graphs = GraphForward(self._forward, heads, predictor.device)
            self._graphs.precapture(range(BUCKET_TOKENS, padded_size(predictor.limit + 1024) + 1, BUCKET_TOKENS))
            self._leaf_states = self._compiled_leaf_states
        else:
            raise ValueError(f"Unknown tree inference mode {mode!r}")

    def _compiled_leaf_states(self, tree: TokenTree) -> torch.Tensor:
        """The compiled forward without a graph, for a call whose heads do not fit the graph's layout."""
        import torch

        size, device = padded_size(tree.size), self._predictor.device
        pad = [0] * (size - tree.size)
        tokens = torch.tensor([tree.tokens + pad], device=device)
        positions = torch.tensor([tree.depth + pad], device=device)
        hidden = self._forward(tokens, positions, torch.tensor(tree.subtree_end(size), device=device))
        return hidden[0, torch.tensor(tree.leaf, device=device)]

    def prepare(self, examples: Sequence[Mapping[str, Any]]) -> TreeCall:
        """The CPU work of scoring ``examples``, the questions of one or more requests, in one forward."""
        import torch

        tree = TokenTree([path for example in examples for path in example["leaf_tokens"]])
        call = TreeCall(list(examples), tree)
        rows = slot_layout(examples) if self._graphs is not None else None
        if rows is not None:
            call.size = padded_size(tree.size)
            choice = [example["type"] == "choice" for example in examples]
            call.packed = torch.tensor(GraphForward.pack(tree, rows, choice, call.size), dtype=torch.long)
        return call

    def score(self, call: TreeCall) -> list[list[float]]:
        """Each example's option probabilities, from one forward over the call's tree."""
        import torch

        precision = self._predictor.precision
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
            if call.packed is not None and self._graphs is not None:
                logits, probabilities = self._graphs(call.packed, call.size).cpu()
                return [
                    _row_probabilities(example, logits[row], probabilities[row])
                    for row, example in enumerate(call.examples)
                ]
            leaves = self._leaf_states(call.tree)
            out, offset = [], 0
            for example in call.examples:
                count = len(example["leaf_tokens"])
                logits = head_logits(self._predictor.model, [example], leaves[offset : offset + count])
                offset += count
                scores = logits[0, : len(example["candidate_ids"])].float()
                if not torch.isfinite(scores).all():
                    raise ValueError("NanoJev produced non-finite logits")
                out.append(scores.softmax(-1).cpu().tolist())
            return out

    def answer(
        self,
        states: Sequence[Mapping[str, Any]],
        examples: Sequence[Mapping[str, Any]],
        probabilities: Sequence[Sequence[float]],
        call: TreeCall,
        requests: int,
    ) -> dict[str, Any]:
        """The response ``DecisionPredictor.predict`` gives for a payload, from its validated ``states``, its
        ``examples`` and their ``probabilities``; ``call`` was the forward it shared with ``requests`` requests."""
        predictor, upstream = self._predictor, self._upstream
        predictor.inference_calls += 1
        outputs = {state["id"]: {"id": state["id"], "answers": {}} for state in states}
        for example, row in zip(examples, probabilities, strict=True):
            outputs[example["state_id"]]["answers"][example["qid"]] = upstream.answer_from_probabilities(
                example, list(row)
            )
        paths = [path for example in examples for path in example["leaf_tokens"]]
        run_config = predictor.run_config
        return {
            "schema_version": "openjev-toy-inference-v1",
            "checkpoint": {
                "directory": str(predictor.root),
                "base_model": run_config.get("model"),
                "base_revision": run_config.get("resolved_model_revision"),
                "set_head": run_config["set_head"],
            },
            "temperature": {
                "value": 1.0,
                "fitted_by_this_command": False,
                "note": "The given scalar is applied as is; the default of 1 does not mean the model is calibrated.",
            },
            "execution": {
                "device": str(predictor.device),
                "parameter_storage": "float32" if self.mode == "tree" else "float32, bf16 copies of matmul weights",
                "precision": predictor.precision,
                "forward_autocast": "bfloat16" if predictor.precision == "bf16" else "disabled",
                "states": len(states),
                "questions": len(examples),
                "candidate_paths": len(paths),
                "forward_passes": 1,
                "batch_questions_limit": "all",
                "autoregressive_decode_steps": 0,
                "prefix_sharing": True,
                "inference": self.mode,
                "heads_in_graph": call.packed is not None,
                "requests_in_forward": requests,
                "candidate_path_tokens": sum(map(len, paths)),
                "token_tree_tokens": call.tree.size,
                "max_length": predictor.limit,
                "disable_native_triton": predictor.disable_native_triton,
                "network_model_calls": 0,
                "persistent_model_load_count": 1,
                "inference_call_index": predictor.inference_calls,
            },
            "states": list(outputs.values()),
        }

    def predict(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """One payload, encoded by NanoJev's ``prepare_examples`` and answered from its own forward."""
        predictor, upstream = self._predictor, self._upstream
        states = upstream.validate_request(payload)
        examples = upstream.prepare_examples(payload, predictor.tokenizer, predictor.limit)
        call = self.prepare(examples)
        return self.answer(states, examples, self.score(call), call, 1)


def _row_probabilities(example: Mapping[str, Any], logits: torch.Tensor, probabilities: torch.Tensor) -> list[float]:
    """One example's probabilities from its row of the graph's output, as ``head_logits`` and a softmax give them."""
    import torch

    count = len(example["candidate_ids"])
    if example["type"] == "boolean":
        scores = torch.stack([logits[0] * 0, logits[0]])
        if not torch.isfinite(scores).all():
            raise ValueError("NanoJev produced non-finite logits")
        return scores.softmax(-1).tolist()
    if not torch.isfinite(logits[:count]).all():
        raise ValueError("NanoJev produced non-finite logits")
    return probabilities[:count].tolist()
