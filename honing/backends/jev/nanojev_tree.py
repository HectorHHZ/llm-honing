"""Score NanoJev's candidate paths as one token tree, in one forward pass per request.

NanoJev scores each option of a question on its own path: the state, the
question, then that candidate and an end token. ``DecisionPredictor`` pads the
paths into rows and runs every row, so the state and the question, which all
paths of a question share, pass through the backbone once per option. A request
with nine options of about 1,250 tokens each runs about 11,000 tokens to score
about 1,700 distinct ones.

``TokenTree`` merges the paths of a request into a trie instead. Each node is
one token; it attends to itself and its ancestors only, and its position id is
its depth. A path's last node therefore sees the same tokens at the same
positions in the same causal order as when its path runs alone, so its hidden
state is the same function of the weights; only the order of floating-point
sums differs. Paths of different questions about one state share the state's
nodes too. The scoring heads then run per question as ``DecisionModel.forward``
runs them.

``NanoJevSettings.inference`` picks the forward over the tree:

- ``tree``: the checkpoint's own Hugging Face backbone, given the tree as a 4D
  attention mask and depth position ids.
- ``tree-compiled``: a Qwen3 forward written out with the dtypes the Hugging
  Face model uses under bf16 autocast, the q/k/v and gate/up projections each as
  one matmul, block-sparse FlexAttention over the tree mask, ``torch.compile``,
  and one CUDA graph per tree size rounded up to ``BUCKET_TOKENS``. Loading
  compiles the forward and captures the graphs, which takes minutes the first
  time a compile cache is empty.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Mapping, Sequence
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
# Graph leaf slots; a request scoring more paths than this runs the compiled forward without a graph.
GRAPH_LEAVES = 256


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


class GraphForward:
    """Leaf hidden states of a token tree, replayed from a CUDA graph captured for the tree's padded size.

    Each call sends one packed int64 buffer (token ids, positions, subtree ends,
    leaf indices) to the GPU in one copy, and the graph rebuilds the mask from
    the subtree ends. Graphs are captured on first use of a size, or ahead of
    time by :meth:`precapture`; trees larger than ``largest`` run the compiled
    forward without a graph. Calls must come from one thread at a time.
    """

    def __init__(self, forward: Callable[..., Any], device: Any, largest: int) -> None:
        self.forward, self.device, self.largest = forward, device, largest
        self.graphs: dict[int, tuple[Any, Any, Any, Any]] = {}
        self.pool: Any = None

    def _run(self, packed: torch.Tensor, size: int, leaves: int) -> torch.Tensor:
        tokens, positions, end, leaf = packed.split((size, size, size, leaves))
        # A fresh tensor rather than a view into ``packed``, as the mask_mod reading it was measured with.
        return self.forward(tokens[None], positions[None], end.clone())[0, leaf]

    def _pack(self, tree: TokenTree, size: int, leaves: int, host: torch.Tensor) -> torch.Tensor:
        pad = size - tree.size
        values = tree.tokens + [0] * pad + tree.depth + [0] * pad + tree.subtree_end(size)
        values += tree.leaf + [0] * (leaves - len(tree.leaf))
        host.copy_(host.new_tensor(values))
        return host

    def capture(self, size: int) -> None:
        import torch

        host = torch.zeros(3 * size + GRAPH_LEAVES, dtype=torch.long).pin_memory()
        packed = self._pack(TokenTree([[0]]), size, GRAPH_LEAVES, host.clone()).to(self.device)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self._run(packed, size, GRAPH_LEAVES)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool):
            out = self._run(packed, size, GRAPH_LEAVES)
        self.pool = graph.pool()
        self.graphs[size] = (graph, host, packed, out)

    def precapture(self, sizes: Sequence[int]) -> None:
        import torch

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for size in sizes:
                self.capture(size)

    def __call__(self, tree: TokenTree) -> torch.Tensor:
        import torch

        size = padded_size(tree.size)
        if size > self.largest or len(tree.leaf) > GRAPH_LEAVES:
            leaves = len(tree.leaf)
            host = torch.zeros(3 * size + leaves, dtype=torch.long)
            return self._run(self._pack(tree, size, leaves, host).to(self.device), size, leaves)
        if size not in self.graphs:
            self.capture(size)
        graph, host, packed, out = self.graphs[size]
        packed.copy_(self._pack(tree, size, GRAPH_LEAVES, host), non_blocking=True)
        graph.replay()
        return out[: len(tree.leaf)].clone()


# ---------------------------------------------------------------- scoring requests


class TreeScorer:
    """Answers a NanoJev payload as ``DecisionPredictor.predict`` does, from one forward over its paths' tree.

    ``predictor`` is the loaded ``DecisionPredictor`` (its tokenizer, model and
    settings are used as they are) and ``upstream`` its module, whose
    ``validate_request``, ``prepare_examples`` and ``answer_from_probabilities``
    encode the request and check the answers. The response has the reference's
    fields; its ``execution`` record names the tree forward.
    """

    def __init__(self, predictor: Any, upstream: ModuleType, mode: str) -> None:
        import torch

        self._predictor, self._upstream, self.mode = predictor, upstream, mode
        self._leaf_states: Callable[[TokenTree], Any]
        if mode == "tree":
            self._leaf_states = functools.partial(hf_leaf_states, predictor.model.backbone, predictor.device)
        elif mode == "tree-compiled":
            if predictor.precision != "bf16":
                raise ValueError("tree-compiled runs under bf16 autocast; use precision bf16, or inference tree")
            forward = torch.compile(FusedQwen3(predictor.model.backbone), dynamic=True)
            # Graphs for trees up to four times the path limit are captured on first use; most requests fit the
            # sizes captured here, a path limit plus room for the other paths' tails.
            graphs = GraphForward(forward, predictor.device, largest=padded_size(4 * predictor.limit))
            graphs.precapture(range(BUCKET_TOKENS, padded_size(predictor.limit + 1024) + 1, BUCKET_TOKENS))
            self._leaf_states = graphs
        else:
            raise ValueError(f"Unknown tree inference mode {mode!r}")

    def predict(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        import torch

        predictor, upstream = self._predictor, self._upstream
        states = upstream.validate_request(payload)
        examples = upstream.prepare_examples(payload, predictor.tokenizer, predictor.limit)
        paths = [path for example in examples for path in example["leaf_tokens"]]
        tree = TokenTree(paths)
        predictor.inference_calls += 1
        outputs = {state["id"]: {"id": state["id"], "answers": {}} for state in states}
        with (
            torch.inference_mode(),
            torch.autocast("cuda", dtype=torch.bfloat16, enabled=predictor.precision == "bf16"),
        ):
            leaves = self._leaf_states(tree)
            offset = 0
            for example in examples:
                count = len(example["leaf_tokens"])
                logits = head_logits(predictor.model, [example], leaves[offset : offset + count])
                offset += count
                scores = logits[0, : len(example["candidate_ids"])].float()
                if not torch.isfinite(scores).all():
                    raise ValueError("NanoJev produced non-finite logits")
                probabilities = scores.softmax(-1).cpu().tolist()
                outputs[example["state_id"]]["answers"][example["qid"]] = upstream.answer_from_probabilities(
                    example, probabilities
                )
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
                "candidate_path_tokens": sum(map(len, paths)),
                "token_tree_tokens": tree.size,
                "max_length": predictor.limit,
                "disable_native_triton": predictor.disable_native_triton,
                "network_model_calls": 0,
                "persistent_model_load_count": 1,
                "inference_call_index": predictor.inference_calls,
            },
            "states": list(outputs.values()),
        }
