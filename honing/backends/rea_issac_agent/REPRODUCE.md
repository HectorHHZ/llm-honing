# Run the selected REA-Issac NanoJev agent

This is the selected **hybrid** baseline for issue #184: the agent harness in
this directory, `model-assisted-combat-v2`, and the trained NanoJev checkpoint
identified by `checkpoint_pin.json`. The game engine, public observations,
actions, and task definitions remain frozen by `agent_frozen.json`.

## Checkpoint location and contents

Download the selected checkpoint into `work/rea_issac_agent/checkpoint/` from
the repository root. `work/` is ignored by Git. Another location is fine:
pass it to `--checkpoint-dir` below. Publish or download these **six files**
with exactly this relative layout:

```text
checkpoint/
├── best.safetensors
├── config.json
├── backbone_config/config.json
└── tokenizer/
    ├── chat_template.jinja
    ├── tokenizer.json
    └── tokenizer_config.json
```

Their individual SHA-256 values are in `checkpoint_pin.json`; the selected
`best.safetensors` SHA-256 is
`a60db30eb906b46e89499e284a40f31f7e1ea1a790f4a2ac744253b9f90a1c51`.
The checkpoint directory is about 2.3 GB. Do not publish the original
training `checkpoint.json` or `report.json`: those historical manifests
contain machine-specific paths. The six files above are sufficient for
serving the selected weights.

The public download URL is pending upload by the checkpoint owner. Until it
is published, a new user cannot independently rerun the selected-weight
agent. The private historical evidence archives are also outside this PR.

## Verify and serve

Use a Python 3.10+ environment with this repository installed. The original
run used Python 3.12.13, PyTorch 2.7.0+cu126, Transformers 5.17.0,
Safetensors 0.8.0, Hugging Face Hub 1.32.0, Accelerate 1.14.0, and NumPy
1.26.4 on a CUDA GPU. Install a CUDA-enabled PyTorch build appropriate to
your machine and the repository's development dependencies.

Use the NanoJev source revision pinned in `integrations/ra2/pins.json`:

```sh
git clone https://github.com/TianyuCodings/NanoJev.git work/rea_issac_agent/NanoJev
git -C work/rea_issac_agent/NanoJev checkout 76fdfc9ecdca45a9bcef17991a07d3041a87685a
```

After placing the six checkpoint files in the default directory, verify all
file hashes and create a **local** serving manifest:

```sh
python -m honing.backends.rea_issac_agent.checkpoint
```

For a downloaded directory elsewhere, run
`python -m honing.backends.rea_issac_agent.checkpoint --checkpoint-dir PATH`.
Use `--manifest-out PATH` if you also want to choose where the local manifest
is written. By default it is `work/rea_issac_agent/checkpoint.local.json`,
next to the checkpoint directory. It contains the absolute path of **your**
download, is ignored by Git, and should not be committed. Hash mismatch,
missing files, or a different base model fails before serving.

Start the decision service on one available GPU, using that manifest:

```sh
python -m honing.backends.jev.server \
  --family nanojev \
  --pins integrations/ra2/pins.json \
  --nanojev-source work/rea_issac_agent/NanoJev \
  --checkpoint-manifest work/rea_issac_agent/checkpoint.local.json \
  --nanojev-inference tree \
  --max-len 8192 \
  --gpu 2 --port 8000
```

The `--gpu` value is an example; select a GPU you are authorized to use.
Keep the service running while you launch the preflight in another terminal:

```sh
python -m honing.backends.rea_issac_agent.model_preflight \
  --decision-server http://127.0.0.1:8000 \
  --out work/rea_issac_agent/eval-8001-8004 \
  --seeds 8001 8002 8003 8004 \
  --max-ticks 1800 --decision-frames 3 \
  --agent-controller model-assisted-combat-v2 \
  --expected-revision a60db30eb906b46e89499e284a40f31f7e1ea1a790f4a2ac744253b9f90a1c51
```

The runner checks the service's checkpoint identity, records the model and
paired rule games, and audits each saved game. It refuses an existing output
directory. The selected agent won all four of these fresh-seed games in the
original Linux run; these are small capability pilots, not a population
estimate. Exact state-hash replay has differed across Linux and macOS by
floating-point rounding in the frozen engine.

## Regenerate training inputs (optional)

The selected checkpoint was trained in two one-epoch decision-head SFT stages.
The deterministic demonstration data can be regenerated on CPU:

```sh
python -m honing.backends.rea_issac_agent.make_agent_demos \
  --out work/rea_issac_agent/demos-v1
python -m honing.backends.rea_issac_agent.make_agent_demos \
  --out work/rea_issac_agent/demos-v3 \
  --per-seed 60 --max-state-chars 12000 --selection balanced_rooms
```

The expected `decisions.jsonl` SHA-256 values are
`9ed213c9977f76e89b851ceaaeb8465ddbdf05e23045c9108be95fb5972c912e`
for v1 and
`4016873759763a7b70b0ae3daed7dc4ffeef401db3ae96fe53d25e49941d8dab`
for v3. Both hashes were reproduced on a separate CPU run. The public clean
NanoJev release is pinned in `integrations/ra2/pins.json`. To train rather
than use the selected weights, run `train_agent` on v1 with `--max-length
4096`, then run it on v3 with `--base-checkpoint` pointing at the v1
checkpoint **directory** containing `best.safetensors` and `--max-length 8192`. The commands take `--dataset`,
`--out`, `--nanojev-source`, and `--gpu`. Training needs a CUDA GPU, and a new
training run is not expected to reproduce the selected weight hash bit for
bit. The selected weights must be used for an exact baseline comparison.

The historical 40-episode fault-attribution collection and Codex Sol
diagnoses have not been run. This document only specifies how to reconstruct
and check the selected agent baseline.
