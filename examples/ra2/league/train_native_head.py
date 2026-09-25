# ruff: noqa: E402
# Standalone example imports follow its explicit sys.path bootstrap.
"""Bounded native gradient/reload diagnostic on approved TRAIN rows only.

This performs real updates to native decision-head weights. It does not optimize
on game development/heldout episodes, select a champion, or establish transfer
quality. The official predictor is compared before updates and after reload.
"""

import argparse
import contextlib
import hashlib
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from integration.models.choice_protocol import canonical_ids, nano_payload, omni_messages, token_digest
from integration.models.head_checkpoint import (
    apply_head_state,
    effective_weight_digest,
    freeze_body_and_select_head,
    head_module,
    head_state,
    load_head_artifact,
)
from integration.models.worker import digest, gpu_lock, load_native, read_json, verify_nano_source, verify_prepared


def checked_dataset(directory, family, runtime):
    manifest = read_json((directory / "manifest.json").read_text())
    verification = read_json((directory / "verification.json").read_text())
    if (
        manifest["kind"] != "exported-native-ra2-supervision"
        or manifest["family"] != family
        or not verification["verified"]
        or not verification["nativeInputSplitsDisjoint"]
        or not verification["classifierValidationFromTrainProtocolOnly"]
        or verification["manifestSha256"] != digest(directory / "manifest.json")
    ):
        raise ValueError("Requires verified native training-only supervision")
    for name, expected in manifest["outputSha256"].items():
        if digest(directory / name) != expected:
            raise ValueError("Native supervision changed")
    for key in ["weightSetSha256", "preparedManifestSha256", "preparedManifestPath"]:
        if runtime[key] != manifest["nativeRuntimeInputs"][key]:
            raise ValueError("Native data was prepared for another parent")
    rows = [read_json(line) for line in (directory / "train.jsonl").read_text().splitlines() if line.strip()]
    if len(rows) != manifest["trainingExamples"] or any(
        r["split"] != "train" or r["metadata"]["protocolSplit"] != "train" for r in rows
    ):
        raise ValueError("Invalid native training records")
    return manifest, rows


def select_rows(rows, count, seed):
    """Deterministic stage coverage, then a seeded sample; diagnostic only."""
    if count <= 0 or len(rows) < count:
        raise ValueError("Insufficient training rows for registered diagnostic")
    generator = random.Random(seed)
    shuffled = list(rows)
    generator.shuffle(shuffled)
    by_stage = {}
    for row in shuffled:
        by_stage.setdefault(row["metadata"]["stage"], row)
    chosen = [by_stage[key] for key in sorted(by_stage)][:count]
    seen = {r["id"] for r in chosen}
    chosen.extend(row for row in shuffled if row["id"] not in seen)
    return chosen[:count]


def prepare_row(adapter, family, row, max_length):
    key = next(iter(row["questions"]))
    question = row["questions"][key]
    expected = row["metadata"]["tokenAudit"][key]
    if family == "nanojev":
        example = adapter.prepare_examples(
            nano_payload(row["state"], row["questions"]), adapter.predictor.tokenizer, max_length
        )[0]
        if [token_digest(p) for p in example["leaf_tokens"]] != expected["candidatePathSha256"] or example[
            "candidate_ids"
        ] != list(question["criteria"]):
            raise ValueError("Training native candidate paths differ")
        return example
    ids = canonical_ids(
        adapter.tokenizer.apply_chat_template(
            omni_messages(row["state"], question), add_generation_prompt=True, tokenize=True, enable_thinking=False
        )
    )
    if len(ids) > max_length or token_digest(ids) != expected["promptSha256"]:
        raise ValueError("Training native numbered prompt differs")
    return ids


def forward_logits(adapter, family, row, encoded):
    import torch

    key = next(iter(row["questions"]))
    count = len(row["questions"][key]["criteria"])
    if family == "nanojev":
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = adapter.predictor.model([encoded], adapter.predictor.tokenizer.pad_token_id)
        return logits[0, :count].float()
    device = next(adapter.model.parameters()).device
    ids = torch.tensor([encoded], device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = adapter.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        hidden = output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0]
    # The pinned official Omni predictor runs Head256 in FP32 outside autocast.
    return adapter.head(hidden[:, -1].float(), torch.tensor([count], device=device))[0, :count].float()


def official_equivalence(adapter, family, row, encoded):
    import torch

    captured = []

    def hook(module, arguments, output):
        values = output[0] if family == "nanojev" else output
        captured.append(values.detach().clone())

    handle = head_module(adapter, family).register_forward_hook(hook)
    try:
        if family == "nanojev":
            adapter.predictor.predict(nano_payload(row["state"], row["questions"]), batch_questions=1, temperature=1.0)
        else:
            question = next(iter(row["questions"].values()))
            adapter.upstream_predict(
                adapter.model,
                adapter.head,
                adapter.tokenizer,
                row["state"],
                question["instructions"],
                list(question["criteria"].values()),
            )
    finally:
        handle.remove()
    if len(captured) != 1:
        raise ValueError("Official predictor did not perform exactly one complete-question forward")
    actual = forward_logits(adapter, family, row, encoded)
    expected = captured[0][0, : len(actual)].float()
    torch.testing.assert_close(actual.detach(), expected, rtol=1e-3, atol=1e-3)
    if not actual.requires_grad:
        raise ValueError("Training path has no native head gradient")
    return {
        "rowId": row["id"],
        "stage": row["metadata"]["stage"],
        "options": len(actual),
        "maxLogitAbsoluteDifference": float((actual.detach() - expected).abs().max()),
        "rtol": 1e-3,
        "atol": 1e-3,
    }


def tensor_set_digest(values):
    import torch

    hashed = hashlib.sha256()
    for name, tensor in sorted(values.items()):
        cpu = tensor.detach().cpu().contiguous()
        header = json.dumps([name, str(cpu.dtype), list(cpu.shape)], separators=(",", ":")).encode()
        hashed.update(len(header).to_bytes(8, "big"))
        hashed.update(header)
        hashed.update(cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
    return hashed.hexdigest()


def body_state(adapter, family):
    body = adapter.predictor.model.backbone if family == "nanojev" else adapter.model
    # Include nonpersistent runtime buffers; freezing parameters alone is insufficient.
    return dict(list(body.named_parameters()) + list(body.named_buffers()))


def train(args):
    import torch
    from safetensors.torch import save_file

    if args.out.exists():
        raise ValueError("Require a fresh output directory")
    pins = read_json((ROOT / "league/model-sources.json").read_text())
    parent, runtime = verify_prepared(args.prepared_manifest, args.family, pins)
    source = None
    if args.family == "nanojev":
        source, _ = verify_nano_source(args.source_root, pins)
    data, rows = checked_dataset(args.data_dir, args.family, runtime)
    selected = select_rows(rows, args.steps * args.batch_questions, args.seed)
    args.out.mkdir(parents=True)
    try:
        with gpu_lock(args.gpu) as uuid:
            torch.set_num_threads(4)
            torch.manual_seed(args.seed)
            with contextlib.redirect_stdout(sys.stderr):
                adapter = load_native(args.family, parent, source, data["maxLength"])
            parameters = freeze_body_and_select_head(adapter, args.family)
            before_head = {k: v.detach().cpu().clone() for k, v in head_state(adapter, args.family).items()}
            initial_head_sha = tensor_set_digest(before_head)
            body_before = tensor_set_digest(body_state(adapter, args.family))
            encoded = [prepare_row(adapter, args.family, row, data["maxLength"]) for row in selected]
            probes = list(range(min(8, len(selected))))
            equivalence = [official_equivalence(adapter, args.family, selected[i], encoded[i]) for i in probes]
            (args.out / "initial-equivalence.json").write_text(json.dumps(equivalence, indent=2) + "\n")
            optimizer = torch.optim.AdamW(list(parameters.values()), lr=args.head_lr, weight_decay=0.01)
            logs = []
            for step in range(args.steps):
                optimizer.zero_grad(set_to_none=True)
                total = 0.0
                start = step * args.batch_questions
                for index in range(start, start + args.batch_questions):
                    row = selected[index]
                    logits = forward_logits(adapter, args.family, row, encoded[index])
                    label = torch.tensor([row["metadata"]["goldIndex"]], device=logits.device)
                    loss = torch.nn.functional.cross_entropy(logits[None], label) / args.batch_questions
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite native head loss")
                    loss.backward()
                    total += float(loss.detach())
                gradient = float(
                    torch.nn.utils.clip_grad_norm_(list(parameters.values()), 1.0, error_if_nonfinite=True)
                )
                optimizer.step()
                record = {
                    "step": step + 1,
                    "loss": total,
                    "gradientNorm": gradient,
                    "questionIds": [row["id"] for row in selected[start : start + args.batch_questions]],
                }
                logs.append(record)
                print(json.dumps(record), flush=True)
            body_after = tensor_set_digest(body_state(adapter, args.family))
            if body_before != body_after:
                raise ValueError("Native backbone parameters or buffers changed")
            final_head = {
                k: v.detach().cpu().contiguous().clone() for k, v in head_state(adapter, args.family).items()
            }
            if tensor_set_digest(final_head) == initial_head_sha:
                raise ValueError("Native head did not update")
            for value in final_head.values():
                if not torch.isfinite(value).all():
                    raise ValueError("Nonfinite native head artifact")
            path = args.out / "head.safetensors"
            save_file(final_head, str(path))
            head_sha = digest(path)
            manifest = {
                "kind": "trained-native-decision-head",
                "family": args.family,
                "trainingScope": "native-head-only",
                "backboneUpdated": False,
                "parent": runtime,
                "weightFilename": "head.safetensors",
                "headSha256": head_sha,
                "effectiveWeightSetSha256": effective_weight_digest(runtime["weightSetSha256"], head_sha),
                "dataManifestPath": str(args.data_dir / "manifest.json"),
                "dataManifestSha256": digest(args.data_dir / "manifest.json"),
                "dataVerificationSha256": digest(args.data_dir / "verification.json"),
                "settings": {
                    "steps": args.steps,
                    "batchQuestions": args.batch_questions,
                    "headLearningRate": args.head_lr,
                    "weightDecay": 0.01,
                    "seed": args.seed,
                    "precision": "float32 parameters / bfloat16 backbone forward",
                    "mode": "eval; head gradients enabled; frozen backbone",
                },
                "sourceSha256": {
                    str(p.relative_to(ROOT)): digest(p)
                    for p in [
                        Path(__file__),
                        ROOT / "integration/models/head_checkpoint.py",
                        ROOT / "integration/models/worker.py",
                        ROOT / "integration/models/choice_protocol.py",
                    ]
                },
                "selectedTrainingRows": [row["id"] for row in selected],
                "initialHeadTensorSha256": initial_head_sha,
                "finalHeadTensorSha256": tensor_set_digest(final_head),
                "frozenBodyTensorSha256": body_before,
                "actualOptimizerSteps": len(logs),
                "gameplayEvaluated": False,
                "notMigrationQualification": True,
                "notChampionEvidence": True,
            }
            (args.out / "checkpoint.json").write_text(json.dumps(manifest, indent=2) + "\n")
            with torch.no_grad():
                final_logits = [forward_logits(adapter, args.family, selected[i], encoded[i]).cpu() for i in probes]
            probe_records = [
                {
                    "rowId": selected[i]["id"],
                    "logits": values.tolist(),
                    "goldIndex": selected[i]["metadata"]["goldIndex"],
                    "candidateIds": list(next(iter(selected[i]["questions"].values()))["criteria"]),
                }
                for i, values in zip(probes, final_logits, strict=True)
            ]
            (args.out / "final-probes.json").write_text(json.dumps(probe_records, indent=2) + "\n")
            manifest["finalProbesSha256"] = digest(args.out / "final-probes.json")
            (args.out / "checkpoint.json").write_text(json.dumps(manifest, indent=2) + "\n")
            apply_head_state(adapter, args.family, before_head)
            if tensor_set_digest(head_state(adapter, args.family)) != initial_head_sha:
                raise ValueError("Parent-head restore failed")
            load_head_artifact(adapter, args.family, args.out / "checkpoint.json", runtime)
            with torch.no_grad():
                for i, expected in zip(probes, final_logits, strict=True):
                    torch.testing.assert_close(
                        forward_logits(adapter, args.family, selected[i], encoded[i]).cpu(), expected, rtol=0, atol=0
                    )
            after_equivalence = [official_equivalence(adapter, args.family, selected[i], encoded[i]) for i in probes]
            if tensor_set_digest(body_state(adapter, args.family)) != body_before:
                raise ValueError("Reload modified native body")
            report = {
                "verified": True,
                "kind": "actual-native-head-gradient-and-reload-diagnostic",
                "family": args.family,
                "actualOptimizerSteps": len(logs),
                "selectedTrainingExamples": len(selected),
                "headChanged": True,
                "frozenBodyAndBuffersUnchanged": True,
                "officialDifferentiablePathVerifiedBeforeAndAfter": True,
                "reloadLogitsBitwiseEqual": True,
                "parentRestoredBeforeReload": True,
                "initialEquivalence": equivalence,
                "afterEquivalence": after_equivalence,
                "checkpointSha256": digest(args.out / "checkpoint.json"),
                "headSha256": head_sha,
                "gpuUuid": uuid,
                "learningPerformed": True,
                "learningScope": "native decision head; technical diagnostic only",
                "notGameplayEvidence": True,
                "notMigrationQualification": True,
                "notChampionEvidence": True,
            }
            (args.out / "training-log.json").write_text(json.dumps(logs, indent=2) + "\n")
            (args.out / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
            print(
                json.dumps({k: v for k, v in report.items() if k not in ["initialEquivalence", "afterEquivalence"]}),
                flush=True,
            )
    except BaseException as error:
        (args.out / "error.json").write_text(json.dumps({"error": str(error)}, indent=2) + "\n")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=["nanojev", "jev-omni"], required=True)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--batch-questions", type=int, default=4)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.steps <= 0 or args.batch_questions <= 0 or not 0 < args.head_lr < 1:
        parser.error("Invalid bounded training settings")
    if args.family == "nanojev" and args.source_root is None:
        parser.error("NanoJev requires its pinned source")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false", USE_TF="0")
    train(args)
