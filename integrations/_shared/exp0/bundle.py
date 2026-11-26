"""Build or check the local Exp 0 v0 release bundle ``<release_root>/<game-id>/v0/`` (Griddly v0 ``bundle.py``).

  python3 bundle.py prepare   ROOT AUDIT OUT [--source NAME=PATH ...] [--exclude-extra]
  python3 bundle.py evaluated ROOT AUDIT OUT --cards DIR [--source ...] [--variant NAME=DIR ...]
  python3 bundle.py release   ROOT AUDIT OUT --cards DIR --code-revision SHA40 [--checkpoint-dir ROLE=PATH]
                              [--artifact-uri ROLE=URI] [--source ...] [--variant NAME=DIR ...]
  python3 bundle.py verify    OUT

``--variant NAME=DIR`` (added 2026-11-26): a pre-registered extra diagnosis run reported separately from the two
primary modes, e.g. ``probe-variant=ROOT/diagnosis-probe-variant`` (``diagnose.py --mode probe --prompt
diagnosis_prompt_probe_variant.txt``). It must use the primary runs' model, CLI, schema, command and inputs, and a
prompt file of this directory. Its settings, replies and attempt summary go to ``evaluation/NAME/``, its prompt next
to the primary prompt, its scores (``score.py`` on the ``read`` replies + its replies, so its "probe" numbers are
the variant's) to ``evaluation/scores-NAME.json``, and the list to ``evaluation/variants.json``; the verifier
recomputes them. Without the option the bundle is exactly as before.

ROOT is the collection root (attempt directories, ``collection.json``, and the evaluation outputs
``read-inputs.jsonl``, ``probe-reference.jsonl``, ``probe-alternate.jsonl``, ``probes/``, ``diagnosis-read/``,
``diagnosis-probe/``, optional ``weights-reask.jsonl``); AUDIT holds ``episodes.json`` and ``attempts.jsonl``.
OUT must end with the plan's ``dataset_dir`` (``<game-id>/v0``) and must not exist yet.

``prepare`` copies the plan, every attempt (kept or not) with its raw evidence into ``evaluator/``, the
game's collector sources (``--source``) and the common tools' sources, writes ``dataset.json`` and the
per-file ``sha256.json``. ``evaluated`` adds the frozen diagnosis inputs, probe captures and identity
records, both modes' settings, replies and raw attempts, ``scores.json`` (recomputed here from the saved
replies) and the game's cards. ``release`` adds the publication metadata and the portable verifier. Local
fields (``path``, ``gpuUuid``) are removed from the checkpoint records; any other absolute local path left in
the bundle is listed in ``dataset.json`` and blocks ``release``. Nothing is uploaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from contract import read_json, read_rows, write_json
from release_metadata import LOCAL_FIELDS, add_release_metadata
from score import MODES, score

HERE = Path(__file__).resolve().parent
RUN_FILES = ("manifest.json", "decisions.jsonl", "public_states.jsonl", "result.json", "error.json")
CODE_SUFFIXES = {
    ".py",
    ".json",
    ".txt",
    ".md",
    ".cs",
    ".js",
    ".mjs",
    ".yaml",
    ".yml",
    ".toml",
    ".sh",
    ".cfg",
    ".ini",
    ".ml",
    ".mli",
    ".c",
    ".h",
    ".cpp",
    ".lua",
    ".jsonl",
}
COMMON_FILES = (
    "contract.py",
    "schema_lite.py",
    "validate_episode.py",
    "adapter.py",
    "audit_set.py",
    "upstream.py",
    "diagnosis_inputs.py",
    "probes.py",
    "diagnose.py",
    "score.py",
    "bundle.py",
    "release_metadata.py",
    "verify_release.py",
    "diagnosis_prompt.txt",
    "diagnosis_schema.json",
    "CONTRACT.md",
    "vendor/probe_a151596.py",
)
CARDS = {
    "PROTOCOL.md": "PROTOCOL.md",
    "RUNBOOK.md": "RUNBOOK.md",
    "RESULTS.md": "RESULTS.md",
    "DATASET.md": "README.md",
    "LABELS.md": "LABELS.md",
    "TODO.md": "TODO.md",
}
# A release holds structured text/JSON evidence and code only: never game binaries, assets, frames,
# screenshots or videos (user decision relayed 2026-11-23 11:35). Everything else is skipped or refused.
TEXT_SUFFIXES = {".json", ".jsonl", ".md", ".txt", ".raw", ".csv", ".log", *CODE_SUFFIXES}
SECRET = re.compile(r"\b(hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{30,})\b")
LOCAL_PATH = re.compile(r"(?<![\w.])(/data/[A-Za-z0-9_]+/|/home/[A-Za-z0-9_]+/|/Users/[A-Za-z0-9_]+/|[A-Z]:\\Users\\)")
PLACEHOLDER = re.compile(r"\{\{[A-Z0-9_]+\}\}")


def expected_episodes(plan: dict[str, Any]) -> int:
    """40 for a v0 release (10 per condition). A plan marked ``dry_run`` (contract 0.2.0) may hold fewer per
    condition to test the whole chain on pilot / smoke data; such a bundle says so and is never uploaded."""
    if plan["per_condition"] != 10 and not plan.get("dry_run"):
        raise ValueError("a v0 release has 10 episodes per condition; only a dry-run plan may hold fewer")
    return 4 * plan["per_condition"]


def file_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != root / "sha256.json"
    }


def text_problem(path: Path) -> str | None:
    """Why a file may not be in a release (not a text/JSON file, not UTF-8, binary content), or ``None``."""
    if path.suffix not in TEXT_SUFFIXES:
        return f"suffix {path.suffix or '(none)'} is not a text/JSON evidence or code type"
    data = path.read_bytes()
    if b"\0" in data:
        return "contains NUL bytes (binary)"
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return "is not UTF-8 text"
    if path.suffix == ".raw":
        try:
            json.loads(text)
        except json.JSONDecodeError:
            return "a raw decision reply that is not JSON"
    if SECRET.search(text):
        return "contains a credential-like string"
    return None


def content_problems(root: Path) -> list[str]:
    return [
        f"{path.relative_to(root).as_posix()}: {problem}"
        for path in sorted(root.rglob("*"))
        if path.is_file() and (problem := text_problem(path))
    ]


def _copy_text_tree(source: Path, target: Path) -> list[dict[str, Any]]:
    """Copy only text/JSON files (``extra/``); return what was left out, with sizes, for the dataset report."""
    skipped = []
    for path in sorted(p for p in source.rglob("*") if p.is_file()):
        problem = text_problem(path)
        if problem:
            skipped.append({"file": path.relative_to(source).as_posix(), "bytes": path.stat().st_size, "why": problem})
            continue
        destination = target / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    return skipped


def _strip_checkpoint(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key not in LOCAL_FIELDS}


def _copy_tree(source: Path, target: Path, suffixes: set[str] | None) -> list[str]:
    skipped = []
    files = [source] if source.is_file() else sorted(p for p in source.rglob("*") if p.is_file())
    for path in files:
        relative = path.name if source.is_file() else path.relative_to(source).as_posix()
        if "__pycache__" in relative or "/.git/" in f"/{relative}" or relative.startswith(".git/"):
            continue
        if suffixes is not None and path.suffix not in suffixes:
            skipped.append(relative)
            continue
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    return skipped


def _model_input_tokens(run: Path, rows: list[dict[str, Any]]) -> int:
    total = 0
    for row in rows:
        if row["kind"] != "model" or not row["response"]["file"]:
            continue
        reply = json.loads((run / row["response"]["file"]).read_bytes())
        usage = reply.get("usage") or (reply.get("modelResult") or {}).get("usage") or {}
        if isinstance(usage.get("input_tokens"), int):
            total += usage["input_tokens"]
        else:
            total += sum(entry.get("inputTokens") or 0 for entry in (reply.get("tokenAudit") or {}).values())
    return total


def local_path_findings(root: Path) -> list[str]:
    found = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in {".json", ".jsonl", ".md", ".txt", ".raw", ".py"}:
            continue
        if (
            path.name in ("dataset.json", "sha256.json")
            or "common-source" in path.parts
            or "collector-source" in path.parts
        ):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        match = LOCAL_PATH.search(text)
        if match:
            found.append(f"{path.relative_to(root).as_posix()}: {match.group(0)}")
    return found


def bundle(root: Path, audit: Path, out: Path, sources: dict[str, Path], exclude_extra: bool) -> None:
    plan = read_json(root / "collection.json")
    if not out.as_posix().rstrip("/").endswith(plan["dataset_dir"].rstrip("/")):
        raise ValueError(f"the bundle directory must end with {plan['dataset_dir']}")
    selected = read_json(audit / "episodes.json")
    attempts = read_rows(audit / "attempts.jsonl")
    expected = expected_episodes(plan)
    if len(selected) != expected or len({row["episode_id"] for row in selected}) != expected:
        raise ValueError(f"the bundle requires the {expected} scored episodes")
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(root / "collection.json", out / "collection.json")
    skipped: dict[str, list[str]] = {}
    for name, path in sources.items():
        skipped[name] = _copy_tree(path, out / "collector-source" / name, CODE_SUFFIXES)
    for name in COMMON_FILES:
        target = out / "common-source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(HERE / name, target)
    _copy_tree(HERE / "schemas", out / "common-source" / "schemas", {".json"})
    evaluator = out / "evaluator"
    evaluator.mkdir()
    shutil.copyfile(audit / "episodes.json", evaluator / "episodes.json")
    shutil.copyfile(audit / "attempts.jsonl", evaluator / "attempts.jsonl")
    if (root / "weights-reask.jsonl").is_file():
        shutil.copyfile(root / "weights-reask.jsonl", evaluator / "weights-reask.jsonl")
    totals = Counter()
    extra_skipped: dict[str, list[dict[str, Any]]] = {}
    for item in attempts:
        run = (root / item["run"]).resolve()
        if run.parent != root.resolve():
            raise ValueError("an attempt path leaves the collection root")
        target = evaluator / run.name
        target.mkdir()
        for name in RUN_FILES:
            if (run / name).is_file():
                shutil.copyfile(run / name, target / name)
        if (target / "manifest.json").is_file():
            manifest = read_json(target / "manifest.json")
            manifest["pins"]["checkpoint"] = _strip_checkpoint(manifest["pins"]["checkpoint"])
            write_json(target / "manifest.json", manifest)
        if (target / "result.json").is_file():
            result = read_json(target / "result.json")
            result["checkpoint_end"] = _strip_checkpoint(result["checkpoint_end"])
            write_json(target / "result.json", result)
        if (run / "responses").is_dir():
            shutil.copytree(run / "responses", target / "responses")
        if (run / "extra").is_dir() and not exclude_extra:
            left_out = _copy_text_tree(run / "extra", target / "extra")
            if left_out:
                extra_skipped[run.name] = left_out
        if item["selected_for_scoring"]:
            rows = read_rows(run / "decisions.jsonl")
            totals["rows"] += len(rows)
            totals["model_decisions"] += sum(row["kind"] == "model" for row in rows)
            totals["model_questions"] += sum(
                len((row["request"] or {}).get("questions") or {}) for row in rows if row["kind"] == "model"
            )
            totals["native_windows"] += sum(len((row["executed"] or {}).get("windows", [])) for row in rows)
            totals["decisions_bytes"] += (run / "decisions.jsonl").stat().st_size
            totals["model_server_input_tokens"] += _model_input_tokens(run, rows)
    report = {
        "schema": "honing-exp0-dataset-v0",
        "game_id": plan["game_id"],
        "dry_run": bool(plan.get("dry_run")),
        "episode_count": len(selected),
        "conditions": dict(Counter(row["condition"] for row in selected)),
        "outcomes": {
            condition: dict(
                Counter(row["native_outcome"]["status"] for row in selected if row["condition"] == condition)
            )
            for condition in ("no_injection", "harness", "weights", "controller")
        },
        "scenario_reset_groups": len({row["scenario_group"] for row in selected}),
        "attempts": len(attempts),
        "excluded_attempts": sum(row["annotation_status"] == "excluded" for row in attempts),
        "scored_episode_ids": [row["episode_id"] for row in selected],
        "scored_totals": dict(totals),
        "extra_included": "text/JSON files only" if not exclude_extra else False,
        "extra_files_skipped": extra_skipped,
        "collector_sources": sorted(sources),
        "collector_source_files_skipped": skipped,
    }
    report["local_path_findings"] = local_path_findings(out)
    write_json(out / "dataset.json", report)
    write_json(out / "sha256.json", file_hashes(out))


def _settings(root: Path) -> list[dict[str, Any]]:
    settings = [read_json(root / f"diagnosis-{mode}" / "settings.json") for mode in MODES]
    for pin in ("model", "backend", "cli_version", "inputs_sha256", "prompt_sha256", "schema_sha256", "command"):
        if settings[0][pin] != settings[1][pin]:
            raise ValueError(f"the diagnosis modes used different {pin}")
    if [item["mode"] for item in settings] != list(MODES) or settings[0]["probe_inputs_sha256"] is not None:
        raise ValueError("the diagnosis settings do not match the read and probe mode boundaries")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()  # noqa: E731
    for path, recorded in (
        (root / "read-inputs.jsonl", settings[0]["inputs_sha256"]),
        (root / "probes" / "probe-inputs.jsonl", settings[1]["probe_inputs_sha256"]),
        (HERE / "diagnosis_prompt.txt", settings[0]["prompt_sha256"]),
        (HERE / "diagnosis_schema.json", settings[0]["schema_sha256"]),
    ):
        if digest(path) != recorded:
            raise ValueError(f"{path.name} differs from the frozen diagnosis input")
    return settings


def _attempt_summary(run: Path) -> list[dict[str, Any]]:
    recorded = []
    for attempt in sorted(run.glob("*-attempt-*.json")):
        data = read_json(attempt)
        messages, usage = [], None
        for line in (data.get("stdout") or "").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "item.completed" and event.get("item", {}).get("type") == "agent_message":
                messages.append(event["item"]["text"])
            if event.get("type") == "turn.completed":
                usage = event.get("usage")
        recorded.append(
            {
                "episode_id": attempt.name.split("-", 1)[0],
                "file": attempt.name,
                "returncode": data.get("returncode"),
                "status": data.get("status", "exited"),
                "wall_seconds": data["wall_seconds"],
                "raw_model_replies": messages,
                "usage": usage,
            }
        )
    return recorded


def _variant(root: Path, primary: list[dict[str, Any]], name: str, directory: Path) -> dict[str, Any]:
    """Check one pre-registered variant run against the primary runs' frozen settings (see the module docstring)."""
    if not re.fullmatch(r"[a-z][a-z0-9-]{1,40}", name) or name in MODES:
        raise ValueError(f"invalid variant name {name!r}")
    settings = read_json(directory / "settings.json")
    if settings["mode"] not in MODES:
        raise ValueError(f"variant {name} has an unknown mode")
    same_mode = primary[MODES.index(settings["mode"])]
    for pin in ("model", "backend", "cli_version", "inputs_sha256", "probe_inputs_sha256", "schema_sha256", "command"):
        if settings[pin] != same_mode[pin]:
            raise ValueError(f"variant {name} used another {pin} than the primary {settings['mode']} run")
    prompt = HERE / settings.get("prompt_file", "diagnosis_prompt.txt")
    if (
        prompt.parent != HERE
        or not prompt.is_file()
        or hashlib.sha256(prompt.read_bytes()).hexdigest() != settings["prompt_sha256"]
    ):
        raise ValueError(f"variant {name}: its prompt is not a frozen prompt file of the common tools")
    return {
        "name": name,
        "mode": settings["mode"],
        "prompt_file": prompt.name,
        "prompt_sha256": settings["prompt_sha256"],
        "diagnoses": f"evaluation/{name}/diagnoses.jsonl",
        "scores": f"evaluation/scores-{name}.json",
        "scored_with": "score.py on evaluation/read/diagnoses.jsonl + this variant's replies (its 'probe' block "
        "and the paired difference are the variant's)",
    }


def evaluated_bundle(
    root: Path,
    audit: Path,
    out: Path,
    cards: Path,
    sources: dict[str, Path],
    exclude_extra: bool,
    *,
    require_filled: bool = False,
    variants: dict[str, Path] | None = None,
) -> None:
    index = read_json(audit / "episodes.json")
    primary = _settings(root)
    declared = [_variant(root, primary, name, directory) for name, directory in sorted((variants or {}).items())]
    diagnoses = [row for mode in MODES for row in read_rows(root / f"diagnosis-{mode}" / "diagnoses.jsonl")]
    plan = read_json(root / "collection.json")
    scores = score(index, diagnoses, expected_episodes(plan) // 4 if plan.get("dry_run") else None)
    bundle(root, audit, out, sources, exclude_extra)
    unfilled = []
    for name, target in CARDS.items():
        if (cards / name).is_file():
            text = (cards / name).read_text(encoding="utf-8")
            if PLACEHOLDER.search(text):
                unfilled.append(name)
            shutil.copyfile(cards / name, out / target)
    if require_filled and unfilled:
        shutil.rmtree(out)
        raise ValueError(f"cards with unfilled placeholders: {unfilled}")
    for name in ("diagnosis_prompt.txt", "diagnosis_schema.json"):
        shutil.copyfile(HERE / name, out / name)
    evaluation = out / "evaluation"
    evaluation.mkdir()
    for name in ("read-inputs.jsonl", "probe-reference.jsonl", "probe-alternate.jsonl"):
        shutil.copyfile(root / name, evaluation / name)
    for name in ("identity.jsonl", "probe-inputs.jsonl"):
        shutil.copyfile(root / "probes" / name, evaluation / name)
    for mode in MODES:
        source, target = root / f"diagnosis-{mode}", evaluation / mode
        target.mkdir()
        shutil.copyfile(source / "settings.json", target / "settings.json")
        shutil.copyfile(source / "diagnoses.jsonl", target / "diagnoses.jsonl")
        # As Griddly: the summary keeps each attempt's raw model reply text, usage, status and timing; the full
        # CLI stdout/stderr files stay in the local collection (stderr can hold local paths).
        write_json(target / "attempts.json", _attempt_summary(source))
    write_json(evaluation / "scores.json", scores)
    if declared:
        read_replies = read_rows(root / "diagnosis-read" / "diagnoses.jsonl")
        for item in declared:
            source, target = variants[item["name"]], evaluation / item["name"]
            target.mkdir()
            shutil.copyfile(source / "settings.json", target / "settings.json")
            shutil.copyfile(source / "diagnoses.jsonl", target / "diagnoses.jsonl")
            write_json(target / "attempts.json", _attempt_summary(source))
            if not (out / item["prompt_file"]).is_file():
                shutil.copyfile(HERE / item["prompt_file"], out / item["prompt_file"])
            variant_rows = read_rows(source / "diagnoses.jsonl")
            write_json(
                evaluation / f"scores-{item['name']}.json",
                score(
                    index, read_replies + variant_rows, expected_episodes(plan) // 4 if plan.get("dry_run") else None
                ),
            )
        write_json(evaluation / "variants.json", declared)
    report = read_json(out / "dataset.json")
    report["unfilled_cards"] = unfilled
    report["local_path_findings"] = local_path_findings(out)
    write_json(out / "dataset.json", report)
    write_json(out / "sha256.json", file_hashes(out))


def release_bundle(
    root: Path,
    audit: Path,
    out: Path,
    cards: Path,
    sources: dict[str, Path],
    exclude_extra: bool,
    code_revision: str,
    checkpoint_dirs: dict[str, Path | None],
    uris: dict[str, str | None],
    variants: dict[str, Path] | None = None,
) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", code_revision):
        raise ValueError("the release code revision must be a full Git commit id")
    evaluated_bundle(root, audit, out, cards, sources, exclude_extra, require_filled=True, variants=variants)
    index = read_json(audit / "episodes.json")
    add_release_metadata(out, index, code_revision, checkpoint_dirs, uris)
    if (cards / "RELEASE_RECEIPT.md").is_file():
        shutil.copyfile(cards / "RELEASE_RECEIPT.md", out / "RELEASE_RECEIPT.md")
    _copy_tree(HERE / "schemas", out / "schemas", {".json"})
    tools = out / "tools"
    tools.mkdir()
    for name in ("verify_release.py", "score.py", "schema_lite.py"):
        shutil.copyfile(HERE / name, tools / name)
    findings = local_path_findings(out)
    problems = content_problems(out)
    if findings or problems:
        shutil.rmtree(out)
        raise ValueError(f"the release would hold local paths {findings[:5]} or non-text content {problems[:5]}")
    write_json(out / "sha256.json", file_hashes(out))


def _pairs(values: list[str], kind: type) -> dict[str, Any]:
    pairs = {}
    for value in values or ():
        name, _, item = value.partition("=")
        if not name or not item:
            raise ValueError(f"expected NAME=VALUE, got {value!r}")
        pairs[name] = kind(item)
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "evaluated", "release"):
        build = commands.add_parser(name)
        build.add_argument("root", type=Path)
        build.add_argument("audit", type=Path)
        build.add_argument("out", type=Path)
        build.add_argument("--source", action="append", default=[], help="NAME=PATH of collector code to snapshot")
        build.add_argument("--exclude-extra", action="store_true")
        if name != "prepare":
            build.add_argument("--cards", type=Path, required=True)
            build.add_argument(
                "--variant",
                action="append",
                default=[],
                help="NAME=DIR of a pre-registered extra diagnosis run (reported separately)",
            )
        if name == "release":
            build.add_argument("--code-revision", required=True)
            build.add_argument("--checkpoint-dir", action="append", default=[], help="ROLE=PATH (clean|alternate)")
            build.add_argument("--artifact-uri", action="append", default=[], help="ROLE=URI (clean|alternate)")
    commands.add_parser("verify").add_argument("out", type=Path)
    args = parser.parse_args()
    if args.command != "verify":
        sources = _pairs(args.source, Path)
        if args.command == "prepare":
            bundle(args.root, args.audit, args.out, sources, args.exclude_extra)
        elif args.command == "evaluated":
            evaluated_bundle(
                args.root,
                args.audit,
                args.out,
                args.cards,
                sources,
                args.exclude_extra,
                variants=_pairs(args.variant, Path),
            )
        else:
            release_bundle(
                args.root,
                args.audit,
                args.out,
                args.cards,
                sources,
                args.exclude_extra,
                args.code_revision,
                _pairs(args.checkpoint_dir, Path),
                _pairs(args.artifact_uri, str),
                variants=_pairs(args.variant, Path),
            )
    expected = read_json(args.out / "sha256.json")
    if expected != file_hashes(args.out):
        raise ValueError("local release file checksums or file set differ")
    print(
        json.dumps(
            {
                "verified_files": len(expected),
                **{
                    k: v
                    for k, v in read_json(args.out / "dataset.json").items()
                    if k
                    in ("episode_count", "attempts", "scenario_reset_groups", "local_path_findings", "unfilled_cards")
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
