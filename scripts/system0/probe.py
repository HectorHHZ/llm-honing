"""Probe a decision model on logged tank decisions: ask again with the text edited or under other weights.

    python scripts/system0/probe.py RUN... --server URL [--policy-checksum SHA]

With no variant it runs the identity probe: every logged request asked again
unchanged, which must reproduce the logged probabilities. A variant is one of:

    --strip TEXT ...                   remove TEXT from every rendered state
    --replace OLD NEW ...              replace OLD with NEW in every rendered state
    --in-questions                     apply --strip and --replace to the question and option text too
    --weights-server URL --weights-checksum SHA
                                       answer the logged requests with other weights

RUN is a tank run directory written by ``integrations/tank-war/scripts/match.mjs``.
``--policy-checksum`` names the weights that played the runs, which the
server's ``weightsSha256`` must match. ``--question`` keeps only the decisions
that ask that question. ``--target QUESTION=OPTION`` probes that option;
without it, each question's most probable logged option is the target.

Prints one JSON summary. ``--rows FILE`` also writes every probed question as a
JSON line. Exits 1 when a check fails: a server holding other weights than it
should, or an identity probe that does not reproduce the logged probabilities.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from honing.backends.tank.traces import load_run
from honing.core import ContractError, DecisionTrace
from honing.harness.probe import (
    DEFAULT_TIE,
    DEFAULT_TOLERANCE,
    IDENTITY,
    DecideClient,
    IdentityError,
    TextEdit,
    TextVariant,
    Variant,
    WeightsVariant,
    probe,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path, help="tank run directories")
    parser.add_argument("--server", required=True, help="the decision server holding the weights that played the runs")
    parser.add_argument("--policy-checksum", help="the checksum of those weights, when the runs do not carry it")
    parser.add_argument("--name", help="a name for the variant in the summary")
    parser.add_argument("--strip", action="append", default=[], metavar="TEXT")
    parser.add_argument("--replace", action="append", nargs=2, default=[], metavar=("OLD", "NEW"))
    parser.add_argument("--in-questions", action="store_true")
    parser.add_argument("--weights-server")
    parser.add_argument("--weights-checksum")
    parser.add_argument("--question", action="append", default=[], help="probe only decisions asking this question")
    parser.add_argument("--target", action="append", default=[], metavar="QUESTION=OPTION")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--tie", type=float, default=DEFAULT_TIE)
    parser.add_argument("--no-identity", action="store_true", help="skip the identity check before a variant")
    parser.add_argument(
        "--batch", type=int, help="requests per /v1/decide call (default: as many as the server takes)"
    )
    parser.add_argument("--rows", type=Path, help="write every probed question here as JSON lines")
    return parser


def _variant(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Variant:
    edits = tuple(TextEdit(text) for text in args.strip) + tuple(TextEdit(old, new) for old, new in args.replace)
    weights = args.weights_server is not None or args.weights_checksum is not None
    if weights and edits:
        parser.error("a probe changes one thing: either the text or the weights")
    if weights:
        if not (args.weights_server and args.weights_checksum):
            parser.error("a weights variant needs --weights-server and --weights-checksum")
        client = DecideClient(args.weights_server, batch=args.batch)
        return WeightsVariant(args.name or f"weights {args.weights_checksum[:12]}", client, args.weights_checksum)
    if edits:
        return TextVariant(args.name or "text", edits, questions=args.in_questions)
    return IDENTITY


def _traces(runs: Sequence[Path], checksum: str | None, questions: Sequence[str]) -> list[DecisionTrace]:
    traces = [trace for run in runs for trace in load_run(run, policy_checksum=checksum)]
    if questions:
        traces = [t for t in traces if any(key in questions for key, _ in t.decision.questions)]
    return traces


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    target: dict[str, str] | None = None
    if args.target:
        pairs = [item.split("=", 1) for item in args.target]
        if any(len(pair) != 2 or not all(pair) for pair in pairs):
            parser.error("--target takes QUESTION=OPTION")
        target = dict(pairs)  # type: ignore[arg-type]
    variant = _variant(args, parser)
    try:
        traces = _traces(args.runs, args.policy_checksum, args.question)
        if not traces:
            raise ContractError("the runs hold no decision asking the questions named")
        result = probe(
            traces,
            variant,
            server=DecideClient(args.server, batch=args.batch),
            target=target,
            policy_checksum=args.policy_checksum,
            check_identity=not args.no_identity or variant is IDENTITY,
            tolerance=args.tolerance,
            tie=args.tie,
        )
    except IdentityError as error:
        print(json.dumps({"identity": error.report.to_json()}, indent=2))
        print(f"probe refused: {error}", file=sys.stderr)
        return 1
    except ContractError as error:
        print(f"probe refused: {error}", file=sys.stderr)
        return 1
    if args.rows is not None:
        args.rows.write_text("".join(json.dumps(row.to_json()) + "\n" for row in result.rows), encoding="utf-8")
    print(json.dumps({"runs": [str(run) for run in args.runs], **result.to_json()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
