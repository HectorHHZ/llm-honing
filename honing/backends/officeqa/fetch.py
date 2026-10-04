"""Download chosen OfficeQA questions and only the bulletins they need, instead of the 460 MB corpus.

The dataset is gated on Hugging Face (approved automatically); ``HF_TOKEN`` must belong to an account that accepted it.

Run: ``python -m honing.backends.officeqa.fetch --out integrations/officeqa/.cache --uids UID0002,UID0006,UID0016``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path

from honing.backends.officeqa.suite import CORPUS_DIR, read_questions

_DATASET = "https://huggingface.co/datasets/databricks/officeqa/resolve/main"
QUESTIONS = "officeqa_full.csv"
_TRANSFORMED = "treasury_bulletins_parsed/transformed"
_ISSUE = re.compile(r"^treasury_bulletin_(\d{4})_(\d{2})\.txt$")


def _download(url: str, target: Path, token: str) -> None:
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)


def neighbors(name: str, count: int) -> list[str]:
    """Names a bulletin and the monthly issues around it, some of which may not exist.

    Args:
        name: A file such as ``treasury_bulletin_1944_01.txt``.
        count: Issues to include on each side.

    Returns:
        The file names, oldest first; just ``name`` if it is not a dated bulletin.
    """
    match = _ISSUE.match(name)
    if match is None or count == 0:
        return [name]
    month = int(match.group(1)) * 12 + int(match.group(2)) - 1
    return [f"treasury_bulletin_{m // 12:04d}_{m % 12 + 1:02d}.txt" for m in range(month - count, month + count + 1)]


def fetch(out: Path, uids: Sequence[str], neighbor_count: int, token: str) -> dict[str, str]:
    """Downloads the questions file and the bulletins the named questions need.

    Writes ``<out>/officeqa_full.csv``, ``<out>/corpus/``, and ``<out>/manifest.json``.
    Files already present are kept.

    Args:
        out: The data directory.
        uids: The questions.
        neighbor_count: Issues to fetch on each side of each source bulletin.
        token: A Hugging Face token.

    Returns:
        The SHA-256 of each bulletin, by file name.

    Raises:
        SystemExit: A uid is not in the dataset.
    """
    csv_path = out / QUESTIONS
    if not csv_path.exists():
        _download(f"{_DATASET}/{QUESTIONS}", csv_path, token)
    questions = read_questions(csv_path)
    unknown = [uid for uid in uids if uid not in questions]
    if unknown:
        raise SystemExit(f"no OfficeQA questions {unknown}")
    wanted = sorted({n for uid in uids for f in questions[uid].source_files for n in neighbors(f, neighbor_count)})
    files: dict[str, str] = {}
    for name in wanted:
        target = out / CORPUS_DIR / name
        if not target.exists():
            try:
                _download(f"{_DATASET}/{_TRANSFORMED}/{name}", target, token)
            except urllib.error.HTTPError as error:
                if error.code == 404 and name not in {f for uid in uids for f in questions[uid].source_files}:
                    continue  # a neighbor that was never published
                raise
        files[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    manifest = {"uids": list(uids), "neighbors": neighbor_count, "corpus": files}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return files


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the fetch from the command line.

    Args:
        argv: The arguments; defaults to ``sys.argv[1:]``.

    Returns:
        The process exit code.
    """
    parser = argparse.ArgumentParser(description="Download OfficeQA questions and the bulletins they need.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--uids", required=True, help="comma-separated question uids, e.g. UID0002,UID0006")
    parser.add_argument("--neighbors", type=int, default=2, help="issues on each side of each source bulletin")
    args = parser.parse_args(argv)
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("set HF_TOKEN to a Hugging Face token that has accepted databricks/officeqa", file=sys.stderr)
        return 2
    files = fetch(args.out, [u.strip() for u in args.uids.split(",") if u.strip()], args.neighbors, token)
    print(f"{len(files)} bulletins in {args.out / CORPUS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
