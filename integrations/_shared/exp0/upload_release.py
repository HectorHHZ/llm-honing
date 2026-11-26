"""Check, plan, upload and download-verify one Exp 0 v0 release on Hugging Face (``honinglab/exp0-data``).

  python3 upload_release.py check  BUNDLE [--work DIR]
      No token, no network. Content policy (structured text/JSON and code only: no binaries, game assets,
      frames, screenshots or videos; UTF-8; raw replies parse as JSON; no credential-like strings; no
      absolute local paths) and ``tools/verify_release.py`` run in a fresh venv without packages.

  python3 upload_release.py plan   BUNDLE --parent-revision REV [--repo honinglab/exp0-data] [--work DIR]
      ``check`` plus the upload plan (repo, path in repo = the plan's ``dataset_dir``, files, bytes, the
      bundle's ``sha256.json`` digest, the expected parent revision). No token, no network.

  python3 upload_release.py upload BUNDLE --parent-revision REV --execute [--repo ...] [--receipt FILE]
      MAIN SESSION ONLY, after it has reviewed the locally verified release. Reads the token from the file
      named by the environment variable ``HF_TOKEN_FILE`` (never an argument, never printed or logged), checks
      that the repo's current revision is REV and that the target directory does not exist yet, uploads the
      bundle as one commit with ``parent_commit=REV``, and writes an upload receipt without the token.
      Without ``--execute`` it only prints the plan.

  python3 upload_release.py download-verify --game-dir GAME/v0/ --revision REV --out DIR --execute
                                            [--local BUNDLE] [--repo ...]
      MAIN SESSION ONLY. Downloads exactly that revision's GAME/v0/ (token from ``HF_TOKEN_FILE``; the repo is
      private), compares its ``sha256.json`` with the local bundle's when given, runs the verifier in a fresh
      venv without packages and without any API key, and writes a verification receipt.

``check``/``plan`` use only the standard library; ``huggingface_hub`` is imported only by the two executing
commands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bundle import content_problems, local_path_findings

DEFAULT_REPO = "honinglab/exp0-data"
TOKEN_VARIABLE = "HF_TOKEN_FILE"


def _sanitize(message: str, secret: str | None) -> str:
    return message.replace(secret, "<token>") if secret else message


def _token() -> str:
    """The Hugging Face token, read from the file named by ``HF_TOKEN_FILE``. Never print or log the value."""
    path = os.environ.get(TOKEN_VARIABLE)
    if not path:
        raise SystemExit(f"set {TOKEN_VARIABLE} to the path of the token file; the token is never an argument")
    token = Path(path).read_text(encoding="utf-8").strip()
    if not token:
        raise SystemExit(f"the file named by {TOKEN_VARIABLE} is empty")
    return token


def fresh_verify(bundle: Path, work: Path) -> dict[str, Any]:
    """Run the bundle's own verifier in a fresh venv without packages (no API key in its environment)."""
    venv = work / "verify-env"
    if not (venv / "bin" / "python").exists():
        subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True)
    environment = {
        key: value for key, value in os.environ.items() if key in ("PATH", "LANG", "LC_ALL", "HOME", "TMPDIR")
    }
    process = subprocess.run(
        [str(venv / "bin" / "python"), "-I", "-B", str(bundle / "tools" / "verify_release.py"), str(bundle)],
        capture_output=True,
        text=True,
        env=environment,
    )
    if process.returncode:
        raise SystemExit(f"the release does not verify: {process.stderr.strip().splitlines()[-1:]}")
    return json.loads(process.stdout)


def check(bundle: Path, work: Path) -> dict[str, Any]:
    bundle = bundle.resolve()
    manifest = bundle / "release-manifest.json"
    if (manifest.is_file() and json.loads(manifest.read_text(encoding="utf-8")).get("dry_run")) or json.loads(
        (bundle / "dataset.json").read_text(encoding="utf-8")
    ).get("dry_run"):
        raise SystemExit(
            "a dry-run bundle (contract 0.2.0) tests the release chain on pilot / smoke data; it is " "never uploaded"
        )
    problems = content_problems(bundle)
    paths = local_path_findings(bundle)
    if problems or paths:
        raise SystemExit(
            f"content policy: {len(problems)} non-text files, {len(paths)} local paths; first: "
            f"{(problems + paths)[:5]}"
        )
    verified = fresh_verify(bundle, work)
    files = [path for path in bundle.rglob("*") if path.is_file()]
    return {
        "content_policy": "ok",
        "verified": verified,
        "files": len(files),
        "bytes": sum(path.stat().st_size for path in files),
        "sha256_json_digest": hashlib.sha256((bundle / "sha256.json").read_bytes()).hexdigest(),
    }


def plan(bundle: Path, repo: str, parent: str, work: Path) -> dict[str, Any]:
    checked = check(bundle, work)
    collection = json.loads((bundle / "collection.json").read_text(encoding="utf-8"))
    release = json.loads((bundle / "release-manifest.json").read_text(encoding="utf-8"))
    if release["hf_directory"] != collection["dataset_dir"] or release["hf_dataset"] != repo:
        raise SystemExit("the release manifest names another repo or directory")
    return {
        **checked,
        "repo": repo,
        "repo_type": "dataset",
        "path_in_repo": collection["dataset_dir"],
        "parent_revision": parent,
        "code_git_commit": release["code_git_commit"],
    }


def upload(bundle: Path, repo: str, parent: str, work: Path, receipt: Path | None) -> dict[str, Any]:
    planned = plan(bundle, repo, parent, work)
    token = _token()
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        current = api.repo_info(repo, repo_type="dataset").sha
        if current != parent:
            raise SystemExit(f"the repo is at {current}, not the reviewed parent revision {parent}")
        existing = [
            name
            for name in api.list_repo_files(repo, repo_type="dataset", revision=parent)
            if name.startswith(planned["path_in_repo"])
        ]
        if existing:
            raise SystemExit(
                f"{planned['path_in_repo']} already exists at {parent}; a v0 release is never overwritten"
            )
        commit = api.upload_folder(
            repo_id=repo,
            repo_type="dataset",
            folder_path=str(bundle.resolve()),
            path_in_repo=planned["path_in_repo"].rstrip("/"),
            parent_commit=parent,
            commit_message=f"Add {planned['path_in_repo']} (Exp 0 v0, code " f"{planned['code_git_commit'][:12]})",
        )
    except SystemExit:
        raise
    except Exception as error:
        raise SystemExit(_sanitize(f"upload failed: {type(error).__name__}: {error}", token)) from None
    record = {
        **planned,
        "new_revision": commit.oid,
        "commit_url": commit.commit_url,
        "uploaded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    target = receipt or bundle.resolve().parents[1] / f"upload-receipt-{commit.oid[:12]}.json"
    target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def download_verify(repo: str, game_dir: str, revision: str, out: Path, local: Path | None) -> dict[str, Any]:
    token = _token()
    out.mkdir(parents=True, exist_ok=False)
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=repo,
            repo_type="dataset",
            revision=revision,
            allow_patterns=[f"{game_dir}*"],
            local_dir=str(out / "snapshot"),
            token=token,
        )
    except Exception as error:
        raise SystemExit(_sanitize(f"download failed: {type(error).__name__}: {error}", token)) from None
    bundle = out / "snapshot" / game_dir.rstrip("/")
    downloaded = hashlib.sha256((bundle / "sha256.json").read_bytes()).hexdigest()
    record: dict[str, Any] = {
        "repo": repo,
        "revision": revision,
        "game_dir": game_dir,
        "downloaded_sha256_json_digest": downloaded,
        "verified": fresh_verify(bundle, out),
        "checked_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if local is not None:
        expected = hashlib.sha256((local / "sha256.json").read_bytes()).hexdigest()
        record["local_sha256_json_digest"] = expected
        record["matches_local_bundle"] = expected == downloaded
        if expected != downloaded:
            raise SystemExit("the downloaded release differs from the local bundle")
    (out / "verification-receipt.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "plan", "upload"):
        sub = commands.add_parser(name)
        sub.add_argument("bundle", type=Path)
        sub.add_argument("--work", type=Path)
        if name != "check":
            sub.add_argument("--parent-revision", required=True)
            sub.add_argument("--repo", default=DEFAULT_REPO)
        if name == "upload":
            sub.add_argument("--execute", action="store_true")
            sub.add_argument("--receipt", type=Path)
    down = commands.add_parser("download-verify")
    down.add_argument("--game-dir", required=True)
    down.add_argument("--revision", required=True)
    down.add_argument("--out", type=Path, required=True)
    down.add_argument("--local", type=Path)
    down.add_argument("--repo", default=DEFAULT_REPO)
    down.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.command == "download-verify":
        if not args.execute:
            raise SystemExit(
                "download-verify reads the token and calls Hugging Face: add --execute (release owner only)"
            )
        print(
            json.dumps(download_verify(args.repo, args.game_dir, args.revision, args.out, args.local), sort_keys=True)
        )
        return
    with tempfile.TemporaryDirectory(prefix="exp0-verify-") as scratch:
        work = args.work or Path(scratch)
        work.mkdir(parents=True, exist_ok=True)
        if args.command == "check":
            result = check(args.bundle, work)
        elif args.command == "plan" or not args.execute:
            result = {**plan(args.bundle, args.repo, args.parent_revision, work), "dry_run": args.command == "upload"}
        else:
            result = upload(args.bundle, args.repo, args.parent_revision, work, args.receipt)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
