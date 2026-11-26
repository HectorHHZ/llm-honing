"""Run-time locations and the decision model's tokenizer for the full-HK Exp 0 adapter, resolved without any absolute
local path in the code (2026-11-26, Exp 0 v0 HK collection; the VVVVVV adapter's runtime.py is the model).

Nothing here is written into a record except `tokenizer_record()` (repo, revision, file, SHA-256 and which source was
used; never the local path).

- WORKSPACE: the repository root (this file is integrations/hollow-knight-forestdango/exp0/v0/runtime.py).
- COMMON:    `integrations/_shared/exp0` (the shared Exp 0 tools).
- `workspace_path(p)`: a plan path (relative to WORKSPACE, as the frozen plan writes them) made absolute.
- FFMPEG:    `$HONING_FFMPEG`, else `ffmpeg` on PATH, else the `cse153` conda environment of the running Python's conda
             installation (where this server's ffmpeg lives). Only videos use it.
- The tokenizer of the decision model (Laya, convaiinnovations/laya-typed-decisions@1a793eb): the harness counts the
  tokens of an option text with it and shortens an option by a fixed rule when it is longer than Laya's 48-token
  limit (harness text.py `fit`), so it changes request bytes. The pinned harness copy holds a neutral placeholder
  instead of the Demo session's absolute local path; `harness_agent(package)` loads the tokenizer from
  `$HONING_LAYA_TOKENIZER`, else through huggingface_hub from the public checkpoint (repo, revision, file below; with
  HF_HUB_OFFLINE=1 only a cached copy is used), checks its SHA-256, and sets it in the harness's text module before
  the first request is rendered. Without the `tokenizers` package or the file every tool refuses to run (the
  harness's own silent fallback, a word count, differs from Laya's count: measured 1 of 511 logged v12 requests).
"""

from __future__ import annotations

import hashlib
import importlib
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[3]
COMMON = HERE.parents[2] / "_shared" / "exp0"

LAYA_TOKENIZER_ENV = "HONING_LAYA_TOKENIZER"
LAYA_REPO = "convaiinnovations/laya-typed-decisions"
LAYA_REVISION = "1a793eb568e6718f15941d08f85432581df534e3"
LAYA_TOKENIZER_PATH_IN_REPO = "tokenizer/tokenizer.json"
LAYA_TOKENIZER_SHA256 = "6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30"
PLACEHOLDER = "<set by the Exp 0 adapter>"  # the constant in the pinned harness copy (text._TOKENIZER_FILE)

_LOADED: dict[str, dict] = {}


class TokenizerMissing(SystemExit):
    pass


def workspace_path(path: str | os.PathLike) -> Path:
    p = Path(path)
    return p if p.is_absolute() else WORKSPACE / p


def _ffmpeg() -> str:
    explicit = os.environ.get("HONING_FFMPEG")
    if explicit:
        return explicit
    found = shutil.which("ffmpeg")
    if found:
        return found
    candidate = Path(sys.base_prefix).resolve() / "envs" / "cse153" / "bin" / "ffmpeg"
    return str(candidate) if candidate.exists() else "ffmpeg"


FFMPEG = _ffmpeg()


def tokenizer_file() -> tuple[Path, str]:
    """(local file, source) of Laya's tokenizer.json: $HONING_LAYA_TOKENIZER, else huggingface_hub."""
    explicit = os.environ.get(LAYA_TOKENIZER_ENV)
    if explicit:
        return Path(explicit), f"${LAYA_TOKENIZER_ENV}"
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise TokenizerMissing(
            f"Laya's tokenizer is needed to render requests: set ${LAYA_TOKENIZER_ENV} to "
            f"{LAYA_REPO}@{LAYA_REVISION[:7]} {LAYA_TOKENIZER_PATH_IN_REPO}, or install "
            "huggingface_hub"
        ) from exc
    try:
        return Path(hf_hub_download(LAYA_REPO, LAYA_TOKENIZER_PATH_IN_REPO, revision=LAYA_REVISION)), "huggingface_hub"
    except Exception as exc:
        raise TokenizerMissing(
            f"Laya's tokenizer could not be resolved through huggingface_hub ({type(exc).__name__}: "
            f"{exc}); set ${LAYA_TOKENIZER_ENV} to the file"
        ) from exc


def load_tokenizer():
    """The loaded Laya tokenizer and its record; refuses (SystemExit) when the package or the file is missing or the
    file is not the pinned one."""
    if "laya" in _LOADED:
        return _LOADED["laya"]["tokenizer"], _LOADED["laya"]["record"]
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise TokenizerMissing(
            "the `tokenizers` package is required (the harness counts option tokens with Laya's "
            "tokenizer); install tokenizers"
        ) from exc
    path, source = tokenizer_file()
    if not path.is_file():
        raise TokenizerMissing(f"Laya's tokenizer file is missing (from {source})")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != LAYA_TOKENIZER_SHA256:
        raise TokenizerMissing(
            f"Laya's tokenizer file (from {source}) has sha256 {digest[:12]}..., not the pinned "
            f"{LAYA_TOKENIZER_SHA256[:12]}... of {LAYA_REPO}@{LAYA_REVISION[:7]}"
        )
    tokenizer = Tokenizer.from_file(str(path))
    record = {
        "repo": LAYA_REPO,
        "revision": LAYA_REVISION,
        "file": LAYA_TOKENIZER_PATH_IN_REPO,
        "sha256": digest,
        "source": source,
    }
    _LOADED["laya"] = {"tokenizer": tokenizer, "record": record, "path": str(path)}
    return tokenizer, record


def setup_text(package: str) -> dict | None:
    """Set Laya's tokenizer in the harness package's text module (when it counts tokens). Returns its record."""
    text_mod = importlib.import_module(f"{package}.text")
    if not hasattr(text_mod, "option_tokens"):
        return None
    tokenizer, record = load_tokenizer()
    text_mod._TOKENIZER_FILE = _LOADED["laya"]["path"]
    text_mod._TOKENIZER = tokenizer
    return record


def harness_agent(package: str):
    """The harness package's agent module, with the tokenizer set up before any request is rendered."""
    agent_mod = importlib.import_module(f"{package}.agent")
    setup_text(package)
    return agent_mod


def tokenizer_record() -> dict | None:
    return dict(_LOADED["laya"]["record"]) if "laya" in _LOADED else None
