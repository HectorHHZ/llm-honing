"""Pinned local decision-model inference over JSON lines; model failure stops the match.

A thin stdio front end for honing.backends.jev (Laya or Kev), the same
implementations the HTTP decision server uses. DECISION_FAMILY picks the family
(default laya). Settings come from the environment:

- Laya: LAYA_GPU, LAYA_MODEL_TYPE, LAYA_SOURCE_DIR, LAYA_MODEL_CONFIG_DIR,
  LAYA_CHECKPOINT_MANIFEST, LAYA_LOCK_DIR, and LAYA_MAX_LEN.
- Kev: LAYA_GPU, KEV_MODEL, KEV_SOURCE_DIR, KEV_CHECKPOINT_MANIFEST, LAYA_LOCK_DIR,
  and KEV_DTYPE.
- NanoJev: LAYA_GPU, NANOJEV_MODEL, NANOJEV_SOURCE_DIR, NANOJEV_CHECKPOINT_MANIFEST,
  LAYA_LOCK_DIR, and NANOJEV_MAX_LEN.
- djev: DJEV_URL (a running deployment), or LAYA_GPU, DJEV_SOURCE_DIR, DJEV_CACHE_DIR, and
  DJEV_PORTS to start one with this interpreter's environment; DJEV_MODEL.
"""

import json
import os
import signal
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# The integration lives inside the Honing repository, so the model environment
# does not need Honing installed to import its backend.
sys.path.insert(0, str(ROOT.parents[1]))

from honing.backends.djev import DjevDecisionModel, DjevRuntimeSettings, DjevSettings  # noqa: E402
from honing.backends.jev.kev import KevDecisionModel, KevSettings  # noqa: E402
from honing.backends.jev.laya import LayaDecisionModel, LayaSettings  # noqa: E402
from honing.backends.jev.nanojev import NanoJevDecisionModel, NanoJevSettings  # noqa: E402


def emit(value):
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


def settings_from_environment():
    manifest = os.environ.get("LAYA_CHECKPOINT_MANIFEST")
    lock_dir = os.environ.get("LAYA_LOCK_DIR")
    return LayaSettings(
        pins_path=ROOT / "pins.json",
        model_type=os.environ.get("LAYA_MODEL_TYPE", "english"),
        model_config_dir=Path(os.environ.get("LAYA_MODEL_CONFIG_DIR", ROOT / ".cache/models")),
        source_dir=Path(os.environ.get("LAYA_SOURCE_DIR", ROOT / "deps/laya")),
        gpu=os.environ.get("LAYA_GPU", "0"),
        checkpoint_manifest=Path(manifest) if manifest else None,
        lock_dir=Path(lock_dir) if lock_dir else None,
        max_len=int(os.environ.get("LAYA_MAX_LEN", "2048")),
    )


def kev_settings_from_environment():
    manifest = os.environ.get("KEV_CHECKPOINT_MANIFEST")
    lock_dir = os.environ.get("LAYA_LOCK_DIR")
    return KevSettings(
        pins_path=ROOT / "pins.json",
        source_dir=Path(os.environ.get("KEV_SOURCE_DIR", ROOT / "deps/kev")),
        model=os.environ.get("KEV_MODEL", "kev-0.8b"),
        gpu=os.environ.get("LAYA_GPU", "0"),
        checkpoint_manifest=Path(manifest) if manifest else None,
        lock_dir=Path(lock_dir) if lock_dir else None,
        dtype=os.environ.get("KEV_DTYPE", "bf16"),
    )


def nanojev_settings_from_environment():
    manifest = os.environ.get("NANOJEV_CHECKPOINT_MANIFEST")
    lock_dir = os.environ.get("LAYA_LOCK_DIR")
    return NanoJevSettings(
        pins_path=ROOT / "pins.json",
        source_dir=Path(os.environ.get("NANOJEV_SOURCE_DIR", ROOT / "deps/nanojev")),
        model=os.environ.get("NANOJEV_MODEL", "nanojev"),
        gpu=os.environ.get("LAYA_GPU", "0"),
        checkpoint_manifest=Path(manifest) if manifest else None,
        lock_dir=Path(lock_dir) if lock_dir else None,
        max_length=int(os.environ.get("NANOJEV_MAX_LEN", "2048")),
    )


def djev_settings_from_environment():
    runtime = None
    if not os.environ.get("DJEV_URL"):
        api_port, model_port = (int(p) for p in os.environ.get("DJEV_PORTS", "8100,8101").split(","))
        cache = os.environ.get("DJEV_CACHE_DIR")
        runtime = DjevRuntimeSettings(
            pins_path=ROOT / "pins.json",
            source_dir=Path(os.environ.get("DJEV_SOURCE_DIR", ROOT / "deps/djev")),
            python=Path(sys.executable),
            gpu=os.environ.get("LAYA_GPU", "0"),
            api_port=api_port,
            model_port=model_port,
            cache_dir=Path(cache) if cache else None,
        )
    return DjevSettings(
        pins_path=ROOT / "pins.json",
        runtime=runtime,
        url=os.environ.get("DJEV_URL") or None,
        model=os.environ.get("DJEV_MODEL", "diffusiongemma-26b-a4b-it"),
    )


def load_model():
    family = os.environ.get("DECISION_FAMILY", "laya")
    if family == "djev":
        return DjevDecisionModel.load(djev_settings_from_environment())
    if family == "nanojev":
        return NanoJevDecisionModel.load(nanojev_settings_from_environment())
    if family == "kev":
        return KevDecisionModel.load(kev_settings_from_environment())
    if family != "laya":
        raise ValueError(f"Unknown decision model family {family!r}")
    return LayaDecisionModel.load(settings_from_environment())


def main():
    # SIGTERM (the match process closing its client) ends the worker through its context managers.
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(0))
    with load_model() as model:
        emit({"ready": True, "metadata": dict(model.metadata)})
        for line in sys.stdin:
            request = json.loads(line)
            emit({"id": request["id"], **model.predict(request["parts"], request["questions"])})


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        emit({"error": str(error)})
        sys.exit(1)
