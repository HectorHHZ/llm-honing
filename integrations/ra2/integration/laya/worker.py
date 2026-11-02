"""Pinned local decision-model inference over JSON lines; model failure stops the match.

A thin stdio front end for honing.backends.jev (Laya, Kev, NanoJev, StartLux or Nimble), the same
implementations the HTTP decision server uses. DECISION_FAMILY picks the family
(default laya). Settings come from the environment:

- Laya: LAYA_GPU, LAYA_MODEL_TYPE, LAYA_SOURCE_DIR, LAYA_MODEL_CONFIG_DIR,
  LAYA_CHECKPOINT_MANIFEST, LAYA_LOCK_DIR, and LAYA_MAX_LEN.
- Kev: LAYA_GPU, KEV_MODEL, KEV_SOURCE_DIR, KEV_CHECKPOINT_MANIFEST, LAYA_LOCK_DIR,
  and KEV_DTYPE.
- NanoJev: LAYA_GPU, NANOJEV_MODEL, NANOJEV_SOURCE_DIR, NANOJEV_CHECKPOINT_MANIFEST,
  LAYA_LOCK_DIR, and NANOJEV_MAX_LEN.
- StartLux: LAYA_GPU, STARTLUX_MODEL, STARTLUX_SOURCE_DIR, LAYA_LOCK_DIR, and STARTLUX_MAX_LEN.
- Nimble: LAYA_GPU, NIMBLE_MODEL, NIMBLE_SOURCE_DIR, NIMBLE_KIT_DIR, LAYA_LOCK_DIR, and NIMBLE_MAX_LEN.
"""

import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# The integration lives inside the Honing repository, so the model environment
# does not need Honing installed to import its backend.
sys.path.insert(0, str(ROOT.parents[1]))

from honing.backends.jev.kev import KevDecisionModel, KevSettings  # noqa: E402
from honing.backends.jev.laya import LayaDecisionModel, LayaSettings  # noqa: E402
from honing.backends.jev.nanojev import NanoJevDecisionModel, NanoJevSettings  # noqa: E402
from honing.backends.jev.nimble import NimbleDecisionModel, NimbleSettings  # noqa: E402
from honing.backends.jev.startlux import StartLuxDecisionModel, StartLuxSettings  # noqa: E402


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


def startlux_settings_from_environment():
    if os.environ.get("STARTLUX_CHECKPOINT_MANIFEST"):
        raise ValueError("StartLux serves the pinned releases only; it has no trained checkpoints yet")
    lock_dir = os.environ.get("LAYA_LOCK_DIR")
    return StartLuxSettings(
        pins_path=ROOT / "pins.json",
        source_dir=Path(os.environ.get("STARTLUX_SOURCE_DIR", ROOT / "deps/startlux")),
        model=os.environ.get("STARTLUX_MODEL", "startlux-4b"),
        gpu=os.environ.get("LAYA_GPU", "0"),
        lock_dir=Path(lock_dir) if lock_dir else None,
        max_length=int(os.environ.get("STARTLUX_MAX_LEN", "8192")),
    )


def nimble_settings_from_environment():
    if os.environ.get("NIMBLE_CHECKPOINT_MANIFEST"):
        raise ValueError("Nimble serves the pinned release only; it has no trained checkpoints yet")
    if not os.environ.get("NIMBLE_SOURCE_DIR"):
        raise ValueError("NIMBLE_SOURCE_DIR must name the pinned source snapshot that scripts/setup-nimble.sh prints")
    lock_dir = os.environ.get("LAYA_LOCK_DIR")
    return NimbleSettings(
        pins_path=ROOT / "pins.json",
        source_dir=Path(os.environ["NIMBLE_SOURCE_DIR"]),
        kit_dir=Path(os.environ.get("NIMBLE_KIT_DIR", ROOT / "deps/decision-index")),
        model=os.environ.get("NIMBLE_MODEL", "nimble-v4-max"),
        gpu=os.environ.get("LAYA_GPU", "0"),
        lock_dir=Path(lock_dir) if lock_dir else None,
        max_length=int(os.environ.get("NIMBLE_MAX_LEN", "8192")),
    )


def load_model():
    family = os.environ.get("DECISION_FAMILY", "laya")
    if family == "startlux":
        return StartLuxDecisionModel.load(startlux_settings_from_environment())
    if family == "nimble":
        return NimbleDecisionModel.load(nimble_settings_from_environment())
    if family == "nanojev":
        return NanoJevDecisionModel.load(nanojev_settings_from_environment())
    if family == "kev":
        return KevDecisionModel.load(kev_settings_from_environment())
    if family != "laya":
        raise ValueError(f"Unknown decision model family {family!r}")
    return LayaDecisionModel.load(settings_from_environment())


def main():
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
