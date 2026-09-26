"""Pinned local Laya inference over JSON lines; model failure stops the match.

A thin stdio front end for honing.backends.jev.laya, the same implementation the
HTTP decision server uses. Settings come from the environment, as before:
LAYA_GPU, LAYA_MODEL_TYPE, LAYA_SOURCE_DIR, LAYA_MODEL_CONFIG_DIR,
LAYA_CHECKPOINT_MANIFEST, LAYA_LOCK_DIR, and LAYA_MAX_LEN.
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

from honing.backends.jev.laya import LayaDecisionModel, LayaSettings  # noqa: E402


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


def main():
    with LayaDecisionModel.load(settings_from_environment()) as model:
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
