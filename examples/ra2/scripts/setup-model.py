"""Download one pinned checkpoint; weights remain outside version control."""

import argparse
import json
import os
from pathlib import Path


def main():
    from huggingface_hub import snapshot_download

    root = Path(__file__).resolve().parents[1]
    pins = json.loads((root / "pins.json").read_text())["models"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(pins), required=True)
    args = parser.parse_args()
    pin = pins[args.model]
    path = snapshot_download(
        pin["repo"],
        revision=pin["revision"],
        token=False,
        allow_patterns=["rl_agent_config.json", "model.safetensors", "encoder/*", "tokenizer/*"],
    )
    directory = Path(os.environ.get("LAYA_MODEL_CONFIG_DIR", root / ".cache/models"))
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{args.model}.json").write_text(json.dumps({**pin, "path": path}, indent=2) + "\n")
    print(f"Prepared {args.model} at revision {pin['revision']}")


if __name__ == "__main__":
    main()
