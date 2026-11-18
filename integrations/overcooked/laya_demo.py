"""Replay a committed base-Laya episode, including its exact requests and probabilities."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tempfile
import threading
import webbrowser
from pathlib import Path
from typing import Any

from integrations.overcooked.engine import ROOT, Scenario
from integrations.overcooked.ui import Application, make_server

EXPERIMENT = ROOT / "experiments/laya-harness-v1"
DEFAULT_EPISODE = "search--seed--asymmetric_advantages"


def model_episodes() -> dict[str, dict[str, Any]]:
    return {
        row["key"]: row
        for row in json.loads((EXPERIMENT / "episodes.json").read_text(encoding="utf-8"))
        if row["model_calls"] > 0
    }


def extract_archive(row: dict[str, Any], out: Path) -> None:
    """Check the committed compressed bytes before extracting into a fresh temporary directory."""
    archive = (EXPERIMENT / "data/episodes" / row["key"]).resolve()
    files = []
    for relative, expected in row["files"].items():
        source = (EXPERIMENT / relative).resolve()
        destination = out / source.relative_to(archive)
        data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("Archive file checksum mismatch: " + relative)
        files.append((destination, data))
    for destination, data in files:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.suffix == ".gz":
            destination = destination.with_suffix("")
            data = gzip.decompress(data)
        with destination.open("xb") as stream:
            stream.write(data)


class ReplayApplication(Application):
    def __init__(self, run: Path, row: dict[str, Any]) -> None:
        # Application verifies all recorded joint actions against the native engine first.
        super().__init__(Scenario(), replay_dir=run)
        self.row = row
        self.decisions = [
            json.loads(line) for line in (run / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        self.steps = [json.loads(line) for line in (run / "steps.jsonl").read_text(encoding="utf-8").splitlines()]

    def state(self) -> dict[str, Any]:
        with self.lock:
            value = super().state()
            index = max(0, self.index - 1)
            metadata = self.replay_data["manifest"]["model"]["metadata"]
            value["demo"] = {
                "repo": metadata["repo"],
                "revision": metadata["revision"],
                "harness": self.row["candidate"],
                "final_score": self.row["score"],
                "decision": self.decisions[index],
                "actions": self.steps[index + 1]["actions"],
                "reward": self.steps[index + 1]["frame"]["reward"],
            }
            return value


def page() -> bytes:
    html = (ROOT / "web/index.html").read_text(encoding="utf-8")
    html = html.replace("<title>Overcooked · Honing</title>", "<title>Base Laya · Verified experiment replay</title>")
    html = html.replace(
        "Honing · Cooperative decision environment",
        "Base Laya · Real recorded decisions · Chef 0 model / Chef 1 greedy",
    )
    html = html.replace("Rule baseline", "Recorded Laya actions").replace("New game", "Rewind")
    html = html.replace("Rewind started; previous recording retained", "Replay rewound to tick 0")
    html = html.replace("<main>", "<main>" + (EXPERIMENT / "demo.html").read_text(encoding="utf-8"))
    script = (EXPERIMENT / "demo.js").read_text(encoding="utf-8")
    return html.replace("</html>", "<script>" + script + "</script></html>").encode("utf-8")


def main() -> None:
    episodes = model_episodes()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", choices=sorted(episodes), default=DEFAULT_EPISODE)
    parser.add_argument("--port", type=int, default=4187)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    cache = ROOT / ".cache"
    cache.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="laya-demo-", dir=cache) as temporary:
        run = Path(temporary)
        row = episodes[args.episode]
        extract_archive(row, run)
        app = ReplayApplication(run, row)
        try:
            server = make_server(app, args.port, html=page())
            try:
                url = f"http://127.0.0.1:{server.server_port}/"
                print(f"Verified base-Laya replay: {url} | {args.episode} | final score {row['score']}", flush=True)
                if args.open:
                    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
                server.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                server.server_close()
        finally:
            app.close()


if __name__ == "__main__":
    main()
