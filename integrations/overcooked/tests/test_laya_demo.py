"""The portable demo replays committed evidence without model inference."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from urllib.request import Request, urlopen

import pytest

from integrations.overcooked import laya_demo
from integrations.overcooked.ui import make_server


def test_corrupt_archive_is_rejected_before_extraction(tmp_path: Path) -> None:
    row = dict(laya_demo.model_episodes()[laya_demo.DEFAULT_EPISODE])
    row["files"] = dict.fromkeys(row["files"], "0" * 64)
    with pytest.raises(ValueError, match="checksum mismatch"):
        laya_demo.extract_archive(row, tmp_path)
    assert not list(tmp_path.iterdir())


def test_recorded_demo_http_seek_keeps_decisions_and_native_frames_aligned(tmp_path: Path) -> None:
    row = laya_demo.model_episodes()[laya_demo.DEFAULT_EPISODE]
    laya_demo.extract_archive(row, tmp_path)
    app = laya_demo.ReplayApplication(tmp_path, row)
    server = make_server(app, 0, html=laya_demo.page())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"

    def get_state(index: int | None = None) -> dict:
        request = url + "/api/state"
        if index is not None:
            request = Request(
                url + "/api/step", json.dumps({"index": index}).encode(), {"Content-Type": "application/json"}
            )
        with urlopen(request, timeout=10) as response:
            return json.load(response)

    try:
        with urlopen(url, timeout=10) as response:
            html = response.read().decode()
        assert "Actual base-Laya experiment recording" in html
        assert "recordedProbabilities" in html
        initial = get_state()
        assert initial["replay"] and not initial["model_available"]
        assert initial["frame"]["step"] == initial["demo"]["decision"]["step"] == 0
        assert initial["demo"]["repo"] == "convaiinnovations/laya"
        for index in (1, 200, 400):
            state = get_state(index)
            assert state["frame"] == app.replay_data["frames"][index]
            assert state["demo"]["decision"]["step"] == index - 1
            assert state["demo"]["actions"] == app.steps[index]["actions"]
        assert state["frame"]["score"] == state["demo"]["final_score"] == 180
        assert get_state(0)["frame"]["score"] == 0
    finally:
        server.shutdown()
        thread.join(timeout=10)
        server.server_close()
        app.close()
