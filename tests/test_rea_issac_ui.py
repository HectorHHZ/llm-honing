import copy
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from honing.backends.rea_issac import ui as server


class HTTPContracts(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.original_root = server.ROOT
        server.ROOT = Path(self.directory.name)
        self.session = server.Session(7)
        self.http = ThreadingHTTPServer(("127.0.0.1", 0), server.handler(self.session))
        self.thread = threading.Thread(target=self.http.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.http.server_port}"

    def tearDown(self):
        self.http.shutdown()
        self.http.server_close()
        self.thread.join()
        self.session.log.close()
        server.ROOT = self.original_root
        self.directory.cleanup()

    def call(self, path, body=None):
        req = urllib.request.Request(
            self.base + path,
            data=None if body is None else json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as error:
            return error.code, json.load(error)

    def test_get_does_not_tick_and_invalid_step_is_atomic(self):
        before = self.session.engine.digest()
        for _ in range(3):
            self.assertEqual(self.call("/api/state")[1]["observation"]["tick"], 0)
        for body in ({"frames": True}, {"action": {"move_x": 100}}, {"frames": 301}):
            self.assertEqual(self.call("/api/step", body)[0], 400)
            self.assertEqual(self.session.engine.digest(), before)

    def test_replay_export_import_and_playback(self):
        for _ in range(10):
            self.call("/api/step", {"action": {"move_x": 1, "shoot_y": -1}, "frames": 3})
        record = self.call("/api/export")[1]
        self.assertEqual(self.call("/api/replay", record)[0], 200)
        self.assertEqual(self.call("/api/state")[1]["observation"]["tick"], 0)
        for _ in record["actions"]:
            status, meta = self.call("/api/step", {})
            self.assertEqual(status, 200)
        self.assertTrue(meta["playback_complete"])
        self.assertEqual(self.call("/api/export")[1]["final_hash"], record["final_hash"])

    def test_invalid_replay_preserves_current_run(self):
        self.call("/api/step", {"action": {"move_x": 1}, "frames": 3})
        record = self.call("/api/export")[1]
        invalid = copy.deepcopy(record)
        invalid["actions"][0]["action"]["move_x"] = -1
        before = self.session.engine.digest()
        run = self.session.run
        self.assertEqual(self.call("/api/replay", invalid)[0], 400)
        self.assertEqual(self.session.run, run)
        self.assertEqual(self.session.engine.digest(), before)

    def test_reset_preserves_previous_artifacts(self):
        self.call("/api/step", {"action": {"move_x": 1}, "frames": 3})
        before = self.session.run
        self.assertEqual(self.call("/api/reset", {"seed": 11})[0], 200)
        self.assertNotEqual(before, self.session.run)
        self.assertTrue((before / "replay.json").is_file())
        self.assertEqual(self.call("/api/state")[1]["observation"]["seed"], 11)


if __name__ == "__main__":
    unittest.main()
