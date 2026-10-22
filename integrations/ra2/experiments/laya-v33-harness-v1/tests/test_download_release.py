"""Offline downloader checks: no requests leave this process."""

import hashlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "download-release.py"
SPEC = importlib.util.spec_from_file_location("v33_download", SCRIPT)
download = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(download)


class DownloadReleaseTest(unittest.TestCase):
    def test_paths_cannot_escape_destination(self):
        for name in ("../escape", "/absolute", "x/../escape", "x\\escape", ""):
            with self.subTest(name=name), self.assertRaises(ValueError):
                download.safe_name(name)

    def test_matching_existing_file_is_reused_without_network(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "file").write_bytes(b"content")
            checksum = hashlib.sha256(b"content").hexdigest()
            with patch.object(download, "urlopen", side_effect=AssertionError("Unexpected network")):
                download.fetch(root, "file", checksum)
                with self.assertRaisesRegex(ValueError, "Existing file differs"):
                    download.fetch(root, "file", "0" * 64)
            self.assertEqual((root / "file").read_bytes(), b"content")

    def test_verified_download_uses_immutable_revision(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checksum = hashlib.sha256(b"content").hexdigest()
            with patch.object(download, "urlopen", return_value=io.BytesIO(b"content")) as request:
                download.fetch(root, "tokenizer/config.json", checksum)
            self.assertIn(f"/resolve/{download.REVISION}/", request.call_args.args[0].full_url)
            self.assertEqual((root / "tokenizer/config.json").read_bytes(), b"content")
            self.assertFalse((root / "tokenizer/config.json.download-part").exists())

    def test_bad_download_never_becomes_a_verified_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (
                patch.object(download, "urlopen", side_effect=lambda *_args, **_kwargs: io.BytesIO(b"wrong")),
                patch.object(download.time, "sleep"),
                self.assertRaisesRegex(ValueError, "Checksum mismatch"),
            ):
                download.fetch(root, "file", hashlib.sha256(b"expected").hexdigest())
            self.assertFalse((root / "file").exists())

    def test_changed_manifest_is_refused_before_fetching_any_release_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary) / "SHA256SUMS").write_text("modified manifest\n", encoding="utf-8")
            with (
                patch("sys.argv", [str(SCRIPT), "--destination", temporary]),
                patch.object(download, "urlopen", side_effect=AssertionError("Unexpected network")),
                self.assertRaisesRegex(ValueError, "Existing file differs"),
            ):
                download.main()


if __name__ == "__main__":
    unittest.main()
