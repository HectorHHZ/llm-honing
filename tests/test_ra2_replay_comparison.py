"""Replay parity ignores save time, while retaining all simulation information."""

import base64
import json
import struct
import zlib
from pathlib import Path

import pytest
from integrations.ra2.parity.replay_compare import canonical_replay, replay_comparison


def replay(path: Path, *, saved_at: int = 10, game_timestamp: int = 0, actions: bytes = b"actions") -> Path:
    head = json.dumps({"game": {"savedAt": saved_at, "gameTimestamp": game_timestamp}}).encode()
    raw = b"RPLX\x01\x00\x00\x00" + struct.pack("<H", 2)
    for tag, data in ((b"HEAD", head), (b"ACTS", actions)):
        raw += struct.pack("<4sHHII", tag, 1, 1, len(data), zlib.crc32(data)) + data
    path.write_bytes(base64.b64encode(raw))
    return path


@pytest.mark.unit
def test_only_save_timestamp_is_ignored(tmp_path: Path) -> None:
    left = replay(tmp_path / "left")
    right = replay(tmp_path / "right", saved_at=987654321)
    result = replay_comparison(left, right)
    assert result["equal"]
    assert result["fileSha256"][0] != result["fileSha256"][1]
    assert result["contentSha256"][0] == result["contentSha256"][1]
    assert not replay_comparison(left, replay(right, game_timestamp=1))["equal"]
    assert not replay_comparison(left, replay(right, actions=b"other action"))["equal"]


@pytest.mark.unit
def test_corrupt_payload_is_rejected_even_if_both_files_match(tmp_path: Path) -> None:
    path = replay(tmp_path / "replay")
    raw = bytearray(base64.b64decode(path.read_bytes()))
    raw[-1] ^= 1
    path.write_bytes(base64.b64encode(raw))
    with pytest.raises(ValueError, match="CRC"):
        replay_comparison(path, path)


@pytest.mark.unit
@pytest.mark.parametrize("mutate", [lambda b: b[:-1], lambda b: b + b"extra", lambda b: b[:10]])
def test_truncated_or_extra_bytes_are_refused(tmp_path: Path, mutate) -> None:
    path = replay(tmp_path / "replay")
    path.write_bytes(base64.b64encode(mutate(base64.b64decode(path.read_bytes()))))
    with pytest.raises(ValueError):
        canonical_replay(path)
