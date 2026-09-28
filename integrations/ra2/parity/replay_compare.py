"""Compare RPLX v1 payloads, excluding only the wall-clock save timestamp.

Every chunk's CRC is checked first. Action bytes, RNG state, hash checkpoints,
chunk order and flags remain part of the comparison, as does gameTimestamp.
"""

from __future__ import annotations

import base64
import hashlib
import json
import struct
import zlib
from pathlib import Path


def canonical_replay(path: Path) -> bytes:
    encoded = path.read_bytes()
    raw = encoded if encoded.startswith(b"RPLX") else base64.b64decode(encoded, validate=True)
    if len(raw) < 10 or raw[:8] != b"RPLX\x01\x00\x00\x00":
        raise ValueError("expected an RPLX v1 header")
    count = struct.unpack_from("<H", raw, 8)[0]
    offset, heads = 10, 0
    result = bytearray(raw[:10])
    for _ in range(count):
        if offset + 16 > len(raw):
            raise ValueError("truncated RPLX chunk header")
        tag, version, flags, size, crc = struct.unpack_from("<4sHHII", raw, offset)
        offset += 16
        payload = raw[offset : offset + size]
        if len(payload) != size or zlib.crc32(payload) != crc:
            raise ValueError(f"invalid RPLX payload or CRC for {tag!r}")
        offset += size
        if tag == b"HEAD":
            heads += 1
            if version != 1 or flags != 1:
                raise ValueError("unsupported RPLX HEAD version or flags")
            header = json.loads(payload)
            saved_at = header["game"].pop("savedAt")
            if type(saved_at) is not int:
                raise ValueError("RPLX savedAt must be an integer timestamp")
            payload = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
        result.extend(struct.pack("<4sHHII", tag, version, flags, len(payload), zlib.crc32(payload)))
        result.extend(payload)
    if offset != len(raw) or heads != 1:
        raise ValueError("unexpected RPLX trailing data or HEAD count")
    return bytes(result)


def replay_comparison(left: Path, right: Path) -> dict[str, object]:
    paths = (left, right)
    payloads = [canonical_replay(path) for path in paths]
    return {
        "equal": payloads[0] == payloads[1],
        "ignoredFields": ["HEAD.game.savedAt"],
        "fileSha256": [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths],
        "contentSha256": [hashlib.sha256(payload).hexdigest() for payload in payloads],
    }
