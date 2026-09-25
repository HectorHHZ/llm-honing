# ruff: noqa: E402
# Standalone example imports follow its explicit sys.path bootstrap.
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "league"))
from gpu_group import bind, try_group

from campaign import try_device_lease


def test_busy_device_never_allocated_and_partial_group_released():
    held, closed = set(), []

    class Lease:
        def __init__(self, key):
            self.key = key
            held.add(key)

        def close(self):
            held.remove(self.key)
            closed.append(self.key)

    def acquire(key):
        return None if key == "u5" else Lease(key)

    result = try_group(["1", "5", "6"], 2, {"1": "u1", "5": "u5", "6": "u6"}, set(), acquire)
    assert result[0] == ["1", "6"]
    assert closed == ["u1"]
    for lease in result[1]:
        lease.close()
    assert not held
    assert try_group(["1", "6"], 2, {"1": "u1", "6": "u6"}, {"u6"}, acquire) is None
    assert not held


def test_real_dispatch_contention_rolls_back_without_deadlock(tmp_path, monkeypatch):
    monkeypatch.setenv("LAYA_LOCK_DIR", str(tmp_path))
    existing = try_device_lease("u5")
    try:
        result = try_group(["1", "5"], 2, {"1": "u1", "5": "u5"}, set())
        assert result is None
        again = try_device_lease("u1")
        assert again is not None
        again.close()
    finally:
        existing.close()
    result = try_group(["1", "5"], 2, {"1": "u1", "5": "u5"}, set())
    assert result[0] == ["1", "5"]
    for lease in result[1]:
        lease.close()


def test_explicit_resource_binding_requires_both_devices():
    assert bind(["node", "--red-gpu={gpu0}", "--blue-gpu={gpu1}"], ["5", "6"]) == [
        "node",
        "--red-gpu=5",
        "--blue-gpu=6",
    ]
    with pytest.raises(ValueError):
        bind(["node", "--gpu={gpu0}"], ["5", "6"])
    with pytest.raises(ValueError):
        bind(["{gpu0}", "{gpu1}", "{gpu2}"], ["5", "6"])


def test_explicit_portable_pool_rejects_ambiguous_or_duplicate_devices():
    from gpu_group import validate_pool

    validate_pool(["0", "2"], 2)
    validate_pool(["1", "5", "6"], 2)
    for pool, count in [
        (["0", "0"], 2),
        (["0"], 2),
        (["01", "2"], 2),
        (["-1", "2"], 2),
        (["0", "2"], 1),
        ([chr(0xFF11), "2"], 2),  # Fullwidth digit is not a canonical GPU index.
    ]:
        with pytest.raises(ValueError):
            validate_pool(pool, count)
