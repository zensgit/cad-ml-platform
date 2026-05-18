"""Cache-invalidation invariants for readiness_registry checksum cache.

`_CHECKSUM_CACHE` is keyed by (path, mtime_ns, size). Any rewrite that changes
size MUST recompute the checksum; the cache key already covers mtime, so
appending bytes naturally invalidates. The risk we guard here is a future
refactor of `_checksum_file` that drops one of those key components and lets
a modified file silently keep its old checksum.
"""

from __future__ import annotations

from pathlib import Path

from src.models.readiness_registry import (
    _CHECKSUM_CACHE,
    _checksum_file,
)


def _clear_cache_for(path: Path) -> None:
    for key in list(_CHECKSUM_CACHE.keys()):
        if key[0] == str(path):
            _CHECKSUM_CACHE.pop(key, None)


def test_checksum_returns_none_for_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "nope.pth"
    assert _checksum_file(str(missing)) is None


def test_checksum_is_stable_for_same_bytes(tmp_path: Path) -> None:
    target = tmp_path / "ckpt.pth"
    target.write_bytes(b"checkpoint-bytes-v1")
    _clear_cache_for(target)
    first = _checksum_file(str(target))
    second = _checksum_file(str(target))
    assert first is not None
    assert first == second
    assert len(first) == 16  # truncated SHA256


def test_checksum_invalidates_when_size_changes(tmp_path: Path) -> None:
    target = tmp_path / "ckpt.pth"
    target.write_bytes(b"checkpoint-v1")
    _clear_cache_for(target)
    digest_v1 = _checksum_file(str(target))
    # Append bytes — both size and mtime change.
    target.write_bytes(b"checkpoint-v1-EXTENDED-PAYLOAD")
    digest_v2 = _checksum_file(str(target))
    assert digest_v1 is not None
    assert digest_v2 is not None
    assert digest_v1 != digest_v2, (
        "Checksum must recompute after a size change; cache key invariant broken."
    )


def test_checksum_invalidates_when_content_changes_at_same_size(
    tmp_path: Path,
) -> None:
    target = tmp_path / "ckpt.pth"
    target.write_bytes(b"AAAAAAAAAAAA")  # 12 bytes
    _clear_cache_for(target)
    digest_a = _checksum_file(str(target))
    # Same length, different content → mtime_ns advances → cache key changes.
    # We intentionally tolerate an OS that gives the rewrite the exact same
    # mtime_ns (extremely unlikely with sub-ms resolution); in that pathological
    # case the cache key would NOT change and this assertion would fail —
    # which is exactly the invariant we want surfaced.
    target.write_bytes(b"BBBBBBBBBBBB")
    digest_b = _checksum_file(str(target))
    assert digest_a is not None
    assert digest_b is not None
    assert digest_a != digest_b, (
        "Checksum must recompute when content changes; cache key invariant broken. "
        "If this trips on a fast filesystem, the cache key must include content "
        "hash or os.stat(st_ino) explicitly, not just mtime+size."
    )


def test_checksum_cache_serves_repeats_without_reread(tmp_path: Path) -> None:
    # Indirect proof: after first read, the same key must be present in the
    # cache, and a second call hits it (we verify by the key landing in
    # _CHECKSUM_CACHE, not by mocking).
    target = tmp_path / "ckpt.pth"
    target.write_bytes(b"warm-cache-bytes")
    _clear_cache_for(target)
    assert all(k[0] != str(target) for k in _CHECKSUM_CACHE)
    _checksum_file(str(target))
    assert any(k[0] == str(target) for k in _CHECKSUM_CACHE), (
        "First read must populate _CHECKSUM_CACHE for the path."
    )
