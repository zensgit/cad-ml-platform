"""Phase-A C1 model-activation core — positive controls + observed-RED discriminators.

Scope: reusable core only. No family wiring, no production pins, no reload.
"""

from __future__ import annotations

import hashlib
import os
import resource
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
from unittest import mock

import pytest

from src.core.model_activation import (
    ActivationRefusal,
    ArtifactKind,
    BoundPolicy,
    ControlledStore,
    PinRecord,
    RefusalReason,
    ResolverImpl,
    last_open_impl,
    openat2_available,
    sha256_hex,
    tree_digest_v1_from_file_bytes,
    validate_raw_pin,
)
from src.core.model_activation.digest import read_fd_bounded_once
from src.core.model_activation.fd_dir import list_dir_fd
from src.core.model_activation.resolver import open_pinned, open_store_root
from src.core.model_activation.types import TerminalKind
import src.core.model_activation.store as store_mod
import src.core.model_activation.fd_dir as fd_dir_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _store(
    root: Path,
    pins: Iterable[PinRecord],
    *,
    bounds: Optional[BoundPolicy] = None,
    impl: Optional[ResolverImpl] = None,
    freeze_parent: Optional[Path] = None,
) -> ControlledStore:
    return ControlledStore(
        str(root),
        list(pins),
        bounds=bounds,
        freeze_parent=str(freeze_parent) if freeze_parent else None,
        resolver_impl=impl or ResolverImpl.COMPONENT,
    )


def _sf_pin(lid: str, aid: str, rel: str, data: bytes) -> PinRecord:
    return PinRecord(
        logical_activation_id=lid,
        artifact_id=aid,
        kind=ArtifactKind.SINGLE_FILE,
        digest=_hex(data),
        store_relpath=rel,
    )


def _bundle_pin(
    lid: str,
    aid: str,
    rel: str,
    files: List[Tuple[str, bytes]],
) -> PinRecord:
    return PinRecord(
        logical_activation_id=lid,
        artifact_id=aid,
        kind=ArtifactKind.BUNDLE,
        digest=tree_digest_v1_from_file_bytes(files),
        store_relpath=rel,
    )


def _loose_bounds(**kwargs: int) -> BoundPolicy:
    base = dict(
        max_single_file_bytes=1024 * 1024,
        max_bundle_file_count=10_000,
        max_bundle_per_file_bytes=1024 * 1024,
        max_bundle_aggregate_bytes=10 * 1024 * 1024,
        max_total_dirents=50_000,
        max_directories=10_000,
        max_depth=64,
        max_relpath_utf8_bytes=4096,
    )
    base.update(kwargs)
    return BoundPolicy(**base)


# ---------------------------------------------------------------------------
# Raw pin domain
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "a//b",
        "a/./b",
        "a/../b",
        "family/model//weights.pt",
        "family/./model.pt",
        "family/../secret.pt",
        "/etc/model.pt",
        "abs/trailing/",
        "",
        ".",
        "..",
        "a/b/",
        "/leading",
        "has\x00nul",
        "\x00",
        "//double",
    ],
)
def test_raw_pin_domain_rejects(raw: str) -> None:
    with pytest.raises(ActivationRefusal) as ei:
        validate_raw_pin(raw)
    assert ei.value.reason is RefusalReason.RAW_PIN_INVALID


@pytest.mark.parametrize(
    "raw",
    [
        "weights.pt",
        "family/model.pt",
        "a/b/c/d.pt",
        "ocr/deepseek/bundle_root",
    ],
)
def test_raw_pin_domain_accepts(raw: str) -> None:
    parts = validate_raw_pin(raw)
    assert parts == tuple(raw.split("/"))


def test_raw_pin_rejects_path_object_type() -> None:
    with pytest.raises(ActivationRefusal) as ei:
        validate_raw_pin(Path("a/b"))  # type: ignore[arg-type]
    assert ei.value.reason is RefusalReason.RAW_PIN_INVALID


# ---------------------------------------------------------------------------
# Single-file
# ---------------------------------------------------------------------------


def test_single_file_exact_hash_green(tmp_path: Path) -> None:
    root = tmp_path / "store"
    data = b"model-bytes-v1"
    rel = "family/model.pt"
    _write(root / rel, data)
    pin = _sf_pin("act.graph2d", "art.g2d.v1", rel, data)
    store = _store(root, [pin], freeze_parent=tmp_path / "freeze")
    try:
        out = store.assert_fixed_hash("act.graph2d", "art.g2d.v1")
        assert out == data
        assert store.load_pinned_file("act.graph2d", "art.g2d.v1") == data
    finally:
        store.close()


def test_single_file_digest_mismatch_red(tmp_path: Path) -> None:
    root = tmp_path / "store"
    data = b"model-bytes-v1"
    rel = "family/model.pt"
    _write(root / rel, data)
    pin = PinRecord(
        logical_activation_id="act.x",
        artifact_id="art.x",
        kind=ArtifactKind.SINGLE_FILE,
        digest=_hex(b"other"),
        store_relpath=rel,
    )
    store = _store(root, [pin])
    try:
        with pytest.raises(ActivationRefusal) as ei:
            store.assert_fixed_hash("act.x", "art.x")
        assert ei.value.reason is RefusalReason.DIGEST_MISMATCH
    finally:
        store.close()


def test_pin_absent_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "store"
    root.mkdir()
    store = _store(root, [])
    try:
        with pytest.raises(ActivationRefusal) as ei:
            store.assert_fixed_hash("missing", "missing")
        assert ei.value.reason is RefusalReason.PIN_ABSENT
    finally:
        store.close()


def test_wrong_kind_single_file_api_on_bundle_pin(tmp_path: Path) -> None:
    root = tmp_path / "store"
    bdir = root / "bundle"
    _write(bdir / "a.bin", b"a")
    pin = _bundle_pin("act.b", "art.b", "bundle", [("a.bin", b"a")])
    store = _store(root, [pin])
    try:
        with pytest.raises(ActivationRefusal) as ei:
            store.assert_fixed_hash("act.b", "art.b")
        assert ei.value.reason is RefusalReason.KIND_MISMATCH
    finally:
        store.close()


def test_wrong_kind_bundle_api_on_single_file_pin(tmp_path: Path) -> None:
    root = tmp_path / "store"
    data = b"x"
    rel = "m.pt"
    _write(root / rel, data)
    pin = _sf_pin("act.s", "art.s", rel, data)
    store = _store(root, [pin])
    try:
        with pytest.raises(ActivationRefusal) as ei:
            store.assert_bundle_digest("act.s", "art.s")
        assert ei.value.reason is RefusalReason.KIND_MISMATCH
    finally:
        store.close()


def test_missing_artifact_red(tmp_path: Path) -> None:
    root = tmp_path / "store"
    root.mkdir()
    pin = _sf_pin("act.m", "art.m", "gone.pt", b"x")
    store = _store(root, [pin])
    try:
        with pytest.raises(ActivationRefusal) as ei:
            store.assert_fixed_hash("act.m", "art.m")
        assert ei.value.reason in (
            RefusalReason.ARTIFACT_MISSING,
            RefusalReason.CONTAINMENT,
        )
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Raw domain via both resolvers
# ---------------------------------------------------------------------------


_RAW_INVALID = [
    "a//b",
    "a/./b",
    "a/../b",
    "has\x00nul",
    "/absolute.pt",
    "trail/",
]


@pytest.mark.parametrize("raw", _RAW_INVALID)
def test_component_resolver_rejects_raw_domain(tmp_path: Path, raw: str) -> None:
    root = tmp_path / "store"
    root.mkdir()
    fd, _ = open_store_root(str(root))
    try:
        with pytest.raises(ActivationRefusal) as ei:
            open_pinned(
                fd,
                raw,
                TerminalKind.REGULAR_FILE,
                impl=ResolverImpl.COMPONENT,
            )
        assert ei.value.reason is RefusalReason.RAW_PIN_INVALID
    finally:
        os.close(fd)


@pytest.mark.parametrize("raw", _RAW_INVALID)
def test_openat2_resolver_rejects_raw_domain_identically(
    tmp_path: Path, raw: str
) -> None:
    if not openat2_available():
        pytest.skip("openat2 syscall unavailable on this kernel/platform")
    root = tmp_path / "store"
    root.mkdir()
    fd, _ = open_store_root(str(root))
    try:
        with pytest.raises(ActivationRefusal) as ei:
            open_pinned(
                fd,
                raw,
                TerminalKind.REGULAR_FILE,
                impl=ResolverImpl.OPENAT2,
            )
        assert ei.value.reason is RefusalReason.RAW_PIN_INVALID
    finally:
        os.close(fd)


def test_resolver_impl_parity_raw_domain_matrix(tmp_path: Path) -> None:
    root = tmp_path / "store"
    root.mkdir()
    fd, _ = open_store_root(str(root))
    try:
        for raw in _RAW_INVALID:
            reasons = []
            with pytest.raises(ActivationRefusal) as ei:
                open_pinned(
                    fd, raw, TerminalKind.REGULAR_FILE, impl=ResolverImpl.COMPONENT
                )
            reasons.append(ei.value.reason)
            if openat2_available():
                with pytest.raises(ActivationRefusal) as ei:
                    open_pinned(
                        fd,
                        raw,
                        TerminalKind.REGULAR_FILE,
                        impl=ResolverImpl.OPENAT2,
                    )
                reasons.append(ei.value.reason)
            assert all(r is RefusalReason.RAW_PIN_INVALID for r in reasons)
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# Containment / special files / TOCTOU
# ---------------------------------------------------------------------------


def test_intermediate_symlink_refused(tmp_path: Path) -> None:
    root = tmp_path / "store"
    outside = tmp_path / "outside"
    outside.mkdir()
    _write(outside / "evil.pt", b"evil")
    family = root / "family"
    family.parent.mkdir(parents=True, exist_ok=True)
    family.symlink_to(outside, target_is_directory=True)
    pin = _sf_pin("act.s", "art.s", "family/evil.pt", b"evil")
    store = _store(root, [pin])
    try:
        with pytest.raises(ActivationRefusal) as ei:
            store.assert_fixed_hash("act.s", "art.s")
        assert ei.value.reason in (
            RefusalReason.SYMLINK_REJECTED,
            RefusalReason.NOT_DIRECTORY,
            RefusalReason.CONTAINMENT,
            RefusalReason.ARTIFACT_MISSING,
        )
    finally:
        store.close()


def test_leaf_symlink_refused_even_inside_store(tmp_path: Path) -> None:
    root = tmp_path / "store"
    real = root / "real.pt"
    _write(real, b"real-bytes")
    link = root / "link.pt"
    link.symlink_to(real)
    pin = _sf_pin("act.s", "art.s", "link.pt", b"real-bytes")
    store = _store(root, [pin])
    try:
        with pytest.raises(ActivationRefusal) as ei:
            store.assert_fixed_hash("act.s", "art.s")
        assert ei.value.reason in (
            RefusalReason.SYMLINK_REJECTED,
            RefusalReason.SPECIAL_FILE,
            RefusalReason.NOT_REGULAR_FILE,
            RefusalReason.CONTAINMENT,
        )
    finally:
        store.close()


def test_parent_swap_to_symlink_mid_walk_refused(tmp_path: Path) -> None:
    root = tmp_path / "store"
    target = root / "family" / "leaf.pt"
    _write(target, b"ok")
    outside = tmp_path / "out"
    outside.mkdir()
    _write(outside / "leaf.pt", b"evil")

    fd, _ = open_store_root(str(root))
    try:
        leaf_fd = open_pinned(
            fd,
            "family/leaf.pt",
            TerminalKind.REGULAR_FILE,
            impl=ResolverImpl.COMPONENT,
        )
        os.close(leaf_fd)

        family = root / "family"
        family_real = root / "family_real"
        family.rename(family_real)
        family.symlink_to(outside, target_is_directory=True)

        with pytest.raises(ActivationRefusal) as ei:
            open_pinned(
                fd,
                "family/leaf.pt",
                TerminalKind.REGULAR_FILE,
                impl=ResolverImpl.COMPONENT,
            )
        assert ei.value.reason in (
            RefusalReason.SYMLINK_REJECTED,
            RefusalReason.NOT_DIRECTORY,
            RefusalReason.CONTAINMENT,
        )
    finally:
        os.close(fd)


def test_fifo_leaf_does_not_block(tmp_path: Path) -> None:
    root = tmp_path / "store"
    root.mkdir()
    os.mkfifo(root / "pipe.pt")
    pin = _sf_pin("act.f", "art.f", "pipe.pt", b"x")
    store = _store(root, [pin])
    try:
        t0 = time.monotonic()
        with pytest.raises(ActivationRefusal) as ei:
            store.assert_fixed_hash("act.f", "art.f")
        assert time.monotonic() - t0 < 2.0
        assert ei.value.reason in (
            RefusalReason.SPECIAL_FILE,
            RefusalReason.NOT_REGULAR_FILE,
            RefusalReason.SYMLINK_REJECTED,
            RefusalReason.CONTAINMENT,
            RefusalReason.UNREADABLE,
        )
    finally:
        store.close()


def test_oversized_single_file_red(tmp_path: Path) -> None:
    root = tmp_path / "store"
    data = b"x" * 100
    rel = "big.pt"
    _write(root / rel, data)
    pin = _sf_pin("act.o", "art.o", rel, data)
    store = _store(root, [pin], bounds=BoundPolicy(max_single_file_bytes=50))
    try:
        with pytest.raises(ActivationRefusal) as ei:
            store.assert_fixed_hash("act.o", "art.o")
        assert ei.value.reason is RefusalReason.OVERSIZE
    finally:
        store.close()


def test_growing_file_refused_on_same_fd(tmp_path: Path) -> None:
    path = tmp_path / "grow.bin"
    path.write_bytes(b"abcd")
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    try:
        st = os.fstat(fd)
        with open(path, "ab") as w:
            w.write(b"EFGH")
        with pytest.raises(ActivationRefusal) as ei:
            read_fd_bounded_once(fd, max_bytes=1024, expect_size=st.st_size)
        assert ei.value.reason is RefusalReason.GROWING_FILE
    finally:
        os.close(fd)


def test_same_fd_toctou_returns_hashed_bytes_not_reread(tmp_path: Path) -> None:
    root = tmp_path / "store"
    rel = "m.pt"
    original = b"original-model-bytes"
    _write(root / rel, original)
    pin = _sf_pin("act.t", "art.t", rel, original)
    store = _store(root, [pin])
    try:
        out = store.assert_fixed_hash("act.t", "art.t")
        assert out == original
        (root / rel).write_bytes(b"swapped-after-hash!!!!!!")
        assert out == original
        assert sha256_hex(out) == pin.digest
        with pytest.raises(ActivationRefusal) as ei:
            store.assert_fixed_hash("act.t", "art.t")
        assert ei.value.reason is RefusalReason.DIGEST_MISMATCH
    finally:
        store.close()


def test_inode_swap_after_open_does_not_affect_same_fd_read(
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"
    rel = "leaf.pt"
    path = root / rel
    _write(path, b"inode-A")
    fd, _ = open_store_root(str(root))
    try:
        leaf = open_pinned(
            fd, rel, TerminalKind.REGULAR_FILE, impl=ResolverImpl.COMPONENT
        )
        try:
            path.unlink()
            path.write_bytes(b"inode-B-swapped")
            data = read_fd_bounded_once(leaf, 1024, expect_size=7)
            assert data == b"inode-A"
        finally:
            os.close(leaf)
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# Bundle: digest from freeze, DFS, preflight, sealed handle
# ---------------------------------------------------------------------------


def test_bundle_exact_digest_and_determinism(tmp_path: Path) -> None:
    root = tmp_path / "store"
    files = [
        ("a.txt", b"aaa"),
        ("sub/b.txt", b"bbb"),
        ("sub/c.txt", b""),
    ]
    for rel, data in files:
        _write(root / "bundle" / rel, data)
    pin = _bundle_pin("act.emb", "art.emb", "bundle", files)
    assert pin.digest == tree_digest_v1_from_file_bytes(list(reversed(files)))

    store = _store(root, [pin], freeze_parent=tmp_path / "freeze")
    try:
        frozen = store.assert_bundle_digest("act.emb", "art.emb")
        try:
            assert frozen.digest == pin.digest
            assert frozen.file_count == 3
            assert frozen.read_member("a.txt") == b"aaa"
            assert frozen.read_member("sub/b.txt") == b"bbb"
            assert frozen.read_member("sub/c.txt") == b""
            # Source mutation GREEN for freeze (reads via held fd).
            (root / "bundle" / "a.txt").write_bytes(b"mutated")
            assert frozen.read_member("a.txt") == b"aaa"
        finally:
            frozen.cleanup()
    finally:
        store.close()


def test_bundle_digest_mismatch_red(tmp_path: Path) -> None:
    root = tmp_path / "store"
    _write(root / "bundle" / "a.txt", b"aaa")
    pin = PinRecord(
        logical_activation_id="act.b",
        artifact_id="art.b",
        kind=ArtifactKind.BUNDLE,
        digest="ab" * 32,
        store_relpath="bundle",
    )
    store = _store(root, [pin], freeze_parent=tmp_path / "freeze")
    try:
        with pytest.raises(ActivationRefusal) as ei:
            store.assert_bundle_digest("act.b", "art.b")
        assert ei.value.reason is RefusalReason.DIGEST_MISMATCH
    finally:
        store.close()


def test_bundle_digest_computed_from_freeze_not_source_buffer(
    tmp_path: Path,
) -> None:
    """If freeze write is corrupted, digest-from-freeze must RED (not accept pin)."""
    root = tmp_path / "store"
    _write(root / "bundle" / "m.bin", b"good")
    pin = _bundle_pin("act.b", "art.b", "bundle", [("m.bin", b"good")])
    store = _store(root, [pin], freeze_parent=tmp_path / "freeze")

    real_write = os.write

    def evil_write(fd: int, data: bytes | memoryview) -> int:  # type: ignore[override]
        # Corrupt freeze payload while reporting the same length.
        if isinstance(data, memoryview):
            raw = data.tobytes()
        else:
            raw = bytes(data)
        if raw == b"good":
            return real_write(fd, b"evil")
        return real_write(fd, data)

    try:
        with mock.patch.object(store_mod.os, "write", side_effect=evil_write):
            with pytest.raises(ActivationRefusal) as ei:
                store.assert_bundle_digest("act.b", "art.b")
            assert ei.value.reason is RefusalReason.DIGEST_MISMATCH
    finally:
        store.close()


def test_bundle_symlink_member_red_and_destroys_partial(
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"
    freeze_parent = tmp_path / "freeze"
    freeze_parent.mkdir()
    _write(root / "bundle" / "good.txt", b"good")
    (root / "bundle" / "link").symlink_to("good.txt")
    pin = _bundle_pin("act.b", "art.b", "bundle", [("good.txt", b"good")])
    store = _store(root, [pin], freeze_parent=freeze_parent)
    try:
        with pytest.raises(ActivationRefusal) as ei:
            store.assert_bundle_digest("act.b", "art.b")
        assert ei.value.reason is RefusalReason.SYMLINK_REJECTED
        leftovers = [
            p
            for p in freeze_parent.iterdir()
            if p.is_dir() and p.name.startswith("cadml-freeze-")
        ]
        assert leftovers == []
    finally:
        store.close()


def test_bundle_fifo_member_red_no_block(tmp_path: Path) -> None:
    root = tmp_path / "store"
    bdir = root / "bundle"
    bdir.mkdir(parents=True)
    _write(bdir / "good.txt", b"good")
    os.mkfifo(bdir / "pipe")
    pin = _bundle_pin("act.b", "art.b", "bundle", [("good.txt", b"good")])
    store = _store(root, [pin], freeze_parent=tmp_path / "freeze")
    try:
        t0 = time.monotonic()
        with pytest.raises(ActivationRefusal) as ei:
            store.assert_bundle_digest("act.b", "art.b")
        assert time.monotonic() - t0 < 2.0
        assert ei.value.reason is RefusalReason.SPECIAL_FILE
    finally:
        store.close()


def test_preflight_metadata_cap_zero_destination_writes(tmp_path: Path) -> None:
    """Metadata-detectable max_directories=1 must refuse before any freeze write."""
    root = tmp_path / "store"
    bdir = root / "bundle"
    _write(bdir / "good.txt", b"goodxx")  # 6 bytes — spy length check
    (bdir / "d1").mkdir(parents=True)
    (bdir / "d2").mkdir(parents=True)
    pin = PinRecord(
        logical_activation_id="act.b",
        artifact_id="art.b",
        kind=ArtifactKind.BUNDLE,
        digest="00" * 32,
        store_relpath="bundle",
    )
    bounds = _loose_bounds(max_directories=1)
    store = _store(
        root, [pin], bounds=bounds, freeze_parent=tmp_path / "freeze"
    )
    writes: list[int] = []
    real_write = os.write

    def spy_write(fd: int, data: bytes | memoryview) -> int:  # type: ignore[override]
        n = real_write(fd, data)
        writes.append(n)
        return n

    try:
        with mock.patch.object(store_mod.os, "write", side_effect=spy_write):
            with pytest.raises(ActivationRefusal) as ei:
                store.assert_bundle_digest("act.b", "art.b")
            assert ei.value.reason is RefusalReason.BUNDLE_DIRECTORY_COUNT
            assert writes == [], f"expected zero freeze writes, got {writes}"
    finally:
        store.close()


def test_preflight_file_count_zero_writes(tmp_path: Path) -> None:
    root = tmp_path / "store"
    bdir = root / "bundle"
    for i in range(5):
        _write(bdir / f"f{i}.bin", b"x")
    pin = PinRecord(
        logical_activation_id="act.b",
        artifact_id="art.b",
        kind=ArtifactKind.BUNDLE,
        digest="00" * 32,
        store_relpath="bundle",
    )
    store = _store(
        root,
        [pin],
        bounds=_loose_bounds(max_bundle_file_count=2),
        freeze_parent=tmp_path / "freeze",
    )
    writes: list[int] = []
    real_write = os.write

    def spy_write(fd: int, data: bytes | memoryview) -> int:  # type: ignore[override]
        n = real_write(fd, data)
        writes.append(n)
        return n

    try:
        with mock.patch.object(store_mod.os, "write", side_effect=spy_write):
            with pytest.raises(ActivationRefusal) as ei:
                store.assert_bundle_digest("act.b", "art.b")
            assert ei.value.reason is RefusalReason.BUNDLE_FILE_COUNT
            assert writes == []
    finally:
        store.close()


def test_preflight_aggregate_zero_writes(tmp_path: Path) -> None:
    root = tmp_path / "store"
    bdir = root / "bundle"
    _write(bdir / "a.bin", b"a" * 30)
    _write(bdir / "b.bin", b"b" * 30)
    pin = PinRecord(
        logical_activation_id="act.b",
        artifact_id="art.b",
        kind=ArtifactKind.BUNDLE,
        digest="00" * 32,
        store_relpath="bundle",
    )
    store = _store(
        root,
        [pin],
        bounds=_loose_bounds(max_bundle_aggregate_bytes=40),
        freeze_parent=tmp_path / "freeze",
    )
    writes: list[int] = []
    real_write = os.write

    def spy_write(fd: int, data: bytes | memoryview) -> int:  # type: ignore[override]
        n = real_write(fd, data)
        writes.append(n)
        return n

    try:
        with mock.patch.object(store_mod.os, "write", side_effect=spy_write):
            with pytest.raises(ActivationRefusal) as ei:
                store.assert_bundle_digest("act.b", "art.b")
            assert ei.value.reason is RefusalReason.BUNDLE_AGGREGATE_BYTES
            assert writes == []
    finally:
        store.close()


def test_bundle_directory_bomb_red(tmp_path: Path) -> None:
    root = tmp_path / "store"
    bdir = root / "bundle"
    bdir.mkdir(parents=True)
    for i in range(20):
        (bdir / f"d{i}").mkdir()
    pin = PinRecord(
        logical_activation_id="act.b",
        artifact_id="art.b",
        kind=ArtifactKind.BUNDLE,
        digest="00" * 32,
        store_relpath="bundle",
    )
    store = _store(
        root,
        [pin],
        bounds=_loose_bounds(max_directories=5),
        freeze_parent=tmp_path / "freeze",
    )
    try:
        with pytest.raises(ActivationRefusal) as ei:
            store.assert_bundle_digest("act.b", "art.b")
        assert ei.value.reason is RefusalReason.BUNDLE_DIRECTORY_COUNT
    finally:
        store.close()


def test_bundle_dirent_bomb_red(tmp_path: Path) -> None:
    root = tmp_path / "store"
    bdir = root / "bundle"
    bdir.mkdir(parents=True)
    for i in range(30):
        (bdir / f"f{i}.txt").write_bytes(b"")
    pin = PinRecord(
        logical_activation_id="act.b",
        artifact_id="art.b",
        kind=ArtifactKind.BUNDLE,
        digest="00" * 32,
        store_relpath="bundle",
    )
    store = _store(
        root,
        [pin],
        bounds=_loose_bounds(max_total_dirents=10),
        freeze_parent=tmp_path / "freeze",
    )
    try:
        with pytest.raises(ActivationRefusal) as ei:
            store.assert_bundle_digest("act.b", "art.b")
        assert ei.value.reason in (
            RefusalReason.BUNDLE_TOTAL_DIRENTS,
            RefusalReason.BUNDLE_FILE_COUNT,
        )
    finally:
        store.close()


def test_bundle_depth_bomb_red(tmp_path: Path) -> None:
    root = tmp_path / "store"
    cur = root / "bundle"
    for i in range(10):
        cur = cur / f"d{i}"
        cur.mkdir(parents=True, exist_ok=True)
    (cur / "leaf.txt").write_bytes(b"x")
    pin = PinRecord(
        logical_activation_id="act.b",
        artifact_id="art.b",
        kind=ArtifactKind.BUNDLE,
        digest="00" * 32,
        store_relpath="bundle",
    )
    store = _store(
        root,
        [pin],
        bounds=_loose_bounds(max_depth=3),
        freeze_parent=tmp_path / "freeze",
    )
    try:
        with pytest.raises(ActivationRefusal) as ei:
            store.assert_bundle_digest("act.b", "art.b")
        assert ei.value.reason is RefusalReason.BUNDLE_DEPTH
    finally:
        store.close()


def test_bundle_relpath_bomb_red(tmp_path: Path) -> None:
    root = tmp_path / "store"
    bdir = root / "bundle"
    bdir.mkdir(parents=True)
    long_name = "n" * 80
    (bdir / long_name).write_bytes(b"x")
    pin = PinRecord(
        logical_activation_id="act.b",
        artifact_id="art.b",
        kind=ArtifactKind.BUNDLE,
        digest="00" * 32,
        store_relpath="bundle",
    )
    store = _store(
        root,
        [pin],
        bounds=_loose_bounds(max_relpath_utf8_bytes=40),
        freeze_parent=tmp_path / "freeze",
    )
    try:
        with pytest.raises(ActivationRefusal) as ei:
            store.assert_bundle_digest("act.b", "art.b")
        assert ei.value.reason is RefusalReason.BUNDLE_RELPATH_BYTES
    finally:
        store.close()


def test_bundle_file_count_red(tmp_path: Path) -> None:
    root = tmp_path / "store"
    bdir = root / "bundle"
    for i in range(5):
        _write(bdir / f"f{i}.bin", b"x")
    pin = PinRecord(
        logical_activation_id="act.b",
        artifact_id="art.b",
        kind=ArtifactKind.BUNDLE,
        digest="00" * 32,
        store_relpath="bundle",
    )
    store = _store(
        root,
        [pin],
        bounds=_loose_bounds(max_bundle_file_count=2),
        freeze_parent=tmp_path / "freeze",
    )
    try:
        with pytest.raises(ActivationRefusal) as ei:
            store.assert_bundle_digest("act.b", "art.b")
        assert ei.value.reason is RefusalReason.BUNDLE_FILE_COUNT
    finally:
        store.close()


def test_partial_freeze_cleanup_on_mid_walk_failure(tmp_path: Path) -> None:
    root = tmp_path / "store"
    freeze_parent = tmp_path / "freeze"
    freeze_parent.mkdir()
    bdir = root / "bundle"
    _write(bdir / "ok.txt", b"ok")
    (bdir / "bad").symlink_to("ok.txt")
    pin = PinRecord(
        logical_activation_id="act.b",
        artifact_id="art.b",
        kind=ArtifactKind.BUNDLE,
        digest="00" * 32,
        store_relpath="bundle",
    )
    store = _store(root, [pin], freeze_parent=freeze_parent)
    try:
        with pytest.raises(ActivationRefusal):
            store.assert_bundle_digest("act.b", "art.b")
        assert list(freeze_parent.glob("cadml-freeze-*")) == []
    finally:
        store.close()


def test_bundle_source_mutation_after_freeze_green(tmp_path: Path) -> None:
    root = tmp_path / "store"
    files = [("a.txt", b"stable")]
    _write(root / "bundle" / "a.txt", b"stable")
    pin = _bundle_pin("act.b", "art.b", "bundle", files)
    store = _store(root, [pin], freeze_parent=tmp_path / "freeze")
    try:
        frozen = store.load_pinned_bundle("act.b", "art.b")
        try:
            (root / "bundle" / "a.txt").write_bytes(b"mutated-source")
            assert frozen.read_member("a.txt") == b"stable"
        finally:
            frozen.cleanup()
    finally:
        store.close()


def test_frozen_bundle_has_no_public_path(tmp_path: Path) -> None:
    root = tmp_path / "store"
    _write(root / "bundle" / "m.bin", b"good")
    pin = _bundle_pin("act.b", "art.b", "bundle", [("m.bin", b"good")])
    store = _store(root, [pin], freeze_parent=tmp_path / "freeze")
    try:
        frozen = store.assert_bundle_digest("act.b", "art.b")
        try:
            assert not hasattr(frozen, "path") or "path" not in getattr(
                frozen, "__dict__", {}
            )
            with pytest.raises(AttributeError):
                _ = frozen.path  # type: ignore[attr-defined]
            assert frozen.read_member("m.bin") == b"good"
        finally:
            frozen.cleanup()
    finally:
        store.close()


def test_freeze_inplace_mutation_red(tmp_path: Path) -> None:
    """Sealed freeze: mode 0400 always; write refusal when DAC applies (non-root)."""
    root = tmp_path / "store"
    _write(root / "bundle" / "m.bin", b"good")
    pin = _bundle_pin("act.b", "art.b", "bundle", [("m.bin", b"good")])
    freeze_parent = tmp_path / "freeze"
    store = _store(root, [pin], freeze_parent=freeze_parent)
    try:
        frozen = store.assert_bundle_digest("act.b", "art.b")
        try:
            # Discover private backing via process (not public API).
            backing = frozen._backing_path  # noqa: SLF001 — test only
            target = Path(backing) / "m.bin"
            # Privilege-honest: seal always sets 0400; root bypasses DAC write refusal.
            assert (target.stat().st_mode & 0o777) == 0o400
            # Write-refusal portion: skip only when geteuid()==0 — root bypasses
            # 0400 mode bits (CAP_DAC_OVERRIDE / traditional root write privilege).
            if os.geteuid() != 0:
                with pytest.raises(OSError):
                    target.write_bytes(b"evil")
            # Held-fd read still sees original sealed bytes (runs under root too).
            assert frozen.read_member("m.bin") == b"good"
        finally:
            frozen.cleanup()
    finally:
        store.close()


def test_freeze_path_redirect_red(tmp_path: Path) -> None:
    """Renaming/repointing the backing path must not affect dir-fd reads."""
    root = tmp_path / "store"
    _write(root / "bundle" / "m.bin", b"good")
    pin = _bundle_pin("act.b", "art.b", "bundle", [("m.bin", b"good")])
    freeze_parent = tmp_path / "freeze"
    store = _store(root, [pin], freeze_parent=freeze_parent)
    try:
        frozen = store.assert_bundle_digest("act.b", "art.b")
        try:
            backing = Path(frozen._backing_path)  # noqa: SLF001
            moved = freeze_parent / "moved-away"
            backing.rename(moved)
            # Attacker recreates path with evil content.
            backing.mkdir()
            (backing / "m.bin").write_bytes(b"evil")
            # Dir-fd still points at original freeze inode.
            assert frozen.read_member("m.bin") == b"good"
        finally:
            # Restore for cleanup: put tree back if needed.
            try:
                if moved.exists() and not Path(frozen._backing_path).exists():  # noqa: SLF001
                    moved.rename(frozen._backing_path)  # noqa: SLF001
            except Exception:
                pass
            frozen.cleanup()
    finally:
        store.close()


def test_dfs_sibling_dirs_under_low_rlimit(tmp_path: Path) -> None:
    """200 sibling dirs must succeed with RLIMIT_NOFILE=128 (O(depth) fds)."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    new_soft = 128
    if hard < new_soft and hard != resource.RLIM_INFINITY:
        pytest.skip(f"hard RLIMIT_NOFILE={hard} < 128")
    root = tmp_path / "store"
    bdir = root / "bundle"
    bdir.mkdir(parents=True)
    n_dirs = 200
    for i in range(n_dirs):
        (bdir / f"d{i:04d}").mkdir()
    # Empty dirs only → empty tree digest.
    pin = _bundle_pin("act.b", "art.b", "bundle", [])
    store = _store(
        root,
        [pin],
        bounds=_loose_bounds(max_directories=500, max_total_dirents=1000),
        freeze_parent=tmp_path / "freeze",
    )
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
        try:
            frozen = store.assert_bundle_digest("act.b", "art.b")
            try:
                assert frozen.file_count == 0
                assert frozen.digest == pin.digest
            finally:
                frozen.cleanup()
        finally:
            resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
    finally:
        store.close()


def test_bundle_non_utf8_entry_red(tmp_path: Path) -> None:
    root = tmp_path / "store"
    bdir = root / "bundle"
    bdir.mkdir(parents=True)
    bad_name = b"bad-\xff-name"
    bdir_fd = os.open(str(bdir), os.O_RDONLY)
    try:
        try:
            fd = os.open(
                bad_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=bdir_fd,
            )
            os.write(fd, b"x")
            os.close(fd)
        except OSError as exc:
            pytest.skip(f"filesystem rejects non-UTF-8 names: {exc}")
    finally:
        os.close(bdir_fd)

    pin = PinRecord(
        logical_activation_id="act.b",
        artifact_id="art.b",
        kind=ArtifactKind.BUNDLE,
        digest="00" * 32,
        store_relpath="bundle",
    )
    store = _store(root, [pin], freeze_parent=tmp_path / "freeze")
    try:
        with pytest.raises(ActivationRefusal) as ei:
            store.assert_bundle_digest("act.b", "art.b")
        assert ei.value.reason in (
            RefusalReason.NON_UTF8_ENTRY,
            RefusalReason.MALFORMED_ENTRY,
            RefusalReason.UNREADABLE,
        )
    finally:
        store.close()


# ---------------------------------------------------------------------------
# fd_dir / openat2 / policy surface
# ---------------------------------------------------------------------------


def test_list_dir_fd_oserror_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "store"
    root.mkdir()
    fd = os.open(str(root), os.O_RDONLY | os.O_DIRECTORY)

    def boom(_dir_fd: int) -> list[str]:
        raise OSError(5, "simulated readdir failure")

    try:
        with mock.patch.object(fd_dir_mod.os, "listdir", side_effect=boom):
            with pytest.raises(ActivationRefusal) as ei:
                list_dir_fd(fd)
            assert ei.value.reason is RefusalReason.UNREADABLE
    finally:
        os.close(fd)


def test_list_dir_fd_happy(tmp_path: Path) -> None:
    d = tmp_path / "d"
    d.mkdir()
    (d / "a").write_bytes(b"x")
    fd = os.open(str(d), os.O_RDONLY | os.O_DIRECTORY)
    try:
        names = set(list_dir_fd(fd))
        assert "a" in names
    finally:
        os.close(fd)


def test_no_repin_api_and_immutable_pins(tmp_path: Path) -> None:
    root = tmp_path / "store"
    root.mkdir()
    store = _store(root, [])
    try:
        assert not hasattr(store, "add_pin")
        assert not hasattr(store, "update_pin")
        assert not hasattr(store, "set_pin")
        assert not hasattr(store, "reload_pins")
        with pytest.raises(TypeError):
            store._pins[("a", "b")] = None  # type: ignore[index]
    finally:
        store.close()


def test_refusal_messages_are_path_safe(tmp_path: Path) -> None:
    root = tmp_path / "store"
    root.mkdir()
    secret = "/tmp/should-not-appear-in-refusal"
    pin = _sf_pin("act.x", "art.x", "missing.pt", b"x")
    store = _store(root, [pin])
    try:
        with pytest.raises(ActivationRefusal) as ei:
            store.assert_fixed_hash("act.x", "art.x")
        text = str(ei.value)
        assert secret not in text
        assert str(root) not in text
        assert "missing.pt" not in text
    finally:
        store.close()


def test_openat2_green_path_records_impl(tmp_path: Path) -> None:
    if not openat2_available():
        pytest.skip(
            "openat2 syscall unavailable — component-walk is the tested fallback"
        )
    root = tmp_path / "store"
    data = b"openat2-bytes"
    rel = "f/model.pt"
    _write(root / rel, data)
    pin = _sf_pin("act.o2", "art.o2", rel, data)
    store = _store(root, [pin], impl=ResolverImpl.OPENAT2)
    try:
        assert store.assert_fixed_hash("act.o2", "art.o2") == data
        assert last_open_impl() is ResolverImpl.OPENAT2
    finally:
        store.close()


def test_openat2_bundle_green_path_records_impl(tmp_path: Path) -> None:
    """Linux: BUNDLE root open must record OPENAT2 and yield a readable freeze."""
    if not openat2_available():
        pytest.skip(
            "openat2 syscall unavailable — component-walk is the tested fallback"
        )
    root = tmp_path / "store"
    member = b"openat2-bundle-member"
    _write(root / "bundle" / "m.bin", member)
    pin = _bundle_pin("act.o2b", "art.o2b", "bundle", [("m.bin", member)])
    store = _store(
        root,
        [pin],
        impl=ResolverImpl.OPENAT2,
        freeze_parent=tmp_path / "freeze",
    )
    try:
        frozen = store.assert_bundle_digest("act.o2b", "art.o2b")
        try:
            # last_open_impl must prove OPENAT2 (not a silent COMPONENT fallback).
            assert last_open_impl() is ResolverImpl.OPENAT2
            assert frozen.read_member("m.bin") == member
        finally:
            frozen.cleanup()
    finally:
        store.close()


def test_openat2_bundle_intermediate_symlink_refused(tmp_path: Path) -> None:
    """Linux: BUNDLE pin through intermediate symlink rejected via OPENAT2."""
    if not openat2_available():
        pytest.skip(
            "openat2 syscall unavailable — component-walk is the tested fallback"
        )
    import src.core.model_activation.resolver as resolver_mod

    root = tmp_path / "store"
    outside = tmp_path / "outside"
    # Real terminal bundle root lives outside the store; only the intermediate
    # pin component (family) is the symlink.
    _write(outside / "bundle" / "m.bin", b"evil")
    family = root / "family"
    family.parent.mkdir(parents=True, exist_ok=True)
    family.symlink_to(outside, target_is_directory=True)
    pin = _bundle_pin(
        "act.o2s", "art.o2s", "family/bundle", [("m.bin", b"evil")]
    )
    store = _store(root, [pin], impl=ResolverImpl.OPENAT2)
    try:
        # last_open_impl is set only on successful open; on refusal, prove the
        # OPENAT2 path ran (and COMPONENT did not) via spy — equivalent guard
        # against a silent component fallback passing this RED case.
        with mock.patch.object(
            resolver_mod,
            "_component_walk",
            side_effect=AssertionError(
                "silent COMPONENT fallback — OPENAT2 path required"
            ),
        ), mock.patch.object(
            resolver_mod,
            "_openat2_syscall",
            wraps=resolver_mod._openat2_syscall,
        ) as openat2_spy:
            with pytest.raises(ActivationRefusal) as ei:
                store.assert_bundle_digest("act.o2s", "art.o2s")
        assert openat2_spy.called, "OPENAT2 syscall must be attempted"
        assert ei.value.reason in (
            RefusalReason.SYMLINK_REJECTED,
            RefusalReason.NOT_DIRECTORY,
            RefusalReason.CONTAINMENT,
            RefusalReason.ARTIFACT_MISSING,
        )
    finally:
        store.close()


def test_component_and_openat2_agree_on_green_when_both(
    tmp_path: Path,
) -> None:
    if not openat2_available():
        pytest.skip("openat2 syscall unavailable")
    root = tmp_path / "store"
    data = b"parity-green"
    rel = "g/m.pt"
    _write(root / rel, data)
    pin = _sf_pin("act.p", "art.p", rel, data)
    for impl in (ResolverImpl.COMPONENT, ResolverImpl.OPENAT2):
        store = _store(root, [pin], impl=impl)
        try:
            assert store.assert_fixed_hash("act.p", "art.p") == data
            assert last_open_impl() is impl
        finally:
            store.close()
