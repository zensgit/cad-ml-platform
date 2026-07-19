"""Server-owned controlled store + Phase-A C1 activation core.

No caller path, no env path-swap, no network fetch, no runtime hot-swap,
no generic re-pin API. Default is fail-closed when a pin is absent.

Bundle activation is two-pass (ratified lock):
  1) descriptor-relative DFS metadata preflight — refuse metadata-detectable
     bounds **before any freeze copy**;
  2) DFS same-fd freeze with full bound recheck; destroy partial freeze on
     any mid-walk/copy failure; digest is recomputed from the **frozen**
     snapshot (not source-side copy buffers).
Live source directory fds are O(depth), never O(directory count).
"""

from __future__ import annotations

import errno
import os
import shutil
import stat
import tempfile
from types import MappingProxyType
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from src.core.model_activation.digest import (
    read_fd_bounded_once,
    sha256_hex,
    tree_digest_v1,
)
from src.core.model_activation.fd_dir import list_dir_fd
from src.core.model_activation.pin_domain import (
    validate_raw_pin,
    validate_readdir_name,
)
from src.core.model_activation.resolver import (
    ResolverImpl,
    open_pinned,
    open_store_root,
)
from src.core.model_activation.types import (
    ActivationRefusal,
    ArtifactKind,
    BoundPolicy,
    FrozenBundle,
    PinRecord,
    RefusalReason,
    TerminalKind,
    seal_freeze_tree,
)

PinKey = Tuple[str, str]

# Member open: O_NOFOLLOW so symlinks fail; O_NONBLOCK so FIFOs do not block.
_MEMBER_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC


def _pin_key(logical_activation_id: str, artifact_id: str) -> PinKey:
    return (logical_activation_id, artifact_id)


class _WalkCounters:
    __slots__ = (
        "file_count",
        "aggregate",
        "total_dirents",
        "dir_count",
    )

    def __init__(self) -> None:
        self.file_count = 0
        self.aggregate = 0
        self.total_dirents = 0
        self.dir_count = 0


def _map_member_open_error(exc: OSError) -> ActivationRefusal:
    if exc.errno in (errno.ELOOP, errno.EOPNOTSUPP) or exc.errno == getattr(
        errno, "EFTYPE", -1
    ):
        return ActivationRefusal(RefusalReason.SYMLINK_REJECTED, "bundle symlink")
    if exc.errno == errno.ENOENT:
        return ActivationRefusal(RefusalReason.UNREADABLE, "entry vanished")
    if exc.errno in (errno.EMFILE, errno.ENFILE):
        return ActivationRefusal(RefusalReason.UNREADABLE, "fd limit")
    return ActivationRefusal(RefusalReason.UNREADABLE, "member open")


def _relpath_join(prefix: str, name: str) -> str:
    return f"{prefix}/{name}" if prefix else name


def _check_relpath(relpath: str, bounds: BoundPolicy) -> None:
    try:
        rel_b = relpath.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ActivationRefusal(RefusalReason.NON_UTF8_ENTRY, "relpath") from exc
    if len(rel_b) > bounds.max_relpath_utf8_bytes:
        raise ActivationRefusal(RefusalReason.BUNDLE_RELPATH_BYTES, "relpath cap")


class ControlledStore:
    """Immutable pin table + store-root-anchored activation."""

    def __init__(
        self,
        store_root: str,
        pins: Iterable[PinRecord],
        *,
        bounds: Optional[BoundPolicy] = None,
        freeze_parent: Optional[str] = None,
        resolver_impl: Optional[ResolverImpl] = None,
    ) -> None:
        self._bounds = bounds or BoundPolicy()
        self._resolver_impl = resolver_impl
        self._freeze_parent = freeze_parent
        self._store_root_fd, self._store_root_abs = open_store_root(store_root)

        table: Dict[PinKey, PinRecord] = {}
        for pin in pins:
            if not isinstance(pin, PinRecord):
                raise TypeError("pins must be PinRecord instances")
            validate_raw_pin(pin.store_relpath)
            key = _pin_key(pin.logical_activation_id, pin.artifact_id)
            if key in table:
                raise ValueError(
                    "duplicate pin for "
                    f"{pin.logical_activation_id!r}/{pin.artifact_id!r}"
                )
            table[key] = pin
        self._pins: Mapping[PinKey, PinRecord] = MappingProxyType(table)

    @property
    def bounds(self) -> BoundPolicy:
        return self._bounds

    @property
    def pin_count(self) -> int:
        return len(self._pins)

    def close(self) -> None:
        fd = getattr(self, "_store_root_fd", None)
        if fd is not None and fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
            self._store_root_fd = -1

    def __del__(self) -> None:  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass

    def _lookup(self, logical_activation_id: str, artifact_id: str) -> PinRecord:
        if not logical_activation_id or not artifact_id:
            raise ActivationRefusal(RefusalReason.PIN_UNKNOWN, "empty id")
        pin = self._pins.get(_pin_key(logical_activation_id, artifact_id))
        if pin is None:
            raise ActivationRefusal(RefusalReason.PIN_ABSENT, "no pin")
        return pin

    def _impl(self) -> Optional[ResolverImpl]:
        return self._resolver_impl

    # ------------------------------------------------------------------
    # Single-file KIND
    # ------------------------------------------------------------------

    def assert_fixed_hash(
        self,
        logical_activation_id: str,
        artifact_id: str,
    ) -> bytes:
        pin = self._lookup(logical_activation_id, artifact_id)
        if pin.kind is not ArtifactKind.SINGLE_FILE:
            raise ActivationRefusal(RefusalReason.KIND_MISMATCH, "not single_file")

        fd = open_pinned(
            self._store_root_fd,
            pin.store_relpath,
            TerminalKind.REGULAR_FILE,
            impl=self._impl(),
        )
        try:
            try:
                st = os.fstat(fd)
            except OSError as exc:
                raise ActivationRefusal(
                    RefusalReason.UNREADABLE, "fstat failed"
                ) from exc
            if not stat.S_ISREG(st.st_mode):
                raise ActivationRefusal(
                    RefusalReason.NOT_REGULAR_FILE, "leaf not regular"
                )
            if st.st_size > self._bounds.max_single_file_bytes:
                raise ActivationRefusal(RefusalReason.OVERSIZE, "single-file cap")

            data = read_fd_bounded_once(
                fd,
                self._bounds.max_single_file_bytes,
                expect_size=st.st_size,
            )
        finally:
            try:
                os.close(fd)
            except OSError:
                pass

        digest = sha256_hex(data)
        if digest != pin.digest:
            raise ActivationRefusal(RefusalReason.DIGEST_MISMATCH, "sha256")
        return data

    def load_pinned_file(
        self,
        logical_activation_id: str,
        artifact_id: str,
    ) -> bytes:
        return self.assert_fixed_hash(logical_activation_id, artifact_id)

    # ------------------------------------------------------------------
    # Bundle KIND
    # ------------------------------------------------------------------

    def assert_bundle_digest(
        self,
        logical_activation_id: str,
        artifact_id: str,
    ) -> FrozenBundle:
        pin = self._lookup(logical_activation_id, artifact_id)
        if pin.kind is not ArtifactKind.BUNDLE:
            raise ActivationRefusal(RefusalReason.KIND_MISMATCH, "not bundle")

        root_fd = open_pinned(
            self._store_root_fd,
            pin.store_relpath,
            TerminalKind.DIRECTORY,
            impl=self._impl(),
        )
        freeze_root: Optional[str] = None
        freeze_dir_fd = -1
        try:
            # Pass 1: metadata preflight — no freeze writes.
            self._preflight_tree(root_fd)

            # Pass 2: same-fd DFS freeze with full recheck.
            freeze_root = self._make_freeze_dir()
            self._freeze_tree(root_fd, freeze_root)

            # Seal so casual path-based in-place writes fail; immutability for
            # activation still rests on the held dir fd + digest-from-freeze.
            seal_freeze_tree(freeze_root)

            # Open freeze root and recompute digest from the frozen snapshot.
            try:
                freeze_dir_fd = os.open(
                    freeze_root,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
                )
            except OSError as exc:
                raise ActivationRefusal(
                    RefusalReason.FREEZE_FAILED, "open freeze"
                ) from exc

            entries, file_count, aggregate = self._digest_frozen_tree(freeze_dir_fd)
            digest = tree_digest_v1(entries)
            if digest != pin.digest:
                raise ActivationRefusal(
                    RefusalReason.DIGEST_MISMATCH, "tree-digest-v1"
                )

            handle = FrozenBundle(
                dir_fd=freeze_dir_fd,
                backing_path=freeze_root,
                digest=digest,
                file_count=file_count,
                aggregate_bytes=aggregate,
            )
            freeze_dir_fd = -1  # owned by handle
            freeze_root = None
            return handle
        except Exception:
            if freeze_dir_fd >= 0:
                try:
                    os.close(freeze_dir_fd)
                except OSError:
                    pass
            if freeze_root is not None:
                try:
                    from src.core.model_activation.types import (
                        make_tree_deletable,
                    )

                    make_tree_deletable(freeze_root)
                except Exception:
                    pass
                shutil.rmtree(freeze_root, ignore_errors=True)
            raise
        finally:
            try:
                os.close(root_fd)
            except OSError:
                pass

    def load_pinned_bundle(
        self,
        logical_activation_id: str,
        artifact_id: str,
    ) -> FrozenBundle:
        return self.assert_bundle_digest(logical_activation_id, artifact_id)

    def _make_freeze_dir(self) -> str:
        parent = self._freeze_parent
        if parent is None:
            parent = tempfile.gettempdir()
        try:
            os.makedirs(parent, mode=0o700, exist_ok=True)
        except OSError as exc:
            raise ActivationRefusal(
                RefusalReason.FREEZE_FAILED, "freeze parent"
            ) from exc
        try:
            path = tempfile.mkdtemp(prefix="cadml-freeze-", dir=parent)
            os.chmod(path, 0o700)
        except OSError as exc:
            raise ActivationRefusal(
                RefusalReason.FREEZE_FAILED, "mkdtemp"
            ) from exc
        return path

    # ----- shared DFS member iteration (O(depth) source fds) -----

    def _iter_members_dfs(
        self,
        dir_fd: int,
        *,
        prefix: str,
        depth: int,
        counters: _WalkCounters,
        on_file: Callable[[int, os.stat_result, str], None],
        on_dir_enter: Optional[Callable[[str, str], None]] = None,
        freeze_dir: Optional[str] = None,
    ) -> None:
        """Depth-first walk: close each child dir before opening the next sibling.

        Live source directory fds ≤ O(depth). Never stacks all sibling dir fds.
        """
        bounds = self._bounds
        if depth > bounds.max_depth:
            raise ActivationRefusal(RefusalReason.BUNDLE_DEPTH, "depth cap")

        names = list_dir_fd(dir_fd)
        for name in names:
            if name in (".", ".."):
                continue

            counters.total_dirents += 1
            if counters.total_dirents > bounds.max_total_dirents:
                raise ActivationRefusal(
                    RefusalReason.BUNDLE_TOTAL_DIRENTS, "dirent cap"
                )

            validate_readdir_name(name)
            relpath = _relpath_join(prefix, name)
            _check_relpath(relpath, bounds)

            try:
                mfd = os.open(name, _MEMBER_FLAGS, dir_fd=dir_fd)
            except OSError as exc:
                raise _map_member_open_error(exc) from exc

            try:
                try:
                    st = os.fstat(mfd)
                except OSError as exc:
                    raise ActivationRefusal(
                        RefusalReason.UNREADABLE, "fstat member"
                    ) from exc
                mode = st.st_mode

                if stat.S_ISDIR(mode):
                    counters.dir_count += 1
                    if counters.dir_count > bounds.max_directories:
                        raise ActivationRefusal(
                            RefusalReason.BUNDLE_DIRECTORY_COUNT, "dir cap"
                        )
                    child_depth = depth + 1
                    if child_depth > bounds.max_depth:
                        raise ActivationRefusal(
                            RefusalReason.BUNDLE_DEPTH, "depth cap"
                        )
                    child_freeze: Optional[str] = None
                    if freeze_dir is not None:
                        child_freeze = os.path.join(freeze_dir, name)
                        try:
                            os.mkdir(child_freeze, 0o700)
                        except OSError as exc:
                            raise ActivationRefusal(
                                RefusalReason.FREEZE_FAILED, "mkdir"
                            ) from exc
                    if on_dir_enter is not None:
                        on_dir_enter(relpath, name)
                    # Recurse while holding only this child dir fd (+ ancestors).
                    self._iter_members_dfs(
                        mfd,
                        prefix=relpath,
                        depth=child_depth,
                        counters=counters,
                        on_file=on_file,
                        on_dir_enter=on_dir_enter,
                        freeze_dir=child_freeze,
                    )
                    # Close before next sibling — O(depth) live source fds.
                elif not stat.S_ISREG(mode):
                    raise ActivationRefusal(
                        RefusalReason.SPECIAL_FILE, "non-regular"
                    )
                else:
                    counters.file_count += 1
                    if counters.file_count > bounds.max_bundle_file_count:
                        raise ActivationRefusal(
                            RefusalReason.BUNDLE_FILE_COUNT, "file cap"
                        )
                    if st.st_size > bounds.max_bundle_per_file_bytes:
                        raise ActivationRefusal(
                            RefusalReason.BUNDLE_PER_FILE_BYTES, "per-file cap"
                        )
                    if (
                        counters.aggregate + st.st_size
                        > bounds.max_bundle_aggregate_bytes
                    ):
                        raise ActivationRefusal(
                            RefusalReason.BUNDLE_AGGREGATE_BYTES, "aggregate cap"
                        )
                    # on_file may read the fd; still counted via st_size first.
                    on_file(mfd, st, relpath)
                    counters.aggregate += st.st_size
                    if counters.aggregate > bounds.max_bundle_aggregate_bytes:
                        raise ActivationRefusal(
                            RefusalReason.BUNDLE_AGGREGATE_BYTES, "aggregate cap"
                        )
            finally:
                try:
                    os.close(mfd)
                except OSError:
                    pass

    def _preflight_tree(self, root_fd: int) -> None:
        """Metadata-only DFS: refuse bounds before any freeze copy."""
        counters = _WalkCounters()

        def on_file(mfd: int, st: os.stat_result, relpath: str) -> None:
            # Preflight: metadata only — do not read body bytes.
            return

        self._iter_members_dfs(
            root_fd,
            prefix="",
            depth=0,
            counters=counters,
            on_file=on_file,
            freeze_dir=None,
        )

    def _freeze_tree(self, root_fd: int, freeze_root: str) -> None:
        """Second pass: same-fd bounded copy into freeze_root with recheck."""
        counters = _WalkCounters()
        bounds = self._bounds

        def on_file(mfd: int, st: os.stat_result, relpath: str) -> None:
            data = read_fd_bounded_once(
                mfd,
                bounds.max_bundle_per_file_bytes,
                expect_size=st.st_size,
            )
            # Write under freeze using the relpath components (already validated).
            dest = os.path.join(freeze_root, *relpath.split("/"))
            parent = os.path.dirname(dest)
            if parent and parent != freeze_root:
                # Parent dirs created on dir enter; defend if empty.
                os.makedirs(parent, mode=0o700, exist_ok=True)
            try:
                dfd = os.open(
                    dest,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
                    0o600,
                )
            except OSError as exc:
                raise ActivationRefusal(
                    RefusalReason.FREEZE_FAILED, "create dest"
                ) from exc
            try:
                view = memoryview(data)
                offset = 0
                while offset < len(data):
                    written = os.write(dfd, view[offset:])
                    if written <= 0:
                        raise ActivationRefusal(
                            RefusalReason.FREEZE_FAILED, "write"
                        )
                    offset += written
            except OSError as exc:
                raise ActivationRefusal(
                    RefusalReason.FREEZE_FAILED, "write"
                ) from exc
            finally:
                try:
                    os.close(dfd)
                except OSError:
                    pass

        self._iter_members_dfs(
            root_fd,
            prefix="",
            depth=0,
            counters=counters,
            on_file=on_file,
            freeze_dir=freeze_root,
        )

    def _digest_frozen_tree(
        self, freeze_dir_fd: int
    ) -> Tuple[List[Tuple[str, str]], int, int]:
        """Recompute tree-digest-v1 entries by reading the **frozen** snapshot.

        Never uses source-side copy buffers. Descriptor-relative from freeze fd.
        """
        bounds = self._bounds
        entries: List[Tuple[str, str]] = []
        file_count = 0
        aggregate = 0

        def walk(dir_fd: int, prefix: str, depth: int) -> None:
            nonlocal file_count, aggregate
            if depth > bounds.max_depth:
                raise ActivationRefusal(RefusalReason.BUNDLE_DEPTH, "freeze depth")
            names = list_dir_fd(dir_fd)
            for name in names:
                if name in (".", ".."):
                    continue
                validate_readdir_name(name)
                relpath = _relpath_join(prefix, name)
                _check_relpath(relpath, bounds)
                try:
                    mfd = os.open(
                        name,
                        os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
                        dir_fd=dir_fd,
                    )
                except OSError as exc:
                    raise ActivationRefusal(
                        RefusalReason.FREEZE_FAILED, "freeze re-open"
                    ) from exc
                try:
                    st = os.fstat(mfd)
                    if stat.S_ISDIR(st.st_mode):
                        walk(mfd, relpath, depth + 1)
                    elif stat.S_ISREG(st.st_mode):
                        data = read_fd_bounded_once(
                            mfd,
                            max(bounds.max_bundle_per_file_bytes, 1),
                            expect_size=st.st_size,
                        )
                        file_count += 1
                        aggregate += len(data)
                        entries.append((relpath, sha256_hex(data)))
                    else:
                        raise ActivationRefusal(
                            RefusalReason.SPECIAL_FILE, "freeze non-regular"
                        )
                finally:
                    try:
                        os.close(mfd)
                    except OSError:
                        pass

        walk(freeze_dir_fd, "", 0)
        return entries, file_count, aggregate


__all__ = ["ControlledStore", "PinKey"]
