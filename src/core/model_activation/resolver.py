"""Store-root-anchored path resolver.

Prefer Linux ``openat2`` via ``syscall(SYS_openat2, …)`` with
``RESOLVE_BENEATH|RESOLVE_NO_SYMLINKS`` when the kernel supports it;
otherwise descriptor-relative per-component ``openat``. Both paths run the
shared raw-pin pre-gate first. ``Path.resolve()`` + open is FORBIDDEN.

glibc does **not** export a portable ``openat2`` wrapper — use the raw
syscall (see openat2(2)).
"""

from __future__ import annotations

import ctypes
import errno
import os
import platform
import stat
from enum import Enum
from typing import Optional, Sequence, Tuple

from src.core.model_activation.pin_domain import validate_raw_pin
from src.core.model_activation.types import (
    ActivationRefusal,
    RefusalReason,
    TerminalKind,
)

# Linux openat2 resolve flags (uapi/linux/openat2.h)
_RESOLVE_NO_SYMLINKS = 0x04
_RESOLVE_BENEATH = 0x08

# Common open flags for leaves / intermediates
_INTERMEDIATE_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC

# SYS_openat2 — Linux architectures that share the same number.
# (x86_64, aarch64, riscv64, s390x use 437; i386 uses 437 as well in modern kernels.)
_SYS_OPENAT2_BY_MACHINE = {
    "x86_64": 437,
    "amd64": 437,
    "aarch64": 437,
    "arm64": 437,
    "riscv64": 437,
    "s390x": 437,
    "ppc64le": 437,
    "i386": 437,
    "i686": 437,
}


class ResolverImpl(str, Enum):
    OPENAT2 = "openat2"
    COMPONENT = "component"


class _OpenHow(ctypes.Structure):
    _fields_ = [
        ("flags", ctypes.c_uint64),
        ("mode", ctypes.c_uint64),
        ("resolve", ctypes.c_uint64),
    ]


# Probe / runtime state for the syscall path.
_openat2_state: Optional[str] = None  # None=unprobed, "ok", "no"
_syscall_fn = None
_sys_openat2_nr: Optional[int] = None
# Last impl that successfully performed an open_pinned walk (for tests).
_last_open_impl: Optional[ResolverImpl] = None


def last_open_impl() -> Optional[ResolverImpl]:
    """Return the resolver impl used by the most recent successful open_pinned."""
    return _last_open_impl


def openat2_available() -> bool:
    """True when Linux SYS_openat2 works on this kernel (not merely a symbol)."""
    global _openat2_state, _syscall_fn, _sys_openat2_nr
    if _openat2_state is not None:
        return _openat2_state == "ok"
    _openat2_state = "no"
    if platform.system() != "Linux":
        return False
    machine = platform.machine().lower()
    nr = _SYS_OPENAT2_BY_MACHINE.get(machine)
    if nr is None:
        return False
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        syscall = libc.syscall
        syscall.restype = ctypes.c_long
        # Probe: openat2(-1, "", …) — expect failure, but not ENOSYS.
        how = _OpenHow(flags=os.O_RDONLY | os.O_CLOEXEC, mode=0, resolve=0)
        ctypes.set_errno(0)
        # Build a flexible syscall call: SYS_openat2, dirfd, path, how*, size
        ret = syscall(
            ctypes.c_long(nr),
            ctypes.c_int(-1),
            b"",
            ctypes.byref(how),
            ctypes.c_size_t(ctypes.sizeof(how)),
        )
        err = ctypes.get_errno() if ret < 0 else 0
        if ret < 0 and err == errno.ENOSYS:
            return False
        # Any other error (EBADF, ENOENT, EINVAL, …) means the syscall exists.
        _syscall_fn = syscall
        _sys_openat2_nr = nr
        _openat2_state = "ok"
        return True
    except (AttributeError, OSError, TypeError, ValueError):
        return False


def default_resolver_impl() -> ResolverImpl:
    return ResolverImpl.OPENAT2 if openat2_available() else ResolverImpl.COMPONENT


def _map_open_error(err: OSError, *, terminal: TerminalKind) -> ActivationRefusal:
    en = err.errno
    if en in (errno.ELOOP, errno.ENOTDIR) or en == getattr(errno, "EOPNOTSUPP", -1):
        return ActivationRefusal(RefusalReason.SYMLINK_REJECTED, "symlink or notdir")
    if en == errno.ENOENT:
        return ActivationRefusal(RefusalReason.ARTIFACT_MISSING, "missing")
    if en == errno.EACCES or en == errno.EPERM:
        return ActivationRefusal(RefusalReason.UNREADABLE, "permission")
    if en == errno.ENOTDIR:
        return ActivationRefusal(RefusalReason.NOT_DIRECTORY, "not directory")
    if terminal == TerminalKind.REGULAR_FILE:
        return ActivationRefusal(RefusalReason.CONTAINMENT, "open failed")
    return ActivationRefusal(RefusalReason.CONTAINMENT, "open failed")


def _fstat_terminal(fd: int, terminal: TerminalKind) -> None:
    try:
        st = os.fstat(fd)
    except OSError as exc:
        os.close(fd)
        raise ActivationRefusal(RefusalReason.UNREADABLE, "fstat failed") from exc
    mode = st.st_mode
    if terminal == TerminalKind.REGULAR_FILE:
        if stat.S_ISLNK(mode):
            os.close(fd)
            raise ActivationRefusal(RefusalReason.SYMLINK_REJECTED, "leaf symlink")
        if not stat.S_ISREG(mode):
            os.close(fd)
            raise ActivationRefusal(RefusalReason.SPECIAL_FILE, "non-regular leaf")
        return
    if not stat.S_ISDIR(mode):
        os.close(fd)
        raise ActivationRefusal(RefusalReason.NOT_DIRECTORY, "bundle root not dir")


def _openat2_syscall(
    store_root_fd: int,
    components: Sequence[str],
    terminal: TerminalKind,
) -> int:
    if _syscall_fn is None or _sys_openat2_nr is None:
        raise ActivationRefusal(RefusalReason.INTERNAL, "openat2 unavailable")
    relpath = "/".join(components)
    flags = _FILE_FLAGS if terminal == TerminalKind.REGULAR_FILE else _DIR_FLAGS
    how = _OpenHow(
        flags=flags,
        mode=0,
        resolve=_RESOLVE_BENEATH | _RESOLVE_NO_SYMLINKS,
    )
    path_b = relpath.encode("utf-8", errors="surrogateescape")
    ctypes.set_errno(0)
    ret = _syscall_fn(
        ctypes.c_long(_sys_openat2_nr),
        ctypes.c_int(store_root_fd),
        path_b,
        ctypes.byref(how),
        ctypes.c_size_t(ctypes.sizeof(how)),
    )
    fd = int(ret)
    if fd < 0:
        err = ctypes.get_errno()
        if err == errno.ENOSYS:
            # Kernel lost the syscall after probe — force re-probe next time.
            global _openat2_state
            _openat2_state = "no"
            raise ActivationRefusal(RefusalReason.INTERNAL, "openat2 ENOSYS")
        raise _map_open_error(OSError(err, os.strerror(err)), terminal=terminal)
    try:
        _fstat_terminal(fd, terminal)
    except ActivationRefusal:
        raise
    return fd


def _component_walk(
    store_root_fd: int,
    components: Sequence[str],
    terminal: TerminalKind,
) -> int:
    if not components:
        raise ActivationRefusal(RefusalReason.RAW_PIN_INVALID, "empty components")

    dir_fd = store_root_fd
    owned: list[int] = []
    try:
        for i, name in enumerate(components):
            is_last = i == len(components) - 1
            if not is_last:
                try:
                    next_fd = os.open(name, _INTERMEDIATE_FLAGS, dir_fd=dir_fd)
                except OSError as exc:
                    raise _map_open_error(exc, terminal=terminal) from exc
                owned.append(next_fd)
                dir_fd = next_fd
                try:
                    st = os.fstat(next_fd)
                except OSError as exc:
                    raise ActivationRefusal(
                        RefusalReason.UNREADABLE, "fstat intermediate"
                    ) from exc
                if not stat.S_ISDIR(st.st_mode):
                    raise ActivationRefusal(
                        RefusalReason.NOT_DIRECTORY, "intermediate not dir"
                    )
                continue

            flags = (
                _FILE_FLAGS
                if terminal == TerminalKind.REGULAR_FILE
                else _DIR_FLAGS
            )
            try:
                leaf_fd = os.open(name, flags, dir_fd=dir_fd)
            except OSError as exc:
                raise _map_open_error(exc, terminal=terminal) from exc
            try:
                _fstat_terminal(leaf_fd, terminal)
            except ActivationRefusal:
                raise
            return leaf_fd
    finally:
        for fd in reversed(owned):
            try:
                os.close(fd)
            except OSError:
                pass

    raise ActivationRefusal(RefusalReason.INTERNAL, "walk produced no fd")


def open_pinned(
    store_root_fd: int,
    raw_pin: str,
    terminal: TerminalKind,
    *,
    impl: Optional[ResolverImpl] = None,
) -> int:
    """Open a pin relative to an already-open store-root directory fd.

    Returns an owned fd. Caller must close it. Shared raw-pin pre-gate always
    runs first for both implementations.
    """
    global _last_open_impl
    components = validate_raw_pin(raw_pin)
    chosen = impl or default_resolver_impl()
    if chosen is ResolverImpl.OPENAT2:
        if not openat2_available():
            raise ActivationRefusal(
                RefusalReason.INTERNAL, "openat2 unavailable"
            )
        fd = _openat2_syscall(store_root_fd, components, terminal)
        _last_open_impl = ResolverImpl.OPENAT2
        return fd
    if chosen is ResolverImpl.COMPONENT:
        fd = _component_walk(store_root_fd, components, terminal)
        _last_open_impl = ResolverImpl.COMPONENT
        return fd
    raise ActivationRefusal(RefusalReason.INTERNAL, "unknown resolver impl")


def open_under_dir(
    dir_fd: int,
    name: str,
    *,
    terminal: TerminalKind = TerminalKind.REGULAR_FILE,
) -> int:
    """Open a single readdir component relative to ``dir_fd`` (O_NOFOLLOW)."""
    flags = _FILE_FLAGS if terminal == TerminalKind.REGULAR_FILE else _DIR_FLAGS
    try:
        fd = os.open(name, flags, dir_fd=dir_fd)
    except OSError as exc:
        raise _map_open_error(exc, terminal=terminal) from exc
    try:
        _fstat_terminal(fd, terminal)
    except ActivationRefusal:
        raise
    return fd


def open_store_root(store_root: str) -> Tuple[int, str]:
    """Open the server-owned store root as a directory fd."""
    if not isinstance(store_root, str) or store_root == "":
        raise ActivationRefusal(RefusalReason.INTERNAL, "store root required")
    if "\x00" in store_root:
        raise ActivationRefusal(RefusalReason.INTERNAL, "store root nul")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    try:
        fd = os.open(store_root, flags)
    except OSError as exc:
        raise ActivationRefusal(
            RefusalReason.INTERNAL, "store root open failed"
        ) from exc
    try:
        abspath = os.path.abspath(store_root)
    except OSError:
        abspath = store_root
    return fd, abspath


__all__ = [
    "ResolverImpl",
    "default_resolver_impl",
    "last_open_impl",
    "open_pinned",
    "open_store_root",
    "open_under_dir",
    "openat2_available",
]
