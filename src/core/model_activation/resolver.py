"""Store-root-anchored path resolver.

Prefer Linux ``openat2`` via ``syscall(SYS_openat2, …)`` with
``RESOLVE_BENEATH|RESOLVE_NO_SYMLINKS`` when the kernel supports it;
otherwise descriptor-relative per-component ``openat``. Both paths run the
shared raw-pin pre-gate first. ``Path.resolve()`` + open is FORBIDDEN.

glibc does **not** export a portable ``openat2`` wrapper — use the raw
syscall (see openat2(2)).

Probe ``EPERM``/``EACCES`` means the syscall is unavailable for this process
→ component fallback. Runtime ``ENOSYS`` after a successful probe is
``INTERNAL``, resets state, and must **not** same-request fall back.
"""

from __future__ import annotations

import ctypes
import errno
import os
import platform
import stat
import threading
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
# O_NONBLOCK required before fstat so FIFOs do not block for a writer.
_FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC

# SYS_openat2 — Linux architectures that share the same number.
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


# Probe / runtime state for the syscall path — published under a lock as a
# coherent (state, fn, nr) snapshot so readers never mix generations.
_state_lock = threading.Lock()
_openat2_state: Optional[str] = None  # None=unprobed, "ok", "no"
_syscall_fn = None
_sys_openat2_nr: Optional[int] = None
# Last impl that successfully performed an open_pinned walk (for tests).
_last_open_impl: Optional[ResolverImpl] = None


def last_open_impl() -> Optional[ResolverImpl]:
    """Return the resolver impl used by the most recent successful open_pinned."""
    return _last_open_impl


def _publish_openat2_state(
    state: Optional[str],
    fn=None,  # type: ignore[no-untyped-def]
    nr: Optional[int] = None,
) -> None:
    """Publish coherent (state, fn, nr). ``state is None`` means unprobed."""
    global _openat2_state, _syscall_fn, _sys_openat2_nr
    with _state_lock:
        _openat2_state = state
        _syscall_fn = fn
        _sys_openat2_nr = nr


def _snapshot_openat2():  # type: ignore[no-untyped-def]
    with _state_lock:
        return _openat2_state, _syscall_fn, _sys_openat2_nr


def openat2_available() -> bool:
    """True when Linux SYS_openat2 works on this kernel (not merely a symbol).

    ``None`` (unprobed) re-runs the probe. Runtime ENOSYS after a prior
    successful probe resets to unprobed so a later request re-probes; it does
    **not** permanently publish ``"no"``.
    """
    state, _, _ = _snapshot_openat2()
    if state is not None:
        return state == "ok"

    if platform.system() != "Linux":
        _publish_openat2_state("no")
        return False
    machine = platform.machine().lower()
    nr = _SYS_OPENAT2_BY_MACHINE.get(machine)
    if nr is None:
        _publish_openat2_state("no")
        return False
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        syscall = libc.syscall
        syscall.restype = ctypes.c_long
        # Probe: openat2(-1, "", …) — expect failure, but not ENOSYS / EPERM / EACCES.
        how = _OpenHow(flags=os.O_RDONLY | os.O_CLOEXEC, mode=0, resolve=0)
        ctypes.set_errno(0)
        ret = syscall(
            ctypes.c_long(nr),
            ctypes.c_int(-1),
            b"",
            ctypes.byref(how),
            ctypes.c_size_t(ctypes.sizeof(how)),
        )
        if int(ret) >= 0:
            # Defensive: probe should not succeed, but never leak a live fd.
            try:
                os.close(int(ret))
            except OSError:
                pass
            _publish_openat2_state("ok", fn=syscall, nr=nr)
            return True
        err = ctypes.get_errno()
        if err in (errno.ENOSYS, errno.EPERM, errno.EACCES):
            # ENOSYS: kernel lacks syscall. EPERM/EACCES: process cannot use it
            # (seccomp / LSM) → treat as unavailable and use component fallback.
            _publish_openat2_state("no")
            return False
        # Any other error (EBADF, ENOENT, EINVAL, …) means the syscall exists.
        _publish_openat2_state("ok", fn=syscall, nr=nr)
        return True
    except (AttributeError, OSError, TypeError, ValueError):
        _publish_openat2_state("no")
        return False


def default_resolver_impl() -> ResolverImpl:
    return ResolverImpl.OPENAT2 if openat2_available() else ResolverImpl.COMPONENT


def _map_open_error(err: OSError, *, terminal: TerminalKind) -> ActivationRefusal:
    en = err.errno
    if en in (errno.ELOOP,) or en == getattr(errno, "EOPNOTSUPP", -1):
        return ActivationRefusal(RefusalReason.SYMLINK_REJECTED, "symlink")
    if en == errno.ENOTDIR:
        return ActivationRefusal(RefusalReason.NOT_DIRECTORY, "not directory")
    if en == errno.ENOENT:
        return ActivationRefusal(RefusalReason.ARTIFACT_MISSING, "missing")
    if en == errno.EACCES or en == errno.EPERM:
        return ActivationRefusal(RefusalReason.UNREADABLE, "permission")
    return ActivationRefusal(RefusalReason.CONTAINMENT, "open failed")


def _fstat_terminal(fd: int, terminal: TerminalKind) -> None:
    mapped: Optional[ActivationRefusal] = None
    st: Optional[os.stat_result] = None
    try:
        st = os.fstat(fd)
    except OSError:
        mapped = ActivationRefusal(RefusalReason.UNREADABLE, "fstat failed")
    if mapped is not None:
        try:
            os.close(fd)
        except OSError:
            pass
        raise mapped
    assert st is not None
    mode = st.st_mode
    if terminal == TerminalKind.REGULAR_FILE:
        if stat.S_ISLNK(mode):
            try:
                os.close(fd)
            except OSError:
                pass
            raise ActivationRefusal(RefusalReason.SYMLINK_REJECTED, "leaf symlink")
        if not stat.S_ISREG(mode):
            try:
                os.close(fd)
            except OSError:
                pass
            raise ActivationRefusal(RefusalReason.SPECIAL_FILE, "non-regular leaf")
        return
    if not stat.S_ISDIR(mode):
        try:
            os.close(fd)
        except OSError:
            pass
        raise ActivationRefusal(RefusalReason.NOT_DIRECTORY, "bundle root not dir")


def _openat2_syscall(
    store_root_fd: int,
    components: Sequence[str],
    terminal: TerminalKind,
) -> int:
    state, syscall_fn, sys_nr = _snapshot_openat2()
    if state != "ok" or syscall_fn is None or sys_nr is None:
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
    ret = syscall_fn(
        ctypes.c_long(sys_nr),
        ctypes.c_int(store_root_fd),
        path_b,
        ctypes.byref(how),
        ctypes.c_size_t(ctypes.sizeof(how)),
    )
    fd = int(ret)
    if fd < 0:
        err = ctypes.get_errno()
        if err == errno.ENOSYS:
            # Kernel lost the syscall after probe — reset to **unprobed**
            # (clear fn/nr) so the *next* request re-probes. Do NOT publish
            # permanent "no", and do NOT same-request fall back to component.
            _publish_openat2_state(None, fn=None, nr=None)
            raise ActivationRefusal(RefusalReason.INTERNAL, "openat2 ENOSYS")
        mapped = _map_open_error(OSError(err, os.strerror(err)), terminal=terminal)
        raise mapped
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
                mapped: Optional[ActivationRefusal] = None
                next_fd = -1
                try:
                    next_fd = os.open(name, _INTERMEDIATE_FLAGS, dir_fd=dir_fd)
                except OSError as exc:
                    mapped = _map_open_error(exc, terminal=terminal)
                if mapped is not None:
                    raise mapped
                owned.append(next_fd)
                dir_fd = next_fd
                mapped_st: Optional[ActivationRefusal] = None
                st: Optional[os.stat_result] = None
                try:
                    st = os.fstat(next_fd)
                except OSError:
                    mapped_st = ActivationRefusal(
                        RefusalReason.UNREADABLE, "fstat intermediate"
                    )
                if mapped_st is not None:
                    raise mapped_st
                assert st is not None
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
            mapped_leaf: Optional[ActivationRefusal] = None
            leaf_fd = -1
            try:
                leaf_fd = os.open(name, flags, dir_fd=dir_fd)
            except OSError as exc:
                mapped_leaf = _map_open_error(exc, terminal=terminal)
            if mapped_leaf is not None:
                raise mapped_leaf
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
    mapped: Optional[ActivationRefusal] = None
    fd = -1
    try:
        fd = os.open(name, flags, dir_fd=dir_fd)
    except OSError as exc:
        mapped = _map_open_error(exc, terminal=terminal)
    if mapped is not None:
        raise mapped
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
    mapped: Optional[ActivationRefusal] = None
    fd = -1
    try:
        fd = os.open(store_root, flags)
    except OSError:
        mapped = ActivationRefusal(RefusalReason.INTERNAL, "store root open failed")
    if mapped is not None:
        raise mapped
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
