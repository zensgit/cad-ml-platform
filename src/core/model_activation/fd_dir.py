"""Directory enumeration helpers for the C1 activation core.

**Security traversal (C1):** use :func:`scandir_dir_fd` only — a lazy, fd-bound
``os.scandir`` iterator (native dir fd, then ``/dev/fd/{n}`` / ``/proc/self/fd/{n}``).
Never materialises an unbounded directory listing. If no fd-bound scandir can
be established, fail closed with a path-safe :class:`ActivationRefusal`.

**Compatibility:** :func:`list_dir_fd` still materialises via ``os.listdir`` and
is **unused by C1 security traversal** (preflight / freeze / digest / seal /
reconcile / cleanup). It remains only as a non-security helper for callers or
tests that explicitly want a full name list; do not use it on bounded walks.
"""

from __future__ import annotations

import os
from typing import Any, Iterator, List, Optional, Union

from src.core.model_activation.types import ActivationRefusal, RefusalReason


def scandir_dir_fd(dir_fd: int) -> Any:
    """Return a lazy ``os.scandir`` iterator bound to an already-open dir fd.

    Prefer native ``os.scandir(dir_fd)`` when the runtime accepts an int fd.
    Fall back only to fd-bound path forms (``/dev/fd/{n}``, ``/proc/self/fd/{n}``)
    that still refer to the held descriptor.

    **Never** materialises ``os.listdir``. Callers must close the iterator on
    every exit path (``try``/``finally``: ``it.close()``).

    Raises:
        ActivationRefusal: path-safe; ``__cause__``/``__context__`` not chained
        from OSError (map inside except, raise outside where applicable).
    """
    if not isinstance(dir_fd, int) or dir_fd < 0:
        raise ActivationRefusal(RefusalReason.INTERNAL, "bad dir_fd")

    mapped: Optional[ActivationRefusal] = None
    it: Any = None
    try:
        it = os.scandir(dir_fd)
    except TypeError:
        it = None
    except OSError:
        it = None
    if it is not None:
        return it

    for template in (f"/dev/fd/{dir_fd}", f"/proc/self/fd/{dir_fd}"):
        try:
            it = os.scandir(template)
        except OSError:
            it = None
            continue
        if it is not None:
            return it

    mapped = ActivationRefusal(RefusalReason.UNREADABLE, "scandir unavailable")
    raise mapped


def dirent_name_str(name: Union[str, bytes, Any]) -> str:
    """Validate one readdir/scandir entry name; return a str (path-safe refusals)."""
    if isinstance(name, bytes):
        mapped_u: Optional[ActivationRefusal] = None
        decoded: Optional[str] = None
        try:
            decoded = name.decode("utf-8")
        except UnicodeDecodeError:
            mapped_u = ActivationRefusal(RefusalReason.NON_UTF8_ENTRY, "entry name")
        if mapped_u is not None:
            raise mapped_u
        assert decoded is not None
        name = decoded
    if not isinstance(name, str):
        raise ActivationRefusal(RefusalReason.MALFORMED_ENTRY, "name type")
    if "\x00" in name:
        raise ActivationRefusal(RefusalReason.MALFORMED_ENTRY, "nul name")
    return name


def iter_dirent_names(dir_fd: int) -> Iterator[str]:
    """Yield directory entry names lazily from :func:`scandir_dir_fd`.

    Closes the underlying scandir iterator when the generator is exhausted,
    closed, or garbage-collected after partial consumption. Prefer an explicit
    ``scandir_dir_fd`` + ``try``/``finally`` when the caller needs hard close
    guarantees on exception mid-walk (C1 walks do that).
    """
    it = scandir_dir_fd(dir_fd)
    try:
        for entry in it:
            yield dirent_name_str(entry.name)
    finally:
        try:
            it.close()
        except Exception:
            pass


def list_dir_fd(dir_fd: int) -> List[str]:
    """Materialising name list via ``os.listdir(dir_fd)``.

    **Not used by C1 security traversal.** Directory-bomb bounds
    (``max_total_dirents``) require lazy enumeration; use
    :func:`scandir_dir_fd` on any preflight / freeze / digest / seal /
    reconcile / cleanup path. Retained only as a non-security compatibility
    helper.
    """
    if not isinstance(dir_fd, int) or dir_fd < 0:
        raise ActivationRefusal(RefusalReason.INTERNAL, "bad dir_fd")
    mapped: Optional[ActivationRefusal] = None
    names: Optional[List[Any]] = None
    try:
        names = os.listdir(dir_fd)
    except OSError:
        mapped = ActivationRefusal(RefusalReason.UNREADABLE, "readdir")
    except TypeError:
        mapped = ActivationRefusal(RefusalReason.UNREADABLE, "readdir")
    if mapped is not None:
        raise mapped
    if not isinstance(names, list):
        raise ActivationRefusal(RefusalReason.UNREADABLE, "readdir type")

    out: List[str] = []
    for name in names:
        out.append(dirent_name_str(name))
    return out


__all__ = [
    "dirent_name_str",
    "iter_dirent_names",
    "list_dir_fd",
    "scandir_dir_fd",
]
