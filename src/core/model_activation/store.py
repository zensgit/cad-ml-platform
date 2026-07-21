"""Server-owned controlled store + Phase-A C1 activation core.

No caller path, no env path-swap, no network fetch, no runtime hot-swap,
no generic re-pin API. Default is fail-closed when a pin is absent.

Bundle activation is two-pass (ratified lock):
  1) descriptor-relative DFS metadata preflight — refuse metadata-detectable
     bounds **before any freeze copy**;
  2) DFS same-fd freeze with full bound recheck from the already-open source
     member fd (no path re-open between fstat and copy); destroy partial freeze
     on any mid-walk/copy failure; digest is recomputed from the **frozen**
     snapshot (not source-side copy buffers).

Freeze resources are owned by :class:`FreezeResourceLease`: an initially empty
lease, creation-time identity ledger, reserve-before-create, adopt-before-
finalize, and descriptor-relative scrub (never ``shutil.rmtree(path)``).

Live source directory fds are O(depth), never O(directory count).
"""

from __future__ import annotations

import errno
import os
import secrets
import stat
import threading
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
)

from src.core.model_activation.digest import (
    read_fd_bounded_once,
    sha256_hex,
    tree_digest_v1,
)
from src.core.model_activation.fd_dir import dirent_name_str
from src.core.model_activation.fd_dir import scandir_dir_fd as _scandir_dir_fd
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
    seal_freeze_tree_fd,
)

PinKey = Tuple[str, str]

# Member open: O_NOFOLLOW so symlinks fail; O_NONBLOCK so FIFOs do not block
# before fstat. Applied to source walk leaves and intermediates-as-files.
_MEMBER_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
# Freeze destination leaf create/open (write path uses O_CREAT|O_EXCL).
_FREEZE_LEAF_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
_FREEZE_CREATE_FLAGS = (
    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NONBLOCK
)

# Hard cap on creation-time identity inventory (files + dirs). Tests may lower.
_MAX_FREEZE_IDENTITY_INVENTORY = 100_000

# Cleanup scan: refuse to materialize more than cap+1 entries under one dir.
_CLEANUP_SCANDIR_CAP = _MAX_FREEZE_IDENTITY_INVENTORY


def _pin_key(logical_activation_id: str, artifact_id: str) -> PinKey:
    return (logical_activation_id, artifact_id)


@dataclass(frozen=True)
class _FileId:
    dev: int
    ino: int


@dataclass
class _PendingNode:
    """Adopted fd not yet finalized into the creation-time ledger."""

    fd: int
    parent_fd: int
    name: str
    is_dir: bool
    reserved: bool = True


@dataclass(eq=False)
class FreezeResourceLease:
    """Service-private freeze tree lease with creation-time identity ledger.

    ``eq=False``: ownership lists (e.g. store ``_pending_leases``) use object
    identity, never field-value coalescence of equal-looking leases.

    Starts empty. Callers must supply an explicit trusted freeze parent — there
    is **no** default temp directory. Parent must be absolute; it is opened
    ``O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC`` and validated (owner euid,
    ``mode & 0o077 == 0``) via **fstat of that fd**. The freeze root is created
    only with ``os.mkdir(name, dir_fd=parent_fd)`` + openat — never
    ``tempfile.mkdtemp`` or pathname ``chmod`` on the parent path (path-swap
    resistant).

    Protocol for every create (file or directory):
      1. reserve ledger capacity;
      2. mkdir / O_CREAT (descriptor-relative);
      3. adopt the newly created fd into the lease **before** fstat/finalize;
      4. fstat and commit identity into the creation-time ledger;
      5. on fstat failure the pending fd remains owned; release retries fstat
         without name-delete or closing the pending fd until fstat succeeds
         and name identity is proven equal, then closes once and scrubs.

    On dest-dir open failure after mkdir: **cancel reservation only** — never
    name-stat/commit the object currently at the name (it may be foreign).

    Cleanup is descriptor-relative (``unlinkat`` / ``rmdir`` via dir fds and
    lazy ``os.scandir``), never ``shutil.rmtree(path)``. Cleanup never adopts
    cleanup-time objects into the ledger. Foreign (observed − ledger) objects
    are left untouched. A safe empty directory shell may remain only where
    portable atomic mkdir+fd binding is impossible (documented residual).
    """

    _root_fd: int = -1
    _parent_fd: int = -1
    _root_name: str = ""
    _backing_path: str = ""
    _ledger: Dict[_FileId, bool] = field(default_factory=dict)
    _pending: List[_PendingNode] = field(default_factory=list)
    _reservations: int = 0
    _released: bool = False
    _cleanup_complete: bool = True
    _max_inventory: int = field(default_factory=lambda: _MAX_FREEZE_IDENTITY_INVENTORY)
    _root_detached: bool = False
    # Serialize release() for all callers (store drain, bundle cleanup, concurrent).
    _release_lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False, hash=False
    )
    # Test/production hook: (parent_dir_fd, name, fid, is_dir) after identity
    # observation and immediately before the pre-remove identity re-check.
    _pre_remove_hook: Optional[Callable[[int, str, "_FileId", bool], None]] = None

    # ----- capacity -----

    def _slots_used(self) -> int:
        return len(self._ledger) + self._reservations + len(self._pending)

    def reserve_identity_slot(self) -> None:
        if self._slots_used() >= self._max_inventory:
            raise ActivationRefusal(
                RefusalReason.FREEZE_FAILED, "identity inventory full"
            )
        self._reservations += 1

    def cancel_reservation(self) -> None:
        if self._reservations > 0:
            self._reservations -= 1

    def _adopt_pending(
        self, fd: int, parent_fd: int, name: str, *, is_dir: bool
    ) -> _PendingNode:
        """Adopt fd under an existing reservation (reservation → pending)."""
        if self._reservations <= 0:
            # Defensive: adopt still takes a slot.
            self.reserve_identity_slot()
        self._reservations -= 1
        node = _PendingNode(fd=fd, parent_fd=parent_fd, name=name, is_dir=is_dir)
        self._pending.append(node)
        return node

    def _finalize_pending(self, node: _PendingNode, st: os.stat_result) -> _FileId:
        is_dir = stat.S_ISDIR(st.st_mode)
        if is_dir != node.is_dir:
            raise ActivationRefusal(RefusalReason.FREEZE_FAILED, "identity kind")
        fid = _FileId(int(st.st_dev), int(st.st_ino))
        if fid in self._ledger:
            raise ActivationRefusal(RefusalReason.FREEZE_FAILED, "duplicate identity")
        self._ledger[fid] = is_dir
        try:
            self._pending.remove(node)
        except ValueError:
            pass
        return fid

    def _map_identity_oserror(self) -> ActivationRefusal:
        return ActivationRefusal(RefusalReason.FREEZE_FAILED, "identity inventory")

    # ----- parent validation + root bind -----

    @staticmethod
    def _validate_freeze_parent_path(parent: str) -> None:
        """Path-string prechecks only (absolute, non-empty, no NUL)."""
        if not isinstance(parent, str) or parent == "":
            raise ActivationRefusal(RefusalReason.FREEZE_FAILED, "freeze parent required")
        if "\x00" in parent:
            raise ActivationRefusal(RefusalReason.FREEZE_FAILED, "freeze parent nul")
        if not os.path.isabs(parent):
            raise ActivationRefusal(
                RefusalReason.FREEZE_FAILED, "freeze parent not absolute"
            )

    @staticmethod
    def _validate_parent_fd(parent_fd: int) -> None:
        """Validate trusted parent via fstat of the already-open dir fd."""
        mapped: Optional[ActivationRefusal] = None
        st: Optional[os.stat_result] = None
        try:
            st = os.fstat(parent_fd)
        except OSError:
            mapped = ActivationRefusal(
                RefusalReason.FREEZE_FAILED, "freeze parent fstat"
            )
        if mapped is not None:
            raise mapped
        assert st is not None
        if stat.S_ISLNK(st.st_mode):
            raise ActivationRefusal(
                RefusalReason.FREEZE_FAILED, "freeze parent symlink"
            )
        if not stat.S_ISDIR(st.st_mode):
            raise ActivationRefusal(
                RefusalReason.FREEZE_FAILED, "freeze parent not dir"
            )
        if int(st.st_uid) != os.geteuid():
            raise ActivationRefusal(
                RefusalReason.FREEZE_FAILED, "freeze parent owner"
            )
        if (st.st_mode & 0o077) != 0:
            raise ActivationRefusal(
                RefusalReason.FREEZE_FAILED, "freeze parent mode"
            )

    def bind_root(self, freeze_parent: str) -> str:
        """Bind freeze root under a trusted parent via dir-fd operations only.

        Opens ``freeze_parent`` with ``O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC``,
        validates owner/mode with **fstat** of that fd, then creates the random
        freeze root with ``os.mkdir(name, dir_fd=parent_fd)`` and openat —
        never ``tempfile.mkdtemp`` or pathname chmod of the parent path.

        Returns a diagnostic backing path string (path-join of the input name
        and child); activation trust rests on the held parent/root fds, not
        that path string.
        """
        if self._root_fd >= 0:
            raise ActivationRefusal(RefusalReason.INTERNAL, "lease already bound")
        self._validate_freeze_parent_path(freeze_parent)

        parent_flags = (
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        )
        mapped_open_p: Optional[ActivationRefusal] = None
        parent_fd = -1
        try:
            parent_fd = os.open(freeze_parent, parent_flags)
        except OSError:
            mapped_open_p = ActivationRefusal(
                RefusalReason.FREEZE_FAILED, "freeze parent open"
            )
        if mapped_open_p is not None:
            raise mapped_open_p

        try:
            self._validate_parent_fd(parent_fd)
        except ActivationRefusal:
            try:
                os.close(parent_fd)
            except OSError:
                pass
            raise

        self._parent_fd = parent_fd
        self.reserve_identity_slot()

        # Random name + mkdir relative to the held parent fd only.
        root_name: Optional[str] = None
        mapped_mk: Optional[ActivationRefusal] = None
        for _ in range(64):
            candidate = f"cadml-freeze-{secrets.token_hex(8)}"
            try:
                os.mkdir(candidate, 0o700, dir_fd=parent_fd)
            except FileExistsError:
                continue
            except OSError:
                mapped_mk = ActivationRefusal(RefusalReason.FREEZE_FAILED, "mkdir root")
                break
            root_name = candidate
            break
        if mapped_mk is not None:
            self.cancel_reservation()
            try:
                os.close(parent_fd)
            except OSError:
                pass
            self._parent_fd = -1
            raise mapped_mk
        if root_name is None:
            self.cancel_reservation()
            try:
                os.close(parent_fd)
            except OSError:
                pass
            self._parent_fd = -1
            raise ActivationRefusal(RefusalReason.FREEZE_FAILED, "mkdir root")

        mapped_open: Optional[ActivationRefusal] = None
        root_fd = -1
        try:
            root_fd = os.open(
                root_name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=parent_fd,
            )
        except OSError:
            mapped_open = ActivationRefusal(
                RefusalReason.FREEZE_FAILED, "open freeze root"
            )
        if mapped_open is not None:
            # mkdir+fd binding is not atomic portably: empty shell may remain
            # under the held parent inode (not under a swapped pathname).
            self.cancel_reservation()
            self._cleanup_complete = False
            raise mapped_open

        # Seal root mode via the held child fd (never pathname chmod of parent).
        mapped_ch: Optional[ActivationRefusal] = None
        try:
            os.fchmod(root_fd, 0o700)
        except OSError:
            mapped_ch = ActivationRefusal(RefusalReason.FREEZE_FAILED, "fchmod root")
        if mapped_ch is not None:
            try:
                os.close(root_fd)
            except OSError:
                pass
            self.cancel_reservation()
            self._cleanup_complete = False
            raise mapped_ch

        node = self._adopt_pending(root_fd, parent_fd, root_name, is_dir=True)
        mapped_st: Optional[ActivationRefusal] = None
        st: Optional[os.stat_result] = None
        try:
            st = os.fstat(root_fd)
        except OSError:
            mapped_st = self._map_identity_oserror()
        if mapped_st is not None:
            # pending remains owned for release scrub
            raise mapped_st
        assert st is not None
        self._finalize_pending(node, st)
        self._root_fd = root_fd
        self._root_name = root_name
        # Diagnostic only — trust is dir-fd, not this rejoinable path string.
        self._backing_path = os.path.join(freeze_parent, root_name)
        return self._backing_path

    # ----- create under freeze (reserve → create → adopt → finalize) -----

    def mkdir_owned(self, parent_dir_fd: int, name: str) -> int:
        """mkdirat + openat directory; return owned dir fd (lease-tracked)."""
        self.reserve_identity_slot()
        mapped_mk: Optional[ActivationRefusal] = None
        try:
            os.mkdir(name, 0o700, dir_fd=parent_dir_fd)
        except OSError:
            mapped_mk = ActivationRefusal(RefusalReason.FREEZE_FAILED, "mkdir")
        if mapped_mk is not None:
            self.cancel_reservation()
            raise mapped_mk

        mapped_open: Optional[ActivationRefusal] = None
        child_fd = -1
        try:
            child_fd = os.open(name, _DIR_FLAGS, dir_fd=parent_dir_fd)
        except OSError:
            # Dest-dir open failure: cancel reservation only. Do not name-stat
            # or commit whatever now sits at ``name`` (may be foreign after a
            # rename race). Owned empty shell created by mkdir may remain if
            # renamed away — documented residual; zero model bytes.
            mapped_open = ActivationRefusal(
                RefusalReason.FREEZE_FAILED, "open dest dir"
            )
        if mapped_open is not None:
            self.cancel_reservation()
            self._cleanup_complete = False
            raise mapped_open

        node = self._adopt_pending(child_fd, parent_dir_fd, name, is_dir=True)
        mapped_st: Optional[ActivationRefusal] = None
        st: Optional[os.stat_result] = None
        try:
            st = os.fstat(child_fd)
        except OSError:
            mapped_st = self._map_identity_oserror()
        if mapped_st is not None:
            raise mapped_st
        assert st is not None
        if not stat.S_ISDIR(st.st_mode):
            raise ActivationRefusal(RefusalReason.NOT_DIRECTORY, "dest not dir")
        self._finalize_pending(node, st)
        return child_fd

    def create_file_owned(self, parent_dir_fd: int, name: str) -> int:
        """O_CREAT|O_EXCL file; adopt before fstat; return owned write fd."""
        self.reserve_identity_slot()
        mapped_cr: Optional[ActivationRefusal] = None
        fd = -1
        try:
            fd = os.open(name, _FREEZE_CREATE_FLAGS, 0o600, dir_fd=parent_dir_fd)
        except OSError:
            mapped_cr = ActivationRefusal(RefusalReason.FREEZE_FAILED, "create dest")
        if mapped_cr is not None:
            self.cancel_reservation()
            raise mapped_cr

        # Adopt before fstat/finalize so fstat failure cannot orphan model bytes.
        node = self._adopt_pending(fd, parent_dir_fd, name, is_dir=False)
        mapped_st: Optional[ActivationRefusal] = None
        st: Optional[os.stat_result] = None
        try:
            st = os.fstat(fd)
        except OSError:
            mapped_st = self._map_identity_oserror()
        if mapped_st is not None:
            # pending remains; release will scrub
            raise mapped_st
        assert st is not None
        if not stat.S_ISREG(st.st_mode):
            raise ActivationRefusal(RefusalReason.SPECIAL_FILE, "dest not regular")
        self._finalize_pending(node, st)
        return fd

    def root_dir_fd(self) -> int:
        if self._root_fd < 0:
            raise ActivationRefusal(RefusalReason.FREEZE_FAILED, "no root fd")
        return self._root_fd

    @property
    def backing_path(self) -> str:
        return self._backing_path

    @property
    def cleanup_complete(self) -> bool:
        """True only when released with no residual ownership.

        Requires empty pending list, empty ledger, and no live root fd — a
        nonempty ledger or retained root never reports complete.
        """
        return (
            self._cleanup_complete
            and self._released
            and not self._pending
            and not self._ledger
            and self._root_fd < 0
            and self._reservations == 0
        )

    # ----- reconcile (exact equality) -----

    def reconcile_observed_against_owned_ledger(self) -> None:
        """Require observed freeze tree identities == creation-time ledger.

        * observed − ledger → foreign/unknown: refuse without adopting/deleting
          and **without descending** into foreign directories.
        * ledger − observed → owned missing/moved: refuse.
        """
        if self._root_fd < 0 and not self._root_detached:
            raise ActivationRefusal(RefusalReason.FREEZE_FAILED, "no root")
        root_fd = self._root_fd
        if root_fd < 0:
            raise ActivationRefusal(RefusalReason.FREEZE_FAILED, "root detached")

        observed: Set[_FileId] = set()
        # Global inventory counter (not per-directory) — bounds whole walk.
        global_entries = [0]
        mapped: Optional[ActivationRefusal] = None
        try:
            self._collect_observed(root_fd, observed, global_entries)
        except ActivationRefusal as ar:
            mapped = ar
        if mapped is not None:
            raise mapped

        # Root itself is in the ledger and must be observed via fstat root.
        mapped_rt: Optional[ActivationRefusal] = None
        st: Optional[os.stat_result] = None
        try:
            st = os.fstat(root_fd)
        except OSError:
            mapped_rt = self._map_identity_oserror()
        if mapped_rt is not None:
            raise mapped_rt
        assert st is not None
        root_id = _FileId(int(st.st_dev), int(st.st_ino))
        observed.add(root_id)

        foreign = observed - set(self._ledger.keys())
        missing = set(self._ledger.keys()) - observed
        if foreign:
            raise ActivationRefusal(
                RefusalReason.FREEZE_MUTATED, "foreign observed"
            )
        if missing:
            raise ActivationRefusal(
                RefusalReason.FREEZE_MUTATED, "owned missing"
            )

    def _collect_observed(
        self,
        dir_fd: int,
        observed: Set[_FileId],
        global_entries: List[int],
    ) -> None:
        """Walk only creation-ledger-owned nodes; refuse foreign before descent.

        ``global_entries[0]`` counts scandir yields across the whole walk
        (cap+1 discrimination). Foreign identities raise FREEZE_MUTATED
        immediately — never recurse into foreign directories (avoids RecursionError
        on deep attacker trees).
        """
        mapped_sc: Optional[ActivationRefusal] = None
        it = None
        try:
            it = _scandir_dir_fd(dir_fd)
        except ActivationRefusal as ar:
            mapped_sc = ar
        except OSError:
            mapped_sc = ActivationRefusal(RefusalReason.UNREADABLE, "scandir")
        if mapped_sc is not None:
            raise mapped_sc
        assert it is not None
        try:
            for entry in it:
                global_entries[0] += 1
                if global_entries[0] > self._max_inventory + 1:
                    # Cap+1 discrimination: stop without exhausting the iterator.
                    raise ActivationRefusal(
                        RefusalReason.FREEZE_FAILED, "scandir cap"
                    )
                name = dirent_name_str(entry.name)
                if name in (".", ".."):
                    continue
                mapped_o: Optional[ActivationRefusal] = None
                cfd = -1
                try:
                    cfd = os.open(name, _MEMBER_FLAGS, dir_fd=dir_fd)
                except OSError:
                    # Directory may need O_DIRECTORY
                    try:
                        cfd = os.open(name, _DIR_FLAGS, dir_fd=dir_fd)
                    except OSError:
                        mapped_o = ActivationRefusal(
                            RefusalReason.UNREADABLE, "observe open"
                        )
                if mapped_o is not None:
                    raise mapped_o
                try:
                    mapped_st: Optional[ActivationRefusal] = None
                    st: Optional[os.stat_result] = None
                    try:
                        st = os.fstat(cfd)
                    except OSError:
                        mapped_st = self._map_identity_oserror()
                    if mapped_st is not None:
                        raise mapped_st
                    assert st is not None
                    fid = _FileId(int(st.st_dev), int(st.st_ino))
                    # Foreign / unknown: refuse before any descent or adoption.
                    if fid not in self._ledger:
                        raise ActivationRefusal(
                            RefusalReason.FREEZE_MUTATED, "foreign observed"
                        )
                    if fid in observed:
                        raise ActivationRefusal(
                            RefusalReason.FREEZE_MUTATED, "dup observe"
                        )
                    observed.add(fid)
                    if len(observed) > self._max_inventory:
                        raise ActivationRefusal(
                            RefusalReason.FREEZE_FAILED, "observe inventory"
                        )
                    # Descend only into ledger-owned directories.
                    if stat.S_ISDIR(st.st_mode):
                        self._collect_observed(cfd, observed, global_entries)
                finally:
                    try:
                        os.close(cfd)
                    except OSError:
                        pass
        finally:
            if it is not None:
                try:
                    it.close()
                except Exception:
                    pass

    # ----- release / scrub -----

    def release(self) -> bool:
        """Descriptor-relative scrub of creation-time owned identities.

        Serialized and idempotent under a per-lease lock so concurrent callers
        (``ControlledStore.drain_pending_leases``, ``FrozenBundle.cleanup``,
        direct ``release``) never race fd/ledger mutation. Never adopts
        cleanup-time names into the ledger. Returns True when no owned residual
        remains that still requires retry.
        """
        with self._release_lock:
            if self.cleanup_complete:
                return True
            self._released = True
            complete = True

            # 1) Finalize / scrub pending fds (creation-time adopt, not cleanup adopt).
            still_pending: List[_PendingNode] = []
            for node in list(self._pending):
                if not self._scrub_pending_node(node):
                    complete = False
                    still_pending.append(node)
            self._pending = still_pending

            # Do not close root/parent while pending fds may reference them.
            if self._pending:
                complete = False
            # 2) Scrub finalized ledger entries via the lease-owned root fd.
            # The lease always retains this original root until it closes it here —
            # FrozenBundle only holds a dup and must never "detach" this fd without
            # close. On incomplete scrub, retain root/parent fds + ledger for retry.
            elif self._root_fd >= 0:
                if not self._scrub_tree(self._root_fd):
                    complete = False
                else:
                    try:
                        os.close(self._root_fd)
                    except OSError:
                        pass
                    self._root_fd = -1
                    self._root_detached = True

                    # 3) rmdir root name under parent if empty and identity matches.
                    if self._parent_fd >= 0 and self._root_name:
                        if not self._rmdir_root_if_owned():
                            complete = False
            elif self._ledger:
                # Root missing but ledger still has owned identities — incomplete.
                complete = False

            if complete and self._parent_fd >= 0:
                try:
                    os.close(self._parent_fd)
                except OSError:
                    pass
                self._parent_fd = -1

            self._cleanup_complete = (
                complete
                and not self._pending
                and not self._ledger
                and self._root_fd < 0
                and self._reservations == 0
            )
            return self._cleanup_complete

    def _scrub_pending_node(self, node: _PendingNode) -> bool:
        """Retry fstat; remove only on proven identity match; then pop+close.

        On release-time fstat failure: do **not** delete by name (foreign may
        sit there) and do **not** close/drop the pending fd — return False so
        the node remains pending for a later retry.

        On successful fstat: record identity, attempt identity-checked remove.
        **Only if** the name was proven same and removed: pop that fid from the
        ledger, close the pending fd once, and return True. If removal fails or
        mismatches, leave the pending node/fd (and ledger entry) for honest
        retry — never silently drop ownership after a failed remove.
        """
        st: Optional[os.stat_result] = None
        try:
            st = os.fstat(node.fd)
        except OSError:
            # Keep pending ownership; no name-delete, no close.
            return False
        fid = _FileId(int(st.st_dev), int(st.st_ino))
        self._ledger.setdefault(fid, node.is_dir)
        # Unlink by name only when name still refers to this identity.
        removed = self._remove_owned_name(
            node.parent_fd, node.name, fid, is_dir=node.is_dir
        )
        if not removed:
            # Foreign/mismatch or unproven absence — keep pending fd for retry.
            return False
        # Proven same-identity remove: drop ledger ownership, close once.
        self._ledger.pop(fid, None)
        try:
            os.close(node.fd)
        except OSError:
            pass
        node.fd = -1
        return True

    def _remove_owned_name(
        self,
        parent_fd: int,
        name: str,
        fid: _FileId,
        *,
        is_dir: bool,
    ) -> bool:
        """Remove ``name`` under ``parent_fd`` only if it still is ``fid``.

        Optional ``_pre_remove_hook`` runs after the caller has observed
        ownership and immediately before this pre-remove identity re-check, so
        tests can inject a post-stat/pre-rmdir replacement. Returns True iff
        the name was removed. Foreign/mismatched objects are never deleted.

        Note: a remaining non-atomic window exists between the re-check and the
        rmdir/unlink syscall (classic empty-dir TOCTOU); portable POSIX cannot
        close that fully. We never delete when the re-check fails.
        """
        hook = self._pre_remove_hook
        if hook is not None:
            hook(parent_fd, name, fid, is_dir)
        st: Optional[os.stat_result] = None
        try:
            st = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except OSError:
            return False  # already gone / unreadable — do not guess
        cur = _FileId(int(st.st_dev), int(st.st_ino))
        if cur != fid:
            # Foreign or replaced — do not delete.
            return False
        try:
            if is_dir:
                os.rmdir(name, dir_fd=parent_fd)
            else:
                os.unlink(name, dir_fd=parent_fd)
        except OSError:
            return False
        return True

    def _unlink_if_same(
        self,
        parent_fd: int,
        name: str,
        fid: _FileId,
        *,
        is_dir: bool,
    ) -> bool:
        """Compatibility alias for :meth:`_remove_owned_name`."""
        return self._remove_owned_name(parent_fd, name, fid, is_dir=is_dir)

    def _scrub_tree(self, dir_fd: int) -> bool:
        """Post-order scrub: delete ledger-matched children only."""
        complete = True
        # Sealed freezes are mode 0500/0400 — restore owner write before unlink.
        try:
            os.fchmod(dir_fd, 0o700)
        except OSError:
            pass
        mapped_sc: Optional[ActivationRefusal] = None
        it = None
        try:
            it = _scandir_dir_fd(dir_fd)
        except ActivationRefusal as ar:
            mapped_sc = ar
        except OSError:
            mapped_sc = ActivationRefusal(RefusalReason.UNREADABLE, "scandir")
        if mapped_sc is not None:
            # Cannot scan lazily — refuse incomplete; caller retains ownership.
            return False
        assert it is not None
        count = 0
        try:
            for entry in it:
                count += 1
                if count > _CLEANUP_SCANDIR_CAP + 1:
                    # Cap+1 discrimination: stop without exhausting the iterator.
                    complete = False
                    break
                name = dirent_name_str(entry.name)
                if name in (".", ".."):
                    continue
                cfd = -1
                is_dir = False
                try:
                    cfd = os.open(name, _DIR_FLAGS, dir_fd=dir_fd)
                    is_dir = True
                except OSError:
                    try:
                        cfd = os.open(name, _FREEZE_LEAF_FLAGS, dir_fd=dir_fd)
                        is_dir = False
                    except OSError:
                        # Unopenable name: do not adopt; leave (may be foreign).
                        complete = False
                        continue
                try:
                    st = os.fstat(cfd)
                    fid = _FileId(int(st.st_dev), int(st.st_ino))
                    owned = fid in self._ledger
                    if is_dir or stat.S_ISDIR(st.st_mode):
                        if owned:
                            if not self._scrub_tree(cfd):
                                complete = False
                            # fchmod writable for rmdir on sealed trees
                            try:
                                os.fchmod(cfd, 0o700)
                            except OSError:
                                pass
                            try:
                                os.close(cfd)
                            except OSError:
                                pass
                            cfd = -1
                            # Re-check identity immediately before rmdir (post-stat
                            # replacement of empty foreign dir must not be removed).
                            if self._remove_owned_name(
                                dir_fd, name, fid, is_dir=True
                            ):
                                self._ledger.pop(fid, None)
                            else:
                                complete = False
                        else:
                            # Foreign directory — leave it; do not recurse-delete.
                            complete = False
                    else:
                        if owned:
                            try:
                                os.fchmod(cfd, 0o600)
                            except OSError:
                                pass
                            try:
                                os.close(cfd)
                            except OSError:
                                pass
                            cfd = -1
                            if self._remove_owned_name(
                                dir_fd, name, fid, is_dir=False
                            ):
                                self._ledger.pop(fid, None)
                            else:
                                complete = False
                        else:
                            complete = False
                except OSError:
                    complete = False
                finally:
                    if cfd >= 0:
                        try:
                            os.close(cfd)
                        except OSError:
                            pass
        finally:
            try:
                it.close()
            except Exception:
                pass
        return complete

    def _rmdir_root_if_owned(self) -> bool:
        """Rmdir freeze root only when the name still matches a ledger identity.

        Never best-effort rmdir an unproven root name (including when the ledger
        is empty after prior pops or races). Foreign / unproven names are left
        alone; return True only when gone-or-removed-as-owned.
        """
        if self._parent_fd < 0 or not self._root_name:
            return True
        st: Optional[os.stat_result] = None
        try:
            st = os.stat(
                self._root_name, dir_fd=self._parent_fd, follow_symlinks=False
            )
        except OSError:
            # Already gone.
            return True
        if st is None:
            return True
        fid = _FileId(int(st.st_dev), int(st.st_ino))
        if fid not in self._ledger:
            # Unproven or foreign at root name — never delete by name alone.
            # Empty-ledger residual shell is an honest non-atomic leftover.
            return True
        try:
            # Ensure mode allows rmdir of sealed root.
            rfd = os.open(
                self._root_name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=self._parent_fd,
            )
            try:
                os.fchmod(rfd, 0o700)
            finally:
                os.close(rfd)
        except OSError:
            pass
        # Identity re-check immediately before rmdir (same as child removal).
        if not self._remove_owned_name(
            self._parent_fd, self._root_name, fid, is_dir=True
        ):
            return False
        self._ledger.pop(fid, None)
        return True


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
    mapped: Optional[ActivationRefusal] = None
    rel_b: Optional[bytes] = None
    try:
        rel_b = relpath.encode("utf-8")
    except UnicodeEncodeError:
        mapped = ActivationRefusal(RefusalReason.NON_UTF8_ENTRY, "relpath")
    if mapped is not None:
        raise mapped
    assert rel_b is not None
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
        self._lock = threading.RLock()
        self._closed = False
        self._in_flight = 0
        self._pending_leases: List[FreezeResourceLease] = []
        self._close_cond = threading.Condition(self._lock)

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

    def _enter_op(self) -> int:
        """Return a caller-owned dup of the store-root fd under the lock."""
        with self._lock:
            if self._closed or self._store_root_fd < 0:
                raise ActivationRefusal(RefusalReason.INTERNAL, "store closed")
            mapped: Optional[ActivationRefusal] = None
            dup = -1
            try:
                dup = os.dup(self._store_root_fd)
            except OSError:
                mapped = ActivationRefusal(RefusalReason.INTERNAL, "dup store root")
            if mapped is not None:
                raise mapped
            self._in_flight += 1
            return dup

    def _leave_op(self, dup_fd: int) -> None:
        if dup_fd >= 0:
            try:
                os.close(dup_fd)
            except OSError:
                pass
        with self._lock:
            self._in_flight = max(0, self._in_flight - 1)
            self._close_cond.notify_all()

    def _retain_pending_lease(self, lease: FreezeResourceLease) -> None:
        """Retain by object identity (leases are eq=False resource owners)."""
        with self._lock:
            if not any(existing is lease for existing in self._pending_leases):
                self._pending_leases.append(lease)

    def drain_pending_leases(self) -> bool:
        """Retry ``release()`` on retained freeze leases from construction failures.

        Leases whose ``release()`` returns False (or raises) stay in
        ``_pending_leases`` for a later :meth:`drain_pending_leases` /
        :meth:`close`. Returns True only when no residual pending ownership
        remains. Freeze leases use their own parent/root fds (independent of
        the store-root fd).
        """
        with self._lock:
            pending = list(self._pending_leases)
        incomplete: List[FreezeResourceLease] = []
        for lease in pending:
            try:
                ok = bool(lease.release())
            except Exception:
                ok = False
            if not ok:
                incomplete.append(lease)
        with self._lock:
            # Keep unsuccessful leases by identity; drop only clean releases.
            kept: List[FreezeResourceLease] = list(incomplete)
            for lease in self._pending_leases:
                if not any(p is lease for p in pending) and not any(
                    k is lease for k in kept
                ):
                    kept.append(lease)
            self._pending_leases = kept
            return len(self._pending_leases) == 0

    def close(self) -> bool:
        """Wait for in-flight ops; drain pending freeze leases; close store root.

        Returns True only when no residual pending freeze ownership remains.
        Unsuccessful ``lease.release()`` results are **retained** in
        ``_pending_leases`` for a later :meth:`close` / :meth:`drain_pending_leases`
        — incomplete construction-failure scrubs are never silently forgotten.

        Freeze leases own independent parent/root directory fds, so the
        store-root fd is closed once after in-flight activations drain even if
        freeze leases remain pending (those retries do not need the store root).
        """
        with self._lock:
            self._closed = True
            while self._in_flight > 0:
                self._close_cond.wait(timeout=0.05)
            fd = self._store_root_fd
            # Close store root once; freeze leases do not depend on this fd.
            if fd is not None and fd >= 0:
                self._store_root_fd = -1
            else:
                fd = -1

        if fd is not None and fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass

        return self.drain_pending_leases()

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

        root_dup = self._enter_op()
        try:
            fd = open_pinned(
                root_dup,
                pin.store_relpath,
                TerminalKind.REGULAR_FILE,
                impl=self._impl(),
            )
            try:
                mapped: Optional[ActivationRefusal] = None
                st: Optional[os.stat_result] = None
                try:
                    st = os.fstat(fd)
                except OSError:
                    mapped = ActivationRefusal(
                        RefusalReason.UNREADABLE, "fstat failed"
                    )
                if mapped is not None:
                    raise mapped
                assert st is not None
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
        finally:
            self._leave_op(root_dup)

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
        if not self._freeze_parent:
            raise ActivationRefusal(
                RefusalReason.FREEZE_FAILED, "freeze parent required"
            )

        root_dup = self._enter_op()
        lease = FreezeResourceLease()
        bundle_dir_fd = -1
        try:
            src_root_fd = open_pinned(
                root_dup,
                pin.store_relpath,
                TerminalKind.DIRECTORY,
                impl=self._impl(),
            )
            try:
                # Pass 1: metadata preflight — no freeze writes.
                self._preflight_tree(src_root_fd)

                # Pass 2: bind empty lease root, then same-fd DFS freeze.
                backing = lease.bind_root(self._freeze_parent)
                self._freeze_tree(src_root_fd, lease)

                # Exact ledger/observation equality before seal/digest.
                lease.reconcile_observed_against_owned_ledger()

                freeze_fd = lease.root_dir_fd()
                seal_freeze_tree_fd(freeze_fd)

                entries, file_count, aggregate = self._digest_frozen_tree(freeze_fd)
                digest = tree_digest_v1(entries)
                if digest != pin.digest:
                    raise ActivationRefusal(
                        RefusalReason.DIGEST_MISMATCH, "tree-digest-v1"
                    )

                # Bundle owns a dup of the freeze root; lease keeps the original
                # root fd and scrubs/closes it on lease.release() (never detach
                # without close).
                mapped_dup: Optional[ActivationRefusal] = None
                try:
                    bundle_dir_fd = os.dup(freeze_fd)
                except OSError:
                    mapped_dup = ActivationRefusal(
                        RefusalReason.FREEZE_FAILED, "dup freeze root"
                    )
                if mapped_dup is not None:
                    raise mapped_dup

                handle = FrozenBundle(
                    dir_fd=bundle_dir_fd,
                    lease=lease,
                    digest=digest,
                    file_count=file_count,
                    aggregate_bytes=aggregate,
                    backing_path=backing,
                )
                bundle_dir_fd = -1
                lease = FreezeResourceLease()  # transferred; empty stub not released
                return handle
            finally:
                try:
                    os.close(src_root_fd)
                except OSError:
                    pass
        except Exception:
            if bundle_dir_fd >= 0:
                try:
                    os.close(bundle_dir_fd)
                except OSError:
                    pass
            # Construction failure: retry cleanup; retain pending if incomplete.
            try:
                done = lease.release()
            except Exception:
                done = False
            if not done:
                self._retain_pending_lease(lease)
            raise
        finally:
            self._leave_op(root_dup)

    def load_pinned_bundle(
        self,
        logical_activation_id: str,
        artifact_id: str,
    ) -> FrozenBundle:
        return self.assert_bundle_digest(logical_activation_id, artifact_id)

    # ----- shared DFS member iteration (O(depth) source fds) -----

    def _iter_members_dfs(
        self,
        dir_fd: int,
        *,
        prefix: str,
        depth: int,
        counters: _WalkCounters,
        on_file: Callable[[int, os.stat_result, str], None],
        on_dir_enter: Optional[Callable[[int, str, str], None]] = None,
        freeze_parent_fd: Optional[int] = None,
        lease: Optional[FreezeResourceLease] = None,
    ) -> None:
        """Depth-first walk: close each child dir before opening the next sibling.

        Live source directory fds ≤ O(depth). Never stacks all sibling dir fds.
        When ``lease`` is set (pass 2), destination dirs are created under
        ``freeze_parent_fd`` via the lease (reserve/adopt/finalize).
        """
        bounds = self._bounds
        if depth > bounds.max_depth:
            raise ActivationRefusal(RefusalReason.BUNDLE_DEPTH, "depth cap")

        it = _scandir_dir_fd(dir_fd)
        try:
            for entry in it:
                name = dirent_name_str(entry.name)
                if name in (".", ".."):
                    continue

                counters.total_dirents += 1
                if counters.total_dirents > bounds.max_total_dirents:
                    # Cap+1: refuse without exhausting the remainder of the iterator.
                    raise ActivationRefusal(
                        RefusalReason.BUNDLE_TOTAL_DIRENTS, "dirent cap"
                    )

                validate_readdir_name(name)
                relpath = _relpath_join(prefix, name)
                _check_relpath(relpath, bounds)

                mapped_open: Optional[ActivationRefusal] = None
                mfd = -1
                try:
                    mfd = os.open(name, _MEMBER_FLAGS, dir_fd=dir_fd)
                except OSError as exc:
                    mapped_open = _map_member_open_error(exc)
                if mapped_open is not None:
                    raise mapped_open

                child_freeze_fd = -1
                try:
                    mapped_st: Optional[ActivationRefusal] = None
                    st: Optional[os.stat_result] = None
                    try:
                        st = os.fstat(mfd)
                    except OSError:
                        mapped_st = ActivationRefusal(
                            RefusalReason.UNREADABLE, "fstat member"
                        )
                    if mapped_st is not None:
                        raise mapped_st
                    assert st is not None
                    mode = st.st_mode

                    if stat.S_ISDIR(mode):
                        # Re-open as directory for safe descent (member flags omit
                        # O_DIRECTORY so FIFOs do not hang; dirs need DIR flags).
                        try:
                            os.close(mfd)
                        except OSError:
                            pass
                        mfd = -1
                        mapped_d: Optional[ActivationRefusal] = None
                        try:
                            mfd = os.open(name, _DIR_FLAGS, dir_fd=dir_fd)
                        except OSError as exc:
                            mapped_d = _map_member_open_error(exc)
                        if mapped_d is not None:
                            raise mapped_d

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
                        next_freeze_parent = freeze_parent_fd
                        if lease is not None and freeze_parent_fd is not None:
                            child_freeze_fd = lease.mkdir_owned(
                                freeze_parent_fd, name
                            )
                            next_freeze_parent = child_freeze_fd
                        if on_dir_enter is not None:
                            on_dir_enter(mfd, relpath, name)
                        self._iter_members_dfs(
                            mfd,
                            prefix=relpath,
                            depth=child_depth,
                            counters=counters,
                            on_file=on_file,
                            on_dir_enter=on_dir_enter,
                            freeze_parent_fd=next_freeze_parent,
                            lease=lease,
                        )
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
                        # Pass 2: recheck + copy from this already-open source fd.
                        on_file(mfd, st, relpath)
                        counters.aggregate += st.st_size
                        if counters.aggregate > bounds.max_bundle_aggregate_bytes:
                            raise ActivationRefusal(
                                RefusalReason.BUNDLE_AGGREGATE_BYTES, "aggregate cap"
                            )
                finally:
                    if child_freeze_fd >= 0:
                        try:
                            os.close(child_freeze_fd)
                        except OSError:
                            pass
                    if mfd >= 0:
                        try:
                            os.close(mfd)
                        except OSError:
                            pass
        finally:
            try:
                it.close()
            except Exception:
                pass

    def _preflight_tree(self, root_fd: int) -> None:
        """Metadata-only DFS: refuse bounds before any freeze copy."""
        counters = _WalkCounters()

        def on_file(mfd: int, st: os.stat_result, relpath: str) -> None:
            return

        self._iter_members_dfs(
            root_fd,
            prefix="",
            depth=0,
            counters=counters,
            on_file=on_file,
            freeze_parent_fd=None,
            lease=None,
        )

    def _freeze_tree(self, root_fd: int, lease: FreezeResourceLease) -> None:
        """Second pass: same-fd bounded copy into lease freeze with recheck."""
        counters = _WalkCounters()
        freeze_root_fd = lease.root_dir_fd()
        self._freeze_walk(
            root_fd,
            freeze_root_fd,
            lease,
            prefix="",
            depth=0,
            counters=counters,
        )

    def _freeze_walk(
        self,
        src_dir_fd: int,
        freeze_dir_fd: int,
        lease: FreezeResourceLease,
        *,
        prefix: str,
        depth: int,
        counters: _WalkCounters,
    ) -> None:
        bounds = self._bounds
        if depth > bounds.max_depth:
            raise ActivationRefusal(RefusalReason.BUNDLE_DEPTH, "depth cap")

        it = _scandir_dir_fd(src_dir_fd)
        try:
            for entry in it:
                name = dirent_name_str(entry.name)
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

                mapped_open: Optional[ActivationRefusal] = None
                mfd = -1
                try:
                    mfd = os.open(name, _MEMBER_FLAGS, dir_fd=src_dir_fd)
                except OSError as exc:
                    mapped_open = _map_member_open_error(exc)
                if mapped_open is not None:
                    raise mapped_open

                child_freeze = -1
                try:
                    mapped_st: Optional[ActivationRefusal] = None
                    st: Optional[os.stat_result] = None
                    try:
                        st = os.fstat(mfd)
                    except OSError:
                        mapped_st = ActivationRefusal(
                            RefusalReason.UNREADABLE, "fstat member"
                        )
                    if mapped_st is not None:
                        raise mapped_st
                    assert st is not None
                    mode = st.st_mode

                    if stat.S_ISDIR(mode):
                        try:
                            os.close(mfd)
                        except OSError:
                            pass
                        mfd = -1
                        mapped_d: Optional[ActivationRefusal] = None
                        try:
                            mfd = os.open(name, _DIR_FLAGS, dir_fd=src_dir_fd)
                        except OSError as exc:
                            mapped_d = _map_member_open_error(exc)
                        if mapped_d is not None:
                            raise mapped_d

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
                        child_freeze = lease.mkdir_owned(freeze_dir_fd, name)
                        self._freeze_walk(
                            mfd,
                            child_freeze,
                            lease,
                            prefix=relpath,
                            depth=child_depth,
                            counters=counters,
                        )
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
                        # Recheck + copy from the already-open source fd (no reopen).
                        data = read_fd_bounded_once(
                            mfd,
                            bounds.max_bundle_per_file_bytes,
                            expect_size=st.st_size,
                        )
                        dfd = lease.create_file_owned(freeze_dir_fd, name)
                        try:
                            view = memoryview(data)
                            offset = 0
                            while offset < len(data):
                                mapped_w: Optional[ActivationRefusal] = None
                                written = 0
                                try:
                                    written = os.write(dfd, view[offset:])
                                except OSError:
                                    mapped_w = ActivationRefusal(
                                        RefusalReason.FREEZE_FAILED, "write"
                                    )
                                if mapped_w is not None:
                                    raise mapped_w
                                if written <= 0:
                                    raise ActivationRefusal(
                                        RefusalReason.FREEZE_FAILED, "write"
                                    )
                                offset += written
                        finally:
                            try:
                                os.close(dfd)
                            except OSError:
                                pass
                        counters.aggregate += st.st_size
                        if counters.aggregate > bounds.max_bundle_aggregate_bytes:
                            raise ActivationRefusal(
                                RefusalReason.BUNDLE_AGGREGATE_BYTES, "aggregate cap"
                            )
                finally:
                    if child_freeze >= 0:
                        try:
                            os.close(child_freeze)
                        except OSError:
                            pass
                    if mfd >= 0:
                        try:
                            os.close(mfd)
                        except OSError:
                            pass
        finally:
            try:
                it.close()
            except Exception:
                pass

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
            it = _scandir_dir_fd(dir_fd)
            try:
                for entry in it:
                    name = dirent_name_str(entry.name)
                    if name in (".", ".."):
                        continue
                    validate_readdir_name(name)
                    relpath = _relpath_join(prefix, name)
                    _check_relpath(relpath, bounds)

                    # Prefer O_DIRECTORY; fall back to nonblock leaf open.
                    mapped_o: Optional[ActivationRefusal] = None
                    mfd = -1
                    is_dir_open = False
                    try:
                        mfd = os.open(name, _DIR_FLAGS, dir_fd=dir_fd)
                        is_dir_open = True
                    except OSError:
                        try:
                            mfd = os.open(name, _FREEZE_LEAF_FLAGS, dir_fd=dir_fd)
                        except OSError:
                            mapped_o = ActivationRefusal(
                                RefusalReason.FREEZE_FAILED, "freeze re-open"
                            )
                    if mapped_o is not None:
                        raise mapped_o
                    try:
                        mapped_st: Optional[ActivationRefusal] = None
                        st: Optional[os.stat_result] = None
                        try:
                            st = os.fstat(mfd)
                        except OSError:
                            mapped_st = ActivationRefusal(
                                RefusalReason.FREEZE_FAILED, "freeze fstat"
                            )
                        if mapped_st is not None:
                            raise mapped_st
                        assert st is not None
                        if is_dir_open or stat.S_ISDIR(st.st_mode):
                            if not stat.S_ISDIR(st.st_mode):
                                raise ActivationRefusal(
                                    RefusalReason.NOT_DIRECTORY, "freeze dir"
                                )
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
            finally:
                try:
                    it.close()
                except Exception:
                    pass

        walk(freeze_dir_fd, "", 0)
        return entries, file_count, aggregate


__all__ = [
    "ControlledStore",
    "FreezeResourceLease",
    "PinKey",
    "_MAX_FREEZE_IDENTITY_INVENTORY",
]
