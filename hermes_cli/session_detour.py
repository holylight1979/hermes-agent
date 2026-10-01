"""Session detour return records — the state behind ``/detour`` and ``/detour-end``.

A *detour* is a round trip: the surface leaves the current conversation for a genuinely fresh
child session (usually on a different model), and later returns to the ORIGINAL parent session.
Neither transcript is ever copied into the other; the only thing that crosses the boundary is the
small record this module owns, which says *where to come back to*.

Why a record on disk at all: the return target cannot be recovered from live state. The child's
``parent_session_id`` column names the parent, but a surface that restarts mid-detour has no live
handle to either, and the parent's own route (model/provider at the time of leaving) is gone the
moment the child's override is applied. The record is the one durable thing that makes the return
leg possible, so it is written BEFORE the child exists and only consumed after a restore succeeds.

Scoping. One record per conversation *lane*, inside the active Hermes home (so profiles are
isolated structurally, not by a field we have to remember to compare):

    <HERMES_HOME>/session-detours/<fingerprint>.json

The fingerprint is a digest of (surface, channel, owner, lane) — never a user-supplied string — so
no caller can name a file outside that directory. The readable scope fields are stored *inside*
the JSON, which is UTF-8 and indented on purpose: an operator inspecting a stuck detour should be
able to read it without a tool.

Ownership is checked on BOTH sides of the digest: the record's own ``scope`` must match the
caller's recomputed scope, and (once the child exists) ``child_session_id`` must be the caller's
*current* session. That second check is what makes a lane-wide key safe — a second process in the
same lane that is not sitting in the child session cannot use the record to reach the parent.

Nothing here stores credentials, prompts or transcript content; ``*_route`` carries a model and
provider name only.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import uuid
from contextlib import contextmanager, suppress
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional

logger = logging.getLogger(__name__)

RECORD_VERSION = 1

#: Written before the child session exists. A record left in this state means the enter leg died
#: between the write and the rotation; the parent id is already known, so returning still works.
STATUS_ENTERING = "entering"
#: The child session exists and is the live conversation.
STATUS_ACTIVE = "active"
#: The parent was restored. Kept (not deleted) so a repeated ``/detour-end`` is a quiet no-op and
#: an operator can still see what happened.
STATUS_RETURNED = "returned"

_OPEN_STATUSES = frozenset({STATUS_ENTERING, STATUS_ACTIVE})
_KNOWN_STATUSES = _OPEN_STATUSES | {STATUS_RETURNED}

RECORD_DIRNAME = "session-detours"

# Session ids are generated (``%Y%m%d_%H%M%S_<hex>``); this is a shape gate, not a parser. It
# exists so a hand-edited record can never smuggle a path fragment into a resume target.
_SESSION_ID_RE = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


class DetourRecordError(Exception):
    """A record exists but cannot be trusted. Carries user-facing copy; the record is left alone.

    Every raise site here is a fail-closed path: the caller reports the message and changes no
    session state.
    """


class LaneLockError(DetourRecordError):
    """The lane's transition lock could not be taken, so no leg may run.

    A subclass of :class:`DetourRecordError` so every existing ``except DetourRecordError`` site
    keeps failing closed; the command entry points catch it by name to report the right copy.
    """


#: Deliberately generic: a lock failure is reported to whoever typed the command, so it names no
#: path, errno or exception type. The details go to the log instead.
LOCK_REFUSAL = "the conversation lane could not be locked, so nothing was changed"


@dataclass(frozen=True)
class DetourScope:
    """The conversation lane a detour belongs to.

    ``lane`` is the surface's own stable routing handle (the gateway ``session_key``) — deliberately
    NOT a session id, which rotates on the very reset the enter leg performs.
    """

    surface: str
    channel: str = ""
    owner: str = ""
    lane: str = ""

    def __post_init__(self) -> None:
        if not str(self.surface).strip():
            raise ValueError("DetourScope.surface is required")

    @property
    def parts(self) -> tuple[str, str, str, str]:
        return (
            str(self.surface).strip().casefold(),
            str(self.channel or "").strip(),
            str(self.owner or "").strip(),
            str(self.lane or "").strip(),
        )

    @property
    def lane_parts(self) -> tuple[str, str, str]:
        """The lane WITHOUT the owner — the unit a session rotation actually belongs to."""
        surface, channel, _owner, lane = self.parts
        return (surface, channel, lane)

    def fingerprint(self) -> str:
        """Filename-safe digest of the lane. Unit separator: no component can forge another."""
        return hashlib.sha256("\x1f".join(self.parts).encode("utf-8")).hexdigest()[:40]

    def lane_fingerprint(self) -> str:
        """Digest of the owner-independent lane, used for the transition lock.

        Records are per (lane, owner) — two people in one group chat each get their own return
        target. The *rotation* they perform is not per-owner: both move the single session this
        lane routes to. So the lock is keyed on the lane alone, or two owners could rotate the
        same lane concurrently and one of them would record a parent that is already gone.
        """
        return hashlib.sha256("\x1f".join(("lane", *self.lane_parts)).encode("utf-8")).hexdigest()[:40]

    def to_json(self) -> Dict[str, str]:
        surface, channel, owner, lane = self.parts
        return {"surface": surface, "channel": channel, "owner": owner, "lane": lane}

    @classmethod
    def from_json(cls, data: Any) -> Optional["DetourScope"]:
        if not isinstance(data, dict) or not str(data.get("surface") or "").strip():
            return None
        return cls(
            surface=str(data.get("surface") or ""), channel=str(data.get("channel") or ""),
            owner=str(data.get("owner") or ""), lane=str(data.get("lane") or ""),
        )


@dataclass(frozen=True)
class DetourRecord:
    """One lane's return record. ``parent_route`` / ``child_route`` are ``{model, provider}`` only."""

    scope: DetourScope
    parent_session_id: str
    status: str = STATUS_ENTERING
    child_session_id: str = ""
    profile: str = ""
    parent_route: Dict[str, str] = None  # type: ignore[assignment]
    child_route: Dict[str, str] = None  # type: ignore[assignment]
    entered_at: str = ""
    returned_at: str = ""
    #: Per-transition ownership token. Every write after :func:`begin_detour` is a compare-and-swap
    #: against it, so a leg that was overtaken (another process already ended and re-entered the
    #: detour) fails loudly instead of stamping its stale view over the live record.
    token: str = ""
    version: int = RECORD_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "parent_route", _route(self.parent_route))
        object.__setattr__(self, "child_route", _route(self.child_route))

    @property
    def is_open(self) -> bool:
        return self.status in _OPEN_STATUSES

    def to_json(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "status": self.status,
            "profile": self.profile,
            "scope": self.scope.to_json(),
            "parent_session_id": self.parent_session_id,
            "child_session_id": self.child_session_id,
            "parent_route": dict(self.parent_route),
            "child_route": dict(self.child_route),
            "entered_at": self.entered_at,
            "returned_at": self.returned_at,
            "token": self.token,
        }


def _route(value: Any) -> Dict[str, str]:
    """``{model, provider}`` as strings. Any other key is dropped: the record is not a config
    channel, and a stray ``api_key`` must never reach the file."""
    source = value if isinstance(value, dict) else {}
    return {key: str(source.get(key) or "") for key in ("model", "provider")}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def active_profile_name() -> str:
    """Readable profile label for the record. Never used for scoping (the home path is)."""
    try:
        from hermes_cli.profiles import get_active_profile_name

        return get_active_profile_name() or "default"
    except Exception:
        return "default"


def detour_root(home: Any = None) -> Path:
    """The active home's record directory. ``home`` is for tests and profile-explicit callers."""
    if home is None:
        from hermes_constants import get_hermes_home

        home = get_hermes_home()
    return Path(home) / RECORD_DIRNAME


def record_path(scope: DetourScope, home: Any = None) -> Path:
    return detour_root(home) / f"{scope.fingerprint()}.json"


def _lock_path(scope: DetourScope, home: Any = None) -> Path:
    return detour_root(home) / f"{scope.lane_fingerprint()}.lock"


def _flock(handle, *, lock: bool) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK if lock else msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX if lock else fcntl.LOCK_UN)


#: One mutex per lane fingerprint, so same-process concurrency (two gateway coroutines, two threads)
#: serializes even on a platform where advisory file locking is unavailable.
_LANE_MUTEXES: Dict[str, threading.Lock] = {}
_LANE_MUTEXES_GUARD = threading.Lock()


def _lane_mutex(scope: DetourScope) -> threading.Lock:
    key = scope.lane_fingerprint()
    with _LANE_MUTEXES_GUARD:
        mutex = _LANE_MUTEXES.get(key)
        if mutex is None:
            mutex = _LANE_MUTEXES[key] = threading.Lock()
        return mutex


class LaneTransitionLock:
    """The whole-transition lock for one conversation lane.

    Explicit ``acquire``/``release`` rather than only a context manager, because an async surface
    must hold this across ``await``s (the session reset, the resume, the model switch) — a ``with``
    block cannot span them. Both calls are blocking and are meant to be run off the event loop via
    ``asyncio.to_thread``, so waiting for the lock never stalls the loop.

    Two layers: a per-lane in-process mutex (``threading.Lock``, released from whichever thread
    ``release`` lands on) and an advisory lock on a lane file for the cross-process case.

    Both layers are mandatory. :meth:`acquire` FAILS CLOSED — if the lane file cannot be opened or
    locked it raises :class:`LaneLockError` instead of running the transition on in-process ordering
    alone. Degrading was silently unsafe: two Hermes processes sharing a home would each read "no
    detour open" and each rotate the same lane, so one child session is orphaned and its record
    overwritten. A refused command loses nothing — the user can retry.
    """

    def __init__(self, scope: DetourScope, home: Any = None) -> None:
        self._scope = scope
        self._home = home
        self._mutex = _lane_mutex(scope)
        self._mutex_held = False
        self._handle = None

    def acquire(self) -> "LaneTransitionLock":
        self._mutex.acquire()
        self._mutex_held = True
        path = _lock_path(self._scope, self._home)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Assigned BEFORE the flock, so a handle opened by a call that then fails to lock is
            # still closed below rather than leaked for the life of the process.
            self._handle = open(path, "a+b")
            _flock(self._handle, lock=True)
        except Exception:
            logger.warning("session_detour: lane lock refused for %s", path, exc_info=True)
            self.release()
            raise LaneLockError(LOCK_REFUSAL) from None
        return self

    def release(self) -> None:
        handle, self._handle = self._handle, None
        if handle is not None:
            with suppress(Exception):
                _flock(handle, lock=False)
            with suppress(Exception):
                handle.close()
        if self._mutex_held:
            self._mutex_held = False
            with suppress(RuntimeError):
                self._mutex.release()

    def __enter__(self) -> "LaneTransitionLock":
        return self.acquire()

    def __exit__(self, *_exc) -> None:
        self.release()


@contextmanager
def scope_lock(scope: DetourScope, home: Any = None) -> Iterator[None]:
    """Serialize read-decide-write on one lane across processes (sync callers).

    Both legs mutate the record after reading it; without this, two concurrent messages on the same
    channel could each see "no detour" and each rotate the session.

    Raises :class:`LaneLockError` (before the body runs) when the lane cannot be locked — the caller
    reports it and changes nothing.
    """
    lock = LaneTransitionLock(scope, home).acquire()
    try:
        yield
    finally:
        lock.release()


def read_record(scope: DetourScope, home: Any = None) -> Optional[DetourRecord]:
    """The lane's record, or None when the lane has never detoured.

    Raises :class:`DetourRecordError` when a file IS present but unusable — an unreadable,
    non-JSON, wrong-version, wrong-scope or malformed record is a fault to report, never a reason
    to start a session.
    """
    path = record_path(scope, home)
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise DetourRecordError(
            f"the detour record at {path} could not be read ({type(exc).__name__})") from None
    try:
        data = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, ValueError):
        raise DetourRecordError(f"the detour record at {path} is not readable UTF-8 JSON") from None
    if not isinstance(data, dict):
        raise DetourRecordError(f"the detour record at {path} is not a JSON object")

    version = data.get("version")
    if version != RECORD_VERSION:
        raise DetourRecordError(
            f"the detour record at {path} has unsupported version {version!r} "
            f"(this build writes {RECORD_VERSION})")
    status = data.get("status")
    if status not in _KNOWN_STATUSES:
        raise DetourRecordError(f"the detour record at {path} has an unknown status {status!r}")
    stored_scope = DetourScope.from_json(data.get("scope"))
    if stored_scope is None:
        raise DetourRecordError(f"the detour record at {path} has no readable scope")
    if stored_scope.parts != scope.parts:
        # The filename is a digest of the scope, so this means the file was moved or hand-edited.
        raise DetourRecordError(
            f"the detour record at {path} belongs to another conversation and was not used")
    parent = str(data.get("parent_session_id") or "")
    if not _SESSION_ID_RE.match(parent):
        raise DetourRecordError(
            f"the detour record at {path} names an invalid parent session id")
    child = str(data.get("child_session_id") or "")
    if child and not _SESSION_ID_RE.match(child):
        raise DetourRecordError(f"the detour record at {path} names an invalid child session id")
    return DetourRecord(
        scope=stored_scope, parent_session_id=parent, status=status, child_session_id=child,
        profile=str(data.get("profile") or ""), parent_route=data.get("parent_route"),
        child_route=data.get("child_route"), entered_at=str(data.get("entered_at") or ""),
        returned_at=str(data.get("returned_at") or ""), token=str(data.get("token") or ""),
        version=RECORD_VERSION,
    )


def write_record(record: DetourRecord, home: Any = None) -> Path:
    """Persist *record* atomically as indented UTF-8 JSON; returns the path written."""
    from utils import atomic_json_write

    path = record_path(record.scope, home)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_write(path, record.to_json(), indent=2)
    return path


def delete_record(scope: DetourScope, home: Any = None, *, expected_token: Optional[str] = None) -> bool:
    """Remove the lane's record (enter-leg rollback only). True when a file was removed.

    With *expected_token*, the file is removed ONLY when it is still the record that token owns —
    a rollback must never delete the record a *different*, already-succeeded detour just wrote.
    """
    path = record_path(scope, home)
    if expected_token is not None:
        try:
            current = read_record(scope, home)
        except DetourRecordError:
            current = None  # unreadable: a rollback may still clear its own failed attempt
        if current is not None and current.token and current.token != expected_token:
            logger.warning(
                "session_detour: refusing to delete %s — it belongs to another transition", path)
            return False
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False
    except OSError:
        logger.warning("session_detour: could not remove %s", path, exc_info=True)
        return False


def update_record(expected: DetourRecord, home: Any = None, **changes: Any) -> DetourRecord:
    """Compare-and-swap *expected* → ``replace(expected, **changes)`` on disk.

    The durable record is the only authority for "is a detour open here", so every write after the
    first re-reads it and refuses when the on-disk state is no longer the one the caller decided
    against — another process ended this detour, or started a new one, in the meantime. Fail-closed:
    the caller reports and leaves the live record alone.
    """
    current = read_record(expected.scope, home)
    if current is None:
        raise DetourRecordError("the detour record vanished while the detour was in progress")
    if expected.token and current.token != expected.token:
        raise DetourRecordError(
            "another detour transition took over this conversation; nothing was overwritten")
    if current.status != expected.status or current.parent_session_id != expected.parent_session_id:
        raise DetourRecordError(
            f"the detour record changed while the detour was in progress "
            f"(expected {expected.status!r}, found {current.status!r})")
    updated = replace(current, **changes)
    write_record(updated, home)
    return updated


def begin_detour(
    scope: DetourScope, *, parent_session_id: str, parent_route: Any = None, child_route: Any = None,
    home: Any = None,
) -> DetourRecord:
    """Write the ``entering`` record. Call BEFORE rotating the session, under :func:`scope_lock`."""
    if not _SESSION_ID_RE.match(str(parent_session_id or "")):
        raise DetourRecordError("the current session has no usable id to return to")
    record = DetourRecord(
        scope=scope, parent_session_id=str(parent_session_id), status=STATUS_ENTERING,
        profile=active_profile_name(), parent_route=parent_route, child_route=child_route,
        entered_at=_now(), token=uuid.uuid4().hex,
    )
    write_record(record, home)
    return record


def confirm_detour(record: DetourRecord, *, child_session_id: str, home: Any = None) -> DetourRecord:
    """Promote ``entering`` → ``active`` once the child session really exists.

    Also writes the child→lane index, so a surface with no stable routing key can find this record
    again from the child session alone after a restart. Index first: a record that says ``active``
    while nothing can find it from the child side is the unrecoverable state.
    """
    if not _SESSION_ID_RE.match(str(child_session_id or "")):
        raise DetourRecordError("the new session reported no usable id")
    write_child_index(str(child_session_id), record.scope, home)
    return update_record(record, home, status=STATUS_ACTIVE, child_session_id=str(child_session_id))


def mark_returned(record: DetourRecord, home: Any = None) -> DetourRecord:
    """Consume the record — ONLY after the parent restore was observed in live state."""
    updated = update_record(record, home, status=STATUS_RETURNED, returned_at=_now())
    if record.child_session_id:
        delete_child_index(record.child_session_id, home)
    return updated


# --------------------------------------------------------------------- child → lane index
#
# The gateway has a stable lane key (``session_key``) and can always recompute its own record path.
# A CLI cannot: its only identity is the session id, and the enter leg rotates it. So the child side
# gets a one-line index file, written under a digest of the child session id, naming the lane whose
# record to load. It is a POINTER, never an authority: the record it names is still read through
# :func:`read_record` (scope must match) and still has to pass every gate in
# :func:`resolve_return_target`. That is what keeps this different from scanning the directory and
# trusting whatever record turns up.

_CHILD_INDEX_VERSION = 1


def _child_index_path(child_session_id: str, home: Any = None) -> Path:
    digest = hashlib.sha256(f"child\x1f{child_session_id}".encode("utf-8")).hexdigest()[:40]
    return detour_root(home) / f"child-{digest}.json"


def write_child_index(child_session_id: str, scope: DetourScope, home: Any = None) -> Path:
    from utils import atomic_json_write

    path = _child_index_path(child_session_id, home)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_write(path, {
        "version": _CHILD_INDEX_VERSION, "child_session_id": str(child_session_id),
        "fingerprint": scope.fingerprint(), "scope": scope.to_json(),
    }, indent=2)
    return path


def scope_for_child(child_session_id: str, home: Any = None) -> Optional[DetourScope]:
    """The lane a child session detoured from, or None when there is no usable index entry."""
    child = str(child_session_id or "")
    if not child or not _SESSION_ID_RE.match(child):
        return None
    try:
        data = json.loads(_child_index_path(child, home).read_bytes().decode("utf-8-sig"))
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError, ValueError):
        logger.debug("session_detour: child index for %s unreadable", child, exc_info=True)
        return None
    if not isinstance(data, dict) or data.get("version") != _CHILD_INDEX_VERSION:
        return None
    if str(data.get("child_session_id") or "") != child:
        return None  # index moved or hand-edited
    scope = DetourScope.from_json(data.get("scope"))
    if scope is None or scope.fingerprint() != str(data.get("fingerprint") or ""):
        return None
    return scope


def delete_child_index(child_session_id: str, home: Any = None) -> bool:
    try:
        _child_index_path(str(child_session_id), home).unlink()
        return True
    except (FileNotFoundError, OSError):
        return False


def is_abandoned_enter(record: Optional[DetourRecord], *, current_session_id: str) -> bool:
    """True when an ``entering`` record's rotation never happened.

    The enter leg died between the record write and the session reset, so no child was ever
    created and the lane is still on the parent. There is nothing to return TO — but the open
    record would otherwise refuse every future ``/detour`` on this lane forever, so the return leg
    treats this one state as "clear it" rather than "restore".
    """
    return bool(
        record is not None
        and record.status == STATUS_ENTERING
        and not record.child_session_id
        and record.parent_session_id == str(current_session_id or "")
    )


def parse_route_args(args: str) -> tuple[Optional[Dict[str, str]], Optional[str]]:
    """``({model, provider}, None)`` for a detour's route arguments, or ``(None, error)``.

    Parsing goes through the single-owner ``/model`` parser so a detour accepts exactly what
    ``/model`` accepts, on every surface. ``--global`` and ``--once`` are refused: a detour's route
    belongs to the detour session and must neither outlive it nor reach ``config.yaml``.
    """
    from hermes_cli.model_switch import parse_model_switch_args

    text = (args or "").strip()
    if not text:
        return {"model": "", "provider": ""}, None
    request = parse_model_switch_args(text)
    if request.errors:
        return None, request.error_messages()[0]
    if request.is_global:
        return None, "a detour route is session-scoped; drop --global"
    if request.is_once:
        return None, "a detour route lasts for the detour; drop --once"
    return {"model": request.target or "", "provider": request.explicit_provider or ""}, None


def route_switch_command(route: Any) -> str:
    """The session-scoped ``/model`` command that applies *route*, or ``""`` for an empty route."""
    resolved = _route(route)
    if not (resolved["model"] or resolved["provider"]):
        return ""
    parts = [resolved["model"]] if resolved["model"] else []
    if resolved["provider"]:
        parts += ["--provider", resolved["provider"]]
    parts.append("--session")
    return "/model " + " ".join(parts)


def routes_match(requested: Any, effective: Any) -> bool:
    """True when *effective* (live runtime) satisfies *requested* (what the detour asked for).

    Runtime truth is not string equality: ``/model sonnet`` may resolve to a fully-qualified id, and
    a provider-only switch auto-detects the model. So an empty requested field means "don't care",
    and a requested model is satisfied by an effective model that is the same name or contains it.
    Anything else — notably "nothing was applied at all" — is a mismatch, which is the case this
    exists to catch: a leg must never report success on the strength of the reply text.
    """
    want, got = _route(requested), _route(effective)
    if not (want["model"] or want["provider"]):
        return True
    if want["provider"] and want["provider"].casefold() != got["provider"].casefold():
        return False
    if want["model"]:
        wanted, actual = want["model"].casefold(), got["model"].casefold()
        if not actual or (wanted not in actual and actual not in wanted):
            return False
    return True


def resolve_return_target(
    record: Optional[DetourRecord], scope: DetourScope, *, current_session_id: str,
    session_exists: Any = None, lineage_parent: Optional[Callable[[str], Optional[str]]] = None,
) -> str:
    """The parent session id a return may restore, or raise :class:`DetourRecordError`.

    Four independent gates, all fail-closed:

    * an open record must exist (a missing or already-returned one is the caller's no-op case);
    * the record's scope must match the caller's lane (also checked in :func:`read_record`);
    * the caller must be sitting IN the recorded child session — this is what stops a different
      conversation in the same lane from using the record as a jump to the parent;
    * the parent session must still exist, per *session_exists* (a callable the surface supplies
      from its own session store).
    """
    if record is None or not record.is_open:
        raise DetourRecordError("no detour is open for this conversation")
    if record.scope.parts != scope.parts:
        raise DetourRecordError("the detour record belongs to another conversation")
    current = str(current_session_id or "")
    if record.child_session_id and current and record.child_session_id != current:
        raise DetourRecordError(
            f"this conversation ({current}) is not the detour's session "
            f"({record.child_session_id}); resume that session first")
    if not record.child_session_id and current and current != record.parent_session_id:
        # The enter leg rotated but never managed to bind the child id (a write failed, or the
        # process died between the two). Lineage is the fallback authority: the session store's own
        # parent link for THIS session must name the record's parent. Without that proof the record
        # is not usable from here — "some open record exists in this lane" is not identity.
        proven = str((lineage_parent(current) if lineage_parent else None) or "")
        if proven != record.parent_session_id:
            raise DetourRecordError(
                f"this conversation ({current}) is not the detour's session and its lineage does "
                f"not lead back to {record.parent_session_id}; resume that session first")
    target = record.parent_session_id
    if not _SESSION_ID_RE.match(target):
        raise DetourRecordError("the detour record names an invalid parent session id")
    if target == current:
        raise DetourRecordError("already on the detour's parent session")
    if session_exists is not None and not session_exists(target):
        raise DetourRecordError(f"the parent session {target} no longer exists")
    return target
