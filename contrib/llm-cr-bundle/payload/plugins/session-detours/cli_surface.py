"""CLI ``/detour`` and ``/detour-end`` — leave for a fresh side session, then come back.

The CLI half of the same round trip the gateway implements, built the same way: out of the native
session commands, never around them.

* ``/detour`` writes the lane's return record, then calls :meth:`new_session` — the ``/new`` path,
  with its memory-boundary flush, session-DB rotation, agent state reset and prompt invalidation —
  then applies the requested route through ``/model ... --session``.
* ``/detour-end`` calls ``_handle_resume_command`` against the recorded parent id, so the parent's
  transcript is replayed by the same code a hand-typed ``/resume`` runs, then restores the parent's
  own route (``/new`` and ``/resume`` both reset model/provider to the config default, so coming
  back to "the session I left" means re-applying what the record saved).

Nothing copies history in either direction: ``new_session`` starts with an empty
``conversation_history`` and the resume leg loads the parent's own rows from the session DB.

Lane scope. A CLI has no routing key and no stable identity across the rotation, so the record is
keyed on the ORIGINATING session id (plus OS user, inside the active Hermes home). Two CLIs that
detour at the same time therefore write two records and never see each other's. The return leg,
which only knows the *child* session, finds its record the way the enter leg left it findable:

1. the in-process handle, when it is the same CLI that entered;
2. the child→lane index file written at confirm time (survives a restart);
3. failing both, the session store's own parent link for the current session.

None of those is trusted on its own — whatever they name is still read through ``read_record``
(scope must match the recomputed digest) and still has to pass every gate in
``resolve_return_target``. The directory is never scanned for "some open record".
"""

from __future__ import annotations

import getpass
import logging
from typing import Optional

from .detour_records import (
    DetourRecord,
    DetourRecordError,
    DetourScope,
    LaneLockError,
    LaneTransitionLock,
    begin_detour,
    confirm_detour,
    delete_child_index,
    delete_record,
    is_abandoned_enter,
    mark_returned,
    parse_route_args,
    read_record,
    record_path,
    resolve_return_target,
    route_switch_command,
    routes_match,
    scope_for_child,
)

logger = logging.getLogger(__name__)

_SURFACE = "cli"


def _os_user() -> str:
    try:
        return getpass.getuser() or ""
    except Exception:
        return ""


class CLIDetourMixin:
    """``/detour`` + ``/detour-end`` for HermesCLI."""

    #: Set while this process holds an open detour, so the return leg does not have to rediscover
    #: the lane it just created. Never the only path: a restart clears it (see the index file).
    _detour_open_scope: Optional[DetourScope] = None

    # ------------------------------------------------------------------ scope / lookups
    def _detour_scope_for_parent(self, parent_session_id: str) -> DetourScope:
        """The lane for a detour that leaves *parent_session_id* — one record per originating session."""
        return DetourScope(surface=_SURFACE, channel="local", owner=_os_user(),
                           lane=str(parent_session_id or ""))

    def _detour_session_exists(self, session_id: str) -> bool:
        db = getattr(self, "_session_db", None)
        if db is None:
            return False
        try:
            return db.get_session(session_id) is not None
        except Exception:
            logger.debug("detour: session existence check failed for %s", session_id, exc_info=True)
            return False

    def _detour_lineage_parent(self, session_id: str) -> Optional[str]:
        """The session row's own ``parent_session_id``, used only to prove a child's lineage."""
        db = getattr(self, "_session_db", None)
        if db is None or not session_id:
            return None
        try:
            row = db.get_session(session_id) or {}
            return str(row.get("parent_session_id") or "") or None
        except Exception:
            logger.debug("detour: lineage lookup failed for %s", session_id, exc_info=True)
            return None

    def _detour_discover_scope(self, current_id: str) -> Optional[DetourScope]:
        """The lane whose detour this session is sitting in, or None."""
        if self._detour_open_scope is not None:
            return self._detour_open_scope
        from_index = scope_for_child(current_id)
        if from_index is not None:
            return from_index
        parent = self._detour_lineage_parent(current_id)
        return self._detour_scope_for_parent(parent) if parent else None

    def _detour_open_lane(self, current_id: str):
        """``(scope, record)`` when *current_id* is already INSIDE an open detour, else None.

        The CLI's lane key is the originating session id, so the lane a ``/detour`` would create
        from inside a detour is a different one from the lane that is already open — checking only
        the new lane would find nothing and happily nest, orphaning the first child session and
        stranding its record (its ``child_session_id`` is no longer the live session, so the return
        leg would refuse forever). This looks at the lane the CURRENT session belongs to.

        A lane whose own key is this session is deliberately NOT reported: that is the abandoned
        enter (record written, rotation never happened), which :meth:`_detour_enter_locked` already
        refuses with the same copy and ``/detour-end`` can clear.
        """
        scope = self._detour_discover_scope(current_id)
        if scope is None or scope.parts == self._detour_scope_for_parent(current_id).parts:
            return None
        try:
            with LaneTransitionLock(scope):
                record = read_record(scope)
        except LaneLockError:
            # Nothing was read and nothing was written. The command refuses as a whole.
            raise
        except DetourRecordError as exc:
            # Unreadable, but something IS open here. Fail closed rather than nest.
            logger.warning("detour: open-lane record unreadable: %s", exc)
            return scope, None
        except Exception:
            logger.debug("detour: open-lane lookup failed for %s", current_id, exc_info=True)
            return None
        return (scope, record) if (record is not None and record.is_open) else None

    def _detour_refuse_nested(self, scope: DetourScope, record: Optional[DetourRecord]) -> None:
        from cli import _cprint

        if record is None:
            return _cprint(
                "  ✗ A detour is already open for this conversation, but its record could not be "
                f"read.\n    Use /detour-end, or inspect {record_path(scope)}.")
        return _cprint(
            f"  ✗ A detour is already open (session {record.child_session_id or '—'}, "
            f"returning to {record.parent_session_id}).\n"
            "    Use /detour-end first — detours do not nest.")

    # ------------------------------------------------------------------ route plumbing
    def _detour_route_args(self, cmd_original: str):
        """``({model, provider}, None)`` for the trailing ``/model`` arguments, or ``(None, error)``."""
        parts = (cmd_original or "").split(None, 1)
        return parse_route_args(parts[1] if len(parts) > 1 else "")

    def _detour_live_route(self) -> dict:
        return {"model": getattr(self, "model", "") or "", "provider": getattr(self, "provider", "") or ""}

    def _detour_apply_route(self, route: dict) -> bool:
        """Apply *route* as a session-scoped switch; True when the LIVE model/provider then match it."""
        command = route_switch_command(route)
        if not command:
            return True
        try:
            self._handle_model_switch(command)
        except Exception as exc:
            logger.warning("detour: route switch failed: %s", exc, exc_info=True)
            return False
        if not routes_match(route, self._detour_live_route()):
            logger.warning("detour: route %s did not take effect (live %s)", route, self._detour_live_route())
            return False
        return True

    # ------------------------------------------------------------------ /detour
    def _handle_detour_command(self, cmd_original: str) -> None:
        """``/detour [model] [--provider name]`` — fresh side session, remembering the way back."""
        from cli import _cprint

        route, error = self._detour_route_args(cmd_original)
        if error is not None:
            return _cprint(f"  ✗ {error}")
        parent_id = getattr(self, "session_id", "") or ""
        scope = self._detour_scope_for_parent(parent_id)
        # One lock for the whole transition: decide, rotate, bind, switch. A lane that cannot be
        # locked refuses here, before any session or record is touched.
        try:
            already_open = self._detour_open_lane(parent_id)
            if already_open is not None:
                return self._detour_refuse_nested(*already_open)
            lock = LaneTransitionLock(scope).acquire()
        except LaneLockError as exc:
            return _cprint(f"  ✗ Detour not started — {exc}.")
        try:
            self._detour_enter_locked(scope, parent_id, route)
        finally:
            lock.release()

    def _detour_enter_locked(self, scope: DetourScope, parent_id: str, route: dict) -> None:
        from cli import _cprint

        try:
            existing = read_record(scope)
            if existing is not None and existing.is_open:
                return _cprint(
                    f"  ✗ A detour is already open (session {existing.child_session_id or '—'}, "
                    f"returning to {existing.parent_session_id}).\n"
                    "    Use /detour-end first — detours do not nest.")
            record = begin_detour(
                scope, parent_session_id=parent_id, parent_route=self._detour_live_route(),
                child_route=route)
        except DetourRecordError as exc:
            return _cprint(f"  ✗ Detour not started — {exc}.")
        except Exception as exc:
            logger.warning("detour: could not write the return record: %s", exc, exc_info=True)
            return _cprint(
                f"  ✗ Detour not started — the return record could not be written ({type(exc).__name__}).")

        try:
            self.new_session(silent=True)
        except Exception as exc:
            # ``new_session`` can fail partway: it may already have ended (and pruned) the parent row
            # before raising. Check live state rather than assuming nothing moved.
            logger.warning("detour: native new_session failed: %s", exc, exc_info=True)
            return self._detour_abort_enter(scope, record, parent_id, route,
                                            "the fresh session could not be created")
        child_id = getattr(self, "session_id", "") or ""
        if not child_id or child_id == parent_id:
            return self._detour_abort_enter(scope, record, parent_id, route,
                                            "the session did not rotate")
        # ``new_session`` prunes a parent row that never gained content, which would leave the return
        # leg with nothing to resume. Put the row back (empty, which is what it was) so an immediate
        # /detour from a brand-new session is still returnable.
        self._detour_ensure_parent_row(parent_id, record)

        try:
            record = confirm_detour(record, child_session_id=child_id)
        except Exception as exc:
            # The child is live but nothing durable binds it to the parent. Do not report success:
            # go back to the parent so the user keeps the conversation they had.
            logger.warning("detour: could not confirm the return record: %s", exc, exc_info=True)
            return self._detour_abort_enter(
                scope, record, parent_id, route,
                "the return record could not be bound to the new session", child_id=child_id)

        self._detour_open_scope = scope
        if not self._detour_apply_route(route):
            return self._detour_abort_enter(
                scope, record, parent_id, route, "the model switch did not take effect",
                child_id=child_id)
        _cprint(f"  ⤳ Detour started — fresh session {child_id}, no history carried over.")
        _cprint(f"    /detour-end returns to {parent_id} and that conversation's history.")

    def _detour_ensure_parent_row(self, parent_id: str, record: DetourRecord) -> None:
        """Make sure the parent session still has a row to come back to."""
        db = getattr(self, "_session_db", None)
        if db is None or not parent_id or self._detour_session_exists(parent_id):
            return
        try:
            import os as _os

            db.ensure_session(
                parent_id, source=_os.environ.get("HERMES_SESSION_SOURCE", "cli"),
                model=record.parent_route.get("model") or None)
            logger.info("detour: restored the pruned empty parent row %s", parent_id)
        except Exception:
            logger.warning("detour: could not restore the parent row %s", parent_id, exc_info=True)

    def _detour_abort_enter(self, scope: DetourScope, record: DetourRecord, parent_id: str,
                            route: dict, reason: str, *, child_id: str = "") -> None:
        """Undo a failed enter leg: back onto the parent through native ``/resume``, record cleared.

        A rollback that cannot land leaves the record in place ON PURPOSE and names it — it is the
        only durable thing that still knows where this conversation came from.
        """
        from cli import _cprint

        self._detour_open_scope = None
        current = getattr(self, "session_id", "") or ""
        if current and current != parent_id:
            self._detour_ensure_parent_row(parent_id, record)
            try:
                self._handle_resume_command(f"/resume {parent_id}")
            except Exception:
                logger.warning("detour: rollback resume failed for %s", parent_id, exc_info=True)
            if (getattr(self, "session_id", "") or "") != parent_id:
                return _cprint(
                    f"  ✗ Detour not started — {reason}, and returning to {parent_id} failed too.\n"
                    f"    This session is {getattr(self, 'session_id', '') or '—'}; "
                    f"recover with /resume {parent_id}\n"
                    f"    Record kept: {record_path(scope)}")
            self._detour_apply_route(record.parent_route)
        delete_record(scope, expected_token=record.token)
        if child_id:
            delete_child_index(child_id)
        _cprint(f"  ✗ Detour not started — {reason}. Still on {parent_id}; nothing was left open.")

    # ------------------------------------------------------------------ /detour-end
    def _handle_detour_end_command(self, cmd_original: str) -> None:
        """``/detour-end [model] [--provider name]`` — restore the recorded parent session."""
        from cli import _cprint

        route, error = self._detour_route_args(cmd_original)
        if error is not None:
            return _cprint(f"  ✗ {error}")
        current_id = getattr(self, "session_id", "") or ""
        scope = self._detour_discover_scope(current_id)
        if scope is None:
            return _cprint("  Nothing to end — no detour is open for this session.")
        try:
            lock = LaneTransitionLock(scope).acquire()
        except LaneLockError as exc:
            # The detour stays exactly as it is, so the return is still retryable.
            return _cprint(f"  ✗ Could not end the detour — {exc}.")
        try:
            self._detour_end_locked(scope, current_id, route)
        finally:
            lock.release()

    def _detour_end_locked(self, scope: DetourScope, current_id: str, route: dict) -> None:
        from cli import _cprint

        try:
            record = read_record(scope)
            if is_abandoned_enter(record, current_session_id=current_id):
                # The enter leg never rotated; clear it so the lane is not wedged.
                mark_returned(record)
                self._detour_open_scope = None
                return _cprint(
                    "  The detour never started — record cleared, this session is unchanged.")
            target = resolve_return_target(
                record, scope, current_session_id=current_id,
                session_exists=self._detour_session_exists,
                lineage_parent=self._detour_lineage_parent)
        except DetourRecordError as exc:
            # Fail closed: no session is created or switched, and the record is left untouched.
            if str(exc) == "no detour is open for this conversation":
                return _cprint("  Nothing to end — no detour is open.")
            return _cprint(f"  ✗ Could not end the detour — {exc}.\n    Record: {record_path(scope)}")
        except Exception as exc:
            logger.warning("detour: return record unreadable: %s", exc, exc_info=True)
            return _cprint(
                f"  ✗ Could not end the detour — the return record is unusable ({type(exc).__name__}).")

        try:
            self._handle_resume_command(f"/resume {target}")
        except Exception as exc:
            logger.warning("detour: native resume failed for %s: %s", target, exc, exc_info=True)
        if (getattr(self, "session_id", "") or "") != target:
            # Keep the record: the return is still retryable.
            return _cprint(f"  ✗ Could not return to {target} — the detour stays open.")

        # ``/resume`` re-derives model/provider from config, so the parent's own route has to be
        # re-applied or "back on the earlier session" would quietly mean "on the default model".
        wanted = route if (route.get("model") or route.get("provider")) else record.parent_route
        applied = self._detour_apply_route(wanted)
        try:
            mark_returned(record)
        except Exception as exc:
            logger.warning("detour: could not mark the record returned: %s", exc, exc_info=True)
        self._detour_open_scope = None
        _cprint("  ↩ Back on the earlier session — the detour session is archived, not deleted.")
        if not applied:
            _cprint(f"    ⚠ Its model was not restored. Re-apply with: {route_switch_command(wanted)}")
