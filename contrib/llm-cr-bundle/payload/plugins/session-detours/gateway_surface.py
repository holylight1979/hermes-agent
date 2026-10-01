"""Gateway ``/detour`` and ``/detour-end`` — leave for a fresh child session, come back to this one.

Both legs are compositions of the NATIVE session commands, never reimplementations:

* ``/detour`` writes the lane's return record, then runs ``_handle_reset_command`` (the same code
  path as ``/new``: run-generation bump, agent eviction, conversation-scope clear, in-flight
  delegation expiry, ``reset_session`` with ``parent_session_id`` set, session hooks), then applies
  the requested route through ``_handle_model_command`` as a ``--session`` override.
* ``/detour-end`` runs ``_handle_resume_command`` — including its IDOR guard — against the recorded
  parent id, then applies the requested route the same way.

Consequences of composing rather than copying: the fresh session starts with an empty transcript
(``reset_session`` creates a new row; nothing is copied in), the parent's transcript is untouched
while the detour runs, and the return leg replays the parent's own history exactly as ``/resume``
would. Neither side ever sees the other's messages.

Success is decided by STATE, not by the reply text: each leg re-reads the routed session id from
the session store and compares it to what it asked for. The record is only promoted to ``active``
after the child id is observed, and only consumed after the parent id is observed back on the lane.
A failed leg leaves the record exactly as it was, so the return is still available.

This is conversation isolation, not a sandbox: the child session has the same tools, permissions
and filesystem reach as any other session.
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import logging
from typing import Any, Optional, Tuple

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
)

logger = logging.getLogger(__name__)

_SURFACE = "gateway"


class GatewayDetourCommandsMixin:
    """``/detour`` + ``/detour-end`` for GatewayRunner."""

    def _detour_scope(self, source: Any, session_key: str) -> DetourScope:
        platform = getattr(getattr(source, "platform", None), "value", "") or ""
        return DetourScope(
            surface=_SURFACE,
            channel=f"{platform}:{getattr(source, 'chat_id', '') or ''}",
            owner=str(getattr(source, "user_id", "") or ""),
            lane=str(session_key or ""),
        )

    def _detour_parent_route(self, session_key: str) -> dict:
        """The lane's live ``/model`` override, or empty strings for "whatever config resolves to"."""
        override = self._session_model_override(session_key) or {}
        return {"model": str(override.get("model") or ""), "provider": str(override.get("provider") or "")}

    def _detour_sync_session_db(self) -> Any:
        """The SYNC ``SessionDB`` behind the runner's async facade.

        ``self._session_db`` is an ``AsyncSessionDB``: every attribute access returns a coroutine
        function. Calling ``get_session`` on it from a worker thread yields a coroutine object, which
        is truthy — so an existence check against the facade would pass for a session that is not
        there. The detour's gates have to be real, so they go through ``_db``.
        """
        db = getattr(self, "_session_db", None)
        sync = getattr(db, "_db", db)
        if inspect.iscoroutinefunction(getattr(sync, "get_session", None)):
            # Still an async facade (the private handle moved, or a double was installed). A
            # coroutine object is truthy, so answering from it would make every existence gate
            # pass. Report "cannot tell" instead and let the gates fail closed.
            logger.warning("detour: no synchronous session store available; gates fail closed")
            return None
        return sync

    def _detour_session_exists(self, session_id: str) -> bool:
        db = self._detour_sync_session_db()
        if db is None:
            return False
        try:
            return db.get_session(session_id) is not None
        except Exception:
            logger.debug("detour: session existence check failed for %s", session_id, exc_info=True)
            return False

    def _detour_lineage_parent(self, session_id: str) -> Optional[str]:
        """The session store's own ``parent_session_id`` for *session_id* (``reset_session`` sets it)."""
        db = self._detour_sync_session_db()
        if db is None:
            return None
        try:
            row = db.get_session(session_id) or {}
            return str(row.get("parent_session_id") or "") or None
        except Exception:
            logger.debug("detour: lineage lookup failed for %s", session_id, exc_info=True)
            return None

    async def _detour_apply_route(self, event: Any, session_key: str, route: dict) -> Tuple[bool, str]:
        """Apply *route* as a session-scoped ``/model`` switch; ``(applied, reply)``.

        ``applied`` is read back from the lane's live override, not from the handler's reply text: a
        leg that claims a model it did not get is worse than one that refuses.
        """
        text = route_switch_command(route)
        if not text:
            return True, ""
        try:
            reply = str(await self._handle_model_command(dataclasses.replace(event, text=text)) or "")
        except Exception as exc:
            logger.warning("detour: route switch failed: %s", exc, exc_info=True)
            return False, f"the model switch failed ({type(exc).__name__})"
        if not routes_match(route, self._session_model_override(session_key) or {}):
            logger.warning("detour: route %s did not take effect on %s", route, session_key)
            return False, f"the model switch did not take effect ({reply.strip() or 'no reason given'})"
        return True, reply

    # ------------------------------------------------------------------ /detour
    async def _handle_detour_command(self, event: Any) -> str:
        """``/detour [model] [--provider name]`` — fresh child session, remembering the way back.

        The lane's transition lock is held for the WHOLE leg — decide, rotate, bind, switch — not
        just the record write. A second ``/detour`` (or a ``/detour-end``) arriving mid-rotation
        would otherwise read a lane that is halfway between two sessions.
        """
        route, error = parse_route_args(event.get_command_args())
        if error is not None:
            return f"❌ {error}"
        source = await asyncio.to_thread(self._normalize_source_for_session_key, event.source)
        session_key = self._session_key_for_source(source)
        scope = self._detour_scope(source, session_key)
        lock = LaneTransitionLock(scope)
        try:
            await asyncio.to_thread(lock.acquire)
        except LaneLockError as exc:
            # Fail closed: no record written, no session rotated, nothing to clean up.
            return f"❌ Detour not started — {exc}."
        try:
            return await self._detour_enter_locked(event, source, session_key, scope, route)
        finally:
            await asyncio.to_thread(lock.release)

    async def _detour_enter_locked(self, event, source, session_key: str, scope: DetourScope,
                                   route: dict) -> str:
        entry = await self.async_session_store.get_or_create_session(source)
        parent_id = getattr(entry, "session_id", "") or ""

        def _open() -> Any:
            existing = read_record(scope)
            if existing is not None and existing.is_open:
                return existing.status, existing
            return None, begin_detour(
                scope, parent_session_id=parent_id,
                parent_route=self._detour_parent_route(session_key), child_route=route,
            )

        try:
            blocked, record = await asyncio.to_thread(_open)
        except DetourRecordError as exc:
            return f"❌ Detour not started — {exc}."
        except Exception as exc:
            logger.warning("detour: could not write the return record: %s", exc, exc_info=True)
            return f"❌ Detour not started — the return record could not be written ({type(exc).__name__})."
        if blocked is not None:
            return (
                f"⚠️ A detour is already open here (session `{record.child_session_id or '—'}`, "
                f"returning to `{record.parent_session_id}`). Use `/detour-end` first — "
                "detours do not nest.")

        child_id = await self._handle_detour_reset(event) or ""
        if not child_id or child_id == parent_id:
            # No child session exists, so the record would point at a detour that never happened.
            # Only this transition's own record is removed (token-checked).
            await asyncio.to_thread(lambda: delete_record(scope, expected_token=record.token))
            if not await asyncio.to_thread(self._detour_session_exists, parent_id):
                return ("❌ Detour not started — the fresh session could not be created, and the "
                        f"original session `{parent_id}` is no longer in the store. "
                        f"Recover with `/resume {parent_id}`.")
            return "❌ Detour not started — the fresh session could not be created. Nothing changed."

        bound = True
        try:
            record = await asyncio.to_thread(lambda: confirm_detour(record, child_session_id=child_id))
        except Exception as exc:
            # The child IS live but nothing durable binds it to the parent. Do not claim success:
            # roll the lane back onto the parent so the user keeps the conversation they had.
            bound = False
            logger.warning("detour: could not confirm the return record: %s", exc, exc_info=True)

        applied, route_note = await self._detour_apply_route(event, session_key, route)
        if bound and applied:
            head = (
                f"🔀 Detour started — fresh session `{child_id}`, no history carried over.\n"
                f"`/detour-end` returns to `{parent_id}` and that conversation's history.")
            return f"{head}\n{route_note}" if route_note else head
        reason = ("the return record could not be bound to the new session"
                  if not bound else route_note)
        return await self._detour_rollback_enter(event, session_key, scope, record, parent_id,
                                                 child_id, reason)

    async def _detour_rollback_enter(self, event, session_key: str, scope: DetourScope,
                                     record: DetourRecord, parent_id: str, child_id: str,
                                     reason: str) -> str:
        """Put the lane back on the parent after a failed enter leg, through native ``/resume``.

        If the rollback itself cannot land, the record is deliberately LEFT in place and named: it is
        the only durable thing that still knows where this conversation came from.
        """
        await self._handle_resume_command(dataclasses.replace(event, text=f"/resume {parent_id}"))
        if (self.session_store.peek_session_id(session_key) or "") != parent_id:
            return (f"❌ Detour not started — {reason}, and returning to `{parent_id}` failed too.\n"
                    f"This conversation is on `{child_id}`. Recover with `/resume {parent_id}`.\n"
                    f"Record kept: `{record_path(scope)}`")

        def _clear() -> None:
            delete_record(scope, expected_token=record.token)
            if child_id:
                delete_child_index(child_id)

        await asyncio.to_thread(_clear)
        await self._detour_apply_route(event, session_key, record.parent_route)
        return (f"❌ Detour not started — {reason}. Back on `{parent_id}` with its history; "
                "nothing was left open.")

    async def _handle_detour_reset(self, event: Any) -> Optional[str]:
        """Run the native ``/new`` reset and report the session id the lane actually routes to."""
        source = await asyncio.to_thread(self._normalize_source_for_session_key, event.source)
        session_key = self._session_key_for_source(source)
        try:
            await self._handle_reset_command(dataclasses.replace(event, text="/new"))
        except Exception as exc:
            logger.warning("detour: native session reset failed: %s", exc, exc_info=True)
            return None
        return self.session_store.peek_session_id(session_key)

    # ------------------------------------------------------------------ /detour-end
    async def _handle_detour_end_command(self, event: Any) -> str:
        """``/detour-end [model] [--provider name]`` — restore the recorded parent session."""
        route, error = parse_route_args(event.get_command_args())
        if error is not None:
            return f"❌ {error}"
        source = await asyncio.to_thread(self._normalize_source_for_session_key, event.source)
        session_key = self._session_key_for_source(source)
        scope = self._detour_scope(source, session_key)
        lock = LaneTransitionLock(scope)
        try:
            await asyncio.to_thread(lock.acquire)
        except LaneLockError as exc:
            # The detour stays open and untouched, so the return is still retryable.
            return f"❌ Could not end the detour — {exc}."
        try:
            return await self._detour_end_locked(event, session_key, scope, route)
        finally:
            await asyncio.to_thread(lock.release)

    async def _detour_end_locked(self, event, session_key: str, scope: DetourScope,
                                 route: dict) -> str:
        current_id = self.session_store.peek_session_id(session_key) or ""

        def _resolve() -> Tuple[Optional[DetourRecord], Optional[str]]:
            record = read_record(scope)
            if is_abandoned_enter(record, current_session_id=current_id):
                # The enter leg never rotated; clear the record so the lane is not wedged.
                mark_returned(record)
                return record, None
            return record, resolve_return_target(
                record, scope, current_session_id=current_id,
                session_exists=self._detour_session_exists,
                lineage_parent=self._detour_lineage_parent,
            )

        try:
            record, target = await asyncio.to_thread(_resolve)
        except DetourRecordError as exc:
            # Fail closed: no session is created or switched, and the record (if any) is untouched.
            return self._detour_end_refusal(scope, str(exc))
        except Exception as exc:
            logger.warning("detour: return record unreadable: %s", exc, exc_info=True)
            return f"❌ Could not end the detour — the return record is unusable ({type(exc).__name__})."
        if target is None:
            return "The detour never started — the record is cleared and this conversation is unchanged."

        reply = await self._handle_resume_command(dataclasses.replace(event, text=f"/resume {target}"))
        if (self.session_store.peek_session_id(session_key) or "") != target:
            # The native resume refused (ownership guard, missing transcript, ...). Keep the record,
            # so the return is still retryable.
            return f"❌ Could not return to `{target}` — the detour stays open.\n{reply}"

        # The resume cleared this lane's conversation-scoped state, the parent's /model override
        # included, so the parent's route has to be re-applied or the user comes back to a session
        # that is silently on the config default. An explicit argument wins over the recorded route.
        wanted = route if (route.get("model") or route.get("provider")) else record.parent_route
        applied, route_note = await self._detour_apply_route(event, session_key, wanted)
        try:
            await asyncio.to_thread(lambda: mark_returned(record))
        except Exception as exc:
            logger.warning("detour: could not mark the record returned: %s", exc, exc_info=True)
        head = f"↩️ Back on `{target}` — the detour session is archived, not deleted."
        if not applied:
            retry = route_switch_command(wanted)
            route_note = (f"⚠️ The conversation is restored but its model was not — {route_note}. "
                          f"Re-apply with `{retry}`.")
        return "\n".join(part for part in (head, reply, route_note) if part)

    def _detour_end_refusal(self, scope: DetourScope, reason: str) -> str:
        """Refusal copy. A plain "nothing open" is a quiet no-op; anything else names the file so an
        operator can inspect (or remove) it."""
        if reason == "no detour is open for this conversation":
            return "Nothing to end — no detour is open for this conversation."
        return f"❌ Could not end the detour — {reason}.\nRecord: `{record_path(scope)}`"
