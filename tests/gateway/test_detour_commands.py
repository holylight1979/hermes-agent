"""``/detour`` + ``/detour-end`` on the gateway, against a real SessionStore and SessionDB.

The runner is the real :class:`GatewayRunner` with only surface plumbing stubbed (adapters, hooks,
agent cache); the session store, the session database, ``reset_session``, ``switch_session`` and
the ``/new`` / ``/resume`` handlers are production code. So these assert what the user experiences:
which session the lane routes to next, which transcript each session holds, and what the return
record says at each step.
"""

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource, SessionStore
from tests.fakes.session_detours_plugin import detour_gateway
from tests.fakes.session_detours_plugin import (
    STATUS_ACTIVE,
    STATUS_ENTERING,
    STATUS_RETURNED,
    DetourScope,
    read_record,
    record_path,
)


def _source(user="u1", chat="c1") -> SessionSource:
    return SessionSource(platform=Platform.TELEGRAM, user_id=user, chat_id=chat,
                         user_name="tester", chat_type="dm")


def _event(text: str, source=None) -> MessageEvent:
    return MessageEvent(text=text, source=source or _source(), message_id="m1")


@pytest.fixture
def runner(tmp_path, monkeypatch):
    """Real GatewayRunner methods over a real store/DB in a temp home."""
    from gateway.run import GatewayRunner
    from hermes_state import AsyncSessionDB, SessionDB

    home = tmp_path / ".hermes"
    (home / "sessions").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    r = object.__new__(GatewayRunner)
    r.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")})
    r.session_store = SessionStore(home / "sessions", r.config)
    sync_db = SessionDB(home / "state.db")
    # Production pins an AsyncSessionDB here (every attribute is a coroutine function); the store
    # keeps the sync handle. Anything that quietly worked against a bare SessionDB would be a test
    # artefact — ``/resume`` awaits these calls.
    r._session_db = AsyncSessionDB(sync_db)
    r.session_store._db = sync_db

    adapter = MagicMock()
    adapter.send = AsyncMock()
    r.adapters = {Platform.TELEGRAM: adapter}
    r.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    r._session_model_overrides = {}
    r._session_reasoning_overrides = {}
    r._pending_model_notes = {}
    r._background_tasks = set()
    r._running_agents = {}
    r._pending_messages = {}
    r._pending_approvals = {}
    r._agent_cache = {}
    r._agent_cache_lock = None
    r._voice_mode = {}
    r._draining = False
    r._is_user_authorized = lambda _s: True
    r._format_session_info = lambda: ""
    r._reset_notice_session_info = lambda _s: ""
    r._telegram_topic_new_header = lambda _s: ""
    r._is_telegram_topic_lane = lambda _s: False
    r._resume_caller_is_admin = lambda _s: True

    # The one thing we do not want in a test: a real model switch (network + credentials). The stand-in
    # still does the part the detour VERIFIES — publish the lane's session-scoped override — so a leg
    # that only looked at the reply text would fail here.
    applied: list[str] = []

    async def _fake_model_command(event):
        applied.append(event.text)
        if getattr(r, "model_switch_is_broken", False):
            return "❌ could not switch"
        parts = event.text.split()[1:]
        model = parts[0] if parts and not parts[0].startswith("--") else ""
        provider = parts[parts.index("--provider") + 1] if "--provider" in parts else ""
        key = r._session_key_for_source(r._normalize_source_for_session_key(event.source))
        r._session_state(key).conversation.model_override = {"model": model, "provider": provider}
        return f"✓ route applied: {event.text}"

    r._handle_model_command = _fake_model_command
    r.model_switch_is_broken = False
    r.applied_routes = applied
    r.home = home
    return r


def _scope(runner, source=None) -> DetourScope:
    source = source or _source()
    return detour_gateway(runner)._detour_scope(source, runner._session_key_for_source(source))


def _routed(runner, source=None) -> str:
    source = source or _source()
    return runner.session_store.peek_session_id(runner._session_key_for_source(source)) or ""


def _session_ids(runner) -> set:
    """Every session row in the temp DB — proves a refused command created nothing."""
    import sqlite3

    with sqlite3.connect(runner.home / "state.db") as conn:
        return {row[0] for row in conn.execute("SELECT id FROM sessions")}


async def _seed_parent(runner, source=None, texts=("hello parent",)):
    """Create the lane's first session and give it a transcript.

    Roles alternate: ``append_to_transcript`` merges consecutive same-role messages, so a
    user/user seed would read back as one blob and hide which turns actually crossed over.
    """
    source = source or _source()
    entry = await runner.async_session_store.get_or_create_session(source)
    for index, text in enumerate(texts):
        runner.session_store.append_to_transcript(
            entry.session_id, {"role": "user" if index % 2 == 0 else "assistant", "content": text})
    return entry.session_id


# --------------------------------------------------------------------------- the round trip
@pytest.mark.asyncio
async def test_detour_rotates_to_a_fresh_empty_session_and_records_the_way_back(runner):
    parent = await _seed_parent(runner)

    reply = await detour_gateway(runner)._handle_detour_command(_event("/detour some-model --provider some-provider"))

    child = _routed(runner)
    assert child and child != parent
    # The child is genuinely fresh: its own row, no messages carried in.
    assert runner.session_store.load_transcript(child) == []
    # The parent's transcript is untouched and still its own.
    assert [m["content"] for m in runner.session_store.load_transcript(parent)] == ["hello parent"]
    # The record names both ends and is active.
    record = read_record(_scope(runner))
    assert record.status == STATUS_ACTIVE
    assert (record.parent_session_id, record.child_session_id) == (parent, child)
    assert child in reply and parent in reply
    # The route was applied as a session-scoped override, never globally.
    assert runner.applied_routes == ["/model some-model --provider some-provider --session"]


@pytest.mark.asyncio
async def test_detour_end_returns_to_the_parent_with_its_own_history(runner):
    parent = await _seed_parent(runner, texts=("hello parent", "second parent turn"))
    await detour_gateway(runner)._handle_detour_command(_event("/detour some-model --provider some-provider"))
    child = _routed(runner)
    runner.session_store.append_to_transcript(child, {"role": "user", "content": "child only"})

    reply = await detour_gateway(runner)._handle_detour_end_command(_event("/detour-end back-model --provider back-provider"))

    assert _routed(runner) == parent
    assert parent in reply
    # Parent history is exactly what it was; the child's turn never crossed over.
    parent_texts = [m["content"] for m in runner.session_store.load_transcript(parent)]
    assert parent_texts == ["hello parent", "second parent turn"]
    assert "child only" not in parent_texts
    # The child keeps its own history too (archived, not deleted).
    assert [m["content"] for m in runner.session_store.load_transcript(child)] == ["child only"]
    assert runner._session_db._db.get_session(child) is not None
    # Record consumed only after the restore landed.
    assert read_record(_scope(runner)).status == STATUS_RETURNED
    assert runner.applied_routes[-1] == "/model back-model --provider back-provider --session"


@pytest.mark.asyncio
async def test_the_child_never_sees_the_parents_history_in_either_direction(runner):
    parent = await _seed_parent(runner, texts=("parent secret",))
    await detour_gateway(runner)._handle_detour_command(_event("/detour"))
    child = _routed(runner)
    runner.session_store.append_to_transcript(child, {"role": "user", "content": "child secret"})
    await detour_gateway(runner)._handle_detour_end_command(_event("/detour-end"))

    parent_texts = [m["content"] for m in runner.session_store.load_transcript(parent)]
    child_texts = [m["content"] for m in runner.session_store.load_transcript(child)]
    assert "child secret" not in parent_texts and "parent secret" not in child_texts


@pytest.mark.asyncio
async def test_a_bare_detour_switches_no_model_at_all(runner):
    await _seed_parent(runner)
    await detour_gateway(runner)._handle_detour_command(_event("/detour"))
    assert runner.applied_routes == []
    assert read_record(_scope(runner)).status == STATUS_ACTIVE


# --------------------------------------------------------------------------- next-turn routing
@pytest.mark.asyncio
async def test_the_next_turn_routes_to_the_child_then_back_to_the_parent(runner):
    """What the next inbound message resolves to — the native route, not just a stored id."""
    source = _source()
    parent = await _seed_parent(runner, source)
    await detour_gateway(runner)._handle_detour_command(_event("/detour", source))
    child = _routed(runner, source)
    assert (await runner.async_session_store.get_or_create_session(source)).session_id == child

    await detour_gateway(runner)._handle_detour_end_command(_event("/detour-end", source))
    assert (await runner.async_session_store.get_or_create_session(source)).session_id == parent


@pytest.mark.asyncio
async def test_both_legs_evict_the_cached_agent_and_clear_conversation_scope(runner):
    """Pending per-conversation state must not survive either boundary (prompt cache + overrides)."""
    source = _source()
    key = runner._session_key_for_source(source)
    await _seed_parent(runner, source)

    runner._agent_cache[key] = ("stale-agent", "sig")
    runner._session_model_overrides[key] = {"model": "stale", "provider": "stale"}
    await detour_gateway(runner)._handle_detour_command(_event("/detour", source))
    assert key not in runner._agent_cache
    assert not runner._session_model_overrides.get(key)

    runner._agent_cache[key] = ("stale-child-agent", "sig")
    runner._session_model_overrides[key] = {"model": "child", "provider": "child"}
    await detour_gateway(runner)._handle_detour_end_command(_event("/detour-end", source))
    assert key not in runner._agent_cache
    # The child's override is gone (the resume cleared conversation scope) and the PARENT's own
    # route — the one the record saved at enter — is deliberately back: returning to a session
    # means returning to its model, not to the config default.
    assert runner._session_model_overrides.get(key) == {"model": "stale", "provider": "stale"}


# --------------------------------------------------------------------------- repeated commands
@pytest.mark.asyncio
async def test_a_second_detour_neither_nests_nor_overwrites_the_record(runner):
    parent = await _seed_parent(runner)
    await detour_gateway(runner)._handle_detour_command(_event("/detour"))
    child, before = _routed(runner), record_path(_scope(runner)).read_bytes()

    reply = await detour_gateway(runner)._handle_detour_command(_event("/detour other-model"))

    assert "already open" in reply
    assert _routed(runner) == child  # no second rotation
    assert record_path(_scope(runner)).read_bytes() == before
    assert read_record(_scope(runner)).parent_session_id == parent
    assert runner.applied_routes == []


@pytest.mark.asyncio
async def test_a_second_detour_end_is_a_quiet_no_op(runner):
    parent = await _seed_parent(runner)
    await detour_gateway(runner)._handle_detour_command(_event("/detour"))
    await detour_gateway(runner)._handle_detour_end_command(_event("/detour-end"))

    reply = await detour_gateway(runner)._handle_detour_end_command(_event("/detour-end back-model"))

    assert "Nothing to end" in reply
    assert _routed(runner) == parent  # still on the parent; nothing created or switched
    assert read_record(_scope(runner)).status == STATUS_RETURNED


@pytest.mark.asyncio
async def test_detour_end_without_any_detour_creates_nothing(runner):
    parent = await _seed_parent(runner)
    sessions_before = _session_ids(runner)

    reply = await detour_gateway(runner)._handle_detour_end_command(_event("/detour-end"))

    assert "Nothing to end" in reply
    assert _routed(runner) == parent
    assert _session_ids(runner) == sessions_before
    assert not record_path(_scope(runner)).exists()


# --------------------------------------------------------------------------- fail-closed
@pytest.mark.asyncio
async def test_a_corrupt_record_refuses_the_return_and_is_preserved(runner):
    parent = await _seed_parent(runner)
    await detour_gateway(runner)._handle_detour_command(_event("/detour"))
    child = _routed(runner)
    path = record_path(_scope(runner))
    path.write_bytes(b"{ truncated")

    reply = await detour_gateway(runner)._handle_detour_end_command(_event("/detour-end"))

    assert reply.startswith("❌")
    assert _routed(runner) == child  # fail closed: no switch, no new session
    assert parent not in reply.split("Record:")[0] or "not readable" in reply
    assert path.read_bytes() == b"{ truncated"  # left for inspection


@pytest.mark.asyncio
async def test_a_missing_parent_session_refuses_the_return_and_keeps_the_record(runner):
    await _seed_parent(runner)
    await detour_gateway(runner)._handle_detour_command(_event("/detour"))
    child = _routed(runner)
    record = read_record(_scope(runner))
    runner._session_db._db.delete_session(record.parent_session_id,
                                     sessions_dir=runner.home / "sessions")

    reply = await detour_gateway(runner)._handle_detour_end_command(_event("/detour-end"))

    assert "no longer exists" in reply
    assert _routed(runner) == child
    assert read_record(_scope(runner)).status == STATUS_ACTIVE  # still open, not consumed


@pytest.mark.asyncio
async def test_an_async_only_session_store_fails_the_gates_closed(runner):
    """A coroutine object is truthy: an existence gate answered from the async facade would pass
    for a session that is not there. The detour must report "cannot tell", not "it exists"."""

    class _AsyncOnly:
        async def get_session(self, _session_id):  # no sync ``_db`` handle to fall back to
            return {"id": "whatever"}

    await _seed_parent(runner)
    await detour_gateway(runner)._handle_detour_command(_event("/detour"))
    child = _routed(runner)
    runner._session_db = _AsyncOnly()

    assert detour_gateway(runner)._detour_sync_session_db() is None
    assert detour_gateway(runner)._detour_session_exists(child) is False
    reply = await detour_gateway(runner)._handle_detour_end_command(_event("/detour-end"))

    assert "no longer exists" in reply
    assert _routed(runner) == child
    assert read_record(_scope(runner)).status == STATUS_ACTIVE  # still open, not consumed


@pytest.mark.asyncio
async def test_a_failed_reset_rolls_the_record_back(runner, monkeypatch):
    parent = await _seed_parent(runner)

    async def _boom(_event):
        raise RuntimeError("reset exploded")

    monkeypatch.setattr(runner, "_handle_reset_command", _boom)
    reply = await detour_gateway(runner)._handle_detour_command(_event("/detour some-model"))

    assert "Nothing changed" in reply
    assert _routed(runner) == parent
    assert not record_path(_scope(runner)).exists()  # rolled back: no dangling return handle
    assert runner.applied_routes == []


@pytest.mark.asyncio
async def test_a_failed_resume_keeps_the_detour_open(runner, monkeypatch):
    await _seed_parent(runner)
    await detour_gateway(runner)._handle_detour_command(_event("/detour"))
    child = _routed(runner)

    async def _refuse(_event):
        return "not allowed"

    monkeypatch.setattr(runner, "_handle_resume_command", _refuse)
    reply = await detour_gateway(runner)._handle_detour_end_command(_event("/detour-end back-model"))

    assert "stays open" in reply
    assert _routed(runner) == child
    assert read_record(_scope(runner)).status == STATUS_ACTIVE
    assert runner.applied_routes == []  # no route applied on a refused return


@pytest.mark.asyncio
async def test_a_route_that_cannot_be_applied_rolls_the_whole_enter_leg_back(runner):
    """Asking for a model the switch cannot deliver must not strand the user in a child session."""
    parent = await _seed_parent(runner)

    async def _boom(_event):
        raise RuntimeError("provider unreachable")

    runner._handle_model_command = _boom
    reply = await detour_gateway(runner)._handle_detour_command(_event("/detour some-model"))

    assert "Detour not started" in reply and "model switch failed" in reply
    assert _routed(runner) == parent  # back on the original conversation
    assert not record_path(_scope(runner)).exists()  # nothing left open


@pytest.mark.asyncio
async def test_a_switch_that_reports_success_but_changes_nothing_is_treated_as_failure(runner):
    """Runtime state decides, not the reply text."""
    parent = await _seed_parent(runner)
    runner.model_switch_is_broken = True

    reply = await detour_gateway(runner)._handle_detour_command(_event("/detour some-model"))

    assert "Detour not started" in reply and "did not take effect" in reply
    assert _routed(runner) == parent
    assert not record_path(_scope(runner)).exists()


def _break_the_lane_lock(monkeypatch):
    """No advisory lock obtainable. Returns the switch so a test can let it work again without
    ``monkeypatch.undo`` (which would also undo the runner fixture's patches)."""
    from tests.fakes.session_detours_plugin import records as detour_mod

    broken = {"on": True}
    real_flock = detour_mod._flock

    def _maybe_flock(handle, *, lock):
        if broken["on"]:
            raise OSError("unsupported")
        return real_flock(handle, lock=lock)

    monkeypatch.setattr(detour_mod, "_flock", _maybe_flock)
    return broken


@pytest.mark.asyncio
async def test_a_lane_that_cannot_be_locked_refuses_the_enter_leg_and_changes_nothing(
        runner, monkeypatch):
    """Fail closed. Running the transition on in-process ordering alone would let a second Hermes
    process on the same home rotate this lane at the same time."""
    parent = await _seed_parent(runner)
    rows_before = _session_ids(runner)
    _break_the_lane_lock(monkeypatch)

    reply = await detour_gateway(runner)._handle_detour_command(_event("/detour some-model"))

    assert "Detour not started" in reply and "could not be locked" in reply
    assert _routed(runner) == parent
    assert not record_path(_scope(runner)).exists()
    assert runner.applied_routes == []
    assert _session_ids(runner) == rows_before  # no child session was created


@pytest.mark.asyncio
async def test_a_lane_that_cannot_be_locked_refuses_the_return_leg_and_keeps_the_detour(
        runner, monkeypatch):
    await _seed_parent(runner)
    await detour_gateway(runner)._handle_detour_command(_event("/detour"))
    child = _routed(runner)
    _break_the_lane_lock(monkeypatch)

    reply = await detour_gateway(runner)._handle_detour_end_command(_event("/detour-end"))

    assert "Could not end the detour" in reply and "could not be locked" in reply
    assert _routed(runner) == child  # no resume, no switch
    assert read_record(_scope(runner)).status == STATUS_ACTIVE  # still retryable


@pytest.mark.asyncio
async def test_a_detour_refused_by_the_lock_succeeds_on_the_next_try(runner, monkeypatch):
    """A refusal must not wedge the lane — the retry is an ordinary round trip."""
    parent = await _seed_parent(runner)
    broken = _break_the_lane_lock(monkeypatch)
    await detour_gateway(runner)._handle_detour_command(_event("/detour"))
    assert _routed(runner) == parent

    broken["on"] = False
    await detour_gateway(runner)._handle_detour_command(_event("/detour"))
    child = _routed(runner)
    assert child and child != parent
    assert read_record(_scope(runner)).status == STATUS_ACTIVE

    await detour_gateway(runner)._handle_detour_end_command(_event("/detour-end"))
    assert _routed(runner) == parent
    assert read_record(_scope(runner)).status == STATUS_RETURNED


@pytest.mark.asyncio
@pytest.mark.parametrize("args", ["some-model --global", "some-model --once"])
async def test_a_route_that_would_outlive_the_detour_is_refused_before_anything_happens(runner, args):
    parent = await _seed_parent(runner)
    reply = await detour_gateway(runner)._handle_detour_command(_event(f"/detour {args}"))
    assert reply.startswith("❌")
    assert _routed(runner) == parent
    assert not record_path(_scope(runner)).exists()


# --------------------------------------------------------------------------- isolation
@pytest.mark.asyncio
async def test_another_users_detour_is_invisible_and_unusable(runner):
    mine, theirs = _source(user="u1"), _source(user="u2", chat="c2")
    my_parent = await _seed_parent(runner, mine)
    their_parent = await _seed_parent(runner, theirs, texts=("their parent",))

    await detour_gateway(runner)._handle_detour_command(_event("/detour", mine))
    # The other lane has no record at all, so its /detour-end is a no-op on its own session.
    assert read_record(_scope(runner, theirs)) is None
    reply = await detour_gateway(runner)._handle_detour_end_command(_event("/detour-end", theirs))
    assert "Nothing to end" in reply
    assert _routed(runner, theirs) == their_parent
    # ...and my detour is still open, pointing at my parent.
    assert read_record(_scope(runner, mine)).parent_session_id == my_parent


@pytest.mark.asyncio
async def test_a_cli_record_cannot_be_used_by_the_gateway_lane(runner):
    """Cross-surface isolation: same owner, different surface, different record."""
    from tests.fakes.session_detours_plugin import begin_detour

    await _seed_parent(runner)
    gateway_scope = _scope(runner)
    cli_scope = DetourScope(surface="cli", channel="local", owner="u1", lane="")
    begin_detour(cli_scope, parent_session_id="20260101_000000_aaa")

    assert record_path(cli_scope) != record_path(gateway_scope)
    assert read_record(gateway_scope) is None
    reply = await detour_gateway(runner)._handle_detour_end_command(_event("/detour-end"))
    assert "Nothing to end" in reply


# --------------------------------------------------------------------------- restart recovery
@pytest.mark.asyncio
async def test_a_detour_survives_a_gateway_restart(runner, tmp_path):
    """A second runner over the same home resumes the lane and can still return."""
    from gateway.run import GatewayRunner
    from hermes_state import AsyncSessionDB, SessionDB

    source = _source()
    parent = await _seed_parent(runner)
    await detour_gateway(runner)._handle_detour_command(_event("/detour", source))
    child = _routed(runner)

    fresh = object.__new__(GatewayRunner)
    for name, value in vars(runner).items():
        setattr(fresh, name, value)
    # New store + DB handles over the same files: nothing is inherited from memory.
    _fresh_sync = SessionDB(runner.home / "state.db")
    fresh._session_db = AsyncSessionDB(_fresh_sync)
    fresh.session_store = SessionStore(runner.home / "sessions", runner.config)
    fresh.session_store._db = _fresh_sync
    fresh.applied_routes = []
    fresh._handle_model_command = runner._handle_model_command

    assert _routed(fresh, source) == child  # the lane still routes to the detour session
    reply = await detour_gateway(fresh)._handle_detour_end_command(_event("/detour-end", source))
    assert _routed(fresh, source) == parent, reply
    assert read_record(_scope(fresh)).status == STATUS_RETURNED


# --------------------------------------------------------------------------- record contents
@pytest.mark.asyncio
async def test_the_written_record_is_readable_json_without_secrets(runner):
    runner._session_model_overrides[runner._session_key_for_source(_source())] = {
        "model": "parent-model", "provider": "parent-provider", "api_key": "sk-leak",
        "base_url": "http://host/v1",
    }
    await _seed_parent(runner)
    await detour_gateway(runner)._handle_detour_command(_event("/detour child-model --provider child-provider"))

    text = record_path(_scope(runner)).read_text(encoding="utf-8")
    assert "sk-leak" not in text
    data = json.loads(text)
    assert data["parent_route"] == {"model": "parent-model", "provider": "parent-provider"}
    assert data["child_route"] == {"model": "child-model", "provider": "child-provider"}
    assert data["scope"]["surface"] == "gateway" and data["scope"]["owner"] == "u1"
    assert set(data) >= {"version", "status", "profile", "scope", "parent_session_id",
                         "child_session_id", "entered_at"}


@pytest.mark.asyncio
async def test_an_interrupted_enter_leaves_a_returnable_entering_record(runner):
    """Crash between the record write and the rotation: the parent id is already durable."""
    from tests.fakes.session_detours_plugin import begin_detour

    parent = await _seed_parent(runner)
    begin_detour(_scope(runner), parent_session_id=parent)
    assert read_record(_scope(runner)).status == STATUS_ENTERING

    # A later /detour refuses (the record is open) rather than nesting.
    assert "already open" in await detour_gateway(runner)._handle_detour_command(_event("/detour"))
    # The return leg clears it instead of wedging the lane, and changes no session.
    reply = await detour_gateway(runner)._handle_detour_end_command(_event("/detour-end"))
    assert "never started" in reply
    assert _routed(runner) == parent
    assert read_record(_scope(runner)).status == STATUS_RETURNED
    # ...and a fresh detour is possible again.
    assert "already open" not in await detour_gateway(runner)._handle_detour_command(_event("/detour"))
    assert _routed(runner) != parent
