"""Regression tests: a DETACHED hygiene compaction that commits after its turn was released
must retire the session's pre-compaction ``last_prompt_tokens``.

Observed production trace (session 20260929_184209_26f7c55b, agent.log)::

    19:01:04 Session hygiene: 50 messages, ~64,167 tokens (actual) — auto-compressing
    19:01:15 ... exceeded turn-hold budget (10.0s) ... the watermark-fenced worker keeps
             its commit admission and the summary will be adopted when it finishes
    19:01:50 context compression done: messages=49->45 rough_tokens=~21,411
    19:01:50 ... summary adopted at the watermark-fenced commit boundary (#97963)
    19:01:52 Session hygiene: 46 messages, ~64,288 tokens (actual) — auto-compressing

``_hmwa_hygiene_plan`` trusts ``session_entry.last_prompt_tokens`` first and unconditionally
("actual"). Inline adoption zeroes it in ``_hmwa_hygiene_adopt_transcript``; the deferred
commit path (the ``add_done_callback`` installed by ``_hmwa_hygiene_on_turn_hold``) did not,
so the already-compacted session was compressed again on the very next message.

These tests drive the real ``_handle_message`` path against a real ``SessionStore`` and a real
``SessionDB`` (the fake compressor calls the genuine ``archive_and_compact``), so the entry
mutation, its persistence and the follow-up hygiene decision are all production code.
"""

import asyncio
import importlib
import sys
import threading
import time
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource, SessionStore

# The stale reading from the trace: above 85% of a 65,536 context (= 55,705).
STALE_PROMPT_TOKENS = 64_288
CONTEXT_LENGTH = 65_536


def _source():
    return SessionSource(
        platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm", user_id="12345",
    )


def _make_history(n_messages: int, content_size: int = 400) -> list:
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": "x" * content_size}
        for i in range(n_messages)
    ]


class _CaptureAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="fake-token"), Platform.TELEGRAM)
        self.sent = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(content)
        return SendResult(success=True, message_id="x")

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


def _write_config(tmp_path):
    (tmp_path / "config.yaml").write_text(
        "compression:\n"
        "  enabled: true\n"
        "  hygiene_timeout_seconds: 60\n"
        "  hygiene_total_ceiling_seconds: 600\n"
        "  hygiene_max_turn_hold_seconds: 0.3\n"
        "  hygiene_failure_cooldown_seconds: 120\n"
    )


def _install_fakes(monkeypatch, gateway_run, tmp_path, agent_cls):
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = agent_cls
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"})
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length", lambda *_a, **_k: CONTEXT_LENGTH,
    )


def _seed_store(tmp_path, n_messages=50):
    """Real SessionStore + real SessionDB carrying a transcript and the stale reading."""
    store = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
    entry = store.get_or_create_session(_source())
    for msg in _make_history(n_messages):
        store.append_to_transcript(entry.session_id, msg)
    store.update_session(entry.session_key, last_prompt_tokens=STALE_PROMPT_TOKENS)
    return store, entry


def _build_runner(gateway_run, adapter, store):
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake-token")}
    )
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner.session_store = store
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = SimpleNamespace(_db=store._db)
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    # The released turn reports the prompt count of the PRE-compaction transcript it ran on —
    # this is the reading that must not survive the detached commit.
    runner._run_agent = AsyncMock(return_value={
        "final_response": "ok", "messages": [], "tools": [], "history_offset": 0,
        "last_prompt_tokens": STALE_PROMPT_TOKENS,
    })
    return runner


def _event(text="hello", message_id="1"):
    return MessageEvent(text=text, source=_source(), message_id=message_id)


def _fenced_agent_class(*, commit: bool, release_worker, started, committed, compacted_rows=5):
    """A hygiene agent whose summary outlives the turn-hold and (optionally) commits in place
    against the REAL SessionDB, exactly as ``_compress_context`` does."""

    class _FencedAgent:
        instances = []

        def __init__(self, **kwargs):
            self.session_id = kwargs.get("session_id")
            self._session_db = kwargs.get("session_db")
            self._last_compaction_in_place = False
            self.context_compressor = SimpleNamespace(
                bind_session_state=lambda *a, **k: None,
                _last_compress_aborted=False,
                _last_aux_model_failure_model=None,
            )
            self.shutdown_memory_provider = lambda *a, **k: None
            self.close = lambda *a, **k: None
            type(self).instances.append(self)

        def _compress_context(self, messages, *_args, commit_fence=None, **_kwargs):
            watermark = self._session_db.get_active_message_watermark(self.session_id)
            if commit_fence is not None:
                commit_fence.mark_commit_watermark_fenced()
            started.set()
            spin_started = time.monotonic()
            while not release_worker.is_set():
                if time.monotonic() - spin_started > 20:
                    return (messages, None)
                if commit_fence is not None:
                    commit_fence.touch_progress()
                time.sleep(0.01)
            if not commit:
                return (messages, None)  # summary failed: nothing committed
            if commit_fence is not None and not commit_fence.begin_commit():
                return (messages, None)
            try:
                rows = [{"role": "assistant", "content": "summary"}] + [
                    {"role": "user" if i % 2 == 0 else "assistant", "content": "tail"}
                    for i in range(compacted_rows - 1)
                ]
                self._session_db.archive_and_compact(self.session_id, rows, watermark=watermark)
                self._last_compaction_in_place = True
                committed.set()
                return (rows, None)
            finally:
                if commit_fence is not None:
                    commit_fence.finish_commit()

    return _FencedAgent


async def _drain_deferred(runner, timeout=10.0):
    tasks = getattr(runner, "_deferred_agent_cleanup_tasks", None) or set()
    if tasks:
        await asyncio.wait_for(asyncio.gather(*list(tasks), return_exceptions=True), timeout)


async def _await_entry_tokens(store, session_key, predicate, timeout=5.0):
    """Poll the live entry until ``predicate`` holds (the retirement rides a done-callback)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate(store._entries[session_key].last_prompt_tokens):
            return True
        await asyncio.sleep(0.02)
    return False


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(tmp_path)
    return tmp_path


@pytest.mark.asyncio
async def test_deferred_commit_retires_stale_pre_compaction_usage(monkeypatch, hermes_home):
    """The detached worker commits after the turn-hold released the turn; the pre-compaction
    ``last_prompt_tokens`` must be retired — in memory AND on the persisted row — so the next
    message does not re-compress a session that was just compacted."""
    started, release_worker, committed = threading.Event(), threading.Event(), threading.Event()
    agent_cls = _fenced_agent_class(
        commit=True, release_worker=release_worker, started=started, committed=committed,
    )
    gateway_run = importlib.import_module("gateway.run")
    _install_fakes(monkeypatch, gateway_run, hermes_home, agent_cls)

    store, entry = _seed_store(hermes_home)
    session_key = entry.session_key
    runner = _build_runner(gateway_run, _CaptureAdapter(), store)

    assert await asyncio.wait_for(runner._handle_message(_event()), timeout=15) == "ok"
    assert started.is_set(), "hygiene must have fired on the stale 'actual' reading"
    # The released turn wrote its own pre-compaction reading back onto the entry.
    assert store._entries[session_key].last_prompt_tokens == STALE_PROMPT_TOKENS

    release_worker.set()
    await asyncio.wait_for(asyncio.to_thread(committed.wait, 5), timeout=6)
    await _drain_deferred(runner)

    assert await _await_entry_tokens(store, session_key, lambda v: v == 0), (
        "a detached hygiene commit must retire the pre-compaction last_prompt_tokens; "
        f"entry still reports {store._entries[session_key].last_prompt_tokens}"
    )
    # Persisted, not just in memory: a restart between the commit and the next message must
    # not resurrect the stale count.
    reloaded = SessionStore(sessions_dir=hermes_home / "sessions", config=GatewayConfig())
    reloaded._ensure_loaded()
    assert reloaded._entries[session_key].last_prompt_tokens == 0


@pytest.mark.asyncio
async def test_next_message_after_deferred_commit_does_not_recompress(monkeypatch, hermes_home):
    """The behavioural consequence: with the stale reading retired, the very next message
    plans against the COMPACTED transcript and no second hygiene attempt is spawned."""
    started, release_worker, committed = threading.Event(), threading.Event(), threading.Event()
    agent_cls = _fenced_agent_class(
        commit=True, release_worker=release_worker, started=started, committed=committed,
    )
    gateway_run = importlib.import_module("gateway.run")
    _install_fakes(monkeypatch, gateway_run, hermes_home, agent_cls)

    store, entry = _seed_store(hermes_home)
    runner = _build_runner(gateway_run, _CaptureAdapter(), store)

    await asyncio.wait_for(runner._handle_message(_event()), timeout=15)
    release_worker.set()
    await asyncio.wait_for(asyncio.to_thread(committed.wait, 5), timeout=6)
    await _drain_deferred(runner)
    await _await_entry_tokens(store, entry.session_key, lambda v: v == 0)

    # The compaction really landed: the live transcript is the compacted set.
    assert len(store.load_transcript(entry.session_id)) < 50

    await asyncio.wait_for(runner._handle_message(_event(message_id="2")), timeout=15)
    assert len(agent_cls.instances) == 1, (
        "the message after a detached hygiene commit must not re-compress the compacted "
        f"session (hygiene agents built: {len(agent_cls.instances)})"
    )


@pytest.mark.asyncio
async def test_commit_before_the_turn_persists_is_not_written_back(monkeypatch, hermes_home):
    """The other completion ordering: the detached worker commits while the released turn is
    STILL talking to the provider. Retiring at the commit is useless on its own — the turn
    finishes afterwards and would persist its pre-compaction reading straight back over it.
    The turn must consume the invalidation mark and persist 'no real reading' instead."""
    started, release_worker, committed = threading.Event(), threading.Event(), threading.Event()
    agent_cls = _fenced_agent_class(
        commit=True, release_worker=release_worker, started=started, committed=committed,
    )
    gateway_run = importlib.import_module("gateway.run")
    _install_fakes(monkeypatch, gateway_run, hermes_home, agent_cls)

    store, entry = _seed_store(hermes_home)
    session_key = entry.session_key
    runner = _build_runner(gateway_run, _CaptureAdapter(), store)

    # The turn runs (and is released by the turn-hold) BEFORE the worker commits: let the worker
    # finish from inside the agent call, so the done-callback lands while the turn is in flight.
    async def _run_agent_that_outlives_the_commit(*_a, **_k):
        release_worker.set()
        await asyncio.wait_for(asyncio.to_thread(committed.wait, 5), timeout=6)
        return {
            "final_response": "ok", "messages": [], "tools": [], "history_offset": 0,
            "last_prompt_tokens": STALE_PROMPT_TOKENS,
        }

    runner._run_agent = _run_agent_that_outlives_the_commit

    assert await asyncio.wait_for(runner._handle_message(_event()), timeout=20) == "ok"
    assert committed.is_set(), "the fenced worker must have committed during the turn"
    await _drain_deferred(runner)

    assert store._entries[session_key].last_prompt_tokens == 0, (
        "a turn that outlived a detached hygiene commit must not persist its pre-compaction "
        f"reading (entry reports {store._entries[session_key].last_prompt_tokens})"
    )
    reloaded = SessionStore(sessions_dir=hermes_home / "sessions", config=GatewayConfig())
    reloaded._ensure_loaded()
    assert reloaded._entries[session_key].last_prompt_tokens == 0
    # The mark is consumed by the turn that carried it, not left to poison a later run.
    assert not getattr(runner, "_hygiene_invalidated_turn_usage", None)


@pytest.mark.asyncio
async def test_deferred_attempt_that_never_commits_keeps_the_usage(monkeypatch, hermes_home):
    """Refused/failed deferral is not a compaction: the reading still prices the live
    transcript and must survive, otherwise hygiene loses its only real token source and falls
    back to the 30-50%-high estimate."""
    started, release_worker, committed = threading.Event(), threading.Event(), threading.Event()
    agent_cls = _fenced_agent_class(
        commit=False, release_worker=release_worker, started=started, committed=committed,
    )
    gateway_run = importlib.import_module("gateway.run")
    _install_fakes(monkeypatch, gateway_run, hermes_home, agent_cls)

    store, entry = _seed_store(hermes_home)
    runner = _build_runner(gateway_run, _CaptureAdapter(), store)

    await asyncio.wait_for(runner._handle_message(_event()), timeout=15)
    assert started.is_set()
    release_worker.set()
    await _drain_deferred(runner)
    # Give the done-callback the same window the adopting test gets.
    assert not await _await_entry_tokens(store, entry.session_key, lambda v: v == 0, timeout=1.0)
    assert store._entries[entry.session_key].last_prompt_tokens == STALE_PROMPT_TOKENS
    assert not committed.is_set()


def test_retirement_skips_a_session_key_rebound_to_a_new_session(hermes_home):
    """The callback fires off-turn: ``/new`` may already have rebound the key onto a fresh
    session carrying its own real reading. The retirement is a compare-and-set on the session
    the compaction actually committed against, so the new conversation is never wiped."""
    from gateway.run import _retire_stale_hygiene_usage

    store, entry = _seed_store(hermes_home, n_messages=6)
    gateway = SimpleNamespace(_session_db=SimpleNamespace(_db=store._db), session_store=store)

    old_session_id = entry.session_id
    new_entry = store.reset_session(entry.session_key)  # /new
    assert new_entry.session_id != old_session_id
    store.update_session(entry.session_key, last_prompt_tokens=1_234)

    _retire_stale_hygiene_usage(gateway, entry.session_key, old_session_id)
    assert store._entries[entry.session_key].last_prompt_tokens == 1_234, (
        "a compaction committed against the PREVIOUS session must not retire the fresh "
        "session's own reading"
    )

    # Same key, matching session id: the retirement applies.
    _retire_stale_hygiene_usage(gateway, entry.session_key, new_entry.session_id)
    assert store._entries[entry.session_key].last_prompt_tokens == 0


def test_turn_usage_invalidation_mark_is_scoped_to_its_own_run(hermes_home):
    """The in-memory mark is keyed by (quick key, run generation): only the marked run consumes
    it, so a later run's genuinely fresh reading is never zeroed."""
    from gateway.run_turn import GatewayTurnMixin

    runner = object.__new__(GatewayTurnMixin)
    runner._mark_turn_usage_invalidated("chat:1", 7)

    assert not runner._consume_turn_usage_invalidated("chat:1", 8)  # next run
    assert not runner._consume_turn_usage_invalidated("chat:2", 7)  # other session key
    assert runner._consume_turn_usage_invalidated("chat:1", 7)
    assert not runner._consume_turn_usage_invalidated("chat:1", 7)  # consumed once


def test_retirement_keeps_a_reading_re_anchored_after_the_commit(hermes_home):
    """A reading captured AFTER the commit prices the live transcript and is not stale.
    Compaction clears the row's usage anchor and only a real provider response re-persists
    one, so a present anchor pins 'real usage has been seen since' — the count is kept."""
    from agent.usage_anchor import USAGE_ANCHOR_MODEL_CONFIG_KEY, capture_usage_anchor
    from gateway.run import _retire_stale_hygiene_usage

    store, entry = _seed_store(hermes_home, n_messages=6)
    gateway = SimpleNamespace(
        _session_db=SimpleNamespace(_db=store._db), session_store=store,
    )

    live = store.load_transcript(entry.session_id)
    store._db.patch_session_model_config(entry.session_id, {
        USAGE_ANCHOR_MODEL_CONFIG_KEY: capture_usage_anchor(STALE_PROMPT_TOKENS, 10, live),
    })
    _retire_stale_hygiene_usage(gateway, entry.session_key, entry.session_id)
    assert store._entries[entry.session_key].last_prompt_tokens == STALE_PROMPT_TOKENS

    # Compaction clears the anchor; now the reading is provably pre-compaction and is retired.
    store._db.patch_session_model_config(
        entry.session_id, {USAGE_ANCHOR_MODEL_CONFIG_KEY: None},
    )
    _retire_stale_hygiene_usage(gateway, entry.session_key, entry.session_id)
    assert store._entries[entry.session_key].last_prompt_tokens == 0
