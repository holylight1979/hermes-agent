"""The DEPLOYED ``text-command-aliases`` plugin, driven end to end through the gateway.

``test_text_command_alias_dispatch.py`` writes its own miniature plugin body, which proves the seam
but not the artifact. Here the fixture COPIES the installed plugin package out of
``$HERMES_HOME/plugins/text-command-aliases`` into a temp home and lets the real discovery path load
it, so what runs is the file the gateway actually loads. Nothing is asserted about the plugin's
source text — only about behaviour:

  rewrite → authorization/access gate → native ``/model`` switch → the override the NEXT turn uses
  → the exit trigger restoring the configured default route

and the negative that makes the feature worth having: a trigger never reaches the agent.
"""

import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.session import AsyncSessionStore, SessionSource

ENTER = "/model model-x --provider alpha-direct --session"
EXIT = "/model gpt-6-astra-900k --provider beta-direct --session"

CONFIG = {
    "plugins": {"enabled": ["text-command-aliases"]},
    "model": {"default": "gpt-6-astra-900k", "provider": "beta-direct"},
    "providers": {
        "alpha-direct": {
            "name": "alpha-direct", "base_url": "http://alpha.invalid/v1",
            "api_key": "alpha-test-key", "api_mode": "chat_completions",
            "models": {"model-x": {"context_length": 65536}},
        },
        "beta-direct": {
            "name": "beta-direct", "base_url": "http://beta.invalid/v1",
            "api_key": "beta-test-key", "api_mode": "chat_completions",
            "models": {"gpt-6-astra-900k": {"context_length": 131072}},
        },
    },
    "text_command_aliases": {
        "enabled": True,
        "aliases": {"llm-cr": ENTER, "llm-cr-end": EXIT},
    },
}


def _installed_plugin_dir() -> Path:
    """The plugin package as installed on this machine (not a copy authored here).

    Resolved from the PLATFORM DEFAULT home, not ``get_hermes_home()``: conftest sandboxes
    ``HERMES_HOME`` before any test module is imported, so the live install is invisible through
    the usual accessor. ``HERMES_DEPLOYED_PLUGINS_HOME`` overrides it for a non-default profile.
    The directory is only ever read and copied.
    """
    from hermes_constants import _get_platform_default_hermes_home

    supplied = os.environ.get("HERMES_DEPLOYED_PLUGINS_HOME", "").strip()
    home = Path(supplied) if supplied else _get_platform_default_hermes_home()
    return home / "plugins" / "text-command-aliases"


@pytest.fixture
def deployed_alias_home(tmp_path, monkeypatch) -> Path:
    import gateway.run as gateway_run

    source_dir = _installed_plugin_dir()
    if not (source_dir / "__init__.py").is_file():
        pytest.skip(f"text-command-aliases plugin is not installed at {source_dir}")

    home = tmp_path / "hermes_home"
    (home / "plugins").mkdir(parents=True)
    shutil.copytree(
        source_dir, home / "plugins" / "text-command-aliases",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    (home / "config.yaml").write_text(yaml.safe_dump(CONFIG), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(gateway_run, "_hermes_home", home)
    return home


class _RecordingStore:
    """Only the bits of SessionStore the /model write-through touches."""

    def __init__(self) -> None:
        self.overrides: dict = {}

    def set_model_override(self, session_key: str, override) -> None:
        self.overrides[session_key] = override

    def get_model_override(self, session_key: str):
        persisted = self.overrides.get(session_key)
        return dict(persisted) if persisted else None


def _make_runner(*, authorized: bool = True, slash_denial: str | None = None):
    """A GatewayRunner with the inbound collaborators stubbed and the /model path left real."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.WHATSAPP: PlatformConfig(enabled=True)})
    runner.adapters = {Platform.WHATSAPP: SimpleNamespace(send=AsyncMock())}
    runner.session_store = _RecordingStore()
    runner._async_session_store = AsyncSessionStore(runner.session_store)
    runner._session_db = None
    runner._session_model_overrides = {}
    runner._pending_model_notes = {}
    runner._last_inbound_at = 0.0
    runner._is_user_authorized_for_source = lambda source, **kw: authorized
    runner._admit_bot_message_for_source = lambda source: True
    runner._pairing_store_for = lambda source: None
    runner._get_unauthorized_dm_behavior = lambda platform, profile=None: "ignore"
    # The per-platform slash access gate is exercised as a decision, not re-derived from config.
    runner._check_slash_access = lambda source, canonical: slash_denial
    runner.evicted: list = []
    runner._cached_agent_for = lambda key: None
    runner._evict_cached_agent = runner.evicted.append
    # Reaching the agent at all would defeat the feature; every test asserts this stayed unused.
    runner._run_agent = AsyncMock()
    return runner


def _event(text: str, *, allow_gateway_control: bool = True) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_id="m1",
        allow_gateway_control=allow_gateway_control,
        source=SessionSource(
            platform=Platform.WHATSAPP,
            user_id="15551234567@s.whatsapp.net",
            chat_id="15551234567@s.whatsapp.net",
            user_name="tester",
            chat_type="dm",
        ),
    )


async def _submit(runner, text: str):
    """Admit + dispatch one bare message the way the idle inbound path does.

    Returns ``(handled, reply, session_key, event)``; ``handled`` False means the turn would have
    fallen through to the agent.
    """
    admitted = await runner._hm_admit_event(_event(text))
    if admitted is None:
        return None, None, None, None
    event, source, _is_internal = admitted
    session_key = runner._session_key_for_source(source)
    handled, reply = await runner._hm_dispatch_idle_commands(event, source, session_key)
    return handled, reply, session_key, event


@pytest.mark.asyncio
async def test_trigger_switches_the_route_without_ever_reaching_the_agent(deployed_alias_home):
    runner = _make_runner()

    handled, reply, session_key, event = await _submit(runner, "llm-cr")

    assert handled is True
    assert event.text == ENTER  # the deployed plugin's rewrite, not a hand-written stand-in
    assert "model-x" in reply and "alpha-direct" in reply
    assert runner._session_model_overrides[session_key]["model"] == "model-x"
    runner._run_agent.assert_not_awaited()
    assert runner.evicted == [session_key]  # the stale cached agent is dropped for the next turn


@pytest.mark.asyncio
async def test_the_next_turn_runs_on_the_switched_route(deployed_alias_home):
    """What the switch is for: the turn after the trigger resolves to the new model + credentials."""
    runner = _make_runner()
    _handled, _reply, session_key, _event_in = await _submit(runner, "llm-cr")

    model, runtime_kwargs = runner._apply_session_model_override(
        session_key, "gpt-6-astra-900k", {})

    assert model == "model-x"
    assert runtime_kwargs["base_url"] == CONFIG["providers"]["alpha-direct"]["base_url"]
    assert runtime_kwargs["api_key"] == CONFIG["providers"]["alpha-direct"]["api_key"]


@pytest.mark.asyncio
async def test_the_exit_trigger_restores_the_configured_default_route(deployed_alias_home):
    runner = _make_runner()
    await _submit(runner, "llm-cr")

    handled, reply, session_key, event = await _submit(runner, "llm-cr-end")

    assert handled is True
    assert event.text == EXIT
    assert CONFIG["model"]["default"] in reply
    model, runtime_kwargs = runner._apply_session_model_override(session_key, "unused", {})
    assert model == CONFIG["model"]["default"]
    assert runtime_kwargs["base_url"] == CONFIG["providers"]["beta-direct"]["base_url"]
    runner._run_agent.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_failing_cached_agent_eviction_does_not_lose_the_new_route(deployed_alias_home):
    """Eviction is the LAST step of the commit, after the override is recorded and written through.

    Pinning the real behaviour: the failure propagates to the caller (the inbound path reports the
    command as failed rather than pretending it worked), and the route is already on the new model —
    both in memory and in the store. The dangerous alternative would be a session that answered
    "switched" while still running the old model, or one silently rolled back.
    """
    runner = _make_runner()
    admitted = await runner._hm_admit_event(_event("llm-cr"))
    assert admitted is not None
    event, source, _is_internal = admitted
    session_key = runner._session_key_for_source(source)

    def _boom(_session_key):
        raise RuntimeError("agent cache is wedged")

    runner._evict_cached_agent = _boom

    with pytest.raises(RuntimeError, match="agent cache is wedged"):
        await runner._hm_dispatch_idle_commands(event, source, session_key)

    assert runner._session_model_overrides[session_key]["model"] == "model-x"
    assert runner.session_store.overrides[session_key]["model"] == "model-x"
    assert runner._apply_session_model_override(session_key, "gpt-6-astra-900k", {})[0] == "model-x"
    runner._run_agent.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_unauthorized_sender_gets_no_switch(deployed_alias_home):
    runner = _make_runner(authorized=False)

    assert await runner._hm_admit_event(_event("llm-cr")) is None
    assert runner._session_model_overrides == {}
    runner._run_agent.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_slash_denied_sender_stays_denied_after_the_rewrite(deployed_alias_home):
    """The rewrite happens before the access gate, and the gate still gets the last word."""
    runner = _make_runner(slash_denial="You are not allowed to use /model here.")

    handled, reply, session_key, event = await _submit(runner, "llm-cr")

    assert handled is True
    assert event.text == ENTER
    assert reply == "You are not allowed to use /model here."
    assert session_key not in runner._session_model_overrides
    assert runner.session_store.overrides == {}
    runner._run_agent.assert_not_awaited()


@pytest.mark.asyncio
async def test_near_miss_text_falls_through_to_the_agent_unchanged(deployed_alias_home):
    runner = _make_runner()

    handled, _reply, session_key, event = await _submit(runner, "llm-cr 這是什麼")

    assert handled is False  # not a command: the idle path hands it to the agent
    assert event.text == "llm-cr 這是什麼"
    assert session_key not in runner._session_model_overrides
