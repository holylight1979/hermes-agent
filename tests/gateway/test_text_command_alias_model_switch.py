"""What an alias-produced ``/model ... --session`` actually does to the gateway's routing state.

The alias seam only rewrites text; the switch itself is the native ``/model`` handler. These tests
drive that handler with the exact command strings the aliases produce and pin the properties the
llm-cr mode depends on: the switch is session-scoped (config.yaml is never written), sessions are
isolated from each other, the exit trigger restores the configured default route, and a switch that
cannot be resolved leaves the route alone instead of silently falling back.
"""

import pytest
import yaml

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import AsyncSessionStore, SessionSource

ENTER = "/model model-x --provider alpha-direct --session"
EXIT = "/model gpt-6-astra-900k --provider beta-direct --session"
BROKEN = "/model model-x --provider no-such-provider --session"

CONFIG = {
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
}


@pytest.fixture
def gateway_home(tmp_path, monkeypatch):
    import gateway.run as gateway_run

    home = tmp_path / "hermes_home"
    home.mkdir()
    (home / "config.yaml").write_text(yaml.safe_dump(CONFIG), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(gateway_run, "_hermes_home", home)
    return home


class _RecordingStore:
    """Stand-in for SessionStore that records only what the /model path persists.

    ``GatewayRunner.async_session_store`` is a read-only property, so the runner is given the
    real :class:`AsyncSessionStore` facade over this store through the property's backing field;
    the async boundary under test is therefore the production one.
    """

    def __init__(self) -> None:
        self.overrides: dict[str, object] = {}

    def set_model_override(self, session_key: str, override) -> None:
        self.overrides[session_key] = override

    def get_model_override(self, session_key: str):
        persisted = self.overrides.get(session_key)
        return dict(persisted) if persisted else None


def _make_runner():
    runner = object.__new__(GatewayRunner)
    runner.config = None
    runner.adapters = {}
    runner._session_model_overrides = {}
    runner._pending_model_notes = {}
    runner._session_db = None
    runner.session_store = _RecordingStore()
    runner._async_session_store = AsyncSessionStore(runner.session_store)
    runner._cached_agent_for = lambda key: None
    runner._evict_cached_agent = lambda key: None
    runner._normalize_source_for_session_key = lambda source: source
    runner._session_key_for_source = lambda source: f"whatsapp:{source.chat_id}"
    return runner


def _event(text: str, chat_id: str = "chat-a") -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id=chat_id, chat_type="dm"),
    )


@pytest.mark.asyncio
async def test_enter_switch_is_session_scoped_and_never_writes_config(gateway_home):
    before = (gateway_home / "config.yaml").read_bytes()
    runner = _make_runner()

    reply = await runner._handle_model_command(_event(ENTER))

    override = runner._session_model_overrides["whatsapp:chat-a"]
    assert override["model"] == "model-x"
    assert override["provider"] == "alpha-direct"
    # The confirmation is program-generated from the resolved route, not model self-reporting.
    assert "model-x" in reply and "alpha-direct" in reply
    assert (gateway_home / "config.yaml").read_bytes() == before


@pytest.mark.asyncio
async def test_exit_trigger_restores_the_configured_default_route(gateway_home):
    before = (gateway_home / "config.yaml").read_bytes()
    runner = _make_runner()

    await runner._handle_model_command(_event(ENTER))
    reply = await runner._handle_model_command(_event(EXIT))

    override = runner._session_model_overrides["whatsapp:chat-a"]
    assert override["model"] == CONFIG["model"]["default"]
    assert override["provider"] == CONFIG["model"]["provider"]
    assert CONFIG["model"]["default"] in reply
    assert (gateway_home / "config.yaml").read_bytes() == before


@pytest.mark.asyncio
async def test_repeated_enter_is_harmless(gateway_home):
    runner = _make_runner()

    first = await runner._handle_model_command(_event(ENTER))
    second = await runner._handle_model_command(_event(ENTER))

    assert "model-x" in first and "model-x" in second
    assert runner._session_model_overrides["whatsapp:chat-a"]["model"] == "model-x"


@pytest.mark.asyncio
async def test_one_session_switch_leaves_other_sessions_on_their_own_route(gateway_home):
    runner = _make_runner()

    await runner._handle_model_command(_event(ENTER, chat_id="chat-a"))

    assert "whatsapp:chat-b" not in runner._session_model_overrides

    await runner._handle_model_command(_event(EXIT, chat_id="chat-b"))

    assert runner._session_model_overrides["whatsapp:chat-a"]["model"] == "model-x"
    assert runner._session_model_overrides["whatsapp:chat-b"]["model"] == "gpt-6-astra-900k"


@pytest.mark.asyncio
async def test_unresolvable_switch_reports_an_error_and_keeps_the_route(gateway_home):
    """An error stays an error: no silent fall back to the previous or default model."""
    runner = _make_runner()
    await runner._handle_model_command(_event(ENTER))

    reply = await runner._handle_model_command(_event(BROKEN))

    assert reply and reply.startswith("Error:")
    assert "no-such-provider" in reply
    assert runner._session_model_overrides["whatsapp:chat-a"]["model"] == "model-x"
    assert runner._session_model_overrides["whatsapp:chat-a"]["provider"] == "alpha-direct"
