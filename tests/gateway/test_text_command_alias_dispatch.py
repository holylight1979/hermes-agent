"""Gateway half of the exact bare-text → slash-command alias seam.

A ``pre_gateway_dispatch`` plugin rewrites an exact bare trigger into a slash command, so the
gateway dispatches it locally instead of handing the text to the agent. These tests load such a
plugin through the REAL discovery path against a temp ``HERMES_HOME`` and drive the real
``_hm_admit_event`` so the ordering that matters — rewrite happens, but authorization still decides
— is exercised rather than asserted about.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource

SWITCH_IN = "/model hf.co/vendor/Some-Model-GGUF:Q4_K_M --provider local-direct --session"
SWITCH_OUT = "/model gpt-6-astra-900k --provider openai-codex --session"

# Mirrors ``$HERMES_HOME/plugins/text-command-aliases/__init__.py``: the shared matcher over the
# shared config section, skipping events that may not control the gateway.
PLUGIN_BODY = '''
def _rewrite(event=None, **_kwargs):
    from hermes_cli.config import read_raw_config_readonly
    from hermes_cli.text_command_aliases import resolve_text_command_alias

    if event is None or not getattr(event, "allow_gateway_control", False):
        return None
    command = resolve_text_command_alias(getattr(event, "text", None), read_raw_config_readonly())
    return {"action": "rewrite", "text": command} if command else None


def register(ctx):
    ctx.register_hook("pre_gateway_dispatch", _rewrite)
'''


@pytest.fixture
def alias_home(tmp_path, monkeypatch) -> Path:
    """Temp HERMES_HOME carrying the alias table plus an enabled alias-rewrite plugin."""
    home = tmp_path / "hermes_home"
    plugin_dir = home / "plugins" / "text-command-aliases"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        yaml.safe_dump({"name": "text-command-aliases", "version": "1.0.0",
                        "description": "alias rewrite", "hooks": ["pre_gateway_dispatch"]}),
        encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(PLUGIN_BODY, encoding="utf-8")
    (home / "config.yaml").write_text(
        yaml.safe_dump({
            "plugins": {"enabled": ["text-command-aliases"]},
            "text_command_aliases": {
                "enabled": True,
                "aliases": {"llm-cr": SWITCH_IN, "llm-cr-end": SWITCH_OUT},
            },
        }),
        encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _make_event(text: str, *, allow_gateway_control: bool = True) -> MessageEvent:
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


def _make_runner(*, authorized: bool):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.WHATSAPP: PlatformConfig(enabled=True)})
    runner.adapters = {Platform.WHATSAPP: SimpleNamespace(send=AsyncMock())}
    runner.session_store = MagicMock()
    runner._last_inbound_at = 0.0
    runner._is_user_authorized_for_source = lambda source, **kw: authorized
    runner._admit_bot_message_for_source = lambda source: True
    runner._pairing_store_for = lambda source: None
    runner._get_unauthorized_dm_behavior = lambda platform, profile=None: "ignore"
    return runner


@pytest.mark.asyncio
async def test_exact_trigger_becomes_a_recognized_slash_command(alias_home):
    """The rewrite lands before dispatch and produces a command the gateway knows."""
    from hermes_cli.commands import is_gateway_known_command

    runner = _make_runner(authorized=True)
    admitted = await runner._hm_admit_event(_make_event("llm-cr"))

    assert admitted is not None
    event, _source, is_internal = admitted
    assert is_internal is False
    assert event.text == SWITCH_IN
    assert event.get_command() == "model"
    assert is_gateway_known_command("model")


@pytest.mark.asyncio
async def test_exit_trigger_rewrites_to_the_restore_command(alias_home):
    runner = _make_runner(authorized=True)
    admitted = await runner._hm_admit_event(_make_event("  llm-cr-end  "))

    assert admitted is not None
    assert admitted[0].text == SWITCH_OUT


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["llm-cr 你好嗎", "請幫我看 llm-cr 的腳本", "llm-crack-talk",
                                 "debug llm-cr-end please", "LLM-CR"])
async def test_near_miss_text_is_admitted_unchanged(alias_home, text):
    runner = _make_runner(authorized=True)
    admitted = await runner._hm_admit_event(_make_event(text))

    assert admitted is not None
    assert admitted[0].text == text
    assert admitted[0].get_command() is None


@pytest.mark.asyncio
async def test_unauthorized_sender_is_dropped_despite_the_rewrite(alias_home):
    """Authorization still owns admission: the trigger buys no one a model switch."""
    runner = _make_runner(authorized=False)

    assert await runner._hm_admit_event(_make_event("llm-cr")) is None


@pytest.mark.asyncio
async def test_events_that_may_not_control_the_gateway_are_not_rewritten(alias_home):
    """``allow_gateway_control=False`` (proactive/untrusted payloads) stays conversational."""
    runner = _make_runner(authorized=True)
    admitted = await runner._hm_admit_event(
        _make_event("llm-cr", allow_gateway_control=False))

    assert admitted is not None
    assert admitted[0].text == "llm-cr"


@pytest.mark.asyncio
async def test_internal_events_never_reach_the_rewrite(alias_home):
    """Internal events short-circuit before the hook, so their text is untouched."""
    runner = _make_runner(authorized=True)
    event = _make_event("llm-cr")
    event.internal = True

    admitted = await runner._hm_admit_event(event)

    assert admitted is not None
    assert admitted[0].text == "llm-cr"
    assert admitted[2] is True
