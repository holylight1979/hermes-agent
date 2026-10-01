"""``pre_platform_message_admission`` — the consume-only Discord ingress hook.

The contract under test, end to end through a real ``PluginManager`` (a real
plugin directory under a temp ``HERMES_HOME``, discovered and registered by the
real loader) and the real ``DiscordAdapter`` dispatch entry points:

* the hook runs BEFORE ``_discord_message_admission``, so a plugin can claim a
  message the allowlist / mention gate would have dropped;
* ``{"action": "consume"}`` stops everything — no admission, no dedup claim, no
  ``_handle_message``;
* every other outcome (``None``, ``"allow"``, ``{"action": "allow"}``, an
  exception, no plugin at all) leaves admission byte-for-byte as it was. In
  particular the hook can never ADMIT anyone: it has no allow verb, so a
  non-allowlisted user stays out no matter what a plugin returns.
"""

from __future__ import annotations

import textwrap
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig

import plugins.platforms.discord.adapter as discord_platform  # noqa: E402
from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402
from plugins.platforms.discord.admission_hook import HOOK_NAME  # noqa: E402

ALLOWED_USER_ID = 7
OUTSIDER_USER_ID = 4242


class _TextChannel:
    """A guild text channel: not a DM, not a thread."""

    def __init__(self, channel_id: int = 100):
        self.id = channel_id
        self.name = "tech"
        self.parent_id = None
        self.guild = SimpleNamespace(name="Test Server", id=555)
        self.topic = None


_UNSET = object()


def _make_message(*, msg_id=42, channel=None, content="hello", author=None,
                  guild=_UNSET, bot=False, webhook_id=None, msg_type=None):
    channel = channel if channel is not None else _TextChannel()
    if author is None:
        author = SimpleNamespace(
            id=ALLOWED_USER_ID, display_name="Alice", name="alice", bot=bot,
        )
    return SimpleNamespace(
        id=msg_id,
        content=content,
        mentions=[],
        attachments=[],
        reference=None,
        message_snapshots=None,
        created_at=datetime.now(timezone.utc),
        channel=channel,
        author=author,
        webhook_id=webhook_id,
        guild=getattr(channel, "guild", None) if guild is _UNSET else guild,
        type=msg_type if msg_type is not None else discord_platform.discord.MessageType.default,
    )


@pytest.fixture
def adapter(monkeypatch):
    for var in (
        "DISCORD_REQUIRE_MENTION", "DISCORD_AUTO_THREAD", "DISCORD_FREE_RESPONSE_CHANNELS",
        "DISCORD_ALLOWED_CHANNELS", "DISCORD_IGNORED_CHANNELS", "DISCORD_ALLOW_BOTS",
        "DISCORD_IGNORE_NO_MENTION", "DISCORD_ALLOWED_USERS", "DISCORD_ALLOWED_ROLES",
        "GATEWAY_ALLOW_ALL_USERS", "GATEWAY_ALLOWED_USERS",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("DISCORD_IGNORE_NO_MENTION", "false")

    a = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    a._client = SimpleNamespace(user=SimpleNamespace(id=999, bot=True))
    a._allowed_user_ids = {str(ALLOWED_USER_ID)}
    a._ready_event.set()
    a._handle_message = AsyncMock(return_value=True)
    return a


# The probe plugin: registered by the real loader, driven from the test through
# its own module object (``module.MODE`` / ``module.CALLS``).
_PROBE_PLUGIN = textwrap.dedent(
    '''
    MODE = None
    CALLS = []

    def register(ctx):
        def on_hook(platform=None, identity=None, message=None, adapter=None, **kwargs):
            CALLS.append({"platform": platform, "identity": identity,
                          "message": message, "adapter": adapter})
            if MODE == "raise":
                raise RuntimeError("probe plugin failure")
            if MODE == "consume":
                return {"action": "consume", "reason": "probe"}
            if MODE == "allow":
                return {"action": "allow", "reason": "probe"}
            if MODE == "allow-string":
                return "allow"
            return None

        ctx.register_hook("pre_platform_message_admission", on_hook)
    '''
)


@pytest.fixture
def probe(tmp_path, monkeypatch):
    """Load the probe plugin through the real discovery + registration path."""
    from hermes_cli import plugins as plugins_mod

    home = tmp_path / ".hermes"
    (home / "plugins" / "probe").mkdir(parents=True)
    (home / "plugins" / "probe" / "plugin.yaml").write_text(
        "name: probe\nversion: \"1.0.0\"\ndescription: probe\n"
        "hooks:\n  - pre_platform_message_admission\n",
        encoding="utf-8",
    )
    (home / "plugins" / "probe" / "__init__.py").write_text(_PROBE_PLUGIN, encoding="utf-8")
    (home / "config.yaml").write_text("plugins:\n  enabled:\n    - probe\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))

    plugins_mod._reset_plugin_managers_for_tests()
    plugins_mod.discover_plugins(force=True)
    manager = plugins_mod.get_plugin_manager()
    state = manager._plugins["probe"]
    assert state.enabled and state.error is None, state.error
    yield state.module
    plugins_mod._reset_plugin_managers_for_tests()


@pytest.fixture
def no_plugins(tmp_path, monkeypatch):
    """A temp home with no plugins at all — the unchanged baseline."""
    from hermes_cli import plugins as plugins_mod

    home = tmp_path / ".hermes"
    (home / "plugins").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    plugins_mod._reset_plugin_managers_for_tests()
    plugins_mod.discover_plugins(force=True)
    yield plugins_mod
    plugins_mod._reset_plugin_managers_for_tests()


class TestConsume:
    @pytest.mark.asyncio
    async def test_consume_stops_admission_dedup_and_dispatch(self, adapter, probe):
        """A consumed message never reaches admission, dedup or the agent."""
        probe.MODE = "consume"
        outsider = SimpleNamespace(
            id=OUTSIDER_USER_ID, display_name="Mallory", name="mallory", bot=False,
        )
        message = _make_message(msg_id=1001, author=outsider)

        assert await adapter._dispatch_discord_message(message) is False
        adapter._handle_message.assert_not_awaited()
        assert len(probe.CALLS) == 1
        # No dedup claim: the adapter never got that far.
        assert adapter._dedup.is_duplicate("1001") is False

    @pytest.mark.asyncio
    async def test_consume_wins_over_an_otherwise_admitted_message(self, adapter, probe):
        """Consume also claims traffic the adapter WOULD have admitted."""
        probe.MODE = "consume"
        assert await adapter._dispatch_discord_message(_make_message(msg_id=1002)) is False
        adapter._handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_identity_is_sdk_derived_not_text_derived(self, adapter, probe):
        """Identity comes from the SDK objects; ``/stop`` in the text is just data."""
        probe.MODE = "consume"
        message = _make_message(msg_id=1003, content="/stop please")
        await adapter._dispatch_discord_message(message)

        call = probe.CALLS[-1]
        identity = call["identity"]
        assert call["platform"] == "discord"
        assert call["message"] is message  # the raw SDK object, unmodified
        assert call["adapter"] is adapter
        assert identity["user_id"] == str(ALLOWED_USER_ID)
        assert identity["guild_id"] == "555"
        assert identity["chat_id"] == "100"
        assert identity["message_id"] == "1003"
        assert identity["user_name"] == "Alice"
        assert (identity["is_bot"], identity["is_webhook"], identity["is_self"]) == (
            False, False, False,
        )
        assert identity["is_dm"] is False
        assert identity["recovered"] is False
        assert all(
            isinstance(value, (str, bool, type(None))) for value in identity.values()
        ), identity

    @pytest.mark.asyncio
    async def test_identity_flags_bots_webhooks_self_and_dms(self, adapter, probe):
        """The flags a policy plugin needs to ignore non-human traffic."""
        probe.MODE = None
        bot_author = SimpleNamespace(id=88, display_name="Botty", name="botty", bot=True)
        await adapter._dispatch_discord_message(
            _make_message(msg_id=2001, author=bot_author, webhook_id=77)
        )
        identity = probe.CALLS[-1]["identity"]
        assert identity["is_bot"] is True and identity["is_webhook"] is True

        await adapter._dispatch_discord_message(
            _make_message(msg_id=2002, author=adapter._client.user)
        )
        assert probe.CALLS[-1]["identity"]["is_self"] is True

        await adapter._dispatch_discord_message(_make_message(msg_id=2003, guild=None))
        assert probe.CALLS[-1]["identity"]["is_dm"] is True
        assert probe.CALLS[-1]["identity"]["guild_id"] is None


class TestNeverGrantsAuthorization:
    @pytest.mark.parametrize("mode", [None, "allow", "allow-string", "raise"])
    @pytest.mark.asyncio
    async def test_non_consume_results_cannot_admit_an_outsider(self, adapter, probe, mode):
        """No verb on this hook lets a non-allowlisted user into the agent."""
        probe.MODE = mode
        outsider = SimpleNamespace(
            id=OUTSIDER_USER_ID, display_name="Mallory", name="mallory", bot=False,
        )
        assert await adapter._dispatch_discord_message(
            _make_message(msg_id=3001, author=outsider)
        ) is False
        adapter._handle_message.assert_not_awaited()
        assert probe.CALLS, "the hook must still have fired"

    @pytest.mark.parametrize("mode", [None, "allow", "allow-string", "raise"])
    @pytest.mark.asyncio
    async def test_non_consume_results_leave_normal_traffic_alone(self, adapter, probe, mode):
        """The unchanged baseline: an allowlisted user is still dispatched."""
        probe.MODE = mode
        await adapter._dispatch_discord_message(_make_message(msg_id=3100 + (0 if mode is None else 1)))
        adapter._handle_message.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_baseline_without_any_plugin(self, adapter, no_plugins):
        """No hook registered: the fast path short-circuits, admission is untouched."""
        assert no_plugins.has_hook(HOOK_NAME) is False
        await adapter._dispatch_discord_message(_make_message(msg_id=4001))
        adapter._handle_message.assert_awaited_once()

        outsider = SimpleNamespace(
            id=OUTSIDER_USER_ID, display_name="Mallory", name="mallory", bot=False,
        )
        assert await adapter._dispatch_discord_message(
            _make_message(msg_id=4002, author=outsider)
        ) is False
        assert adapter._handle_message.await_count == 1


class TestRecoveredPath:
    @pytest.mark.asyncio
    async def test_recovered_dispatch_fires_the_hook_and_can_be_consumed(self, adapter, probe):
        """Missed-message backfill must not walk bridged messages into the agent."""
        probe.MODE = "consume"
        message = _make_message(msg_id=5001)
        assert await adapter._dispatch_recovered_message(message) is False
        adapter._handle_message.assert_not_awaited()
        assert probe.CALLS[-1]["identity"]["recovered"] is True

    @pytest.mark.asyncio
    async def test_recovered_dispatch_baseline_when_not_consumed(self, adapter, probe, monkeypatch):
        monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")  # past the backfill mention gate
        probe.MODE = None
        await adapter._dispatch_recovered_message(_make_message(msg_id=5002))
        adapter._handle_message.assert_awaited_once()
        assert probe.CALLS[-1]["identity"]["recovered"] is True
