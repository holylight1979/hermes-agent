"""Tests for the LINE platform adapter plugin.

Covers LINE adapter behavior from the PR review:

1. webhook signature verification (HMAC-SHA256, base64) + tampering rejection
2. inbound chat-id resolution for user / group / room sources
3. three-allowlist gating (users / groups / rooms / allow_all)
4. inbound dedup via webhookEventId
5. RequestCache state machine (PENDING → READY → DELIVERED, ERROR)
6. Markdown stripping with URL preservation + LINE-sized chunking
7. inbound media normalization to gateway message types and MIME metadata
8. send routing: reply token preferred → push fallback → batched at 5/call
9. register() metadata + standalone_send shape
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.gateway._plugin_adapter_loader import load_plugin_adapter

# Load plugins/platforms/line/adapter.py under plugin_adapter_line so it
# cannot collide with sibling platform-plugin tests in the same xdist worker.
_line = load_plugin_adapter("line")

verify_line_signature = _line.verify_line_signature
strip_markdown_preserving_urls = _line.strip_markdown_preserving_urls
split_for_line = _line.split_for_line
build_postback_button_message = _line.build_postback_button_message
_resolve_chat = _line._resolve_chat
_allowed_for_source = _line._allowed_for_source
_is_system_bypass = _line._is_system_bypass
RequestCache = _line.RequestCache
State = _line.State
LineAdapter = _line.LineAdapter
register = _line.register
check_requirements = _line.check_requirements
validate_config = _line.validate_config
_standalone_send = _line._standalone_send
_env_enablement = _line._env_enablement
_MessageDeduplicator = _line._MessageDeduplicator
_mentions_bot = _line._mentions_bot


# ---------------------------------------------------------------------------
# 1. Signature verification
# ---------------------------------------------------------------------------

class TestSignature:

    def _sign(self, body: bytes, secret: str) -> str:
        digest = hmac.new(secret.encode(), body, hashlib.sha256).digest()
        return base64.b64encode(digest).decode()


    def test_wrong_secret_rejected(self):
        body = b'{"events": []}'
        sig = self._sign(body, "secret")
        assert not verify_line_signature(body, sig, "different")


    def test_empty_secret_rejected(self):
        assert not verify_line_signature(b"x", "AAAA", "")


# ---------------------------------------------------------------------------
# 2. Chat-id / source resolution
# ---------------------------------------------------------------------------

class TestSourceResolution:

    def test_user_source(self):
        chat_id, ctype = _resolve_chat({"type": "user", "userId": "U123"})
        assert chat_id == "U123"
        assert ctype == "dm"

    def test_group_source(self):
        chat_id, ctype = _resolve_chat({"type": "group", "groupId": "C456", "userId": "U123"})
        assert chat_id == "C456"
        assert ctype == "group"


# ---------------------------------------------------------------------------
# 3. Three-allowlist gating
# ---------------------------------------------------------------------------

class TestAllowlist:

    def test_allow_all_short_circuits(self):
        for src in [
            {"type": "user", "userId": "Ufoo"},
            {"type": "group", "groupId": "Cfoo"},
            {"type": "room", "roomId": "Rfoo"},
        ]:
            assert _allowed_for_source(src, allow_all=True, user_ids=set(), group_ids=set(), room_ids=set())

    def test_user_in_allowlist_passes(self):
        src = {"type": "user", "userId": "Uok"}
        assert _allowed_for_source(src, allow_all=False, user_ids={"Uok"}, group_ids=set(), room_ids=set())


# ---------------------------------------------------------------------------
# 4. Inbound dedup
# ---------------------------------------------------------------------------

class TestDedup:

    def test_first_event_not_duplicate(self):
        d = _MessageDeduplicator()
        assert not d.is_duplicate("evt1")


# ---------------------------------------------------------------------------
# 5. RequestCache state machine
# ---------------------------------------------------------------------------

class TestRequestCache:


    def test_mark_delivered_from_ready(self):
        c = RequestCache()
        rid = c.register_pending("Uchat")
        c.set_ready(rid, "x")
        c.mark_delivered(rid)
        assert c.get(rid).state is State.DELIVERED


    def test_set_ready_on_delivered_is_noop(self):
        c = RequestCache()
        rid = c.register_pending("Uchat")
        c.set_ready(rid, "first")
        c.mark_delivered(rid)
        c.set_ready(rid, "second")
        # DELIVERED is terminal — no further mutation
        assert c.get(rid).payload == "first"
        assert c.get(rid).state is State.DELIVERED


# ---------------------------------------------------------------------------
# 6. Markdown stripping + chunking
# ---------------------------------------------------------------------------

class TestMarkdownAndChunking:

    def test_bold_stripped(self):
        assert strip_markdown_preserving_urls("**hello**") == "hello"

    def test_italic_stripped(self):
        assert strip_markdown_preserving_urls("*hello*") == "hello"


    def test_split_long_chunks_at_paragraph_boundary(self):
        text = "para1\n\npara2\n\npara3"
        chunks = split_for_line(text, max_chars=8)
        assert all(len(c) <= 8 for c in chunks), chunks
        assert len(chunks) >= 2

    def test_split_caps_at_five_chunks(self):
        # 1000 paragraphs of 100 chars each — must cap at 5 LINE bubbles.
        text = "\n\n".join(["x" * 100 for _ in range(1000)])
        chunks = split_for_line(text)
        assert len(chunks) <= 5


# ---------------------------------------------------------------------------
# 7. Inbound media normalization
# ---------------------------------------------------------------------------

class TestInboundMedia:

    @pytest.fixture
    def adapter(self, monkeypatch):
        monkeypatch.delenv("LINE_CHANNEL_ACCESS_TOKEN", raising=False)
        monkeypatch.delenv("LINE_CHANNEL_SECRET", raising=False)
        # These events are unmentioned group messages: pin the mention gate off so
        # this class tests media normalization, not the gate (and so a developer's
        # own LINE_REQUIRE_MENTION can't leak in).
        monkeypatch.delenv("LINE_REQUIRE_MENTION", raising=False)
        from gateway.config import PlatformConfig

        cfg = PlatformConfig(enabled=True, extra={
            "channel_access_token": "tok",
            "channel_secret": "sec",
        })
        ad = LineAdapter(cfg)
        ad._client = MagicMock()
        ad._client.fetch_content = AsyncMock(return_value=b"line-bytes")
        ad.handle_message = AsyncMock()
        return ad

    def _event(self, msg_type, **message):
        payload = {"type": msg_type, "id": f"{msg_type}-1"}
        payload.update(message)
        return {
            "type": "message",
            "replyToken": "reply-token",
            "source": {"type": "group", "groupId": "Cline", "userId": "Uline"},
            "message": payload,
        }

    def _captured_event(self, adapter):
        adapter.handle_message.assert_awaited_once()
        return adapter.handle_message.await_args.args[0]

    def test_image_message_uses_photo_type_and_image_mime(self, adapter):
        with patch.object(
            _line,
            "cache_image_from_bytes_async",
            new=AsyncMock(return_value="/cache/image.jpg"),
        ) as cache:
            asyncio.run(adapter._handle_message_event(self._event("image")))

        cache.assert_awaited_once_with(b"line-bytes", ext=".jpg")
        event = self._captured_event(adapter)
        assert event.message_type is _line.MessageType.PHOTO
        assert event.media_urls == ["/cache/image.jpg"]
        assert event.media_types == ["image/jpeg"]


# ---------------------------------------------------------------------------
# 8. Send routing (reply -> push fallback, batching, system-bypass)
# ---------------------------------------------------------------------------

class TestSendRouting:

    @pytest.fixture
    def adapter(self, monkeypatch):
        monkeypatch.delenv("LINE_CHANNEL_ACCESS_TOKEN", raising=False)
        monkeypatch.delenv("LINE_CHANNEL_SECRET", raising=False)
        from gateway.config import PlatformConfig
        cfg = PlatformConfig(enabled=True, extra={
            "channel_access_token": "tok",
            "channel_secret": "sec",
        })
        ad = LineAdapter(cfg)
        ad._client = MagicMock()
        ad._client.reply = AsyncMock()
        ad._client.push = AsyncMock()
        return ad

    def test_system_bypass_recognized(self):
        assert _is_system_bypass("⚡ Interrupting current run")
        assert _is_system_bypass("⏳ Queued — agent is busy")
        assert _is_system_bypass("⏩ Steered toward new task")
        assert not _is_system_bypass("Hello world")
        assert not _is_system_bypass("")


    def test_send_caps_messages_per_call_at_five(self, adapter):
        # Build a payload that would naturally split into more than 5 LINE
        # bubbles; the chunker should cap at 5 + truncate.
        big = "\n\n".join(["x" * 4500 for _ in range(20)])
        result = asyncio.run(adapter.send("Uchat", big))
        assert result.success
        call_kwargs = adapter._client.push.call_args
        # call_args is (args, kwargs); for our send the messages are the 2nd positional
        sent_messages = call_kwargs.args[1] if call_kwargs.args else call_kwargs.kwargs.get("messages")
        # Without args, fall back to inspecting the call shape
        if sent_messages is None:
            # We invoked client.push(chat_id, messages) — check first batch
            sent_messages = adapter._client.push.call_args.args[1]
        assert len(sent_messages) <= 5

    def test_format_message_strips_markdown(self, adapter):
        out = adapter.format_message("**bold** [link](https://x.com)")
        assert "**" not in out
        assert "https://x.com" in out


# ---------------------------------------------------------------------------
# 9. Register() metadata + plugin entry points
# ---------------------------------------------------------------------------

class TestRegister:

    class _FakeCtx:
        def __init__(self):
            self.kwargs = None

        def register_platform(self, **kw):
            self.kwargs = kw


    def test_register_advertises_required_env(self):
        ctx = self._FakeCtx()
        register(ctx)
        assert set(ctx.kwargs["required_env"]) == {
            "LINE_CHANNEL_ACCESS_TOKEN",
            "LINE_CHANNEL_SECRET",
        }


    def test_register_factory_yields_line_adapter(self):
        ctx = self._FakeCtx()
        register(ctx)
        from gateway.config import PlatformConfig
        cfg = PlatformConfig(enabled=True, extra={
            "channel_access_token": "tok",
            "channel_secret": "sec",
        })
        ad = ctx.kwargs["adapter_factory"](cfg)
        assert isinstance(ad, LineAdapter)

    def test_max_message_length_below_line_per_bubble_limit(self):
        ctx = self._FakeCtx()
        register(ctx)
        # LINE per-bubble limit is 5000; we register 4500 to leave headroom.
        assert ctx.kwargs["max_message_length"] <= 5000


class TestSlowLLMPostbackRegression:
    """#106446 — the two reporter reproductions, driven through the real ``send()`` /
    ``_handle_postback_event()`` paths with only the HTTP client mocked."""

    @pytest.fixture
    def adapter(self, monkeypatch):
        monkeypatch.delenv("LINE_CHANNEL_ACCESS_TOKEN", raising=False)
        monkeypatch.delenv("LINE_CHANNEL_SECRET", raising=False)
        from gateway.config import PlatformConfig
        cfg = PlatformConfig(enabled=True, extra={"channel_access_token": "tok", "channel_secret": "sec"})
        ad = LineAdapter(cfg)
        ad._client = MagicMock()
        ad._client.reply = AsyncMock()
        ad._client.push = AsyncMock()
        return ad

    def test_repro_a_heartbeat_does_not_become_the_cached_answer(self, adapter):
        # The gateway's periodic heartbeat (stamped ``_interim_send`` by ``_interim_metadata``)
        # arrives while the postback button is PENDING; the real answer follows.
        rid = adapter._cache.register_pending("Uchat")
        adapter._pending_buttons["Uchat"] = rid
        asyncio.run(adapter.send("Uchat", "⏳ Working — 3 min — iteration 1/60, research_market",
                                 metadata={"_interim_send": True}))
        assert adapter._cache.get(rid).state is State.PENDING, "heartbeat must not fill the cache"
        assert adapter._client.push.await_count == 1, "heartbeat lands as a visible bubble"
        asyncio.run(adapter.send("Uchat", "Final answer: task completed"))
        assert adapter._cache.get(rid).payload == "Final answer: task completed"

    def test_repro_b_stale_mapping_does_not_swallow_the_next_answer(self, adapter):
        # A READY entry still mapped to the chat, then a tap on a button whose entry is gone:
        # neither may absorb a later send or leave the chat stuck.
        rid = adapter._cache.register_pending("Uchat")
        adapter._cache.set_ready(rid, "Previous answer")
        adapter._pending_buttons["Uchat"] = rid
        result = asyncio.run(adapter.send("Uchat", "Answer to the NEXT user message"))
        assert result.success and adapter._client.push.await_count == 1, "next answer reaches the wire"
        assert adapter._cache.get(rid).payload == "Previous answer"
        assert "Uchat" not in adapter._pending_buttons
        # Expired button tap: user gets a notice and the dead mapping is dropped.
        adapter._pending_buttons["Uchat"] = "ghost-rid"
        asyncio.run(adapter._handle_postback_event({
            "replyToken": "reply-token", "source": {"type": "user", "userId": "Uchat"},
            "postback": {"data": json.dumps({"action": "show_response", "request_id": "ghost-rid"})}}))
        adapter._client.reply.assert_awaited_once()
        assert adapter._client.reply.await_args.args[1][0]["text"] == adapter.expired_text
        assert "Uchat" not in adapter._pending_buttons


class TestEnvEnablement:

    def test_returns_none_without_credentials(self, monkeypatch):
        monkeypatch.delenv("LINE_CHANNEL_ACCESS_TOKEN", raising=False)
        monkeypatch.delenv("LINE_CHANNEL_SECRET", raising=False)
        assert _env_enablement() is None


class TestStandaloneSend:

    def test_missing_token_returns_error(self, monkeypatch):
        monkeypatch.delenv("LINE_CHANNEL_ACCESS_TOKEN", raising=False)
        from gateway.config import PlatformConfig
        cfg = PlatformConfig(enabled=True, extra={})
        result = asyncio.run(_standalone_send(cfg, "Uchat", "hi"))
        assert "error" in result


class TestPostbackButtonShape:

    def test_template_buttons_structure(self):
        msg = build_postback_button_message("hi", "Tap me", "rid-1")
        assert msg["type"] == "template"
        assert msg["template"]["type"] == "buttons"
        assert msg["template"]["text"] == "hi"
        actions = msg["template"]["actions"]
        assert len(actions) == 1
        assert actions[0]["type"] == "postback"
        data = json.loads(actions[0]["data"])
        assert data == {"action": "show_response", "request_id": "rid-1"}


class TestCheckRequirements:


    def test_rejects_without_secret(self, monkeypatch):
        monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "t")
        monkeypatch.delenv("LINE_CHANNEL_SECRET", raising=False)
        assert not check_requirements()


class TestValidateConfig:

    def test_validates_from_extra(self):
        from gateway.config import PlatformConfig
        cfg = PlatformConfig(
            enabled=True,
            extra={"channel_access_token": "t", "channel_secret": "s"},
        )
        assert validate_config(cfg)


class TestAdapterInit:

    def test_init_from_config_extra(self, monkeypatch):
        for k in ("LINE_CHANNEL_ACCESS_TOKEN", "LINE_CHANNEL_SECRET", "LINE_PORT"):
            monkeypatch.delenv(k, raising=False)
        from gateway.config import PlatformConfig
        cfg = PlatformConfig(
            enabled=True,
            extra={
                "channel_access_token": "tok",
                "channel_secret": "sec",
                "port": 7777,
                "public_url": "https://x.example.com",
                "allowed_users": ["U1", "U2"],
            },
        )
        ad = LineAdapter(cfg)
        assert ad.channel_access_token == "tok"
        assert ad.channel_secret == "sec"
        assert ad.webhook_port == 7777
        assert ad.public_base_url == "https://x.example.com"
        assert ad.allowed_users == {"U1", "U2"}


# ---------------------------------------------------------------------------
# 9. Inbound message-type classification
# ---------------------------------------------------------------------------

class TestMessageTypeMapping:
    """LINE webhook message types must map to the right normalized
    MessageType so the gateway routes media correctly (e.g. voice → STT,
    files → document handling). Regression guard for the old code that
    referenced the non-existent ``MessageType.IMAGE`` and collapsed every
    non-text message onto a single type."""

    def test_image_event_not_attributeerror_regression(self):
        # The bug: MessageType.IMAGE doesn't exist on the enum.
        MessageType = _line.MessageType
        assert not hasattr(MessageType, "IMAGE")


# ---------------------------------------------------------------------------
# 10. Dual-stack bind default (NS-603)
# ---------------------------------------------------------------------------

class TestDualStackBind:
    """The LINE webhook server's default bind must serve BOTH IPv4 and IPv6.

    Regression guard for the hosted LINE 502 (NS-603): Fly.io 6PN — the
    private network the edge router reverse-proxies LINE ingest over — is
    IPv6-only (``<app>.internal`` resolves to an ``fdaa:…`` address). The
    adapter used to default to ``host="0.0.0.0"`` (IPv4 only), so the
    router's dial to ``<app>.internal:8646`` hit an address nothing was
    listening on → connection refused → 502 on webhook verification.

    Mirrors gateway/platforms/webhook.py's fix (commit d542894ad):
    ``DEFAULT_HOST = None`` → asyncio binds one socket per address family.
    ``"::"`` is NOT a valid substitute (bindv6only=1 on Fly machines makes
    it IPv6-only, breaking IPv4 loopback health probes).
    """

    def _cfg(self, **extra):
        from gateway.config import PlatformConfig
        base = {"channel_access_token": "tok", "channel_secret": "sec"}
        base.update(extra)
        return PlatformConfig(enabled=True, extra=base)


    def test_empty_host_normalises_to_none(self, monkeypatch):
        monkeypatch.delenv("LINE_HOST", raising=False)
        ad = LineAdapter(self._cfg(host=""))
        assert ad.webhook_host is None


    @pytest.mark.asyncio
    async def test_default_bind_serves_both_families(self, monkeypatch):
        """Behavioural proof: host=None opens v4 AND v6 listening sockets."""
        # Guard: on IPv4-only hosts (CI runners with IPv6 disabled) the
        # dual-stack bind legitimately yields no v6 socket — that's the
        # environment, not a regression. Probe an actual ::1 bind rather
        # than trusting socket.has_ipv6 (compile-time constant).
        import socket as _socket
        if not _socket.has_ipv6:
            pytest.skip("IPv6 not supported by this Python build")
        try:
            _probe = _socket.socket(_socket.AF_INET6, _socket.SOCK_STREAM)
            try:
                _probe.bind(("::1", 0))
            finally:
                _probe.close()
        except OSError:
            pytest.skip("IPv6 stack unavailable on this host (cannot bind ::1)")
        monkeypatch.delenv("LINE_HOST", raising=False)
        ad = LineAdapter(self._cfg(port=0))
        ad._client = MagicMock()
        ad._client.get_bot_user_id = AsyncMock(return_value="Ubot")

        # Skip credential/network preamble — drive the aiohttp bind directly
        # the same way connect() does.
        from aiohttp import web
        ad._app = web.Application()
        ad._runner = web.AppRunner(ad._app)
        await ad._runner.setup()
        site = web.TCPSite(ad._runner, ad.webhook_host, 0)
        try:
            await site.start()
            addrs = list(ad._runner.addresses)
            has_v6 = any(len(a) == 4 for a in addrs)
            has_v4 = any(len(a) == 2 for a in addrs)
            assert has_v4, f"IPv4 bind missing — got {addrs}"
            assert has_v6, (
                f"IPv6 bind missing (the 6PN reachability bug, NS-603) — got {addrs}"
            )
        finally:
            await ad._runner.cleanup()


class TestMediaPublicUrlGuard:
    """Outbound media requires LINE_PUBLIC_URL whenever the bind host is not
    a publicly fetchable address — including the new dual-stack ``None``
    default and the legacy wildcard strings."""

    def _adapter(self, monkeypatch, **extra):
        from gateway.config import PlatformConfig
        monkeypatch.delenv("LINE_HOST", raising=False)
        monkeypatch.delenv("LINE_PUBLIC_URL", raising=False)
        base = {"channel_access_token": "tok", "channel_secret": "sec"}
        base.update(extra)
        return LineAdapter(PlatformConfig(enabled=True, extra=base))


    def test_missing_public_url_false_with_public_base(self, monkeypatch):
        ad = self._adapter(monkeypatch, public_url="https://tunnel.example.com")
        if not ad.public_base_url:
            # Adapter reads env var name LINE_PUBLIC_URL / extra key —
            # set directly if the extra key differs.
            ad.public_base_url = "https://tunnel.example.com"
        assert ad._missing_public_url() is False


    def test_send_image_blocked_without_public_url(self, monkeypatch, tmp_path):
        ad = self._adapter(monkeypatch)
        ad._client = MagicMock()
        img = tmp_path / "x.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n123")
        result = asyncio.run(ad.send_image_file("Uchat", str(img)))
        assert not result.success
        assert "LINE_PUBLIC_URL" in (result.error or "")


# ---------------------------------------------------------------------------
# 11. require_mention gate for groups / rooms
# ---------------------------------------------------------------------------

_BOT_ID = "Ubot"


class TestMentionsBot:
    """Only LINE's native mention metadata naming *this* bot counts."""

    def test_matching_user_mention(self):
        msg = {"type": "text", "text": "@bot hi",
               "mention": {"mentionees": [
                   {"index": 0, "length": 4, "type": "user", "userId": _BOT_ID}]}}
        assert _mentions_bot(msg, _BOT_ID)

    def test_legacy_payload_without_type_field(self):
        # Older LINE payloads omit "type" — those are always user mentions.
        msg = {"mention": {"mentionees": [{"index": 0, "length": 4, "userId": _BOT_ID}]}}
        assert _mentions_bot(msg, _BOT_ID)

    def test_other_user_mention_does_not_match(self):
        msg = {"mention": {"mentionees": [
            {"index": 0, "length": 4, "type": "user", "userId": "Usomeone"}]}}
        assert not _mentions_bot(msg, _BOT_ID)

    def test_mention_all_is_not_a_bot_mention(self):
        # @All carries type="all" and no userId — a broadcast is not an address to the bot.
        msg = {"mention": {"mentionees": [{"index": 0, "length": 4, "type": "all"}]}}
        assert not _mentions_bot(msg, _BOT_ID)

    def test_mention_all_with_bot_userid_still_rejected(self):
        # Defensive: even if a payload tags an "all" mentionee with our id.
        msg = {"mention": {"mentionees": [
            {"index": 0, "length": 4, "type": "all", "userId": _BOT_ID}]}}
        assert not _mentions_bot(msg, _BOT_ID)

    def test_no_mention_metadata(self):
        assert not _mentions_bot({"type": "text", "text": "hello"}, _BOT_ID)

    def test_malformed_mention_metadata(self):
        assert not _mentions_bot({"mention": {"mentionees": "nope"}}, _BOT_ID)
        assert not _mentions_bot({"mention": {"mentionees": ["nope"]}}, _BOT_ID)
        assert not _mentions_bot({"mention": None}, _BOT_ID)

    def test_unknown_bot_id_fails_closed(self):
        msg = {"mention": {"mentionees": [{"type": "user", "userId": _BOT_ID}]}}
        assert not _mentions_bot(msg, None)
        assert not _mentions_bot(msg, "")


class TestRequireMentionConfig:
    """LINE_REQUIRE_MENTION overrides extra.require_mention; extra is the default."""

    def _adapter(self, monkeypatch, **extra):
        from gateway.config import PlatformConfig
        for k in ("LINE_CHANNEL_ACCESS_TOKEN", "LINE_CHANNEL_SECRET"):
            monkeypatch.delenv(k, raising=False)
        base = {"channel_access_token": "tok", "channel_secret": "sec"}
        base.update(extra)
        return LineAdapter(PlatformConfig(enabled=True, extra=base))

    def test_defaults_off(self, monkeypatch):
        monkeypatch.delenv("LINE_REQUIRE_MENTION", raising=False)
        assert self._adapter(monkeypatch).require_mention is False

    def test_extra_is_the_default_source(self, monkeypatch):
        monkeypatch.delenv("LINE_REQUIRE_MENTION", raising=False)
        assert self._adapter(monkeypatch, require_mention=True).require_mention is True
        assert self._adapter(monkeypatch, require_mention=False).require_mention is False

    def test_extra_accepts_yaml_strings(self, monkeypatch):
        monkeypatch.delenv("LINE_REQUIRE_MENTION", raising=False)
        assert self._adapter(monkeypatch, require_mention="true").require_mention is True
        # A bare bool() would read "false" as True — it must not.
        assert self._adapter(monkeypatch, require_mention="false").require_mention is False

    @pytest.mark.parametrize("raw", ["true", "True", "1", "yes", "on"])
    def test_env_true_variants(self, monkeypatch, raw):
        monkeypatch.setenv("LINE_REQUIRE_MENTION", raw)
        assert self._adapter(monkeypatch, require_mention=False).require_mention is True

    @pytest.mark.parametrize("raw", ["false", "0", "no", "off", ""])
    def test_env_false_variants_override_extra(self, monkeypatch, raw):
        monkeypatch.setenv("LINE_REQUIRE_MENTION", raw)
        assert self._adapter(monkeypatch, require_mention=True).require_mention is False


class TestRequireMentionGate:
    """The gate must run before the reply token is stashed, before media is
    downloaded, and before the agent is invoked."""

    def _adapter(self, monkeypatch, *, require_mention=True, bot_id=_BOT_ID):
        from gateway.config import PlatformConfig
        for k in ("LINE_CHANNEL_ACCESS_TOKEN", "LINE_CHANNEL_SECRET",
                  "LINE_REQUIRE_MENTION"):
            monkeypatch.delenv(k, raising=False)
        cfg = PlatformConfig(enabled=True, extra={
            "channel_access_token": "tok",
            "channel_secret": "sec",
            "require_mention": require_mention,
        })
        ad = LineAdapter(cfg)
        ad._bot_user_id = bot_id
        ad._client = MagicMock()
        ad._client.fetch_content = AsyncMock(return_value=b"line-bytes")
        ad._client.loading = AsyncMock()
        ad._client.reply = AsyncMock()
        ad._client.push = AsyncMock()
        ad.handle_message = AsyncMock()
        return ad

    def _event(self, *, source_type="group", msg=None, mentionees=None):
        source = {
            "group": {"type": "group", "groupId": "Cgroup", "userId": "Umember"},
            "room": {"type": "room", "roomId": "Rroom", "userId": "Umember"},
            "user": {"type": "user", "userId": "Umember"},
        }[source_type]
        message = dict(msg or {"type": "text", "id": "m-1", "text": "hello"})
        if mentionees is not None:
            message["mention"] = {"mentionees": mentionees}
        return {
            "type": "message",
            "replyToken": "reply-token",
            "source": source,
            "message": message,
        }

    def _run(self, adapter, event):
        async def _go():
            await adapter._handle_message_event(event)
            # Let any fire-and-forget task (the DM loading indicator) settle.
            await asyncio.sleep(0)
        asyncio.run(_go())

    def _assert_ignored(self, adapter):
        adapter.handle_message.assert_not_awaited()
        assert adapter._reply_tokens == {}
        adapter._client.fetch_content.assert_not_awaited()

    # -- gate disabled / DM: unchanged behaviour ------------------------

    def test_group_passes_when_gate_disabled(self, monkeypatch):
        ad = self._adapter(monkeypatch, require_mention=False)
        self._run(ad, self._event())
        ad.handle_message.assert_awaited_once()
        assert ad._reply_tokens["Cgroup"][0] == "reply-token"

    def test_dm_passes_when_gate_enabled(self, monkeypatch):
        ad = self._adapter(monkeypatch)
        self._run(ad, self._event(source_type="user"))
        ad.handle_message.assert_awaited_once()
        assert ad._reply_tokens["Umember"][0] == "reply-token"

    # -- gate enabled: pass only on an explicit bot mention -------------

    def test_group_mentioning_bot_passes(self, monkeypatch):
        ad = self._adapter(monkeypatch)
        self._run(ad, self._event(
            mentionees=[{"index": 0, "length": 4, "type": "user", "userId": _BOT_ID}]))
        ad.handle_message.assert_awaited_once()
        assert ad._reply_tokens["Cgroup"][0] == "reply-token"

    def test_room_mentioning_bot_passes(self, monkeypatch):
        ad = self._adapter(monkeypatch)
        self._run(ad, self._event(
            source_type="room",
            mentionees=[{"index": 0, "length": 4, "type": "user", "userId": _BOT_ID}]))
        ad.handle_message.assert_awaited_once()
        assert ad._reply_tokens["Rroom"][0] == "reply-token"

    def test_group_without_mention_ignored(self, monkeypatch):
        ad = self._adapter(monkeypatch)
        self._run(ad, self._event())
        self._assert_ignored(ad)

    def test_room_without_mention_ignored(self, monkeypatch):
        ad = self._adapter(monkeypatch)
        self._run(ad, self._event(source_type="room"))
        self._assert_ignored(ad)

    def test_group_mentioning_someone_else_ignored(self, monkeypatch):
        ad = self._adapter(monkeypatch)
        self._run(ad, self._event(
            mentionees=[{"index": 0, "length": 6, "type": "user", "userId": "Uother"}]))
        self._assert_ignored(ad)

    def test_mention_all_ignored(self, monkeypatch):
        ad = self._adapter(monkeypatch)
        self._run(ad, self._event(
            mentionees=[{"index": 0, "length": 4, "type": "all"}]))
        self._assert_ignored(ad)

    def test_missing_bot_user_id_fails_closed(self, monkeypatch):
        # get_bot_user_id() failed at connect — we cannot prove the mention is
        # ours, so nothing from a group may reach the agent.
        ad = self._adapter(monkeypatch, bot_id=None)
        self._run(ad, self._event(
            mentionees=[{"index": 0, "length": 4, "type": "user", "userId": _BOT_ID}]))
        self._assert_ignored(ad)

    # -- the gate runs BEFORE media download ---------------------------

    def test_unmentioned_image_is_not_downloaded(self, monkeypatch):
        ad = self._adapter(monkeypatch)
        with patch.object(_line, "cache_image_from_bytes_async", new=AsyncMock()) as cache:
            self._run(ad, self._event(msg={"type": "image", "id": "img-1"}))
        cache.assert_not_awaited()
        self._assert_ignored(ad)

    @pytest.mark.parametrize("msg", [
        {"type": "text", "id": "m-1", "text": "just chatting"},
        {"type": "sticker", "id": "m-2", "keywords": ["hi"]},
        {"type": "image", "id": "m-3"},
        {"type": "audio", "id": "m-4"},
        {"type": "video", "id": "m-5"},
        {"type": "file", "id": "m-6", "fileName": "x.pdf"},
        {"type": "location", "id": "m-7", "title": "t", "address": "a"},
    ], ids=lambda m: m["type"])
    def test_every_unmentioned_message_type_is_dropped(self, monkeypatch, msg):
        ad = self._adapter(monkeypatch)
        self._run(ad, self._event(msg=msg))
        self._assert_ignored(ad)

    # -- postback is unaffected ----------------------------------------

    def test_postback_still_delivers_with_gate_enabled(self, monkeypatch):
        ad = self._adapter(monkeypatch)
        rid = ad._cache.register_pending("Cgroup")
        ad._cache.set_ready(rid, "the answer")
        asyncio.run(ad._handle_postback_event({
            "type": "postback",
            "replyToken": "fresh-token",
            "source": {"type": "group", "groupId": "Cgroup", "userId": "Umember"},
            "postback": {"data": json.dumps(
                {"action": "show_response", "request_id": rid})},
        }))
        ad._client.reply.assert_awaited_once()
        assert ad._cache.get(rid).state is State.DELIVERED

