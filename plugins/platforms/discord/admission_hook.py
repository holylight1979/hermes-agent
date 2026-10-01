"""``pre_platform_message_admission``: the consume-only ingress hook for Discord.

Adapter admission (``_discord_message_admission``) drops a message long before a
``MessageEvent`` exists, so ``pre_gateway_dispatch`` never sees anything the
allowlist or the mention gate rejected. A plugin that wants to own a channel's
traffic — a translation bridge, an archiver — therefore had no reachable seam.

This hook opens exactly one: it runs *before* admission and its only verb is
``consume``. A callback can remove a message from the pipeline; it can never add
one. There is no ``allow``/``admit``/``rewrite`` directive, so no plugin can use
this surface to walk a non-allowlisted user into the agent, a slash command or an
admin path. Identity is built here from the SDK objects, never from message text.

Failure is fail-open *to the unchanged baseline*: a raising callback, a missing
subsystem or a malformed return value all leave admission exactly as it was.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

HOOK_NAME = "pre_platform_message_admission"
PLATFORM = "discord"
CONSUME_ACTION = "consume"


def _text(value: Any) -> Optional[str]:
    return None if value is None else str(value)


def _message_type_name(message: Any) -> Optional[str]:
    kind = getattr(message, "type", None)
    if kind is None:
        return None
    name = getattr(kind, "name", None)
    return name if isinstance(name, str) else str(kind)


def _build_identity(adapter: Any, message: Any, *, recovered: bool) -> Dict[str, Any]:
    """Trusted identity for one Discord message, read only from SDK objects."""
    import discord

    channel = getattr(message, "channel", None)
    guild = getattr(message, "guild", None)
    author = getattr(message, "author", None)
    is_dm = isinstance(channel, discord.DMChannel) or guild is None

    thread_id, chat_id = adapter._thread_id_and_chat_for_channel(channel)
    parent_chat_id = None
    if not is_dm:
        try:
            parent_chat_id = adapter._get_parent_channel_id(channel)
        except Exception:
            parent_chat_id = None

    is_self = False
    client_user = getattr(getattr(adapter, "_client", None), "user", None)
    if client_user is not None and author is not None:
        try:
            is_self = bool(author == client_user)
        except Exception:
            is_self = False

    try:
        mentions_self = bool(adapter._self_is_explicitly_mentioned(message))
    except Exception:
        mentions_self = False

    attachments = getattr(message, "attachments", None)

    return {
        "guild_id": _text(getattr(guild, "id", None)),
        "chat_id": _text(chat_id),
        "parent_chat_id": _text(parent_chat_id),
        "thread_id": _text(thread_id),
        "message_id": _text(getattr(message, "id", None)),
        "user_id": _text(getattr(author, "id", None)),
        "user_name": _text(
            getattr(author, "display_name", None) or getattr(author, "name", None)
        ),
        "is_bot": bool(getattr(author, "bot", False)),
        "is_webhook": bool(getattr(message, "webhook_id", None)),
        "is_self": is_self,
        "is_dm": bool(is_dm),
        "message_type": _message_type_name(message),
        "mentions_self": mentions_self,
        "has_attachments": bool(attachments),
        "recovered": bool(recovered),
    }


def _is_consume(result: Any) -> bool:
    return (
        isinstance(result, dict)
        and str(result.get("action") or "").strip().lower() == CONSUME_ACTION
    )


def consume_claimed(adapter: Any, message: Any, *, recovered: bool = False) -> bool:
    """Whether a plugin consumed *message*, so the adapter must stop processing it.

    ``True`` means: no admission, no dedup claim, no auth, no dispatch, no
    command handling. Anything other than a ``{"action": "consume"}`` dict —
    including every error — means the caller proceeds unchanged.
    """
    try:
        from hermes_cli.lifecycle import has_hook, invoke_hook

        if not has_hook(HOOK_NAME):
            return False
        results = invoke_hook(
            HOOK_NAME,
            platform=PLATFORM,
            identity=_build_identity(adapter, message, recovered=recovered),
            message=message,
            adapter=adapter,
        )
    except Exception:
        logger.debug(
            "[%s] %s failed; falling back to unchanged admission",
            getattr(adapter, "name", PLATFORM), HOOK_NAME, exc_info=True,
        )
        return False

    for result in results or ():
        if _is_consume(result):
            logger.info(
                "[%s] message %s consumed by a %s plugin (%s)",
                getattr(adapter, "name", PLATFORM), getattr(message, "id", None),
                HOOK_NAME, result.get("reason") or "no reason given",
            )
            return True
    return False
