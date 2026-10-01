"""Gateway half of the exact bare-text → slash-command alias seam.

Reads the SAME ``text_command_aliases`` section of ``config.yaml`` the CLI ingress reads, and uses
the SAME matcher (``hermes_cli.text_command_aliases.resolve_text_command_alias``), so a trigger
behaves identically on Discord/LINE/etc. and in the CLI.

Why a plugin and not a core edit: ``pre_gateway_dispatch`` is an existing documented hook that can
rewrite ``event.text`` before auth, pairing and slash dispatch, so the gateway needs no patch that a
``hermes update`` could revert.

What the rewrite does NOT change:
  * The rewritten text is an ordinary slash command. Authorization, the per-platform slash access
    gate, the destructive-confirm flow and the busy policy all run afterwards, unchanged.
  * Events with ``allow_gateway_control`` false (proactive/untrusted plugin payloads) are skipped —
    that flag exists precisely so such text stays conversational.
  * Nothing is ever dropped. A non-matching message returns None and dispatches normally.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _raw_config() -> dict:
    """Active profile's raw ``config.yaml`` (mtime-cached, read-only, no defaults merged)."""
    try:
        from hermes_cli.config import read_raw_config_readonly

        return read_raw_config_readonly()
    except Exception:
        logger.debug("text-command-aliases: could not read config.yaml", exc_info=True)
        return {}


def _rewrite_exact_text_alias(event=None, **_kwargs):
    """``pre_gateway_dispatch``: rewrite an exact bare-text alias into its slash command."""
    if event is None or not getattr(event, "allow_gateway_control", False):
        return None
    try:
        from hermes_cli.text_command_aliases import resolve_text_command_alias

        command = resolve_text_command_alias(getattr(event, "text", None), _raw_config())
    except Exception:
        # Never take the gateway's inbound path down over an alias lookup.
        logger.warning("text-command-aliases: alias resolution failed", exc_info=True)
        return None
    if not command:
        return None
    logger.info("text-command-aliases: rewrote %r to %r", (event.text or "").strip(), command)
    return {"action": "rewrite", "text": command}


def register(ctx):
    ctx.register_hook("pre_gateway_dispatch", _rewrite_exact_text_alias)
