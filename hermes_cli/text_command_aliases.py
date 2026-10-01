"""Exact bare-text → slash-command aliases (``text_command_aliases`` in config.yaml).

A surface's ingress resolves the submitted text through :func:`resolve_text_command_alias` before
anything reaches the model. An EXACT match (surrounding whitespace stripped) is replaced by the
configured slash command and then dispatched through that surface's ordinary slash path, so
authorization, the busy policy and the command's own program-generated output all stay in force.
Anything else — a substring, a prefix, extra words, different case — is returned unchanged and runs
as a normal prompt.

Consumers: ``cli.py`` (interactive ingress) and the gateway ``text-command-aliases`` plugin
(``pre_gateway_dispatch`` rewrite). The matching rule lives here once so every surface agrees.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Optional

logger = logging.getLogger(__name__)

CONFIG_SECTION = "text_command_aliases"


def _section(config: Any) -> Optional[Mapping]:
    if not isinstance(config, Mapping):
        return None
    section = config.get(CONFIG_SECTION)
    return section if isinstance(section, Mapping) else None


def alias_table(config: Any) -> dict[str, str]:
    """``{bare text: slash command}`` declared in *config*; invalid entries are dropped + logged."""
    section = _section(config)
    if section is None or section.get("enabled") is False:
        return {}
    raw = section.get("aliases")
    if not isinstance(raw, Mapping):
        return {}
    table: dict[str, str] = {}
    for key, target in raw.items():
        trigger = key.strip() if isinstance(key, str) else ""
        command = target.strip() if isinstance(target, str) else ""
        # A "/..." trigger could shadow a real slash command; ingress resolves slash input itself.
        if not trigger or "\n" in trigger or trigger.startswith("/"):
            logger.warning("%s: ignoring entry with an invalid trigger %r", CONFIG_SECTION, key)
            continue
        if not command.startswith("/") or "\n" in command:
            logger.warning(
                "%s[%r]: target must be a single-line slash command, got %r",
                CONFIG_SECTION, trigger, target)
            continue
        table[trigger] = command
    return table


def resolve_text_command_alias(text: Any, config: Any) -> Optional[str]:
    """The slash command *text* is an exact bare alias for, else ``None``.

    Matching is whole-message and case-sensitive: only ``text.strip()`` equal to a configured
    trigger resolves. Text that is already slash input is never rewritten.
    """
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    if not stripped or stripped.startswith("/"):
        return None
    return alias_table(config).get(stripped)
