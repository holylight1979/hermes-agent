"""Server-side enforcement of the crack-talk route's two-tool allowlist.

Advertising only ``web_search`` and ``web_extract`` (see :mod:`light_web`) is presentation. A model
can still emit a call for a tool it was never shown — directly, or by naming one inside a generic
dispatcher like ``tool_call`` / ``execute_code`` / ``terminal``. So the allowlist is also applied at
the seam where a tool actually runs: ``hermes_cli.middleware.run_tool_execution_middleware``.

That seam is given a tool name and a ``session_id``, but not a route. The route is known one layer
up, in the LLM execution middleware, so that layer stamps the verdict here: every request records
whether the session it belongs to is, right now, a lightweight crack-talk session. Because a tool
call can only exist as the result of a model response, the stamp is always set before any call of
that turn is executed, and a session that has left the route (``llm-cr-end``) clears its own stamp
on its very next request.

Sessions with no id are never gated here: they cannot be matched to a route, so the LLM middleware
refuses to run the route at all without one (``injector``) rather than leaving an ungated session.

Lifecycle, and why nothing is ever evicted: a stamp is *added* by a crack-talk request and *removed*
by the same session's next non-crack-talk request. Dropping a stamp for any other reason — age, table
size — would silently hand a still-running crack-talk session its whole toolset back, which is the
one failure this module exists to prevent. So the table only ever shrinks through that one exit, and
its size is bounded by the number of sessions that entered the route and never left; a growth warning
is logged instead of evicting.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Callable, Dict

try:  # inside the plugin loader's package context
    from .light_web import ALLOWED_TOOL_NAME_SET
except ImportError:  # imported as a top-level module (the plugin dir is on sys.path)
    from light_web import ALLOWED_TOOL_NAME_SET

logger = logging.getLogger("hermes_cli.plugins.llm-cr-prompt")

# Not a cap: nothing is evicted (see the module docstring). Crossing it means sessions are entering
# the route without ever leaving it, which is worth one log line, not a silent loss of enforcement.
_GROWTH_WARN_AT = 256

_lock = threading.Lock()
_light_sessions: Dict[str, bool] = {}
_warned_growth = False

REFUSAL = (
    "This tool is not available in llm-cr chat mode. Only web_search and web_extract can run here; "
    "answer from the conversation or use those two."
)


def set_session_light(session_id: Any, light: bool) -> None:
    """Record (or clear) the lightweight-route verdict for ``session_id``."""
    key = str(session_id or "")
    if not key:
        return
    global _warned_growth
    with _lock:
        if light:
            _light_sessions[key] = True
            if len(_light_sessions) > _GROWTH_WARN_AT and not _warned_growth:
                _warned_growth = True
                logger.warning(
                    "llm-cr-prompt: %d sessions are still stamped lightweight; stamps are kept "
                    "(never evicted) so their tool restrictions stay in force",
                    len(_light_sessions),
                )
        else:
            _light_sessions.pop(key, None)


def is_session_light(session_id: Any) -> bool:
    key = str(session_id or "")
    if not key:
        return False
    with _lock:
        return bool(_light_sessions.get(key))


def tracked_session_count() -> int:
    with _lock:
        return len(_light_sessions)


def reset() -> None:
    """Drop all stamps (tests; also makes a plugin reload start from a clean slate)."""
    global _warned_growth
    with _lock:
        _light_sessions.clear()
        _warned_growth = False


def refusal_result(tool_name: str) -> str:
    """The tool result a denied call gets: a normal error body, not an exception.

    A refusal has to read as a tool result so the turn continues and the model can answer in words.
    Raising would abort the turn, which is a worse outcome for a chat route than "that tool isn't
    here".
    """
    try:
        from tools.registry import tool_error

        return tool_error(REFUSAL, tool=str(tool_name or ""))
    except Exception:  # noqa: BLE001 - registry optional; the body is the same shape either way
        return json.dumps({"error": REFUSAL, "tool": str(tool_name or "")}, ensure_ascii=False)


def make_pre_tool_call_hook(_ctx: Any) -> Callable[..., Any]:
    """Build the ``pre_tool_call`` callback: the same allowlist, one seam earlier.

    The execution middleware below is the seam that actually runs a handler, so it is the one that
    *must* hold. This hook is a second, independent layer on the host's generic policy seam — it
    fires before dispatch on every path that consults plugin policy (including the agent loop, which
    fires it itself and then tells the registry not to fire it again). Both layers read the same
    stamp table, so they cannot disagree about what is allowed.
    """

    def llm_cr_pre_tool_call(tool_name=None, args=None, **context):
        if not is_session_light(context.get("session_id")):
            return None
        name = tool_name if isinstance(tool_name, str) else ""
        if name in ALLOWED_TOOL_NAME_SET:
            return None
        return {"action": "block", "message": REFUSAL}

    return llm_cr_pre_tool_call


def make_middleware(_ctx: Any) -> Callable[..., Any]:
    """Build the ``tool_execution`` callback. Deny-by-default on a stamped session."""

    def llm_cr_tool_execution(tool_name=None, args=None, next_call=None, **context):
        session_id = context.get("session_id")
        if not is_session_light(session_id):
            return next_call(args)
        name = tool_name if isinstance(tool_name, str) else ""
        if name in ALLOWED_TOOL_NAME_SET:
            return next_call(args)
        # Never reaches ``next_call``: the handler is not dispatched at all.
        logger.info("llm-cr-prompt: denied tool %r on lightweight session", name or "<unnamed>")
        return refusal_result(name)

    return llm_cr_tool_execution
