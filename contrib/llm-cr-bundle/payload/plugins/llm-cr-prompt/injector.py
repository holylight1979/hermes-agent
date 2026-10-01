"""Load the local crack-talk instruction file and inject it into the outbound system message.

Route-as-identity: there is no second "mode" flag or dispatcher. The ONLY question asked is "is
this very request going to the configured provider + model + base URL in the configured api_mode?".
The ``llm-cr`` / ``llm-cr-end`` text aliases already switch the native route (``/model ... --provider
... --session``), so the effective route IS the mode. A consequence worth knowing: selecting that
same route by hand (``/model`` with the same three values) gets the same persona — by design.

The route is checked twice, against two different sources: the turn's own context, and then the
outbound payload's ``model`` — an upstream request middleware may have rewritten the latter after
the former was built, and only the payload says where the text would really land.

Fail-closed: a matched route whose payload model disagrees, or whose instruction file is missing,
empty, unreadable or not UTF-8, aborts the request (``MiddlewareAbort``) *before* any network call.
The alternative — letting the call through — would quietly send an un-prompted request to the
crack-talk model, which is a different action, not a degraded one.

Nothing here logs, formats or re-raises instruction text: failures carry a fixed reason code only.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlsplit

from hermes_cli.middleware import MiddlewareAbort

logger = logging.getLogger("hermes_cli.plugins.llm-cr-prompt")

# Relative to a skills root; the absolute default is derived at call time (profile-safe).
_INSTRUCTION_RELPATH = Path("skills") / "productivity" / "llm-crack-talk" / "llm-cr-instruction.md"

_DEFAULT_API_MODE = "chat_completions"


@dataclass(frozen=True)
class RouteSettings:
    """``plugins.entries.llm-cr-prompt.settings`` as the gate needs it."""

    enabled: bool
    provider: str
    model: str
    base_url: str
    api_mode: str
    instruction_path: Optional[Path]

    @property
    def configured(self) -> bool:
        """A gate with an unset provider/model/base URL can never be *proven* to match, so it
        never injects (rather than guessing an endpoint for the instruction)."""
        return bool(self.enabled and self.provider and self.model and self.base_url)


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _normalize_url(value: Any) -> str:
    """Compare base URLs tolerating a trailing slash and scheme/host casing — and ONLY those.

    The path stays byte-exact. Folding it would make ``/V1`` match a gate configured for ``/v1``,
    i.e. let a *different* endpoint on the same host pull in the instruction; paths are
    case-sensitive to the server, so they are case-sensitive here too.
    """
    text = _text(value).rstrip("/")
    parts = urlsplit(text)
    if not parts.scheme or not parts.netloc:
        # Not an absolute ``scheme://host`` URL: no component is safely case-insensitive.
        return text
    # ``urlsplit`` only reports a netloc when the text literally reads ``scheme://netloc``, so this
    # slice is the untouched remainder (path + query + fragment).
    remainder = text[len(parts.scheme) + 3 + len(parts.netloc):]
    return f"{parts.scheme.casefold()}://{parts.netloc.casefold()}{remainder}"


def resolve_settings(ctx: Any) -> RouteSettings:
    """Read this plugin's settings live, so a config edit needs no reload of the plugin."""
    raw_path = _text(ctx.get_config("instruction_path", ""))
    return RouteSettings(
        enabled=ctx.get_config("enabled", True) is not False,
        provider=_text(ctx.get_config("route_provider", "")).casefold(),
        model=_text(ctx.get_config("route_model", "")),
        base_url=_normalize_url(ctx.get_config("route_base_url", "")),
        api_mode=_text(ctx.get_config("route_api_mode", _DEFAULT_API_MODE)).casefold() or _DEFAULT_API_MODE,
        instruction_path=Path(raw_path) if raw_path else None,
    )


def default_instruction_path() -> Path:
    """The instruction file of the ACTIVE Hermes home — the single candidate, existing or not.

    Deliberately no second candidate. A checkout-relative (or any other-profile) fallback would
    mean that a profile whose own prompt is missing silently sends *another* profile's — or the
    repository's — instruction to the model. Returning the one path lets the caller's read fail
    closed instead. No user-specific absolute path is baked into this module.
    """
    import hermes_constants

    try:
        home = Path(hermes_constants.get_hermes_home())
    except Exception as exc:
        raise MiddlewareAbort(
            f"llm-cr-prompt: instruction file unavailable (home unresolved: {type(exc).__name__})"
        ) from None
    return home / _INSTRUCTION_RELPATH


def route_matches(settings: RouteSettings, *, provider: Any, model: Any, base_url: Any, api_mode: Any) -> bool:
    """True only for the exact configured native route of this very request."""
    if not settings.configured:
        return False
    return (
        _text(provider).casefold() == settings.provider
        and _text(model) == settings.model
        and _normalize_url(base_url) == settings.base_url
        and (_text(api_mode).casefold() or _DEFAULT_API_MODE) == settings.api_mode
    )


def load_instruction(path: Path) -> str:
    """Return the instruction text (UTF-8, BOM tolerated) or raise a content-free abort."""
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        raise MiddlewareAbort("llm-cr-prompt: instruction file unavailable (missing)") from None
    except OSError:
        raise MiddlewareAbort("llm-cr-prompt: instruction file unavailable (unreadable)") from None
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise MiddlewareAbort("llm-cr-prompt: instruction file unavailable (not utf-8)") from None
    if not text.strip():
        raise MiddlewareAbort("llm-cr-prompt: instruction file unavailable (empty)")
    return text


def inject_instruction(request: Dict[str, Any], text: str) -> Dict[str, Any]:
    """Return a COPY of ``request`` whose first system message carries ``text`` appended.

    The input mapping, its message objects and the caller's stored history are never touched: only
    the messages list and the one system message being extended are copied. Because the copy is
    discarded with the attempt, a retry or a tool round re-injects exactly once from the pristine
    original — the text can never accumulate.
    """
    messages = request.get("messages")
    if not isinstance(messages, list):
        raise MiddlewareAbort("llm-cr-prompt: cannot inject (request carries no messages list)")

    new_messages = list(messages)
    index = next(
        (i for i, m in enumerate(new_messages) if isinstance(m, dict) and m.get("role") == "system"),
        None,
    )
    if index is None:
        new_messages.insert(0, {"role": "system", "content": text})
    else:
        message = dict(new_messages[index])
        content = message.get("content")
        if content is None or isinstance(content, str):
            base = content or ""
            message["content"] = f"{base}\n\n{text}" if base.strip() else text
        elif isinstance(content, list):
            blocks = deepcopy(content)
            blocks.append({"type": "text", "text": text})
            message["content"] = blocks
        else:
            raise MiddlewareAbort("llm-cr-prompt: cannot inject (unsupported system content type)")
        new_messages[index] = message

    injected = dict(request)
    injected["messages"] = new_messages
    return injected


def make_middleware(ctx: Any) -> Callable[..., Any]:
    """Build the ``llm_execution`` callback bound to this plugin's settings reader."""

    def llm_cr_prompt_execution(request=None, next_call=None, **context):
        settings = resolve_settings(ctx)
        if not route_matches(
            settings,
            provider=context.get("provider"),
            model=context.get("model"),
            base_url=context.get("base_url"),
            api_mode=context.get("api_mode"),
        ):
            # Not the crack-talk route: the instruction file is not even opened.
            return next_call(request)

        # The context above describes the route the turn *selected*; the request mapping describes
        # where this payload is actually going. An upstream request middleware can rewrite
        # ``request["model"]`` after that context was built, so the context alone cannot authorise a
        # read: the payload itself has to name the crack-talk model. A mismatch (or no model at all,
        # including a non-mapping request) fails closed here — before the file is opened and before
        # ``next_call`` — so neither the instruction nor a reason to go looking for it escapes.
        if (_text(request.get("model")) if isinstance(request, dict) else "") != settings.model:
            raise MiddlewareAbort(
                "llm-cr-prompt: request payload model does not match the configured route"
            )

        path = settings.instruction_path or default_instruction_path()
        try:
            text = load_instruction(path)
            injected = inject_instruction(request, text)
        except MiddlewareAbort:
            raise
        except Exception as exc:
            # Any unexpected failure on a MATCHED route must still fail closed; the message names
            # the exception type only, never file content.
            raise MiddlewareAbort(
                f"llm-cr-prompt: instruction injection failed ({type(exc).__name__})"
            ) from None
        logger.info("llm-cr-prompt: injected instruction for session %s", context.get("session_id") or "-")
        return next_call(injected)

    return llm_cr_prompt_execution
