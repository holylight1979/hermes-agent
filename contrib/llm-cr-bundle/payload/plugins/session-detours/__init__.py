"""``/detour`` and ``/detour-end`` as a plugin: leave for a fresh side session, then come back.

This plugin owns the whole feature — the return-record store, the CLI leg and the gateway leg.
Core keeps only the generic seam that makes a plugin slash command able to compose native session
behavior at all: ``PluginContext.register_command`` handlers that declare a ``context`` parameter
receive the dispatching surface's host (see ``hermes_cli.plugins.bind_plugin_command_context``).

Both legs are compositions of the NATIVE session commands, never reimplementations: ``/detour``
runs the surface's own ``/new`` path and then its own ``/model ... --session``; ``/detour-end``
runs the surface's own ``/resume`` against the recorded parent id and then restores the parent's
route. Nothing copies a transcript in either direction.

How the host reaches the surface code without touching core classes:

The two surface modules are still written as mixins against the host (``self.new_session``,
``self._handle_resume_command``, ``self._handle_reset_command``, ``self.session_store``, ...) —
that is what keeps them compositions of native commands rather than copies. Nothing is patched
onto the host class. Instead each surface gets a per-host ADAPTER object that subclasses the mixin
and delegates every attribute it does not define itself to the live host. So:

* the mixin's own methods resolve on the adapter (defined on its class),
* everything else — session db, session store, native command handlers, console, model switch —
  reads through ``__getattr__`` to the real CLI / GatewayRunner,
* the mixin's single piece of per-process state (``_detour_open_scope``) lives on the adapter,
  which is cached per host so it survives between the enter and the return leg.

``busy_policy="reject"``: a detour rotates the session, so like the built-in commands that do
(``/model``, ``/resume``) it refuses mid-turn instead of interrupting the running agent.

Disabled plugin ⇒ no ``/detour`` at all: the commands are gone from completion and from gateway
dispatch, the ``llm-cr`` text alias resolves to a command nothing handles, and every native
command handler is untouched.
"""

from __future__ import annotations

import functools
import logging
import weakref
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: host -> adapter, keyed and valued weakly in both directions: the key is weak so a finished
#: CLI/gateway host is not pinned by this cache, and the adapter holds only a weakref back to its
#: host so the value does not resurrect the key (the classic WeakKeyDictionary strong-cycle leak).
#: A long-lived process that creates many disposable hosts therefore retains none of them.
_CLI_ADAPTERS: "weakref.WeakKeyDictionary[Any, Any]" = weakref.WeakKeyDictionary()
_GATEWAY_ADAPTERS: "weakref.WeakKeyDictionary[Any, Any]" = weakref.WeakKeyDictionary()


class _HostDelegatingAdapter:
    """Bind a surface mixin to a live host without touching the host's class.

    Attribute reads that the mixin does not define itself fall through to the host, so the mixin
    keeps calling native session commands exactly as it did when it was mixed into the host class.
    Writes land on the adapter (the mixin only ever writes its own ``_detour_open_scope``).
    """

    def __init__(self, host: Any) -> None:
        object.__setattr__(self, "_detour_host_ref", weakref.ref(host))

    def _host(self) -> Any:
        host = object.__getattribute__(self, "_detour_host_ref")()
        if host is None:  # only reachable if the host died while a leg was still running
            raise RuntimeError("detour: the session host this command was dispatched from is gone")
        return host

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") or name.startswith("_detour_host"):
            raise AttributeError(name)
        return getattr(_HostDelegatingAdapter._host(self), name)


def _adapter(cache: "weakref.WeakKeyDictionary", host: Any, factory) -> Any:
    """The cached per-host adapter, so ``_detour_open_scope`` survives across the two legs."""
    try:
        existing = cache.get(host)
    except TypeError:  # unhashable host: no cache, so each leg gets a fresh adapter
        return factory(host)
    if existing is not None:
        return existing
    adapter = factory(host)
    try:
        cache[host] = adapter
    except TypeError:  # host does not support weak references; never cache it strongly
        logger.debug("detour: host %r is not weak-referenceable, adapter not cached", type(host))
    return adapter


@functools.lru_cache(maxsize=1)
def _cli_adapter_class() -> type:
    from .cli_surface import CLIDetourMixin

    class _CLIDetourAdapter(_HostDelegatingAdapter, CLIDetourMixin):
        pass

    return _CLIDetourAdapter


@functools.lru_cache(maxsize=1)
def _gateway_adapter_class() -> type:
    from .gateway_surface import GatewayDetourCommandsMixin

    class _GatewayDetourAdapter(_HostDelegatingAdapter, GatewayDetourCommandsMixin):
        pass

    return _GatewayDetourAdapter


def _cli_adapter(host: Any) -> Any:
    return _adapter(_CLI_ADAPTERS, host, _cli_adapter_class())


def _gateway_adapter(host: Any) -> Any:
    return _adapter(_GATEWAY_ADAPTERS, host, _gateway_adapter_class())


def _refusal(command: str, surface: Optional[str]) -> str:
    return (f"❌ /{command} is not available on this surface "
            f"({surface or 'unknown'}): it needs the session host.")


def _dispatch(command: str, raw_args: str, context: Optional[dict]):
    """Route ``/detour`` / ``/detour-end`` to the surface leg that owns the live session."""
    ctx = context if isinstance(context, dict) else {}
    surface, host = ctx.get("surface"), ctx.get("host")
    if host is None:
        # Fail closed. A surface that cannot hand us a session host cannot start or end a detour,
        # and guessing one (the "current" CLI, "some" gateway) would rotate the wrong session.
        return _refusal(command, surface)
    if surface == "cli":
        adapter = _cli_adapter(host)
        handler = (adapter._handle_detour_command if command == "detour"
                   else adapter._handle_detour_end_command)
        # The CLI leg parses its own route arguments off the full command line.
        handler(f"/{command} {raw_args}".strip())
        return None  # the CLI leg prints its own program-generated output
    if surface == "gateway":
        event = ctx.get("event")
        if event is None:
            return _refusal(command, surface)
        adapter = _gateway_adapter(host)
        handler = (adapter._handle_detour_command if command == "detour"
                   else adapter._handle_detour_end_command)
        return handler(event)  # awaitable; the gateway resolves it
    return _refusal(command, surface)


def _detour_command(raw_args: str = "", *, context: Optional[dict] = None):
    """``/detour [model] [--provider name]`` — fresh side session, remembering the way back."""
    return _dispatch("detour", raw_args, context)


def _detour_end_command(raw_args: str = "", *, context: Optional[dict] = None):
    """``/detour-end [model] [--provider name]`` — restore the session the detour left."""
    return _dispatch("detour-end", raw_args, context)


def register(ctx) -> None:
    ctx.register_command(
        "detour", _detour_command,
        description="Leave for a fresh side session (this one is kept to return to)",
        args_hint="[model] [--provider name]", argument_mode="mixed", busy_policy="reject",
    )
    ctx.register_command(
        "detour-end", _detour_end_command,
        description="Return to the session the detour left, restoring its history",
        args_hint="[model] [--provider name]", argument_mode="mixed", busy_policy="reject",
    )
