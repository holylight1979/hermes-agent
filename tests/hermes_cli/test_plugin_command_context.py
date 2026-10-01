"""The generic core seam a plugin slash command needs to compose native session behavior.

Two additive pieces of ``hermes_cli.plugins`` are under test, both feature-agnostic:

* :func:`bind_plugin_command_context` — a handler that declares a ``context`` parameter receives
  the dispatching surface's host; every handler that does not is called exactly as before.
* ``register_command(busy_policy=...)`` / :func:`get_plugin_command_busy_policy` — how a plugin
  declares that its command must not run while the session's agent is running.

Backwards compatibility is the point of most of this file: the pre-existing one-argument plugin
command contract (``fn(raw_args) -> str | None``) must be untouched, because every plugin written
before the seam existed uses it.
"""

from __future__ import annotations

import pytest

from hermes_cli.plugins import (
    VALID_PLUGIN_COMMAND_BUSY_POLICIES,
    bind_plugin_command_context,
    get_plugin_command_busy_policy,
    get_plugin_command_handler,
)

CONTEXT = {"surface": "cli", "command": "demo", "host": object(), "session_key": "s1"}


# --------------------------------------------------------------- context binding (compatibility)
def test_an_old_one_argument_handler_is_returned_untouched():
    """The historical contract: no ``context`` parameter means no behavior change at all."""
    calls = []

    def old_handler(raw_args: str):
        calls.append(raw_args)
        return f"ran:{raw_args}"

    bound = bind_plugin_command_context(old_handler, CONTEXT)

    # Not merely equivalent — the SAME object, so nothing about the call can have changed.
    assert bound is old_handler
    assert bound("a b") == "ran:a b"
    assert calls == ["a b"]


def test_a_handler_that_declares_context_receives_it_as_a_keyword():
    seen = {}

    def new_handler(raw_args: str = "", *, context=None):
        seen["raw_args"], seen["context"] = raw_args, context
        return "ok"

    assert bind_plugin_command_context(new_handler, CONTEXT)("x --y") == "ok"
    assert seen == {"raw_args": "x --y", "context": CONTEXT}


def test_a_handler_with_no_arguments_at_all_still_works():
    """``register_command`` has always accepted a zero-arg handler; binding must not break it."""
    def bare_handler():
        return "bare"

    bound = bind_plugin_command_context(bare_handler, CONTEXT)
    assert bound is bare_handler
    assert bound() == "bare"


def test_a_handler_taking_kwargs_but_not_context_is_not_given_one():
    """``**kwargs`` is not a declaration of interest: only a named ``context`` parameter is."""
    seen = {}

    def kwargs_handler(raw_args: str = "", **kwargs):
        seen.update(kwargs)
        return "ok"

    bound = bind_plugin_command_context(kwargs_handler, CONTEXT)
    assert bound is kwargs_handler
    assert bound("x") == "ok"
    assert seen == {}


def test_a_handler_whose_signature_cannot_be_read_is_returned_untouched():
    """C callables have no introspectable signature; they must not be treated as context-takers."""
    assert bind_plugin_command_context(len, CONTEXT) is len
    assert bind_plugin_command_context(str.strip, CONTEXT) is str.strip


@pytest.mark.asyncio
async def test_an_async_context_handler_keeps_returning_its_coroutine():
    """The gateway awaits what the handler returns; binding must stay transparent to that."""
    async def async_handler(raw_args: str = "", *, context=None):
        return f"{context['surface']}:{raw_args}"

    result = bind_plugin_command_context(async_handler, CONTEXT)("z")
    assert await result == "cli:z"


def test_a_bound_handler_is_callable_with_no_arguments():
    """Surfaces that dispatch a parameterless command call ``handler()``; the default must hold."""
    def new_handler(raw_args: str = "", *, context=None):
        return f"[{raw_args}]{context['command']}"

    assert bind_plugin_command_context(new_handler, CONTEXT)() == "[]demo"


# ------------------------------------------------------------------------------- the busy policy
def test_only_reject_is_an_accepted_plugin_busy_policy():
    """A plugin command has no mid-run handler table, so "dispatch" could not mean anything."""
    assert VALID_PLUGIN_COMMAND_BUSY_POLICIES == {"reject"}


@pytest.mark.parametrize("declared,expected", [
    ("reject", "reject"),
    (None, None),
    ("dispatch", None),              # not meaningful for a plugin command → not recorded
    ("interrupt_then_dispatch", None),
    ("", None),
    ("REJECT", None),                # exact token only; no silent case-folding
])
def test_the_declared_busy_policy_is_recorded_only_when_it_is_valid(
    tmp_path, monkeypatch, declared, expected,
):
    """Registration goes through a real discovery pass over a real plugin on disk."""
    _install_demo_plugin(tmp_path, monkeypatch, busy_policy=declared)

    assert get_plugin_command_handler("demo-cmd") is not None
    assert get_plugin_command_busy_policy("demo-cmd") == expected


def test_an_unknown_command_declares_no_busy_policy(tmp_path, monkeypatch):
    _install_demo_plugin(tmp_path, monkeypatch, busy_policy="reject")

    assert get_plugin_command_busy_policy("no-such-command") is None


def test_the_session_detours_plugin_declares_reject_for_both_legs(tmp_path, monkeypatch):
    """The feature's own declaration, read back through the public core accessor."""
    from tests.fakes import session_detours_plugin as sdp

    home = tmp_path / ".hermes"
    (home / "sessions").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    sdp.install_into_home(home)
    try:
        assert get_plugin_command_busy_policy("detour") == "reject"
        assert get_plugin_command_busy_policy("detour-end") == "reject"
    finally:
        sdp.reset_discovery()


# ------------------------------------------------------------------------------------- the glue
PLUGIN_SOURCE = '''
def _handler(raw_args="", *, context=None):
    return "demo:" + raw_args + ":" + str((context or {}).get("surface"))


def register(ctx):
    ctx.register_command("demo-cmd", _handler, description="demo"@BUSY@)
'''


def _install_demo_plugin(tmp_path, monkeypatch, *, busy_policy):
    """A real plugin on disk, consented in config.yaml, discovered by the real PluginManager."""
    from hermes_cli import plugins as plugins_mod

    home = tmp_path / ".hermes"
    plugin_dir = home / "plugins" / "demo-plugin"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        'name: demo-plugin\nversion: "1.0.0"\ndescription: "demo"\n', encoding="utf-8")
    busy_kwarg = "" if busy_policy is None else f", busy_policy={busy_policy!r}"
    (plugin_dir / "__init__.py").write_text(
        PLUGIN_SOURCE.replace("@BUSY@", busy_kwarg), encoding="utf-8")
    (home / "config.yaml").write_text(
        "plugins:\n  enabled:\n    - demo-plugin\n  disabled: []\n", encoding="utf-8")

    monkeypatch.setenv("HERMES_HOME", str(home))
    plugins_mod._reset_plugin_managers_for_tests()
    return plugins_mod._ensure_plugins_discovered(force=True)


@pytest.fixture(autouse=True)
def _unpin_discovered_plugins():
    """Plugin discovery is process-global; never let one test's temp home leak into the next."""
    yield
    from hermes_cli import plugins as plugins_mod

    plugins_mod._reset_plugin_managers_for_tests()
