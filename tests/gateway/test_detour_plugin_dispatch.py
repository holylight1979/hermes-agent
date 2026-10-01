"""``/detour`` as a plugin command on the gateway's own dispatch paths.

Where ``test_detour_commands.py`` drives the feature's surface leg directly, this file drives the
GATEWAY: ``_hm_dispatch_quick_and_plugin_commands`` (the cold path), ``_hm_busy_slash_or_photo``
(the mid-run fast path) and ``_hm_unknown_slash_reply`` (what a user sees when the feature is off).
The plugin is really installed into a temp home and really discovered, so the handler these paths
find is the one a production discovery pass would find.

The guards asserted here are the ones that stopped being built-in guards when ``/detour`` stopped
being a built-in: the slash-access gate, the busy rejection, and the unknown-command notice that
keeps a disabled feature from reaching the LLM as free text. Each "refused" case also asserts that
nothing happened — no new session row, no return record — because a guard that refuses after the
session has already rotated is not a guard.
"""

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionSource
from tests.fakes.session_detours_plugin import STATUS_ACTIVE, read_record
from tests.gateway.test_detour_commands import (  # noqa: F401  (fixture import)
    _event,
    _routed,
    _scope,
    _seed_parent,
    _session_ids,
    _source,
    runner,
)


@pytest.fixture
def installed_plugin(runner):  # noqa: F811
    """The ``session-detours`` plugin installed into the runner's temp home and discovered."""
    from tests.fakes import session_detours_plugin as sdp

    manager = sdp.install_into_home(runner.home)
    yield manager
    sdp.reset_discovery()


@pytest.fixture
def disabled_plugin(runner):  # noqa: F811
    """The plugin on disk but not consented — the "feature off" state."""
    from tests.fakes import session_detours_plugin as sdp

    manager = sdp.install_into_home(runner.home, enabled=False)
    yield manager
    sdp.reset_discovery()


def _gate_slash_commands(runner, *, admin_ids=("admin",), user_allowed=()):  # noqa: F811
    """Turn on per-platform slash gating for this runner's platform (DM scope)."""
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(
        enabled=True, token="***",
        extra={"allow_admin_from": list(admin_ids),
               "user_allowed_commands": list(user_allowed)})})


async def _dispatch(runner, event):  # noqa: F811
    """The gateway's real cold-path plugin dispatch → ``(handled, result)``."""
    handled, result, _command = await runner._hm_dispatch_quick_and_plugin_commands(
        event, event.source, event.get_command())
    return handled, result


# ------------------------------------------------------------------- the cold path, end to end
@pytest.mark.asyncio
async def test_the_gateway_dispatches_detour_to_the_plugin_and_the_lane_rotates(
    runner, installed_plugin,  # noqa: F811
):
    parent = await _seed_parent(runner)

    handled, result = await _dispatch(runner, _event("/detour side-model --provider side-provider"))

    assert handled is True
    assert result and "❌" not in result
    child = _routed(runner)
    assert child and child != parent
    record = read_record(_scope(runner))
    assert (record.status, record.parent_session_id, record.child_session_id) == (
        STATUS_ACTIVE, parent, child)
    assert runner.applied_routes == ["/model side-model --provider side-provider --session"]


@pytest.mark.asyncio
async def test_the_underscored_autocomplete_form_reaches_the_same_handler(
    runner, installed_plugin,  # noqa: F811
):
    """Telegram offers ``/detour_end``; it must resolve to the hyphenated registration."""
    parent = await _seed_parent(runner)
    await _dispatch(runner, _event("/detour"))
    assert _routed(runner) != parent

    handled, result = await _dispatch(runner, _event("/detour_end"))

    assert handled is True
    assert result and "❌" not in result
    assert _routed(runner) == parent


@pytest.mark.asyncio
async def test_the_handler_receives_this_gateway_and_this_lane_as_its_context(
    runner, installed_plugin,  # noqa: F811
):
    """The context the surface binds is the live host, the routed session key and the event."""
    from hermes_cli.plugins import bind_plugin_command_context, get_plugin_command_handler

    seen = {}

    def _probe(raw_args="", *, context=None):
        seen.update(context or {})
        return "probed"

    assert get_plugin_command_handler("detour") is not None  # the real registration exists
    source = _source()
    event = _event("/detour x", source)
    bound = bind_plugin_command_context(_probe, {
        "surface": "gateway", "command": "detour", "host": runner,
        "session_key": runner._plugin_command_session_key(source),
        "event": event, "source": source,
    })
    assert bound("x") == "probed"
    assert seen["host"] is runner and seen["event"] is event and seen["surface"] == "gateway"
    assert seen["session_key"] == runner._session_key_for_source(
        runner._normalize_source_for_session_key(source))


# --------------------------------------------------------------------------- the access gate
@pytest.mark.asyncio
async def test_a_non_admin_is_refused_before_the_lane_moves(runner, installed_plugin):  # noqa: F811
    parent = await _seed_parent(runner)
    _gate_slash_commands(runner)
    before = _session_ids(runner)

    handled, result = await _dispatch(runner, _event("/detour side-model", _source(user="nobody")))

    assert handled is True
    assert result and result.startswith("⛔")
    # Refused BEFORE any effect: same routed session, no new session row, no record at all.
    assert _routed(runner) == parent
    assert _session_ids(runner) == before
    assert read_record(_scope(runner, _source(user="nobody"))) is None
    assert runner.applied_routes == []


@pytest.mark.asyncio
async def test_the_underscored_form_is_gated_too(runner, installed_plugin):  # noqa: F811
    """The cold-path gate runs on the typed name, so the normalized name is re-checked here."""
    parent = await _seed_parent(runner)
    _gate_slash_commands(runner)

    handled, result = await _dispatch(runner, _event("/detour_end", _source(user="nobody")))

    assert handled is True and result.startswith("⛔")
    assert _routed(runner) == parent


@pytest.mark.asyncio
async def test_an_admin_passes_the_gate(runner, installed_plugin):  # noqa: F811
    parent = await _seed_parent(runner, source=_source(user="admin"))
    _gate_slash_commands(runner)

    handled, result = await _dispatch(runner, _event("/detour", _source(user="admin")))

    assert handled is True and "⛔" not in (result or "")
    assert _routed(runner, _source(user="admin")) != parent


@pytest.mark.asyncio
async def test_an_explicitly_allowed_non_admin_passes_the_gate(runner, installed_plugin):  # noqa: F811
    parent = await _seed_parent(runner, source=_source(user="helper"))
    _gate_slash_commands(runner, user_allowed=("detour",))

    handled, result = await _dispatch(runner, _event("/detour", _source(user="helper")))

    assert handled is True and "⛔" not in (result or "")
    assert _routed(runner, _source(user="helper")) != parent


# ------------------------------------------------------------------------- the busy fast path
@pytest.mark.asyncio
async def test_a_detour_is_refused_mid_run_instead_of_interrupting_the_turn(
    runner, installed_plugin,  # noqa: F811
):
    parent = await _seed_parent(runner)
    before = _session_ids(runner)
    key = runner._session_key_for_source(_source())

    handled, result = await runner._hm_busy_slash_or_photo(_event("/detour side-model"), _source(), key)

    assert handled is True
    assert "can't run" in result and "/detour" in result
    # Nothing ran: the lane is where it was, no session was created, no record was opened.
    assert _routed(runner) == parent
    assert _session_ids(runner) == before
    assert read_record(_scope(runner)) is None
    assert runner.applied_routes == []


@pytest.mark.asyncio
async def test_the_busy_refusal_covers_the_return_leg_and_its_underscored_form(
    runner, installed_plugin,  # noqa: F811
):
    await _seed_parent(runner)
    key = runner._session_key_for_source(_source())

    for text in ("/detour-end", "/detour_end"):
        handled, result = await runner._hm_busy_slash_or_photo(_event(text), _source(), key)
        assert handled is True, text
        assert "can't run" in result, text


@pytest.mark.asyncio
async def test_the_busy_path_checks_access_before_it_even_explains_the_refusal(
    runner, installed_plugin,  # noqa: F811
):
    """A non-admin gets the denial, not the busy notice — the gate is not bypassable mid-run."""
    await _seed_parent(runner)
    _gate_slash_commands(runner)
    source = _source(user="nobody")
    key = runner._session_key_for_source(source)

    handled, result = await runner._hm_busy_slash_or_photo(_event("/detour"), source, key)

    assert handled is True and result.startswith("⛔")


@pytest.mark.asyncio
async def test_a_command_the_plugin_did_not_declare_is_left_to_the_normal_busy_path(
    runner, installed_plugin,  # noqa: F811
):
    """The busy fast path must only claim commands that really declared ``reject``."""
    await _seed_parent(runner)
    key = runner._session_key_for_source(_source())

    handled, result = await runner._hm_busy_slash_or_photo(_event("/not-a-command"), _source(), key)

    assert (handled, result) == (False, None)


# ------------------------------------------------------------------ the feature switched off
@pytest.mark.asyncio
async def test_a_disabled_plugin_means_no_detour_dispatch_at_all(runner, disabled_plugin):  # noqa: F811
    parent = await _seed_parent(runner)
    before = _session_ids(runner)

    handled, result = await _dispatch(runner, _event("/detour side-model"))

    assert (handled, result) == (False, None)
    assert _routed(runner) == parent
    assert _session_ids(runner) == before
    assert read_record(_scope(runner)) is None


@pytest.mark.asyncio
async def test_a_disabled_plugin_also_leaves_the_busy_fast_path_alone(runner, disabled_plugin):  # noqa: F811
    await _seed_parent(runner)
    key = runner._session_key_for_source(_source())

    handled, result = await runner._hm_busy_slash_or_photo(_event("/detour"), _source(), key)

    assert (handled, result) == (False, None)


@pytest.mark.asyncio
async def test_a_bare_alias_for_a_disabled_detour_fails_loudly_and_never_reaches_the_llm(
    runner, disabled_plugin,  # noqa: F811
):
    """``llm-cr`` rewrites to ``/detour``; with the feature off that must be an error, not a prompt.

    This is the case that would be worst to get wrong: a slash command nobody handles, forwarded
    to the model as ordinary text, becomes an invented answer about a detour that never happened.
    """
    for command in ("detour", "detour-end"):
        reply = runner._hm_unknown_slash_reply(command, _source())
        assert reply is not None, command
        assert f"Unknown command `/{command}`" in reply


@pytest.mark.asyncio
async def test_an_enabled_detour_never_reaches_that_notice(runner, installed_plugin):  # noqa: F811
    """With the plugin on, plugin dispatch claims the command first, so the notice is never asked.

    ``_hm_unknown_slash_reply`` itself only knows the built-in registry — it would still call an
    enabled ``/detour`` unknown if it were consulted. It isn't: this is the ordering the enabled
    feature depends on, so assert the ordering rather than the stale lookup.
    """
    await _seed_parent(runner)

    handled, _result = await _dispatch(runner, _event("/detour"))

    assert handled is True


# ------------------------------------------------------------ the rest of the gateway is intact
@pytest.mark.asyncio
async def test_the_native_session_commands_are_still_the_gateways_own(runner, installed_plugin):  # noqa: F811
    """The plugin composes native handlers; it must not have replaced or wrapped any of them."""
    from gateway.run import GatewayRunner

    for name in ("_handle_reset_command", "_handle_resume_command", "_handle_model_command",
                 "reset_session", "switch_session"):
        if name == "_handle_model_command":
            continue  # the fixture's own offline stand-in
        assert getattr(type(runner), name, None) is getattr(GatewayRunner, name, None), name
    # /new and /resume are still built-ins with their own registry entries.
    from hermes_cli.commands import resolve_command
    assert resolve_command("new") is not None and resolve_command("resume") is not None
    # ...and /detour is not one, on either surface.
    assert resolve_command("detour") is None and resolve_command("detour-end") is None


@pytest.mark.asyncio
async def test_a_plugin_command_without_a_context_parameter_still_dispatches(runner):  # noqa: F811
    """Backwards compatibility on the real gateway path, not just at the binder."""
    from hermes_cli import plugins as plugins_mod

    plugin_dir = runner.home / "plugins" / "legacy-plugin"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        'name: legacy-plugin\nversion: "1.0.0"\ndescription: "legacy"\n', encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(
        'def _handler(raw_args=""):\n'
        '    return "legacy ran with " + repr(raw_args)\n\n\n'
        'def register(ctx):\n'
        '    ctx.register_command("legacy-cmd", _handler, description="legacy")\n',
        encoding="utf-8")
    (runner.home / "config.yaml").write_text(
        "plugins:\n  enabled:\n    - legacy-plugin\n  disabled: []\n", encoding="utf-8")
    plugins_mod._reset_plugin_managers_for_tests()
    plugins_mod._ensure_plugins_discovered(force=True)
    try:
        handled, result = await _dispatch(runner, _event("/legacy-cmd hello there"))
    finally:
        plugins_mod._reset_plugin_managers_for_tests()

    assert (handled, result) == (True, "legacy ran with 'hello there'")
