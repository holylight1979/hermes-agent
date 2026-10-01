"""``/detour`` + ``/detour-end`` on the interactive CLI, against a real SessionDB.

The object under test is a real :class:`HermesCLI` whose session plumbing is production code:
``new_session`` (the ``/new`` path — end_session, empty-row pruning, id rotation, config-default
model reset, agent rebind, ``create_session``) and ``_handle_resume_command`` (the ``/resume``
path — flush, end, ``get_resume_conversations``, cwd/yolo/model restore). The session store is a
real ``SessionDB`` in a temp ``HERMES_HOME``, so every assertion below is about what the user
actually ends up with: which session id the CLI is on, which transcript it holds, and what the
durable return record says.

Only three things are stood in for, none of them on the path a detour decides against:

* the agent — a small double with ``_memory_manager = None``, so no memory provider (LLM-bound) is
  reached at the session boundary;
* ``switch_model`` — the config-default reset inside ``new_session`` would otherwise resolve a
  provider and its credentials;
* ``_handle_model_switch`` — the route application. The stand-in still publishes the live
  ``model`` / ``provider`` the detour VERIFIES, so a leg that trusted its own reply text instead of
  runtime state fails here (``test_a_switch_that_reports_success_but_changes_nothing_...``).
"""

from __future__ import annotations

import os
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from hermes_cli.session_detour import (
    STATUS_ACTIVE,
    STATUS_RETURNED,
    DetourScope,
    read_record,
    record_path,
)

CONFIG_MODEL, CONFIG_PROVIDER = "config-default-model", "config-provider"
PARENT_MODEL, PARENT_PROVIDER = "parent-model", "parent-provider"


class _FakeAgent:
    """Enough agent for the session boundary: no memory manager, no provider, no network."""

    def __init__(self, session_id: str, session_start) -> None:
        self.session_id = session_id
        self.session_start = session_start
        self.model = PARENT_MODEL
        self._memory_manager = None
        self._last_flushed_db_idx = 0
        self._session_db_created = True
        self.reasoning_config = {}
        self.switch_model = MagicMock()
        self._invalidate_system_prompt = MagicMock()
        self._flush_messages_to_session_db = MagicMock()

    def reset_session_state(self) -> None:
        pass


class _SwitchResult:
    """What ``hermes_cli.model_switch.switch_model`` returns on success."""

    def __init__(self, model: str, provider: str) -> None:
        self.success = True
        self.new_model, self.target_provider = model, provider
        self.api_key = self.base_url = self.api_mode = ""
        self.runtime_capabilities = None


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """A temp Hermes home for the whole test: records, breadcrumbs and the state DB all land here."""
    home = tmp_path / ".hermes"
    (home / "sessions").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))

    import cli as cli_mod
    import hermes_cli.model_switch as model_switch

    # /new re-derives model+provider from config.yaml. Keep that real, but resolve it offline.
    monkeypatch.setitem(cli_mod.CLI_CONFIG, "model",
                        {"default": CONFIG_MODEL, "provider": CONFIG_PROVIDER})
    monkeypatch.setitem(cli_mod.CLI_CONFIG, "agent", {})
    monkeypatch.setattr(model_switch, "switch_model",
                        lambda **kw: _SwitchResult(kw.get("raw_input") or CONFIG_MODEL,
                                                   kw.get("explicit_provider") or CONFIG_PROVIDER))
    yield home
    # ``new_session`` / ``/resume`` publish the live id process-wide; unpin it for the next test.
    from gateway.session_context import _UNSET, _VAR_MAP

    os.environ.pop("HERMES_SESSION_ID", None)
    _VAR_MAP["HERMES_SESSION_ID"].set(_UNSET)


@pytest.fixture
def db(_isolated_home):
    """The real session store, at the path a bare ``SessionDB()`` also resolves to."""
    from hermes_state import SessionDB

    database = SessionDB(db_path=_isolated_home / "state.db")
    yield database
    database.close()


def _new_session_id(suffix: str) -> str:
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{suffix}"


def _make_cli(db, session_id: str, *, texts=()):
    """A real HermesCLI wired to *db*, sitting on *session_id* with *texts* already persisted."""
    from cli import HermesCLI

    db.create_session(session_id=session_id, source="cli", model=PARENT_MODEL)
    for index, text in enumerate(texts):
        # Alternate roles: the resume projection repairs a durable user;user run, which would
        # otherwise hide which turns really came back.
        db.append_message(session_id, role="user" if index % 2 == 0 else "assistant", content=text)

    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = session_id
    cli.session_start = datetime.now()
    cli._session_db = db
    cli._session_db_unavailable = False
    cli.conversation_history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": t} for i, t in enumerate(texts)]
    cli._resume_display_history = list(cli.conversation_history)
    cli.resume_display = "minimal"
    cli._pending_title = None
    cli._pending_resume_sessions = None
    cli._resumed = False
    cli._explicit_model_override = False
    cli._pending_one_turn_model_restore = None
    cli.reasoning_config = {}
    cli.service_tier = ""
    cli.max_turns = 25
    cli.model, cli.provider = PARENT_MODEL, PARENT_PROVIDER
    cli.requested_provider = PARENT_PROVIDER
    cli.base_url, cli.api_key, cli.api_mode = "", "", ""
    cli._explicit_api_key = cli._explicit_base_url = None
    cli.config = {}
    # ``/resume``'s dim "model restored from session" notice goes through the Rich console.
    cli.console = MagicMock()
    cli.agent = _FakeAgent(session_id, cli.session_start)
    _install_route_stub(cli)
    return cli


def _install_route_stub(cli) -> None:
    """Replace only the provider-touching half of ``/model``: the live route still moves."""
    cli.applied_routes = []
    cli.model_switch_is_broken = False

    def _switch(cmd_original: str):
        cli.applied_routes.append(cmd_original)
        if cli.model_switch_is_broken:
            return  # reports nothing, changes nothing — the live-state check must catch it
        parts = cmd_original.split()[1:]
        if parts and not parts[0].startswith("--"):
            cli.model = parts[0]
        if "--provider" in parts:
            cli.provider = parts[parts.index("--provider") + 1]

    cli._handle_model_switch = _switch


def _scope(parent_session_id: str) -> DetourScope:
    """The lane a CLI detour from *parent_session_id* belongs to, recomputed the way production does
    (``_os_user`` degrades to ``""`` when the environment names no user — as it does under the
    hermetic test env — which is a scope component, not an identity check)."""
    from hermes_cli.cli_detour_mixin import _os_user

    return DetourScope(surface="cli", channel="local", owner=_os_user(), lane=parent_session_id)


def _texts(cli) -> list:
    return [m.get("content") for m in cli.conversation_history]


def _session_ids(home) -> set:
    """Every session row in the temp DB — proves a refused command created nothing."""
    import sqlite3

    with sqlite3.connect(home / "state.db") as conn:
        return {row[0] for row in conn.execute("SELECT id FROM sessions")}


def _db_texts(db, session_id: str) -> list:
    model_history, _display = db.get_resume_conversations(session_id)
    return [m.get("content") for m in model_history if m.get("role") in ("user", "assistant")]


# --------------------------------------------------------------------------- entry
def test_detour_rotates_to_a_fresh_child_session_and_records_the_way_back(db):
    parent = _new_session_id("par001")
    cli = _make_cli(db, parent, texts=("hello parent",))

    cli._handle_detour_command("/detour child-model --provider child-provider")

    child = cli.session_id
    assert child and child != parent
    # Genuinely fresh: its own row in the store, and nothing carried in.
    assert db.get_session(child) is not None
    assert cli.conversation_history == []
    assert _db_texts(db, child) == []
    # The record names both ends and is active.
    record = read_record(_scope(parent))
    assert record.status == STATUS_ACTIVE
    assert (record.parent_session_id, record.child_session_id) == (parent, child)
    assert record.parent_route == {"model": PARENT_MODEL, "provider": PARENT_PROVIDER}
    # The route was applied session-scoped — never globally, never once.
    assert cli.applied_routes == ["/model child-model --provider child-provider --session"]
    assert (cli.model, cli.provider) == ("child-model", "child-provider")


def test_the_parent_session_is_preserved_with_its_own_transcript(db):
    parent = _new_session_id("par002")
    cli = _make_cli(db, parent, texts=("hello parent", "parent reply"))

    cli._handle_detour_command("/detour child-model")

    row = db.get_session(parent)
    assert row is not None  # ended, not deleted
    assert _db_texts(db, parent) == ["hello parent", "parent reply"]


def test_a_bare_detour_switches_no_model_at_all(db):
    parent = _new_session_id("par003")
    cli = _make_cli(db, parent, texts=("hello parent",))

    cli._handle_detour_command("/detour")

    assert cli.session_id != parent
    assert cli.applied_routes == []
    # /new still reset the route to the config default; the detour just did not add its own.
    assert cli.model == CONFIG_MODEL
    assert read_record(_scope(parent)).status == STATUS_ACTIVE


def test_detour_is_reachable_through_the_real_slash_dispatch_and_bare_alias(db):
    """Ingress: the configured bare alias resolves to ``/detour ...`` and the CLI dispatches it."""
    from hermes_cli.text_command_aliases import resolve_text_command_alias

    parent = _new_session_id("par004")
    cli = _make_cli(db, parent, texts=("hello parent",))
    cli.config = {"text_command_aliases": {
        "enabled": True,
        "aliases": {"llm-cr": "/detour child-model --provider child-provider",
                    "llm-cr-end": "/detour-end back-model --provider back-provider"}}}

    assert cli._slash_handler("detour") == ("_handle_detour_command", True)
    assert cli._slash_handler("detour-end") == ("_handle_detour_end_command", True)

    command = resolve_text_command_alias("llm-cr", cli.config)
    assert command == "/detour child-model --provider child-provider"
    assert cli.process_command(command) is not False

    child = cli.session_id
    assert child != parent
    assert read_record(_scope(parent)).child_session_id == child
    assert (cli.model, cli.provider) == ("child-model", "child-provider")

    assert cli.process_command(resolve_text_command_alias("llm-cr-end", cli.config)) is not False
    assert cli.session_id == parent
    assert (cli.model, cli.provider) == ("back-model", "back-provider")


# --------------------------------------------------------------------------- exit
def test_detour_end_restores_the_parents_own_history_and_route(db):
    parent = _new_session_id("par005")
    cli = _make_cli(db, parent, texts=("hello parent", "parent reply"))
    cli._handle_detour_command("/detour child-model --provider child-provider")
    child = cli.session_id
    db.append_message(child, role="user", content="child only")
    cli.conversation_history = [{"role": "user", "content": "child only"}]

    cli._handle_detour_end_command("/detour-end")

    assert cli.session_id == parent
    # The actual transcript came back, loaded from the parent's own rows.
    assert _texts(cli) == ["hello parent", "parent reply"]
    assert "child only" not in _texts(cli)
    # ...and the parent's route, not the config default /resume would otherwise leave behind.
    assert (cli.model, cli.provider) == (PARENT_MODEL, PARENT_PROVIDER)
    assert cli.applied_routes[-1] == f"/model {PARENT_MODEL} --provider {PARENT_PROVIDER} --session"
    # The record is consumed only after the restore landed.
    assert read_record(_scope(parent)).status == STATUS_RETURNED
    # The child is archived, not deleted, and keeps its own turn.
    assert db.get_session(child) is not None
    assert _db_texts(db, child) == ["child only"]


def test_detour_end_prefers_an_explicit_route_over_the_recorded_one(db):
    parent = _new_session_id("par006")
    cli = _make_cli(db, parent, texts=("hello parent",))
    cli._handle_detour_command("/detour child-model")

    cli._handle_detour_end_command("/detour-end back-model --provider back-provider")

    assert cli.session_id == parent
    assert _texts(cli) == ["hello parent"]
    assert (cli.model, cli.provider) == ("back-model", "back-provider")


def test_neither_transcript_ever_crosses_the_boundary(db):
    parent = _new_session_id("par007")
    cli = _make_cli(db, parent, texts=("parent secret",))
    cli._handle_detour_command("/detour")
    child = cli.session_id
    db.append_message(child, role="user", content="child secret")
    cli.conversation_history = [{"role": "user", "content": "child secret"}]

    cli._handle_detour_end_command("/detour-end")

    assert _db_texts(db, parent) == ["parent secret"]
    assert _db_texts(db, child) == ["child secret"]
    assert _texts(cli) == ["parent secret"]


def test_a_second_detour_end_is_a_quiet_no_op(db, capsys):
    parent = _new_session_id("par008")
    cli = _make_cli(db, parent, texts=("hello parent",))
    cli._handle_detour_command("/detour")
    cli._handle_detour_end_command("/detour-end")
    capsys.readouterr()

    cli._handle_detour_end_command("/detour-end back-model")

    assert "Nothing to end" in capsys.readouterr().out
    assert cli.session_id == parent
    assert read_record(_scope(parent)).status == STATUS_RETURNED
    assert "/model back-model --session" not in cli.applied_routes


def test_detour_end_without_any_detour_creates_nothing(db, capsys, _isolated_home):
    parent = _new_session_id("par009")
    cli = _make_cli(db, parent, texts=("hello parent",))
    before = _session_ids(_isolated_home)

    cli._handle_detour_end_command("/detour-end")

    assert "Nothing to end" in capsys.readouterr().out
    assert cli.session_id == parent
    assert _session_ids(_isolated_home) == before
    assert not record_path(_scope(parent)).exists()


# --------------------------------------------------------------------------- repeated entry
def test_a_second_detour_neither_nests_nor_overwrites_the_record(db, capsys, _isolated_home):
    """The CLI's lane key is the ORIGINATING session, so the nesting check has to look at the lane
    the current session belongs to — not at the lane a new detour would create, which is empty."""
    parent = _new_session_id("par010")
    cli = _make_cli(db, parent, texts=("hello parent",))
    cli._handle_detour_command("/detour child-model")
    child, before = cli.session_id, record_path(_scope(parent)).read_bytes()
    sessions_before = _session_ids(_isolated_home)
    capsys.readouterr()

    cli._handle_detour_command("/detour other-model")

    assert "already open" in capsys.readouterr().out
    assert cli.session_id == child  # no second rotation
    assert _session_ids(_isolated_home) == sessions_before  # and no orphaned third session
    assert record_path(_scope(parent)).read_bytes() == before
    assert not record_path(_scope(child)).exists()  # no second lane either
    assert "/model other-model --session" not in cli.applied_routes

    # The one detour that IS open is still returnable.
    cli._handle_detour_end_command("/detour-end")
    assert cli.session_id == parent
    assert _texts(cli) == ["hello parent"]


# --------------------------------------------------------------------------- empty parent
def test_a_detour_from_an_empty_parent_is_still_returnable(db):
    """``/new`` prunes a parent row that never gained content; the enter leg puts it back."""
    parent = _new_session_id("par011")
    cli = _make_cli(db, parent)  # no transcript at all
    assert db.get_session(parent) is not None

    cli._handle_detour_command("/detour child-model")

    child = cli.session_id
    assert child != parent
    # The row the native reset pruned is back, so there is something to resume.
    assert db.get_session(parent) is not None
    assert read_record(_scope(parent)).parent_session_id == parent

    cli._handle_detour_end_command("/detour-end")

    assert cli.session_id == parent
    assert cli.conversation_history == []
    assert read_record(_scope(parent)).status == STATUS_RETURNED


# --------------------------------------------------------------------------- isolation
def test_two_concurrent_cli_sessions_detour_and_return_independently(db):
    first_parent, second_parent = _new_session_id("parA01"), _new_session_id("parB01")
    first = _make_cli(db, first_parent, texts=("first parent",))
    second = _make_cli(db, second_parent, texts=("second parent",))

    first._handle_detour_command("/detour first-child-model")
    second._handle_detour_command("/detour second-child-model")
    first_child, second_child = first.session_id, second.session_id

    # Two lanes, two records, neither visible to the other.
    assert record_path(_scope(first_parent)) != record_path(_scope(second_parent))
    assert read_record(_scope(first_parent)).child_session_id == first_child
    assert read_record(_scope(second_parent)).child_session_id == second_child
    assert first_child != second_child

    # Ending the second one must not touch the first.
    second._handle_detour_end_command("/detour-end")
    assert second.session_id == second_parent
    assert _texts(second) == ["second parent"]
    assert first.session_id == first_child
    assert read_record(_scope(first_parent)).status == STATUS_ACTIVE

    first._handle_detour_end_command("/detour-end")
    assert first.session_id == first_parent
    assert _texts(first) == ["first parent"]


def test_a_gateway_record_is_not_usable_from_the_cli_lane(db, capsys):
    from hermes_cli.session_detour import begin_detour

    parent = _new_session_id("par012")
    cli = _make_cli(db, parent, texts=("hello parent",))
    begin_detour(DetourScope(surface="gateway", channel="telegram:c1", owner="u1", lane="k1"),
                 parent_session_id=parent)

    cli._handle_detour_end_command("/detour-end")

    assert "Nothing to end" in capsys.readouterr().out
    assert cli.session_id == parent


# --------------------------------------------------------------------------- restart
def test_a_detour_survives_a_cli_restart_and_still_returns(db):
    """A second CLI process has no in-memory handle; the child→lane index is what finds the record."""
    parent = _new_session_id("par013")
    cli = _make_cli(db, parent, texts=("hello parent", "parent reply"))
    cli._handle_detour_command("/detour child-model --provider child-provider")
    child = cli.session_id

    # "Restart": a brand-new CLI object on a brand-new session, nothing inherited from memory.
    fresh = _make_cli(db, _new_session_id("par014"))
    assert fresh._detour_open_scope is None
    fresh._handle_resume_command(f"/resume {child}")
    assert fresh.session_id == child

    fresh._handle_detour_end_command("/detour-end")

    assert fresh.session_id == parent
    assert _texts(fresh) == ["hello parent", "parent reply"]
    assert (fresh.model, fresh.provider) == (PARENT_MODEL, PARENT_PROVIDER)
    assert read_record(_scope(parent)).status == STATUS_RETURNED


# --------------------------------------------------------------------------- fail-closed
def test_a_route_that_cannot_be_applied_rolls_the_whole_enter_leg_back(db, capsys):
    parent = _new_session_id("par015")
    cli = _make_cli(db, parent, texts=("hello parent",))

    def _boom(_cmd):
        raise RuntimeError("provider unreachable")

    cli._handle_model_switch = _boom
    cli._handle_detour_command("/detour child-model")

    out = capsys.readouterr().out
    assert "Detour not started" in out and "model switch" in out
    assert cli.session_id == parent  # back on the original conversation
    assert _texts(cli) == ["hello parent"]  # with its history
    assert not record_path(_scope(parent)).exists()  # nothing left open


def test_a_switch_that_reports_success_but_changes_nothing_is_treated_as_failure(db, capsys):
    """Runtime state decides, not the handler's silence."""
    parent = _new_session_id("par016")
    cli = _make_cli(db, parent, texts=("hello parent",))
    cli.model_switch_is_broken = True

    cli._handle_detour_command("/detour child-model")

    assert "Detour not started" in capsys.readouterr().out
    assert cli.session_id == parent
    assert _texts(cli) == ["hello parent"]
    assert not record_path(_scope(parent)).exists()


def test_a_failed_native_reset_leaves_the_lane_untouched(db, capsys, monkeypatch):
    parent = _new_session_id("par017")
    cli = _make_cli(db, parent, texts=("hello parent",))

    def _boom(**_kw):
        raise RuntimeError("reset exploded")

    monkeypatch.setattr(cli, "new_session", _boom)
    cli._handle_detour_command("/detour child-model")

    assert "Detour not started" in capsys.readouterr().out
    assert cli.session_id == parent
    assert not record_path(_scope(parent)).exists()
    assert cli.applied_routes == []


def test_a_failed_resume_keeps_the_detour_open(db, capsys, monkeypatch):
    parent = _new_session_id("par018")
    cli = _make_cli(db, parent, texts=("hello parent",))
    cli._handle_detour_command("/detour")
    child = cli.session_id
    capsys.readouterr()

    monkeypatch.setattr(cli, "_handle_resume_command", lambda _cmd: None)  # refuses, quietly
    cli._handle_detour_end_command("/detour-end back-model")

    assert "Could not return" in capsys.readouterr().out
    assert cli.session_id == child
    assert read_record(_scope(parent)).status == STATUS_ACTIVE  # still retryable
    assert "/model back-model --session" not in cli.applied_routes


def test_a_deleted_parent_session_refuses_the_return_and_keeps_the_record(db, capsys, _isolated_home):
    parent = _new_session_id("par019")
    cli = _make_cli(db, parent, texts=("hello parent",))
    cli._handle_detour_command("/detour")
    child = cli.session_id
    db.delete_session(parent, sessions_dir=_isolated_home / "sessions")
    capsys.readouterr()

    cli._handle_detour_end_command("/detour-end")

    out = capsys.readouterr().out
    assert "no longer exists" in out
    assert cli.session_id == child
    assert read_record(_scope(parent)).status == STATUS_ACTIVE


@pytest.mark.parametrize("payload", [b"{ truncated", b"[]", b'{"version": 99}'])
def test_an_unusable_record_refuses_the_return_and_is_preserved(db, capsys, payload):
    parent = _new_session_id("par020")
    cli = _make_cli(db, parent, texts=("hello parent",))
    cli._handle_detour_command("/detour")
    child = cli.session_id
    path = record_path(_scope(parent))
    path.write_bytes(payload)
    capsys.readouterr()

    cli._handle_detour_end_command("/detour-end")

    assert "Could not end the detour" in capsys.readouterr().out
    assert cli.session_id == child  # fail closed: no switch, no new session
    assert path.read_bytes() == payload  # left for inspection


def test_an_unwritable_record_directory_refuses_the_enter_leg(db, capsys, monkeypatch):
    parent = _new_session_id("par021")
    cli = _make_cli(db, parent, texts=("hello parent",))

    import hermes_cli.session_detour as detour_mod

    def _boom(*_a, **_kw):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(detour_mod, "write_record", _boom)
    cli._handle_detour_command("/detour child-model")

    assert "could not be written" in capsys.readouterr().out
    assert cli.session_id == parent  # never rotated
    assert _texts(cli) == ["hello parent"]
    assert cli.applied_routes == []


def _break_the_lane_lock(monkeypatch):
    """No advisory lock obtainable — the platform refuses, as a locked-down filesystem would.

    Returns the switch, so a test can let the platform cooperate again without ``monkeypatch.undo``
    (which would also undo the autouse home fixture's patches).
    """
    import hermes_cli.session_detour as detour_mod

    broken = {"on": True}
    real_flock = detour_mod._flock

    def _maybe_flock(handle, *, lock):
        if broken["on"]:
            raise OSError("unsupported")
        return real_flock(handle, lock=lock)

    monkeypatch.setattr(detour_mod, "_flock", _maybe_flock)
    return broken


def test_a_lane_that_cannot_be_locked_refuses_the_enter_leg_and_changes_nothing(
        db, capsys, monkeypatch, _isolated_home):
    """Fail closed: without the lane lock the enter leg cannot be serialized against another
    process, so it refuses before any record, session or route moves."""
    parent = _new_session_id("par023")
    cli = _make_cli(db, parent, texts=("hello parent",))
    rows_before = _session_ids(_isolated_home)
    _break_the_lane_lock(monkeypatch)

    cli._handle_detour_command("/detour child-model")

    out = capsys.readouterr().out
    assert "Detour not started" in out and "could not be locked" in out
    assert cli.session_id == parent
    assert _texts(cli) == ["hello parent"]
    assert (cli.model, cli.provider) == (PARENT_MODEL, PARENT_PROVIDER)
    assert cli.applied_routes == []
    assert not record_path(_scope(parent)).exists()
    assert _session_ids(_isolated_home) == rows_before


def test_a_lane_that_cannot_be_locked_refuses_the_return_leg_and_keeps_the_detour(
        db, capsys, monkeypatch):
    """The open detour is left exactly as it was, so the return stays retryable."""
    parent = _new_session_id("par024")
    cli = _make_cli(db, parent, texts=("hello parent",))
    cli._handle_detour_command("/detour")
    child = cli.session_id
    capsys.readouterr()
    _break_the_lane_lock(monkeypatch)

    cli._handle_detour_end_command("/detour-end")

    out = capsys.readouterr().out
    assert "Could not end the detour" in out and "could not be locked" in out
    assert cli.session_id == child  # no resume, no switch
    assert read_record(_scope(parent)).status == STATUS_ACTIVE


def test_a_detour_refused_by_the_lock_succeeds_on_the_next_try(db, capsys, monkeypatch):
    """A refusal must not wedge the lane: the retry is an ordinary detour, with a real record."""
    parent = _new_session_id("par025")
    cli = _make_cli(db, parent, texts=("hello parent",))
    broken = _break_the_lane_lock(monkeypatch)
    cli._handle_detour_command("/detour child-model")
    assert cli.session_id == parent
    capsys.readouterr()

    broken["on"] = False  # the platform cooperates again
    cli._handle_detour_command("/detour child-model")

    child = cli.session_id
    assert child and child != parent
    record = read_record(_scope(parent))
    assert record.status == STATUS_ACTIVE
    assert (record.parent_session_id, record.child_session_id) == (parent, child)

    # ...and so does the return leg that was refused a moment earlier.
    capsys.readouterr()
    cli._handle_detour_end_command("/detour-end")
    assert cli.session_id == parent
    assert read_record(_scope(parent)).status == STATUS_RETURNED


@pytest.mark.parametrize("args", ["child-model --global", "child-model --once"])
def test_a_route_that_would_outlive_the_detour_is_refused_before_anything_happens(db, capsys, args):
    parent = _new_session_id("par022")
    cli = _make_cli(db, parent, texts=("hello parent",))

    cli._handle_detour_command(f"/detour {args}")

    assert capsys.readouterr().out.strip().startswith("✗")
    assert cli.session_id == parent
    assert not record_path(_scope(parent)).exists()
    assert cli.applied_routes == []
