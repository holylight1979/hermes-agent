"""Return-record contracts for ``/detour`` / ``/detour-end`` (the ``session-detours`` plugin's ``detour_records``).

These exercise the real store against a temp home: no surface, no sessions, just the durable
state that makes the return leg possible and the gates that stop it from being abused.
"""

import json
import os
import threading

import pytest

from tests.fakes.session_detours_plugin import (
    LOCK_REFUSAL,
    RECORD_VERSION,
    STATUS_ACTIVE,
    STATUS_ENTERING,
    STATUS_RETURNED,
    DetourRecordError,
    DetourScope,
    LaneLockError,
    LaneTransitionLock,
    begin_detour,
    confirm_detour,
    delete_record,
    detour_root,
    mark_returned,
    parse_route_args,
    read_record,
    record_path,
    resolve_return_target,
    route_switch_command,
    scope_lock,
    write_record,
)


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / ".hermes"
    h.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(h))
    return h


def _scope(**kw):
    base = {"surface": "gateway", "channel": "discord:chan1", "owner": "user1", "lane": "key1"}
    base.update(kw)
    return DetourScope(**base)


def _enter(scope, parent="20260101_000000_aaa", child="20260101_000100_bbb", **kw):
    record = begin_detour(scope, parent_session_id=parent, **kw)
    return confirm_detour(record, child_session_id=child)


# --------------------------------------------------------------------------- storage shape
def test_record_lands_under_the_active_home_as_readable_utf8_json(home):
    scope = _scope()
    _enter(scope, child="20260101_000100_bbb")

    path = record_path(scope)
    assert path.parent == detour_root(home)
    assert path.parent.resolve() == (home / "session-detours").resolve()
    # Readable without a tool: indented, decodes as UTF-8, round-trips through json.
    text = path.read_text(encoding="utf-8")
    assert "\n  " in text
    data = json.loads(text)
    assert data["version"] == RECORD_VERSION
    assert data["status"] == STATUS_ACTIVE
    assert data["parent_session_id"] == "20260101_000000_aaa"
    assert data["child_session_id"] == "20260101_000100_bbb"
    # The readable scope is stored alongside the digest-named file.
    assert data["scope"]["channel"] == "discord:chan1"


def test_record_never_carries_credentials_or_transcript(home):
    scope = _scope()
    _enter(
        scope,
        parent_route={"model": "gpt-6-astra-900k", "provider": "openai-codex",
                      "api_key": "sk-must-not-persist", "messages": [{"role": "user", "content": "secret"}]},
    )
    text = record_path(scope).read_text(encoding="utf-8")
    assert "sk-must-not-persist" not in text and "secret" not in text
    stored = json.loads(text)["parent_route"]
    assert set(stored) == {"model", "provider"}
    assert stored["model"] == "gpt-6-astra-900k"


def test_filename_is_a_digest_so_scope_text_cannot_escape_the_directory(home):
    scope = _scope(channel="../../etc", owner="../../..", lane="..\\..\\windows")
    path = record_path(scope)
    assert path.name.endswith(".json") and ".." not in path.name
    assert path.parent.resolve() == detour_root(home).resolve()
    _enter(scope)
    assert path.exists()
    # Nothing was written outside the record directory.
    assert {p.parent.resolve() for p in home.rglob("*.json")} == {detour_root(home).resolve()}


# --------------------------------------------------------------------------- scope isolation
@pytest.mark.parametrize("different", [
    {"surface": "cli"},
    {"channel": "discord:chan2"},
    {"owner": "user2"},
    {"lane": "key2"},
])
def test_each_lane_owner_and_surface_gets_its_own_record(home, different):
    mine, theirs = _scope(), _scope(**different)
    assert record_path(mine) != record_path(theirs)
    _enter(mine, parent="20260101_000000_aaa")
    # The other lane sees no detour at all — records are not a shared global.
    assert read_record(theirs) is None
    assert read_record(mine).parent_session_id == "20260101_000000_aaa"


def test_a_record_moved_into_another_lanes_slot_is_refused(home):
    mine, theirs = _scope(), _scope(owner="user2")
    _enter(mine)
    # Simulate tampering: same bytes, other lane's filename.
    record_path(theirs).write_bytes(record_path(mine).read_bytes())
    with pytest.raises(DetourRecordError, match="belongs to another conversation"):
        read_record(theirs)


def test_different_homes_do_not_share_records(tmp_path, monkeypatch):
    scope = _scope()
    for name in ("profile-a", "profile-b"):
        (tmp_path / name).mkdir()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile-a"))
    _enter(scope, parent="20260101_000000_aaa")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile-b"))
    assert read_record(scope) is None


# --------------------------------------------------------------------------- fail-closed reads
def test_absent_record_reads_as_none_not_an_error(home):
    assert read_record(_scope()) is None


@pytest.mark.parametrize("payload", [
    b"not json at all",
    b"[]",
    b'{"version": 1}',
    b'{"version": 99, "status": "active", "scope": {"surface": "gateway"}, "parent_session_id": "x"}',
    b'\xff\xfe\x00bad utf8',
])
def test_a_corrupt_record_raises_instead_of_reading_as_absent(home, payload):
    scope = _scope()
    path = record_path(scope)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    with pytest.raises(DetourRecordError):
        read_record(scope)


def test_a_record_naming_a_path_as_the_parent_session_is_refused(home):
    scope = _scope()
    record = _enter(scope)
    data = json.loads(record_path(scope).read_text(encoding="utf-8"))
    data["parent_session_id"] = "../../other-home/sessions/stolen"
    record_path(scope).write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(DetourRecordError, match="invalid parent session id"):
        read_record(scope)
    assert record.parent_session_id == "20260101_000000_aaa"


def test_corrupt_record_is_preserved_for_inspection(home):
    scope = _scope()
    path = record_path(scope)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"{truncated")
    with pytest.raises(DetourRecordError):
        read_record(scope)
    assert path.read_bytes() == b"{truncated"


# --------------------------------------------------------------------------- lifecycle
def test_enter_then_return_moves_entering_to_active_to_returned(home):
    scope = _scope()
    opened = begin_detour(scope, parent_session_id="20260101_000000_aaa")
    assert read_record(scope).status == STATUS_ENTERING
    assert read_record(scope).parent_session_id == "20260101_000000_aaa"

    live = confirm_detour(opened, child_session_id="20260101_000100_bbb")
    assert read_record(scope).status == STATUS_ACTIVE and read_record(scope).is_open

    mark_returned(live)
    done = read_record(scope)
    assert done.status == STATUS_RETURNED and not done.is_open
    assert done.returned_at and done.parent_session_id == "20260101_000000_aaa"


def test_a_returned_record_survives_as_history(home):
    scope = _scope()
    mark_returned(_enter(scope))
    assert record_path(scope).exists()
    assert read_record(scope).child_session_id == "20260101_000100_bbb"


def test_delete_record_is_the_rollback_path_and_is_idempotent(home):
    scope = _scope()
    begin_detour(scope, parent_session_id="20260101_000000_aaa")
    assert delete_record(scope) is True
    assert read_record(scope) is None
    assert delete_record(scope) is False


# --------------------------------------------------------------------------- return-target gates
def test_return_target_is_the_recorded_parent_when_every_gate_passes(home):
    scope = _scope()
    record = _enter(scope)
    target = resolve_return_target(
        record, scope, current_session_id="20260101_000100_bbb", session_exists=lambda _s: True)
    assert target == "20260101_000000_aaa"


def test_return_refuses_when_no_detour_is_open(home):
    scope = _scope()
    with pytest.raises(DetourRecordError, match="no detour is open"):
        resolve_return_target(None, scope, current_session_id="x", session_exists=lambda _s: True)


def test_repeated_return_after_a_successful_one_is_the_same_quiet_refusal(home):
    scope = _scope()
    mark_returned(_enter(scope))
    with pytest.raises(DetourRecordError, match="no detour is open"):
        resolve_return_target(
            read_record(scope), scope, current_session_id="20260101_000000_aaa",
            session_exists=lambda _s: True)


def test_return_refuses_from_a_conversation_that_is_not_the_detour_session(home):
    """The record is a return handle for its own child session only — not a jump into the parent."""
    scope = _scope()
    record = _enter(scope)
    with pytest.raises(DetourRecordError, match="is not the detour's session"):
        resolve_return_target(
            record, scope, current_session_id="20260101_009999_zzz", session_exists=lambda _s: True)


def test_return_refuses_when_the_parent_session_no_longer_exists(home):
    scope = _scope()
    record = _enter(scope)
    with pytest.raises(DetourRecordError, match="no longer exists"):
        resolve_return_target(
            record, scope, current_session_id="20260101_000100_bbb", session_exists=lambda _s: False)


def test_return_refuses_when_already_on_the_parent(home):
    scope = _scope()
    record = _enter(scope, child="20260101_000000_aaa")
    with pytest.raises(DetourRecordError, match="already on the detour's parent"):
        resolve_return_target(
            record, scope, current_session_id="20260101_000000_aaa", session_exists=lambda _s: True)


def test_a_record_from_another_lane_is_refused_even_if_handed_in_directly(home):
    record = _enter(_scope())
    with pytest.raises(DetourRecordError, match="belongs to another conversation"):
        resolve_return_target(
            record, _scope(owner="user2"), current_session_id="20260101_000100_bbb",
            session_exists=lambda _s: True)


# --------------------------------------------------------------------------- restart recovery
def test_the_record_outlives_the_process_that_wrote_it(home):
    """Restart recovery: a detour opened earlier is still returnable from a cold read."""
    scope = _scope()
    _enter(scope)
    reread = read_record(DetourScope(**_scope().to_json()))
    assert reread.status == STATUS_ACTIVE
    assert resolve_return_target(
        reread, scope, current_session_id="20260101_000100_bbb",
        session_exists=lambda _s: True) == "20260101_000000_aaa"


# --------------------------------------------------------------------------- concurrency
def test_the_lane_lock_serializes_read_decide_write(home):
    """Two concurrent enters on one lane: exactly one writes a record, the other sees it open."""
    scope = _scope()
    start, outcomes, guard = threading.Barrier(2), [], threading.Lock()

    def _enter_once(index):
        start.wait(timeout=10)
        with scope_lock(scope):
            existing = read_record(scope)
            if existing is not None and existing.is_open:
                with guard:
                    outcomes.append(("blocked", index))
                return
            begin_detour(scope, parent_session_id=f"2026010{index}_000000_aaa")
            with guard:
                outcomes.append(("opened", index))

    threads = [threading.Thread(target=_enter_once, args=(i,)) for i in (1, 2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)
        assert not thread.is_alive()

    assert sorted(kind for kind, _ in outcomes) == ["blocked", "opened"]
    opened_by = next(i for kind, i in outcomes if kind == "opened")
    assert read_record(scope).parent_session_id == f"2026010{opened_by}_000000_aaa"


def _unlockable(monkeypatch):
    """Make the advisory lock unobtainable, the way a platform without it would."""
    from tests.fakes.session_detours_plugin import records as sd

    monkeypatch.setattr(sd, "_flock", lambda *a, **k: (_ for _ in ()).throw(OSError("unsupported")))
    return sd


def test_a_lane_that_cannot_be_locked_refuses_instead_of_running_the_body(home, monkeypatch):
    """Fail closed. Degrading to in-process ordering was silently unsafe: a second process on the
    same home would read "no detour open" on a lane this one is mid-transition on, rotate it too,
    and overwrite the record — orphaning a child session with no way back."""
    _unlockable(monkeypatch)
    scope = _scope()
    ran = []

    with pytest.raises(LaneLockError):
        with scope_lock(scope):
            ran.append("body")
            begin_detour(scope, parent_session_id="20260101_000000_aaa")

    assert ran == []  # the body never ran, so nothing decided and nothing wrote
    assert not record_path(scope).exists()
    assert read_record(scope) is None


def test_the_lock_refusal_names_no_path_or_errno(home, monkeypatch):
    """The copy reaches whoever typed the command, so the details stay in the log."""
    _unlockable(monkeypatch)
    scope = _scope()

    with pytest.raises(LaneLockError) as caught:
        LaneTransitionLock(scope).acquire()

    message = str(caught.value)
    assert message == LOCK_REFUSAL
    assert str(home) not in message and "unsupported" not in message
    # Still a DetourRecordError, so every pre-existing fail-closed handler catches it.
    assert isinstance(caught.value, DetourRecordError)


def test_a_lock_that_fails_after_the_file_is_open_closes_the_handle(home, monkeypatch):
    """The handle is bound BEFORE the flock, so a flock that raises still closes its file rather
    than leaking it for the life of the process."""
    sd = _unlockable(monkeypatch)
    opened = []
    real_open = open

    def _spy_open(*args, **kwargs):
        handle = real_open(*args, **kwargs)
        opened.append(handle)
        return handle

    monkeypatch.setattr(sd, "open", _spy_open, raising=False)

    with pytest.raises(LaneLockError):
        LaneTransitionLock(_scope()).acquire()

    assert opened and all(handle.closed for handle in opened)


def test_a_refused_lock_leaves_the_lane_acquirable_again(home, monkeypatch):
    """The in-process mutex is released on the failure path, so a retry is a normal acquisition —
    a refused command must not wedge the lane."""
    from tests.fakes.session_detours_plugin import records as sd

    calls = []
    real_flock = sd._flock

    def _fails_once(handle, *, lock):
        calls.append(lock)
        if len(calls) == 1:
            raise OSError("unsupported")
        return real_flock(handle, lock=lock)

    monkeypatch.setattr(sd, "_flock", _fails_once)
    scope = _scope()
    with pytest.raises(LaneLockError):
        with scope_lock(scope):
            pass

    assert not sd._lane_mutex(scope).locked()  # asserted before the retry, so a leak is not a hang
    with scope_lock(scope):
        begin_detour(scope, parent_session_id="20260101_000000_aaa")
    assert read_record(scope).parent_session_id == "20260101_000000_aaa"


# --------------------------------------------------------------------------- route arguments
def test_route_args_reuse_the_model_parser_and_stay_session_scoped():
    route, error = parse_route_args("some-model --provider some-provider")
    assert error is None and route == {"model": "some-model", "provider": "some-provider"}
    assert route_switch_command(route) == "/model some-model --provider some-provider --session"


def test_an_empty_route_asks_for_no_model_switch_at_all():
    route, error = parse_route_args("")
    assert error is None and route_switch_command(route) == ""


@pytest.mark.parametrize("flag", ["--global", "--once"])
def test_a_route_that_would_outlive_the_detour_is_refused(flag):
    route, error = parse_route_args(f"some-model {flag}")
    assert route is None and error and flag.lstrip("-") in error


def test_a_route_never_carries_an_api_key_into_the_switch_command():
    assert route_switch_command({"model": "m", "provider": "p", "api_key": "sk-x"}) == (
        "/model m --provider p --session")


# --------------------------------------------------------------------------- writes are atomic
def test_write_is_atomic_so_a_reader_never_sees_a_half_file(home):
    scope = _scope()
    record = _enter(scope)
    path = record_path(scope)
    before = path.read_bytes()
    write_record(record.__class__(**{**record.to_json(), "scope": scope,
                                     "parent_route": record.parent_route,
                                     "child_route": record.child_route}))
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == RECORD_VERSION
    assert before  # the original was complete too
    # No temp files left behind in the record directory.
    assert all(p.suffix == ".json" or p.suffix == ".lock" for p in detour_root(home).iterdir())


def test_record_directory_is_created_on_demand(home):
    assert not detour_root(home).exists()
    begin_detour(_scope(), parent_session_id="20260101_000000_aaa")
    assert detour_root(home).is_dir()
    assert os.path.isfile(record_path(_scope()))


# ------------------------------------------------------- records written before the plugin move
#: A record exactly as the pre-plugin core module (``hermes_cli/session_detour.py``) wrote it.
#: Byte-for-byte: the move to ``contrib/.../session-detours/detour_records.py`` changed no line of
#: that module, so a user upgrading mid-detour must find their way back, not an error.
LEGACY_RECORD_JSON = """{
  "version": 1,
  "status": "active",
  "profile": "default",
  "scope": {
    "surface": "gateway",
    "channel": "discord:chan1",
    "owner": "user1",
    "lane": "key1"
  },
  "parent_session_id": "20260101_000000_aaa",
  "child_session_id": "20260101_000100_bbb",
  "parent_route": {
    "model": "parent-model",
    "provider": "parent-provider"
  },
  "child_route": {
    "model": "child-model",
    "provider": "child-provider"
  },
  "entered_at": "2026-01-01T00:01:00+00:00",
  "returned_at": "",
  "token": "legacytoken0001"
}"""


def _write_legacy_record(home, scope) -> None:
    path = record_path(scope, home)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(LEGACY_RECORD_JSON, encoding="utf-8")


def test_a_record_written_before_the_plugin_move_is_still_read(home):
    scope = _scope()
    _write_legacy_record(home, scope)

    record = read_record(scope)

    assert record is not None
    assert record.version == RECORD_VERSION  # no schema churn across the move
    assert (record.status, record.parent_session_id, record.child_session_id) == (
        STATUS_ACTIVE, "20260101_000000_aaa", "20260101_000100_bbb")
    assert record.parent_route == {"model": "parent-model", "provider": "parent-provider"}
    assert record.token == "legacytoken0001"


def test_an_in_flight_detour_from_before_the_move_still_resolves_its_return_target(home):
    """The point of the compatibility: a user mid-detour at upgrade time can still come back."""
    scope = _scope()
    _write_legacy_record(home, scope)

    record = read_record(scope)
    target = resolve_return_target(
        record, scope, current_session_id="20260101_000100_bbb",
        session_exists=lambda sid: sid == "20260101_000000_aaa")

    assert target == "20260101_000000_aaa"
    assert record.parent_route == {"model": "parent-model", "provider": "parent-provider"}


def test_closing_a_pre_move_detour_rewrites_it_in_place_without_schema_change(home):
    scope = _scope()
    _write_legacy_record(home, scope)

    closed = mark_returned(read_record(scope), home)

    assert closed.status == STATUS_RETURNED
    stored = json.loads(record_path(scope, home).read_text(encoding="utf-8"))
    assert stored["version"] == RECORD_VERSION
    # Same key set as the record the old core wrote: nothing added, nothing dropped.
    assert set(stored) == set(json.loads(LEGACY_RECORD_JSON))
    assert stored["returned_at"] and stored["parent_session_id"] == "20260101_000000_aaa"
