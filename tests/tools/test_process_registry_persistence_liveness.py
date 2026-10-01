"""``_move_to_finished`` keeps ``_lock`` free while the completion receipt is written.

The receipt (``save_completed_result`` → ``atomic_json_write`` into
``logs/process-results``) is disk I/O on a profile directory that can stall for a long
time; ``_lock`` sits on the gateway event loop's path (``_run_process_watcher`` calls
``get()`` from the loop thread), so writing under the lock froze every heartbeat.

The contract this pins is the whole ordering, not just the liveness half:

* queries (``get`` / ``list_sessions``) answer while the receipt is blocked,
* the session stays in ``_running`` and unsignalled until the receipt lands
  (a finite parent must not observe completion and exit mid-write),
* concurrent finalizers claim the move exactly once — one receipt write, one
  completion notification, one ``_completion_event``.
"""

from __future__ import annotations

import threading
import time

import pytest

import tools.process_registry as process_registry
from tools.process_registry import ProcessRegistry, ProcessSession


@pytest.fixture()
def registry() -> ProcessRegistry:
    return ProcessRegistry()


def _run(fn, *args) -> threading.Thread:
    t = threading.Thread(target=fn, args=args, daemon=True)
    t.start()
    return t


def test_blocked_receipt_write_keeps_registry_live_and_finalizes_exactly_once(
    registry: ProcessRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = ProcessSession(
        id="proc_stalled", command="sleep 1", task_id="t-stall",
        started_at=time.time(), exited=True, exit_code=0, notify_on_complete=True,
    )
    registry._running[session.id] = session

    writing = threading.Event()   # receipt write entered
    release = threading.Event()   # let the receipt finish
    receipts: list[str] = []

    def stalled_save(s: ProcessSession) -> None:
        receipts.append(s.id)
        writing.set()
        assert release.wait(30), "test never released the stalled receipt write"

    monkeypatch.setattr(process_registry, "save_completed_result", stalled_save)

    finisher = _run(registry._move_to_finished, session)
    assert writing.wait(10), "receipt write never started"

    # --- while the receipt is blocked -------------------------------------------------
    queries: dict = {}

    def query() -> None:
        queries["get"] = registry.get(session.id)
        queries["list"] = registry.list_sessions(task_id="t-stall")

    reader = _run(query)
    reader.join(timeout=10)
    assert not reader.is_alive(), "get()/list_sessions() blocked on the receipt write"
    assert queries["get"] is session
    assert [e["session_id"] for e in queries["list"]] == [session.id]

    # Durability ordering: still tracked, still unsignalled, nothing delivered yet.
    assert session.id in registry._running
    assert not session._completion_event.is_set()
    assert registry.completion_queue.empty()

    # A racing finalizer (kill_process vs. the reader thread) must neither duplicate the
    # move nor block behind it.
    racers = [_run(registry._move_to_finished, session) for _ in range(2)]
    for racer in racers:
        racer.join(timeout=10)
        assert not racer.is_alive(), "a racing _move_to_finished blocked on the receipt write"
    assert receipts == [session.id], "a racer started a second receipt write"
    assert not session._completion_event.is_set()
    assert registry.completion_queue.empty()

    # --- after the receipt lands ------------------------------------------------------
    release.set()
    finisher.join(timeout=10)
    assert not finisher.is_alive()

    assert receipts == [session.id]
    assert session._completion_event.wait(10)
    assert session.id not in registry._running
    assert registry._finished[session.id] is session
    assert session.id not in registry._persisting

    notification = registry.completion_queue.get_nowait()
    assert notification["type"] == "completion"
    assert notification["session_id"] == session.id
    assert registry.completion_queue.empty(), "completion delivered more than once"


def test_a_failed_receipt_still_delivers_the_completion(
    registry: ProcessRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Losing durability must not also lose live delivery.

    ``save_completed_result`` already degrades a disk ``OSError`` to a warning for exactly
    this reason.  Any other failure has to degrade the same way: the process really did
    exit, so the move completes, the notification is queued and every
    ``_completion_event`` waiter is released.
    """
    session = ProcessSession(
        id="proc_bad_receipt", command="sleep 1", task_id="t-bad",
        started_at=time.time(), exited=True, exit_code=0, notify_on_complete=True,
    )
    registry._running[session.id] = session

    def exploding_save(_s: ProcessSession) -> None:
        raise RuntimeError("redactor blew up")

    monkeypatch.setattr(process_registry, "save_completed_result", exploding_save)

    registry._move_to_finished(session)

    assert session.id not in registry._running
    assert session.id not in registry._persisting
    assert registry._finished[session.id] is session
    assert session._completion_event.is_set()
    assert registry.completion_queue.get_nowait()["session_id"] == session.id
    assert registry.completion_queue.empty()
