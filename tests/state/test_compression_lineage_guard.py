"""Regression tests for stale writes after a compression session split."""

from __future__ import annotations

import json

import pytest

from hermes_state import COMPRESSION_FIDELITY_CONFIG_KEY, SessionDB


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    try:
        yield session_db
    finally:
        session_db.close()


def _compression_parent(db: SessionDB, session_id: str = "parent") -> None:
    db.create_session(session_id, source="webui")
    db.append_message(session_id, "user", "before split")
    db.end_session(session_id, "compression")


def test_find_live_compression_child_returns_unique_direct_child(db: SessionDB) -> None:
    _compression_parent(db)
    db.create_session("child", source="webui", parent_session_id="parent")

    child = db.find_live_compression_child("parent")

    assert child is not None
    assert child["id"] == "child"
    assert child["parent_session_id"] == "parent"
    assert child["ended_at"] is None


def test_find_live_compression_child_fails_closed_when_ambiguous(db: SessionDB) -> None:
    _compression_parent(db)
    db.create_session("child-a", source="webui", parent_session_id="parent")
    db.create_session("child-b", source="webui", parent_session_id="parent")

    assert db.find_live_compression_child("parent") is None




def test_find_live_compression_child_ignores_non_continuation_children(
    db: SessionDB,
) -> None:
    _compression_parent(db)
    db.create_session("canonical", source="webui", parent_session_id="parent")
    db.create_session(
        "branch",
        source="webui",
        parent_session_id="parent",
        model_config={"_branched_from": "parent"},
    )
    db.create_session(
        "delegate",
        source="webui",
        parent_session_id="parent",
        model_config={"_delegate_from": "parent"},
    )
    db.create_session("tool-child", source="tool", parent_session_id="parent")

    child = db.find_live_compression_child("parent")

    assert child is not None
    assert child["id"] == "canonical"








def test_publish_compression_child_is_atomic_on_handoff_failure(
    db: SessionDB, monkeypatch
) -> None:
    db.create_session("atomic-parent", source="webui")
    db.append_message("atomic-parent", "user", "original")
    assert db.try_acquire_compression_lock("atomic-parent", "winner", ttl_seconds=60)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("handoff insert failed")

    monkeypatch.setattr(db, "_insert_message_rows", _boom)
    with pytest.raises(RuntimeError, match="handoff insert failed"):
        db.publish_compression_child(
            parent_session_id="atomic-parent",
            child_session_id="atomic-child",
            source="webui",
            messages=[{"role": "user", "content": "summary"}],
            compression_lock_holder="winner",
        )

    parent = db.get_session("atomic-parent")
    assert parent is not None
    assert parent["ended_at"] is None
    assert db.get_session("atomic-child") is None


def test_publish_compression_child_exposes_complete_child(db: SessionDB) -> None:
    db.create_session("atomic-parent", source="webui")
    db.append_message("atomic-parent", "user", "original")
    assert db.try_acquire_compression_lock("atomic-parent", "winner", ttl_seconds=60)

    db.publish_compression_child(
        parent_session_id="atomic-parent",
        child_session_id="atomic-child",
        source="webui",
        system_prompt="compressed system",
        messages=[{"role": "user", "content": "summary"}],
        compression_lock_holder="winner",
    )

    assert db.get_session("atomic-parent")["end_reason"] == "compression"
    child = db.find_live_compression_child("atomic-parent")
    assert child is not None
    assert child["id"] == "atomic-child"
    assert child["system_prompt"] == "compressed system"
    assert [m["content"] for m in db.get_messages("atomic-child")] == ["summary"]


def test_publish_compression_child_persists_provenance_without_mutating_config(
    db: SessionDB,
) -> None:
    db.create_session("provenance-parent", source="webui")
    assert db.try_acquire_compression_lock(
        "provenance-parent", "winner", ttl_seconds=60
    )
    model_config = {"temperature": 0.2}
    provenance = {
        "schema_version": 1,
        "parent_session_id": "provenance-parent",
        "child_session_id": "provenance-child",
        "manifest_digest": "digest",
    }

    db.publish_compression_child(
        parent_session_id="provenance-parent",
        child_session_id="provenance-child",
        source="webui",
        messages=[{"role": "user", "content": "summary"}],
        model_config=model_config,
        compression_fidelity_provenance=provenance,
        compression_lock_holder="winner",
    )

    assert model_config == {"temperature": 0.2}
    child_config = json.loads(db.get_session("provenance-child")["model_config"])
    assert child_config["temperature"] == 0.2
    assert child_config[COMPRESSION_FIDELITY_CONFIG_KEY] == provenance


def test_publish_without_provenance_preserves_existing_model_config_shape(
    db: SessionDB,
) -> None:
    db.create_session("plain-parent", source="webui")
    assert db.try_acquire_compression_lock("plain-parent", "winner", ttl_seconds=60)

    db.publish_compression_child(
        parent_session_id="plain-parent",
        child_session_id="plain-child",
        source="webui",
        messages=[{"role": "user", "content": "summary"}],
        model_config={"temperature": 0.1},
        compression_lock_holder="winner",
    )

    child_config = json.loads(db.get_session("plain-child")["model_config"])
    assert child_config == {"temperature": 0.1}


def test_provenance_rolls_back_with_failed_handoff(db: SessionDB, monkeypatch) -> None:
    db.create_session("rollback-parent", source="webui")
    assert db.try_acquire_compression_lock(
        "rollback-parent", "winner", ttl_seconds=60
    )

    def _boom(*_args, **_kwargs):
        raise RuntimeError("handoff insert failed")

    monkeypatch.setattr(db, "_insert_message_rows", _boom)
    with pytest.raises(RuntimeError, match="handoff insert failed"):
        db.publish_compression_child(
            parent_session_id="rollback-parent",
            child_session_id="rollback-child",
            source="webui",
            messages=[{"role": "user", "content": "summary"}],
            compression_fidelity_provenance={"schema_version": 1},
            compression_lock_holder="winner",
        )

    assert db.get_session("rollback-parent")["ended_at"] is None
    assert db.get_session("rollback-child") is None


def test_publish_compression_child_rejects_lost_or_expired_lease(db: SessionDB) -> None:
    db.create_session("lease-parent", source="webui")
    db.append_message("lease-parent", "user", "new durable turn")
    assert db.try_acquire_compression_lock("lease-parent", "new-winner", ttl_seconds=60)

    with pytest.raises(RuntimeError, match="lease lost"):
        db.publish_compression_child(
            parent_session_id="lease-parent",
            child_session_id="stale-child",
            source="webui",
            messages=[{"role": "user", "content": "stale summary"}],
            compression_lock_holder="old-loser",
        )

    parent = db.get_session("lease-parent")
    assert parent is not None
    assert parent["ended_at"] is None
    assert db.get_session("stale-child") is None
    assert [m["content"] for m in db.get_messages("lease-parent")] == [
        "new durable turn"
    ]


def test_compression_lease_blocks_non_owner_but_allows_owner_flush(
    db: SessionDB,
) -> None:
    db.create_session("leased", source="webui")
    assert db.try_acquire_compression_lock("leased", "winner", ttl_seconds=60)

    with pytest.raises(RuntimeError, match="being compressed"):
        db.append_message("leased", "user", "late stale turn")

    db.append_message(
        "leased",
        "assistant",
        "winner flush",
        compression_lock_holder="winner",
    )
    assert [m["content"] for m in db.get_messages("leased")] == ["winner flush"]
