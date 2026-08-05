"""Content-free compression fidelity provenance contract."""

from __future__ import annotations

import json
from unittest.mock import patch

from agent.context_compressor import (
    ContinuityFact,
    ContinuityManifest,
    FIDELITY_SEMANTIC_REJECTED,
)
from tests.agent.test_compression_fidelity_semantic_gate import (
    _Router,
    _accept_verdict,
    _compressor,
    _reject_verdict,
    _window,
    _summary_body,
)


def _run_guarded(compressor, verdict: str):
    router = _Router([_summary_body()], [verdict])
    with patch("agent.context_compressor.call_llm", side_effect=router):
        return compressor._generate_verified_summary(_window())


def test_guard_off_has_no_provenance():
    compressor = _compressor(fidelity_guard=False)

    assert compressor.get_last_fidelity_provenance("parent", "child") is None


def test_accepted_provenance_is_content_free_and_json_serializable():
    compressor = _compressor()
    assert _run_guarded(compressor, _accept_verdict()) is not None

    record = compressor.get_last_fidelity_provenance("parent", "child")

    assert record is not None
    assert record["schema_version"] == 1
    assert record["parent_session_id"] == "parent"
    assert record["child_session_id"] == "child"
    assert record["accepted"] is True
    assert record["failure_codes"] == []
    assert record["generation_count"] == 1
    assert record["source_count"] == len(_window())
    assert record["manifest_digest"]
    assert record["facts"]
    rendered = json.dumps(record, sort_keys=True)
    assert "Approved: proceed with the retry ladder" not in rendered
    assert "config.yaml while you work" not in rendered


def test_rejected_provenance_records_only_failure_codes():
    compressor = _compressor(fidelity_max_retries=0)
    assert _run_guarded(compressor, _reject_verdict()) is None

    record = compressor.get_last_fidelity_provenance("parent", "child")

    assert record is not None
    assert record["accepted"] is False
    assert FIDELITY_SEMANTIC_REJECTED in record["failure_codes"]
    assert "reverses the config.yaml prohibition" not in json.dumps(record)


def test_provenance_never_exposes_fact_text_or_secret():
    compressor = _compressor()
    secret = "sk-secret-value-that-must-not-persist"
    manifest = compressor._build_continuity_manifest(
        [{"role": "user", "content": f"Do not expose {secret}."}]
    )
    compressor._last_fidelity_manifest = manifest
    compressor._last_fidelity_generations = 1

    rendered = json.dumps(
        compressor.get_last_fidelity_provenance("parent", "child"),
        sort_keys=True,
    )

    assert secret not in rendered
    assert "Do not expose" not in rendered
    assert all(set(fact) == {"kind", "ordinal", "digest"} for fact in json.loads(rendered)["facts"])


def test_provenance_bounds_ids_codes_and_facts():
    compressor = _compressor()
    facts = tuple(
        ContinuityFact(
            kind="user_directive",
            text=f"private-{index}",
            ordinal=index,
            digest=f"digest-{index}",
        )
        for index in range(75)
    )
    compressor._last_fidelity_manifest = ContinuityManifest(
        facts=facts,
        source_count=75,
        digest="manifest-digest",
    )
    compressor._last_fidelity_failure_codes = tuple(f"code-{i}" for i in range(20))

    record = compressor.get_last_fidelity_provenance("p" * 200, "c" * 200)

    assert record is not None
    assert len(record["parent_session_id"]) == 128
    assert len(record["child_session_id"]) == 128
    assert len(record["failure_codes"]) == 10
    assert len(record["facts"]) == 60
    assert record["facts_truncated"] is True


def test_new_compress_call_clears_stale_provenance_even_on_early_return():
    compressor = _compressor()
    compressor._last_fidelity_manifest = ContinuityManifest(
        facts=(), source_count=0, digest="stale"
    )
    compressor._last_fidelity_accepted = True
    compressor._last_fidelity_generations = 3

    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "too short to compress"},
    ]
    compressor.compress(messages)

    assert compressor.get_last_fidelity_provenance("parent", "child") is None
    assert compressor._last_fidelity_accepted is False
    assert compressor._last_fidelity_generations == 0
