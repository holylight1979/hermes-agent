"""Deterministic continuation after fidelity rejection."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from agent.context_compressor import FIDELITY_SEMANTIC_MALFORMED
from tests.agent.test_compression_fidelity_semantic_gate import (
    _Router,
    _compressor,
    _live_messages,
    _live_summary_body,
    _reject_verdict,
)


def _render(messages) -> str:
    return "\n".join(str(message.get("content", "")) for message in messages)


def test_rejected_candidate_reanchors_even_with_abort_enabled():
    compressor = _compressor(
        abort_on_summary_failure=True,
        fidelity_max_retries=0,
    )
    messages = _live_messages()
    router = _Router([_live_summary_body("rejected-generation")], [_reject_verdict()])

    with patch("agent.context_compressor.call_llm", side_effect=router):
        result = compressor.compress(messages)

    rendered = _render(result)
    assert result != messages
    assert "## Compression Fidelity Re-anchor" in rendered
    assert "rejected-generation" not in rendered
    assert compressor._last_compress_aborted is False
    assert compressor._last_summary_fallback_used is True
    assert compressor._last_fidelity_reanchor_used is True
    provenance = compressor.get_last_fidelity_provenance("parent", "child")
    assert provenance["accepted"] is False
    assert provenance["reanchor_used"] is True


@pytest.mark.parametrize(
    "verdict",
    [
        "not json",
        RuntimeError("verifier unavailable"),
        ConnectionError("verifier network unavailable"),
    ],
)
def test_malformed_or_unreachable_verifier_uses_reanchor(verdict):
    compressor = _compressor(
        abort_on_summary_failure=True,
        fidelity_max_retries=0,
    )
    router = _Router([_live_summary_body()], [verdict])

    with patch("agent.context_compressor.call_llm", side_effect=router):
        result = compressor.compress(_live_messages())

    assert "## Compression Fidelity Re-anchor" in _render(result)
    assert compressor._last_compress_aborted is False
    assert compressor._last_fidelity_reanchor_used is True


def test_reanchor_does_not_leak_source_secret():
    secret = "sk-secret-value-that-must-not-survive"
    messages = _live_messages()
    messages[3] = {
        "role": "user",
        "content": f"Do not expose {secret} while you work.",
    }
    compressor = _compressor(
        abort_on_summary_failure=True,
        fidelity_max_retries=0,
    )
    router = _Router([_live_summary_body("must-not-commit")], [_reject_verdict()])

    with patch("agent.context_compressor.call_llm", side_effect=router):
        result = compressor.compress(messages)

    rendered = _render(result)
    assert "## Compression Fidelity Re-anchor" in rendered
    assert secret not in rendered
    assert "must-not-commit" not in rendered


def test_zero_user_reanchor_does_not_invent_user_request():
    compressor = _compressor()
    turns = [
        {"role": "assistant", "content": "internal progress"},
        {"role": "tool", "content": "failed with exit 2"},
    ]
    compressor._last_fidelity_manifest = compressor._build_continuity_manifest(turns)
    compressor._last_fidelity_failure_codes = (FIDELITY_SEMANTIC_MALFORMED,)

    summary = compressor._build_fidelity_reanchor_summary(turns)

    assert "None. This session contains no user-authored turns." in summary
    assert "User asked:" not in summary


def test_provider_summary_failure_still_aborts_unchanged():
    compressor = _compressor(abort_on_summary_failure=True)
    messages = _live_messages()

    with patch.object(compressor, "_generate_summary", return_value=None):
        result = compressor.compress(messages)

    assert result == messages
    assert compressor._last_compress_aborted is True
    assert compressor._last_fidelity_reanchor_used is False
    assert "## Compression Fidelity Re-anchor" not in _render(result)


# ---------------------------------------------------------------------------
# A rejected candidate followed by a terminal generation failure.
#
# The re-anchor is licensed by a fidelity VERDICT: the source was read, the
# manifest was built, only the model's wording was unsafe.  An access/quota or
# network failure on a later rewrite in the same retry cycle carries no verdict
# — the summary route is simply gone — and must abort unchanged regardless of
# abort_on_summary_failure.  The stale failure codes from the earlier rejection
# must not be mistaken for a verdict on the run as a whole.
# ---------------------------------------------------------------------------


_AUTH_ERROR = RuntimeError(
    "Provider 'opencode-zen' is set in config.yaml but no API key was found."
)
_NETWORK_ERROR = ConnectionError("Connection error.")


def _local_reject_body(marker: str) -> str:
    """A candidate the deterministic LOCAL gate refuses.

    The historical task snapshot is dropped wholesale — grounding only rewrites
    a section that exists, so its absence is a local failure the verifier never
    gets to see.
    """
    return "## Goal" + _live_summary_body(marker).split("## Goal", 1)[1]


@pytest.mark.parametrize("abort_on_summary_failure", [True, False])
@pytest.mark.parametrize(
    "error,failure_flag",
    [
        (_AUTH_ERROR, "_last_summary_auth_failure"),
        (_NETWORK_ERROR, "_last_summary_network_failure"),
    ],
    ids=["auth", "network"],
)
@pytest.mark.parametrize("reject_gate", ["local", "semantic"])
def test_rejection_then_terminal_generation_failure_aborts_unchanged(
    reject_gate, error, failure_flag, abort_on_summary_failure
):
    marker = f"{reject_gate}-rejected-candidate"
    if reject_gate == "local":
        first_candidate, verdicts = _local_reject_body(marker), [_reject_verdict()]
    else:
        first_candidate, verdicts = _live_summary_body(marker), [_reject_verdict()]
    compressor = _compressor(
        abort_on_summary_failure=abort_on_summary_failure,
        fidelity_max_retries=1,
    )
    messages = _live_messages()
    # Candidate 1 is rejected and buys a rewrite; the rewrite hits the terminal
    # failure, so the run ends with stale codes from the rejection.
    router = _Router([first_candidate, error], verdicts)

    with patch("agent.context_compressor.call_llm", side_effect=router):
        result = compressor.compress(messages)

    rendered = _render(result)
    assert result == messages
    assert getattr(compressor, failure_flag) is True
    assert compressor._last_compress_aborted is True
    assert compressor._last_fidelity_reanchor_used is False
    assert compressor._last_summary_fallback_used is False
    assert "## Compression Fidelity Re-anchor" not in rendered
    # The rejected candidate is committed nowhere — not into the returned
    # messages, and not as the next compaction's iterative-update base.
    assert marker not in rendered
    assert marker not in (compressor._previous_summary or "")
    assert len(router.summary_prompts) == 2
    assert len(router.verifier_prompts) == (0 if reject_gate == "local" else 1)


def test_auth_abort_does_not_disable_reanchor_on_next_healthy_compress():
    compressor = _compressor(
        abort_on_summary_failure=True,
        fidelity_max_retries=0,
    )
    messages = _live_messages()

    def _auth_failure(*_args, **_kwargs):
        compressor._last_summary_auth_failure = True
        return None

    with patch.object(compressor, "_generate_summary", side_effect=_auth_failure):
        first = compressor.compress(messages)

    assert first == messages
    assert compressor._last_compress_aborted is True

    router = _Router([_live_summary_body("healthy-second-run")], [_reject_verdict()])
    with patch("agent.context_compressor.call_llm", side_effect=router):
        second = compressor.compress(messages)

    assert compressor._last_summary_auth_failure is False
    assert compressor._last_compress_aborted is False
    assert compressor._last_fidelity_reanchor_used is True
    assert "## Compression Fidelity Re-anchor" in _render(second)
    assert "healthy-second-run" not in _render(second)


def test_rejection_then_nonterminal_generation_failure_still_reanchors():
    compressor = _compressor(
        abort_on_summary_failure=True,
        fidelity_max_retries=1,
    )
    candidates = iter([_live_summary_body("rejected-first"), None])

    with (
        patch.object(compressor, "_generate_summary", side_effect=lambda *_a, **_k: next(candidates)),
        patch.object(
            compressor,
            "_verify_summary_semantics",
            return_value=compressor._parse_semantic_verdict(_reject_verdict()),
        ),
    ):
        result = compressor.compress(_live_messages())

    rendered = _render(result)
    assert compressor._last_summary_auth_failure is False
    assert compressor._last_summary_network_failure is False
    assert compressor._last_compress_aborted is False
    assert compressor._last_fidelity_reanchor_used is True
    assert "## Compression Fidelity Re-anchor" in rendered
    assert "rejected-first" not in rendered
