"""Bounded semantic verification of a candidate handoff summary.

The local gate (``_validate_summary_fidelity``) is a string-level check: it
proves a manifest fact is still *mentioned*, not that the summary still *means*
the same thing.  A second, bounded verifier closes that gap — one strict-JSON
auxiliary call over the redacted candidate plus the source-backed manifest.

Contract exercised here:

* Both gates must accept before a summary can be committed.
* A rejected verdict buys a bounded number of regenerations, each fed the
  verifier's findings.
* A malformed verdict, an unavailable verifier, or an exhausted retry budget
  fails closed — the candidate is never committed.
* The verifier rides the existing ``compression`` auxiliary route and its
  timeout plumbing; it adds no credential and reads no environment variable.
* With the guard disabled (the default) nothing changes: one summary call, no
  verifier call, byte-identical committed output.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from agent.context_compressor import (
    FIDELITY_MAX_RETRIES_CAP,
    FIDELITY_MISSING_LATEST_REQUEST,
    FIDELITY_SEMANTIC_MALFORMED,
    FIDELITY_SEMANTIC_REJECTED,
    FIDELITY_SEMANTIC_UNAVAILABLE,
    SUMMARY_PREFIX,
    ContextCompressor,
    _FIDELITY_FINDINGS_HEADER,
    _FIDELITY_VERIFIER_PROMPT_HEADER,
    _FIDELITY_VERIFIER_PROMPT_MAX_CHARS,
    resolve_fidelity_settings,
)

SECRET = "sk-live-4b8f2c1d9e7a6b5c4d3e2f1a0b9c8d7e"


# ---------------------------------------------------------------------------
# Fixtures: one compression window plus a candidate the LOCAL gate accepts, so
# every assertion below is about the SEMANTIC gate rather than string checks.
# ---------------------------------------------------------------------------


def _window() -> list[dict]:
    return [
        {
            "role": "user",
            "content": "Refactor agent/context_compressor.py so compaction keeps the plan.",
        },
        {
            "role": "assistant",
            "content": "Updating the plan.",
            "tool_calls": [
                {
                    "id": "call-todo",
                    "function": {
                        "name": "todo",
                        "arguments": json.dumps(
                            {
                                "todos": [
                                    {
                                        "id": "1",
                                        "content": "Draft the manifest builder",
                                        "status": "completed",
                                    },
                                    {
                                        "id": "2",
                                        "content": "Wire the verdict into conversation_loop.py",
                                        "status": "pending",
                                    },
                                ]
                            }
                        ),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-todo", "content": "todo list updated"},
        {"role": "user", "content": "Do not modify config.yaml while you work."},
        {
            "role": "assistant",
            "content": "Running it.",
            "tool_calls": [
                {
                    "id": "call-test",
                    "function": {
                        "name": "terminal",
                        "arguments": json.dumps(
                            {"command": "python -m pytest tests/agent/test_widget.py"}
                        ),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-test",
            "content": "2 failed, 5 passed\nFAILED tests/agent/test_widget.py::test_alpha",
        },
        {"role": "user", "content": "Approved: proceed with the retry ladder."},
    ]


def _summary_body(marker: str = "generation-1") -> str:
    return "\n\n".join(
        [
            "## Historical Task Snapshot\n"
            "User asked: 'Approved: proceed with the retry ladder.'",
            f"## Goal\nKeep compaction faithful to the compacted source turns. [{marker}]",
            "## Constraints & Preferences\n"
            "- User said do not modify config.yaml while working.",
            "## Completed Actions\n"
            "1. TEST `python -m pytest tests/agent/test_widget.py` — "
            "2 failed (test_alpha), 5 passed [tool: terminal]",
            "## Active State\n"
            "- Wire the verdict into conversation_loop.py — still pending, not started.",
            "## Blocked\n- test_alpha still failing.",
        ]
    )


def _accept_verdict() -> str:
    return json.dumps(
        {"accepted": True, "missing_source_ordinals": [], "contradictions": []}
    )


def _reject_verdict(
    contradiction: str = "The summary reverses the config.yaml prohibition.",
) -> str:
    return json.dumps(
        {
            "accepted": False,
            "missing_source_ordinals": [3],
            "contradictions": [contradiction],
        }
    )


def _response(content: str):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


class _Router:
    """Route mocked ``call_llm`` calls to summary vs verifier scripts.

    Each script entry is a response string or an ``Exception`` to raise; the
    last entry repeats once the script is exhausted, so "always rejects" is a
    one-element list.
    """

    def __init__(self, summaries, verdicts):
        self._summaries = list(summaries)
        self._verdicts = list(verdicts)
        self.summary_prompts: list[str] = []
        self.verifier_prompts: list[str] = []

    @staticmethod
    def _next(script, log_len):
        if not script:
            raise AssertionError("unscripted call_llm invocation")
        item = script[min(log_len - 1, len(script) - 1)]
        if isinstance(item, Exception):
            raise item
        return _response(item)

    def __call__(self, **kwargs):
        prompt = kwargs["messages"][0]["content"]
        if _FIDELITY_VERIFIER_PROMPT_HEADER in prompt:
            self.verifier_prompts.append(prompt)
            return self._next(self._verdicts, len(self.verifier_prompts))
        self.summary_prompts.append(prompt)
        return self._next(self._summaries, len(self.summary_prompts))


def _compressor(**kwargs) -> ContextCompressor:
    params = {
        "model": "test/model",
        "threshold_percent": 0.85,
        "protect_first_n": 1,
        "protect_last_n": 1,
        "quiet_mode": True,
        "fidelity_guard": True,
    }
    params.update(kwargs)
    with patch(
        "agent.context_compressor.get_model_context_length", return_value=100000
    ):
        return ContextCompressor(**params)


# ---------------------------------------------------------------------------
# Strict JSON verdict parsing
# ---------------------------------------------------------------------------


class TestVerdictParsing:
    def test_strict_accept_verdict_parses(self):
        verdict = ContextCompressor._parse_semantic_verdict(_accept_verdict())

        assert verdict.accepted is True
        assert verdict.failure_codes == ()
        assert verdict.missing_source_ordinals == ()
        assert verdict.contradictions == ()

    def test_strict_reject_verdict_carries_findings(self):
        verdict = ContextCompressor._parse_semantic_verdict(_reject_verdict())

        assert verdict.accepted is False
        assert FIDELITY_SEMANTIC_REJECTED in verdict.failure_codes
        assert verdict.missing_source_ordinals == (3,)
        assert verdict.contradictions == (
            "The summary reverses the config.yaml prohibition.",
        )

    def test_fenced_json_is_tolerated(self):
        """A whole-content code fence is a formatting artifact, not looseness."""
        verdict = ContextCompressor._parse_semantic_verdict(
            f"```json\n{_accept_verdict()}\n```"
        )

        assert verdict.accepted is True

    @pytest.mark.parametrize(
        "payload",
        [
            "the summary looks fine to me",
            "",
            "[]",
            json.dumps({"accepted": True}),
            json.dumps({"accepted": "yes", "missing_source_ordinals": [], "contradictions": []}),
            json.dumps({"accepted": True, "missing_source_ordinals": "none", "contradictions": []}),
            json.dumps({"accepted": True, "missing_source_ordinals": [], "contradictions": [{"a": 1}]}),
            'prose before {"accepted": true, "missing_source_ordinals": [], "contradictions": []}',
        ],
    )
    def test_malformed_verdicts_fail_closed(self, payload):
        verdict = ContextCompressor._parse_semantic_verdict(payload)

        assert verdict.accepted is False
        assert FIDELITY_SEMANTIC_MALFORMED in verdict.failure_codes


# ---------------------------------------------------------------------------
# The verifier call itself
# ---------------------------------------------------------------------------


class TestVerifierCall:
    def test_uses_existing_compression_route_and_timeout_plumbing(self):
        compressor = _compressor()
        manifest = ContextCompressor._build_continuity_manifest(_window())

        with patch(
            "agent.context_compressor.call_llm",
            return_value=_response(_accept_verdict()),
        ) as mock_call:
            verdict = compressor._verify_summary_semantics(_summary_body(), manifest)

        assert verdict.accepted is True
        kwargs = mock_call.call_args.kwargs
        assert kwargs["task"] == "compression"
        # The compression task resolves (and floors) its own timeout in
        # call_llm; an explicit override would bypass that floor.
        assert "timeout" not in kwargs
        assert [m["role"] for m in kwargs["messages"]] == ["user"]
        assert kwargs["main_runtime"]["model"] == compressor.model

    def test_verifier_input_is_bounded_and_redacted(self, monkeypatch):
        compressor = _compressor()
        window = _window() + [
            {"role": "user", "content": f"here is the key {SECRET} for the run " * 200}
        ]
        manifest = ContextCompressor._build_continuity_manifest(window)
        bloated = _summary_body() + f"\n\n## Critical Context\nkey {SECRET}\n" + "x" * 80000

        with patch(
            "agent.context_compressor.call_llm",
            return_value=_response(_accept_verdict()),
        ) as mock_call:
            compressor._verify_summary_semantics(bloated, manifest)

        prompt = mock_call.call_args.kwargs["messages"][0]["content"]
        assert SECRET not in prompt
        assert len(prompt) <= _FIDELITY_VERIFIER_PROMPT_MAX_CHARS

    @pytest.mark.parametrize(
        "error",
        [
            RuntimeError("Request timed out."),
            RuntimeError("No LLM provider configured"),
        ],
    )
    def test_unavailable_verifier_fails_closed(self, error):
        compressor = _compressor()
        manifest = ContextCompressor._build_continuity_manifest(_window())

        with patch("agent.context_compressor.call_llm", side_effect=error):
            verdict = compressor._verify_summary_semantics(_summary_body(), manifest)

        assert verdict.accepted is False
        assert FIDELITY_SEMANTIC_UNAVAILABLE in verdict.failure_codes

    def test_empty_verifier_response_fails_closed(self):
        compressor = _compressor()
        manifest = ContextCompressor._build_continuity_manifest(_window())

        with patch(
            "agent.context_compressor.call_llm", return_value=_response("   ")
        ):
            verdict = compressor._verify_summary_semantics(_summary_body(), manifest)

        assert verdict.accepted is False
        assert not verdict.accepted


# ---------------------------------------------------------------------------
# Generate → validate → verify → (regenerate) orchestration
# ---------------------------------------------------------------------------


class TestVerifiedGeneration:
    def test_guard_disabled_makes_no_verifier_call(self):
        compressor = _compressor(fidelity_guard=False)
        router = _Router([_summary_body()], [])

        with patch("agent.context_compressor.call_llm", side_effect=router):
            summary = compressor._generate_verified_summary(_window())

        assert summary is not None
        assert len(router.summary_prompts) == 1
        assert router.verifier_prompts == []

    def test_accepted_candidate_commits_after_one_generation(self):
        compressor = _compressor()
        router = _Router([_summary_body()], [_accept_verdict()])

        with patch("agent.context_compressor.call_llm", side_effect=router):
            summary = compressor._generate_verified_summary(_window())

        assert summary is not None
        assert "generation-1" in summary
        assert len(router.summary_prompts) == 1
        assert len(router.verifier_prompts) == 1

    def test_rejected_verdict_regenerates_once_with_findings(self):
        compressor = _compressor()
        contradiction = "Summary claims the widget tests passed."
        router = _Router(
            [_summary_body("generation-1"), _summary_body("generation-2")],
            [_reject_verdict(contradiction), _accept_verdict()],
        )

        with patch("agent.context_compressor.call_llm", side_effect=router):
            summary = compressor._generate_verified_summary(_window())

        assert summary is not None
        assert "generation-2" in summary
        assert len(router.summary_prompts) == 2
        assert _FIDELITY_FINDINGS_HEADER not in router.summary_prompts[0]
        assert _FIDELITY_FINDINGS_HEADER in router.summary_prompts[1]
        assert contradiction in router.summary_prompts[1]
        # The rejected candidate must not survive as the iterative-update base.
        assert "generation-1" not in (compressor._previous_summary or "")
        assert "generation-2" in (compressor._previous_summary or "")

    def test_exhausted_retry_budget_fails_closed(self):
        compressor = _compressor()
        router = _Router([_summary_body()], [_reject_verdict()])

        with patch("agent.context_compressor.call_llm", side_effect=router):
            summary = compressor._generate_verified_summary(_window())

        assert summary is None
        # default fidelity_max_retries=1 → one regeneration, then closed.
        assert len(router.summary_prompts) == 2
        assert len(router.verifier_prompts) == 2
        assert FIDELITY_SEMANTIC_REJECTED in compressor._last_fidelity_failure_codes
        assert compressor._previous_summary is None

    def test_configured_retry_count_is_honoured(self):
        compressor = _compressor(fidelity_max_retries=2)
        router = _Router([_summary_body()], [_reject_verdict()])

        with patch("agent.context_compressor.call_llm", side_effect=router):
            summary = compressor._generate_verified_summary(_window())

        assert summary is None
        assert len(router.summary_prompts) == 3

    def test_zero_retries_means_one_generation(self):
        compressor = _compressor(fidelity_max_retries=0)
        router = _Router([_summary_body()], [_reject_verdict()])

        with patch("agent.context_compressor.call_llm", side_effect=router):
            summary = compressor._generate_verified_summary(_window())

        assert summary is None
        assert len(router.summary_prompts) == 1

    def test_malformed_verdict_fails_closed_without_regenerating(self):
        """Rewriting the summary cannot repair a broken verifier."""
        compressor = _compressor()
        router = _Router([_summary_body()], ["not json at all"])

        with patch("agent.context_compressor.call_llm", side_effect=router):
            summary = compressor._generate_verified_summary(_window())

        assert summary is None
        assert len(router.summary_prompts) == 1
        assert FIDELITY_SEMANTIC_MALFORMED in compressor._last_fidelity_failure_codes

    def test_verifier_timeout_fails_closed_without_regenerating(self):
        compressor = _compressor()
        router = _Router([_summary_body()], [RuntimeError("Request timed out.")])

        with patch("agent.context_compressor.call_llm", side_effect=router):
            summary = compressor._generate_verified_summary(_window())

        assert summary is None
        assert len(router.summary_prompts) == 1
        assert FIDELITY_SEMANTIC_UNAVAILABLE in compressor._last_fidelity_failure_codes

    def test_local_rejection_skips_the_verifier_and_regenerates(self):
        compressor = _compressor()
        broken = _summary_body().replace(
            "User asked: 'Approved: proceed with the retry ladder.'", "None."
        )
        # Grounding re-anchors the snapshot, so drop the whole section to make
        # the LOCAL gate the thing that rejects.
        broken = broken.split("## Goal", 1)[1]
        broken = "## Goal" + broken
        router = _Router([broken, _summary_body("generation-2")], [_accept_verdict()])

        with patch("agent.context_compressor.call_llm", side_effect=router):
            summary = compressor._generate_verified_summary(_window())

        assert summary is not None
        assert "generation-2" in summary
        assert len(router.summary_prompts) == 2
        # Only the second candidate ever reached the verifier.
        assert len(router.verifier_prompts) == 1

    def test_generation_failure_never_reaches_the_verifier(self):
        compressor = _compressor()
        router = _Router([RuntimeError("provider exploded")], [_accept_verdict()])

        with patch("agent.context_compressor.call_llm", side_effect=router):
            summary = compressor._generate_verified_summary(_window())

        assert summary is None
        assert router.verifier_prompts == []

    def test_failure_codes_are_content_free(self):
        compressor = _compressor()
        router = _Router(
            [_summary_body()],
            [_reject_verdict("config.yaml was modified per the retry ladder")],
        )

        with patch("agent.context_compressor.call_llm", side_effect=router):
            compressor._generate_verified_summary(_window())

        rendered = repr(compressor._last_fidelity_failure_codes)
        assert "config.yaml" not in rendered
        assert "retry ladder" not in rendered


# ---------------------------------------------------------------------------
# compress() wiring
# ---------------------------------------------------------------------------


def _live_messages() -> list[dict]:
    return [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "Refactor agent/context_compressor.py so compaction keeps the plan."},
        {"role": "assistant", "content": "Working through it."},
        {"role": "user", "content": "Do not modify config.yaml while you work."},
        {"role": "assistant", "content": "Understood."},
        {"role": "user", "content": "Approved: proceed with the retry ladder."},
        {"role": "assistant", "content": "Proceeding with the ladder."},
        {"role": "user", "content": "latest tail request stays protected"},
    ]


def _live_summary_body(marker: str = "generation-1") -> str:
    return "\n\n".join(
        [
            "## Historical Task Snapshot\nUser asked: 'Approved: proceed with the retry ladder.'",
            f"## Goal\nKeep compaction faithful. [{marker}]",
            "## Constraints & Preferences\n- User said do not modify config.yaml while working.",
            "## Completed Actions\n1. READ agent/context_compressor.py — reviewed [tool: read_file]",
            "## Active State\n- Retry ladder not yet written.",
        ]
    )


class TestCompressWiring:
    def test_accepted_summary_matches_guard_disabled_output(self):
        """A verified commit is byte-identical to the unguarded commit.

        Same handoff, same roles, same ordering — so alternation and the
        prompt-cache prefix behave exactly as before the guard existed.
        """
        baseline = _compressor(fidelity_guard=False)
        with patch(
            "agent.context_compressor.call_llm",
            side_effect=_Router([_live_summary_body()], []),
        ):
            expected = baseline.compress(_live_messages())

        guarded = _compressor()
        router = _Router([_live_summary_body()], [_accept_verdict()])
        with patch("agent.context_compressor.call_llm", side_effect=router):
            actual = guarded.compress(_live_messages())

        assert actual == expected
        assert len(router.verifier_prompts) == 1
        assert guarded._last_compress_aborted is False

    def test_rejected_summary_reanchors_even_when_abort_on_summary_failure(self):
        compressor = _compressor(abort_on_summary_failure=True)
        messages = _live_messages()
        router = _Router([_live_summary_body()], [_reject_verdict()])

        with patch("agent.context_compressor.call_llm", side_effect=router):
            result = compressor.compress(messages)

        rendered = "\n".join(str(m.get("content")) for m in result)
        assert result != messages
        assert compressor._last_compress_aborted is False
        assert compressor._last_fidelity_reanchor_used is True
        assert "## Compression Fidelity Re-anchor" in rendered
        assert "generation-1" not in rendered

    def test_rejected_summary_never_commits_without_abort_flag(self):
        compressor = _compressor(abort_on_summary_failure=False)
        router = _Router([_live_summary_body()], [_reject_verdict()])

        with patch("agent.context_compressor.call_llm", side_effect=router):
            result = compressor.compress(_live_messages())

        rendered = "\n".join(str(m.get("content")) for m in result)
        assert "generation-1" not in rendered
        assert compressor._last_summary_fallback_used is True

    def test_default_configuration_makes_no_verifier_call(self):
        compressor = _compressor(fidelity_guard=False)
        with patch(
            "agent.context_compressor.call_llm",
            return_value=_response(_live_summary_body()),
        ) as mock_call:
            compressor.compress(_live_messages())

        assert mock_call.call_count == 1


# ---------------------------------------------------------------------------
# config.yaml resolution
# ---------------------------------------------------------------------------


class TestConfigResolution:
    def test_defaults_are_off_and_one_retry(self):
        assert resolve_fidelity_settings(None) == (False, 1)
        assert resolve_fidelity_settings({}) == (False, 1)

    @pytest.mark.parametrize("raw", [True, "true", "True", "yes", "1", 1])
    def test_truthy_config_values_enable_the_guard(self, raw):
        guard, _ = resolve_fidelity_settings({"fidelity_guard": raw})
        assert guard is True

    @pytest.mark.parametrize("raw", [False, "false", "no", "0", "", None, "maybe"])
    def test_other_values_leave_the_guard_off(self, raw):
        guard, _ = resolve_fidelity_settings({"fidelity_guard": raw})
        assert guard is False

    @pytest.mark.parametrize(
        "raw,expected",
        [
            (0, 0),
            (2, 2),
            ("2", 2),
            (99, FIDELITY_MAX_RETRIES_CAP),
            (-1, 0),
            (None, 1),
            ("many", 1),
            (2.0, 2),
        ],
    )
    def test_retry_count_is_parsed_and_clamped(self, raw, expected):
        _, retries = resolve_fidelity_settings({"fidelity_max_retries": raw})
        assert retries == expected

    def test_environment_variables_are_not_consulted(self, monkeypatch):
        for name in (
            "HERMES_FIDELITY_GUARD",
            "FIDELITY_GUARD",
            "HERMES_COMPRESSION_FIDELITY_GUARD",
            "HERMES_FIDELITY_MAX_RETRIES",
        ):
            monkeypatch.setenv(name, "1")

        assert resolve_fidelity_settings({}) == (False, 1)
        assert _compressor(fidelity_guard=False).fidelity_guard is False

    def test_constructor_accepts_config_yaml_shaped_values(self):
        compressor = _compressor(fidelity_guard="true", fidelity_max_retries="99")

        assert compressor.fidelity_guard is True
        assert compressor.fidelity_max_retries == FIDELITY_MAX_RETRIES_CAP


def test_committed_summary_keeps_the_handoff_prefix():
    """The verified summary is still a normal handoff message."""
    compressor = _compressor()
    router = _Router([_summary_body()], [_accept_verdict()])

    with patch("agent.context_compressor.call_llm", side_effect=router):
        summary = compressor._generate_verified_summary(_window())

    assert summary.startswith(SUMMARY_PREFIX)
