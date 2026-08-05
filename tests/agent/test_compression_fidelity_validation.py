"""Local, deterministic fidelity invariants for a candidate handoff summary.

These checks run before a summary is allowed to become live context. They are
string-level and source-backed: every verdict is derived from the continuity
manifest of the exact compression window, so an omission or a reversal of
critical state fails closed without any model call.
"""

import json

from agent.context_compressor import (
    FIDELITY_FABRICATED_SUCCESS,
    FIDELITY_INVENTED_USER_ATTRIBUTION,
    FIDELITY_MISSING_LATEST_REQUEST,
    FIDELITY_MISSING_SECTIONS,
    FIDELITY_PENDING_STATE_DISTORTED,
    FIDELITY_REVERSED_PROHIBITION,
    SUMMARY_PREFIX,
    ContextCompressor,
)


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


def _sections() -> dict:
    return {
        "## Historical Task Snapshot": (
            "User asked (deterministic, from compacted turns): "
            "'Approved: proceed with the retry ladder.'\n"
            "Historical only; newer protected-tail messages after this summary win."
        ),
        "## Goal": "Keep compaction faithful to the compacted source turns.",
        "## Constraints & Preferences": (
            "- User said do not modify config.yaml while working."
        ),
        "## Completed Actions": (
            "1. TEST `python -m pytest tests/agent/test_widget.py` — "
            "2 failed (test_alpha), 5 passed [tool: terminal]"
        ),
        "## Active State": (
            "- Wire the verdict into conversation_loop.py — still pending, not started."
        ),
        "## Blocked": "- test_alpha still failing.",
    }


def _render(sections: dict) -> str:
    body = "\n\n".join(f"{heading}\n{text}" for heading, text in sections.items())
    return f"{SUMMARY_PREFIX}\n{body}"


def test_faithful_summary_is_accepted():
    verdict = ContextCompressor._validate_summary_fidelity(
        _render(_sections()), _window()
    )

    assert verdict.accepted is True
    assert verdict.failure_codes == ()
    assert verdict.source_ordinals == ()


def test_dropped_latest_user_request_is_rejected():
    sections = _sections()
    sections["## Historical Task Snapshot"] = "None."

    verdict = ContextCompressor._validate_summary_fidelity(
        _render(sections), _window()
    )

    assert verdict.accepted is False
    assert FIDELITY_MISSING_LATEST_REQUEST in verdict.failure_codes
    assert 6 in verdict.source_ordinals


def test_reversed_prohibition_is_rejected():
    sections = _sections()
    sections["## Constraints & Preferences"] = (
        "- User approved modifying config.yaml as needed."
    )

    verdict = ContextCompressor._validate_summary_fidelity(
        _render(sections), _window()
    )

    assert verdict.accepted is False
    assert FIDELITY_REVERSED_PROHIBITION in verdict.failure_codes
    assert 3 in verdict.source_ordinals


def test_pending_work_reported_as_completed_is_rejected():
    sections = _sections()
    sections["## Active State"] = (
        "- Wire the verdict into conversation_loop.py — completed."
    )

    verdict = ContextCompressor._validate_summary_fidelity(
        _render(sections), _window()
    )

    assert verdict.accepted is False
    assert FIDELITY_PENDING_STATE_DISTORTED in verdict.failure_codes
    assert 1 in verdict.source_ordinals


def test_fabricated_test_success_is_rejected():
    sections = _sections()
    sections["## Completed Actions"] = (
        "1. TEST `python -m pytest tests/agent/test_widget.py` — "
        "all 7 passed [tool: terminal]"
    )
    sections["## Blocked"] = "None."

    verdict = ContextCompressor._validate_summary_fidelity(
        _render(sections), _window()
    )

    assert verdict.accepted is False
    assert FIDELITY_FABRICATED_SUCCESS in verdict.failure_codes
    assert 5 in verdict.source_ordinals


def test_invented_user_attribution_is_rejected():
    window = [
        {
            "role": "assistant",
            "content": "Scheduled run starting.",
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {
                        "name": "terminal",
                        "arguments": json.dumps({"command": "ls"}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "docs\nagent"},
    ]
    sections = {
        "## Historical Task Snapshot": "User asked: 'ship the release tonight'",
        "## Goal": "Historical cron objective.",
        "## Completed Actions": "1. Listed the repository root [tool: terminal].",
        "## Active State": "Nothing running.",
    }

    verdict = ContextCompressor._validate_summary_fidelity(_render(sections), window)

    assert verdict.accepted is False
    assert FIDELITY_INVENTED_USER_ATTRIBUTION in verdict.failure_codes


def test_missing_required_sections_fails_closed():
    verdict = ContextCompressor._validate_summary_fidelity(
        f"{SUMMARY_PREFIX}\nEverything went fine.", _window()
    )

    assert verdict.accepted is False
    assert FIDELITY_MISSING_SECTIONS in verdict.failure_codes


def test_verdict_carries_no_source_text():
    sections = _sections()
    sections["## Historical Task Snapshot"] = "None."

    verdict = ContextCompressor._validate_summary_fidelity(
        _render(sections), _window()
    )

    rendered = repr(verdict)
    assert "retry ladder" not in rendered
    assert "config.yaml" not in rendered
    assert all(isinstance(code, str) for code in verdict.failure_codes)
    assert all(isinstance(ordinal, int) for ordinal in verdict.source_ordinals)


def test_manifest_can_be_supplied_by_the_caller():
    window = _window()
    manifest = ContextCompressor._build_continuity_manifest(window)

    verdict = ContextCompressor._validate_summary_fidelity(
        _render(_sections()), window, manifest=manifest
    )

    assert verdict.accepted is True
