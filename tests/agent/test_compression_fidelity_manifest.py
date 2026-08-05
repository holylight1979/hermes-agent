"""Deterministic continuity manifest derived from the compression window.

The manifest answers "what must survive this compaction?" using only the exact
messages being compacted — no LLM call, no message mutation. Every fact carries
the window ordinal it came from plus a SHA-256 digest of its normalized text so
a later gate can prove a candidate summary still carries it.
"""

import copy
import hashlib
import json

from agent.context_compressor import (
    COMPRESSION_CONTINUATION_USER_CONTENT,
    CONTINUITY_KIND_APPROVAL,
    CONTINUITY_KIND_FILE_PATH,
    CONTINUITY_KIND_PENDING_TODO,
    CONTINUITY_KIND_PROHIBITION,
    CONTINUITY_KIND_TEST_EVIDENCE,
    CONTINUITY_KIND_TOOL_FAILURE,
    CONTINUITY_KIND_USER_DIRECTIVE,
    SUMMARY_PREFIX,
    ContextCompressor,
)
from tools.todo_tool import TODO_INJECTION_HEADER


def _todo_call(call_id: str, todos: list[dict]) -> dict:
    return {
        "role": "assistant",
        "content": "Updating the plan.",
        "tool_calls": [
            {
                "id": call_id,
                "function": {
                    "name": "todo",
                    "arguments": json.dumps({"todos": todos}),
                },
            }
        ],
    }


def _terminal_call(call_id: str, command: str) -> dict:
    return {
        "role": "assistant",
        "content": "Running it.",
        "tool_calls": [
            {
                "id": call_id,
                "function": {
                    "name": "terminal",
                    "arguments": json.dumps({"command": command}),
                },
            }
        ],
    }


def _window() -> list[dict]:
    return [
        {
            "role": "user",
            "content": "Refactor agent/context_compressor.py so compaction keeps the plan.",
        },
        _todo_call(
            "call-todo",
            [
                {"id": "1", "content": "Draft the manifest builder", "status": "completed"},
                {"id": "2", "content": "Sketch the prompt", "status": "cancelled"},
                {
                    "id": "3",
                    "content": "Wire the verdict into conversation_loop.py",
                    "status": "pending",
                },
            ],
        ),
        {"role": "tool", "tool_call_id": "call-todo", "content": "todo list updated"},
        {"role": "user", "content": "Do not modify config.yaml while you work."},
        _terminal_call("call-test", "python -m pytest tests/agent/test_widget.py"),
        {
            "role": "tool",
            "tool_call_id": "call-test",
            "content": "2 failed, 5 passed\nFAILED tests/agent/test_widget.py::test_alpha",
        },
        {"role": "user", "content": "Approved: proceed with the retry ladder."},
    ]


def test_manifest_anchors_latest_user_directive_with_ordinal_and_digest():
    window = _window()

    manifest = ContextCompressor._build_continuity_manifest(window)

    latest = manifest.latest_user_directive()
    assert latest is not None
    assert latest.kind == CONTINUITY_KIND_USER_DIRECTIVE
    assert latest.text == "Approved: proceed with the retry ladder."
    assert latest.ordinal == 6
    assert latest.digest == hashlib.sha256(latest.text.encode("utf-8")).hexdigest()
    assert manifest.source_count == len(window)

    directives = manifest.of_kind(CONTINUITY_KIND_USER_DIRECTIVE)
    assert [fact.ordinal for fact in directives] == [0, 3, 6]


def test_manifest_records_prohibitions_and_approvals():
    manifest = ContextCompressor._build_continuity_manifest(_window())

    prohibitions = manifest.of_kind(CONTINUITY_KIND_PROHIBITION)
    assert [fact.text for fact in prohibitions] == [
        "Do not modify config.yaml while you work."
    ]
    assert prohibitions[0].ordinal == 3

    approvals = manifest.of_kind(CONTINUITY_KIND_APPROVAL)
    assert [fact.text for fact in approvals] == [
        "Approved: proceed with the retry ladder."
    ]
    assert approvals[0].ordinal == 6


def test_manifest_keeps_only_unfinished_todo_items():
    manifest = ContextCompressor._build_continuity_manifest(_window())

    pending = manifest.of_kind(CONTINUITY_KIND_PENDING_TODO)
    assert [fact.text for fact in pending] == [
        "Wire the verdict into conversation_loop.py"
    ]
    assert pending[0].ordinal == 1

    all_text = " ".join(fact.text for fact in manifest.facts)
    assert "Draft the manifest builder" not in all_text
    assert "Sketch the prompt" not in all_text


def test_manifest_uses_the_latest_todo_write():
    window = _window() + [
        _todo_call(
            "call-todo-2",
            [
                {
                    "id": "3",
                    "content": "Wire the verdict into conversation_loop.py",
                    "status": "completed",
                },
                {"id": "4", "content": "Document the fidelity gate", "status": "in_progress"},
            ],
        ),
    ]

    manifest = ContextCompressor._build_continuity_manifest(window)

    pending = manifest.of_kind(CONTINUITY_KIND_PENDING_TODO)
    assert [fact.text for fact in pending] == ["Document the fidelity gate"]
    assert pending[0].ordinal == 7


def test_manifest_records_test_evidence_and_tool_failures():
    window = _window() + [
        _terminal_call("call-build", "npm run build"),
        {
            "role": "tool",
            "tool_call_id": "call-build",
            "content": "Error: ENOENT while reading web/package.json",
        },
    ]

    manifest = ContextCompressor._build_continuity_manifest(window)

    evidence = manifest.of_kind(CONTINUITY_KIND_TEST_EVIDENCE)
    assert len(evidence) == 1
    assert evidence[0].ordinal == 5
    assert "tests/agent/test_widget.py" in evidence[0].text
    assert "2 failed" in evidence[0].text

    failures = manifest.of_kind(CONTINUITY_KIND_TOOL_FAILURE)
    assert len(failures) == 1
    assert failures[0].ordinal == 8
    assert "npm run build" in failures[0].text
    assert "ENOENT" in failures[0].text


def test_manifest_collects_file_paths_from_tool_calls():
    window = [
        {"role": "user", "content": "Check the compressor."},
        {
            "role": "assistant",
            "content": "Reading.",
            "tool_calls": [
                {
                    "id": "call-read",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({"path": "agent/context_compressor.py"}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-read", "content": "ok"},
    ]

    manifest = ContextCompressor._build_continuity_manifest(window)

    paths = manifest.of_kind(CONTINUITY_KIND_FILE_PATH)
    assert "agent/context_compressor.py" in [fact.text for fact in paths]


def test_manifest_paths_are_never_truncated_to_a_leading_separator():
    """A repo-relative path mentioned in prose or a command stays whole."""
    window = [
        {"role": "user", "content": "Refactor agent/context_compressor.py and 24/7 keep it green."},
        _terminal_call("call-test", "python -m pytest tests/agent/test_widget.py"),
        {"role": "tool", "tool_call_id": "call-test", "content": "1 failed"},
    ]

    manifest = ContextCompressor._build_continuity_manifest(window)

    paths = [fact.text for fact in manifest.of_kind(CONTINUITY_KIND_FILE_PATH)]
    assert "agent/context_compressor.py" in paths
    assert "tests/agent/test_widget.py" in paths
    assert not any(path.startswith("/") for path in paths)
    assert "24/7" not in paths


def test_manifest_ignores_synthetic_user_rows():
    window = [
        {
            "role": "user",
            "content": (
                f"{SUMMARY_PREFIX}\n## Historical Task Snapshot\n"
                "User asked: 'ship the old release'"
            ),
        },
        {"role": "user", "content": f"{TODO_INJECTION_HEADER}\n[ ] stale item"},
        {"role": "user", "content": COMPRESSION_CONTINUATION_USER_CONTENT},
        {
            "role": "user",
            "content": "[System: Your previous response was truncated mid-sentence]",
        },
        {
            "role": "user",
            "content": "Do not delete /tmp/data.db",
            "_todo_snapshot_synthetic": True,
        },
        {"role": "assistant", "content": "Continuing."},
    ]

    manifest = ContextCompressor._build_continuity_manifest(window)

    assert manifest.latest_user_directive() is None
    assert manifest.of_kind(CONTINUITY_KIND_USER_DIRECTIVE) == ()
    assert manifest.of_kind(CONTINUITY_KIND_PROHIBITION) == ()


def test_manifest_redacts_secrets_before_digesting():
    secret = "sk-ant-api03-ABCDEFGHIJKLMNOPQRSTUV1234567890"
    window = [
        {"role": "user", "content": f"Deploy with {secret} and do not rotate it yet."},
    ]

    manifest = ContextCompressor._build_continuity_manifest(window)

    assert manifest.facts
    for fact in manifest.facts:
        assert secret not in fact.text
        assert fact.digest == hashlib.sha256(fact.text.encode("utf-8")).hexdigest()


def test_manifest_digest_is_deterministic_and_source_bound():
    window = _window()

    first = ContextCompressor._build_continuity_manifest(window)
    second = ContextCompressor._build_continuity_manifest(_window())
    assert first.digest == second.digest
    assert len(first.digest) == 64

    changed = _window()
    changed[-1] = {"role": "user", "content": "Stop and revert the retry ladder."}
    assert ContextCompressor._build_continuity_manifest(changed).digest != first.digest

    trimmed = ContextCompressor._build_continuity_manifest(window[:-1])
    assert trimmed.digest != first.digest


def test_manifest_build_does_not_mutate_the_window():
    window = _window()
    before = copy.deepcopy(window)

    ContextCompressor._build_continuity_manifest(window)

    assert window == before
