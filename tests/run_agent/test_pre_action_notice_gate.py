"""Pre-action notice gate: no qualifying notice, no tool side effects.

When ``agent.require_pre_action_notice`` is on, a turn that carries tool calls
must first tell the user, in plain Traditional Chinese, what it is about to do
(``執行目標：...``) and roughly how long it will take (``預估``/``概估``). The
conversation loop checks this BEFORE dispatch, so a non-conforming batch is
discarded rather than half-executed.

The invariant under test is absolute: a failing turn reaches the tool handler
zero times. Everything else (the ephemeral re-prompt scaffolding, the retry
bound, the stop message) exists to make that invariant survivable.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent

GOOD_NOTICE = "執行目標：執行無害的測試命令並回報結果。預估少於 1 分鐘。"


def _tool_defs(*names):
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": "test tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def _tool_call(name, call_id):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments="{}"),
    )


def _response(*, content, finish_reason, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _build_agent(*, enabled: bool, max_retries: int = 2):
    """A loop-ready agent with the gate explicitly configured.

    The flags are set on the instance rather than through config.yaml so the
    test never depends on the user's real profile — Task 2 already covers the
    config → attribute resolution.
    """
    with (
        patch("run_agent.get_tool_definitions", return_value=_tool_defs("terminal")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.valid_tool_names = {"terminal"}
    agent.require_pre_action_notice = enabled
    agent.pre_action_notice_max_retries = max_retries
    agent._pre_action_notice_retries = 0
    agent.client = MagicMock()
    return agent


def _run(agent, handler, message="請執行測試命令"):
    with (
        patch("run_agent.handle_function_call", side_effect=handler) as mock_hfc,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation(message)
    return result, mock_hfc


def _capture_interim(agent):
    """Record every text the agent pushes through the interim callback."""
    seen = []

    def _cb(text, already_streamed=False):
        seen.append(text)

    agent.interim_assistant_callback = _cb
    return seen


def _durable(messages):
    """The rows that would survive into the session DB."""
    from run_agent import _is_ephemeral_scaffolding

    return [
        m for m in messages
        if isinstance(m, dict) and not _is_ephemeral_scaffolding(m)
    ]


def _ok_handler(*_args, **_kwargs):
    return "PREACTION_SMOKE_OK"


class TestGateBlocksDispatch:
    def test_blank_turn_never_reaches_the_handler_then_recovers(self):
        """Turn 1: tool calls with no notice. Turn 2: the same call, announced.

        The first batch must be discarded before dispatch, and the unexecuted
        assistant(tool_calls) row must never enter the transcript — an
        assistant tool-call row with no matching tool result is an illegal
        pairing for every provider.
        """
        agent = _build_agent(enabled=True)
        agent.client.chat.completions.create.side_effect = [
            _response(
                content="",
                finish_reason="tool_calls",
                tool_calls=[_tool_call("terminal", "call_blank")],
            ),
            _response(
                content=GOOD_NOTICE,
                finish_reason="tool_calls",
                tool_calls=[_tool_call("terminal", "call_announced")],
            ),
            _response(content="命令已執行完畢。", finish_reason="stop"),
        ]

        result, mock_hfc = _run(agent, _ok_handler)

        assert mock_hfc.call_count == 1, (
            "Only the announced batch may run — the blank turn's tool call "
            "must never reach the handler."
        )
        assert agent.client.chat.completions.create.call_count == 3

        executed_ids = {
            m.get("tool_call_id")
            for m in result["messages"]
            if isinstance(m, dict) and m.get("role") == "tool"
        }
        assert executed_ids == {"call_announced"}

        # No assistant row may carry the discarded call.
        for msg in result["messages"]:
            if not isinstance(msg, dict):
                continue
            for tc in msg.get("tool_calls") or []:
                tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                assert tc_id != "call_blank", (
                    "The unexecuted tool call must not be written back into "
                    "the conversation — it would have no tool result."
                )

    def test_retry_scaffolding_carries_no_tool_calls(self):
        """The ephemeral assistant half of the re-prompt pair must be plain
        text. Re-sending the tool_calls would invite the provider to pair them
        with a result that will never exist.

        Two layers are checked, because they want opposite things. In live
        memory the pair must carry ``_pre_action_notice_synthetic`` so
        persistence and compression can recognise and skip it. On the wire that
        flag must be gone: Hermes' request sanitizer drops every
        ``_``-prefixed key, since strict gateways reject unknown message fields
        outright.
        """
        from run_agent import _is_ephemeral_scaffolding

        agent = _build_agent(enabled=True)
        responses = [
            _response(
                content="",
                finish_reason="tool_calls",
                tool_calls=[_tool_call("terminal", "call_blank")],
            ),
            _response(content="改用文字回答。", finish_reason="stop"),
        ]
        live_snapshots = []

        def _snapshot_then_respond(*_args, **_kwargs):
            live_snapshots.append(
                [dict(m) for m in agent._session_messages if isinstance(m, dict)]
            )
            return responses.pop(0)

        agent.client.chat.completions.create.side_effect = _snapshot_then_respond

        _run(agent, _ok_handler)

        second_call = agent.client.chat.completions.create.call_args_list[1]
        msgs = second_call.kwargs.get("messages") or second_call.args[0].get("messages")

        nudge = msgs[-1]
        assert nudge["role"] == "user"
        assert "執行目標" in nudge["content"]
        assert "預估" in nudge["content"]

        scaffold_assistant = msgs[-2]
        assert scaffold_assistant["role"] == "assistant"
        assert not scaffold_assistant.get("tool_calls"), (
            "The recovery assistant turn must not re-send the discarded calls."
        )

        # Nothing internal may ride along to the provider.
        for msg in msgs:
            if not isinstance(msg, dict):
                continue
            assert not [
                k for k in msg if isinstance(k, str) and k.startswith("_")
            ], f"internal field leaked to the provider: {msg!r}"

        # In memory, though, the very same pair must still be flagged — that is
        # what keeps it out of the durable transcript and out of compression.
        live = live_snapshots[1]
        live_nudge, live_assistant = live[-1], live[-2]
        assert live_nudge["role"] == "user"
        assert live_nudge["content"] == nudge["content"]
        assert live_nudge.get("_pre_action_notice_synthetic") is True
        assert live_assistant["role"] == "assistant"
        assert live_assistant.get("_pre_action_notice_synthetic") is True
        assert not live_assistant.get("tool_calls")
        assert _is_ephemeral_scaffolding(live_nudge)
        assert _is_ephemeral_scaffolding(live_assistant)

    @pytest.mark.parametrize(
        "content",
        [
            "",
            "準備執行工具，請稍候。",
            "執行目標：讀取設定。",
            "預估約 2 分鐘。",
            "執行目標：   預估：   ",
        ],
    )
    def test_incomplete_notices_are_all_rejected(self, content):
        """Missing goal, missing estimate, or label-only text all fail."""
        agent = _build_agent(enabled=True, max_retries=0)
        agent.client.chat.completions.create.side_effect = [
            _response(
                content=content,
                finish_reason="tool_calls",
                tool_calls=[_tool_call("terminal", "call_1")],
            ),
        ]

        result, mock_hfc = _run(agent, _ok_handler)

        assert mock_hfc.call_count == 0
        assert result["completed"] is False

    def test_multi_tool_batch_is_blocked_whole(self):
        """A batch is gated as a unit — no partial execution."""
        agent = _build_agent(enabled=True, max_retries=0)
        agent.client.chat.completions.create.side_effect = [
            _response(
                content="",
                finish_reason="tool_calls",
                tool_calls=[
                    _tool_call("terminal", "call_a"),
                    _tool_call("terminal", "call_b"),
                    _tool_call("terminal", "call_c"),
                ],
            ),
        ]

        result, mock_hfc = _run(agent, _ok_handler)

        assert mock_hfc.call_count == 0, "Not one call of the batch may run."
        assert not [
            m for m in result["messages"]
            if isinstance(m, dict) and m.get("role") == "tool"
        ]


class TestRetryExhaustion:
    def test_exhausted_retries_stop_the_turn_with_zero_side_effects(self, tmp_path):
        """Every turn blank: the handler must be called zero times and the
        user must be told why nothing ran."""
        agent = _build_agent(enabled=True, max_retries=2)
        agent.client.chat.completions.create.side_effect = [
            _response(
                content="",
                finish_reason="tool_calls",
                tool_calls=[_tool_call("terminal", f"call_{i}")],
            )
            for i in range(6)
        ]

        sentinel = tmp_path / "PREACTION_SIDE_EFFECT"

        def _side_effecting_handler(*_args, **_kwargs):
            sentinel.write_text("executed", encoding="utf-8")
            return "ran"

        result, mock_hfc = _run(agent, _side_effecting_handler)

        assert mock_hfc.call_count == 0
        assert not sentinel.exists(), (
            "The gate must stop the turn before any tool touches the disk."
        )
        # 1 initial call + 2 re-prompts, then the turn stops.
        assert agent.client.chat.completions.create.call_count == 3
        assert "已停止" in result["final_response"]
        assert "沒有執行工具" in result["final_response"]
        assert result["completed"] is False

        # No unpaired tool call may survive into the transcript.
        for msg in result["messages"]:
            if isinstance(msg, dict):
                assert not msg.get("tool_calls")

    def test_stop_message_is_the_visible_answer(self):
        """The turn must not present itself as a success."""
        agent = _build_agent(enabled=True, max_retries=0)
        agent.client.chat.completions.create.side_effect = [
            _response(
                content="",
                finish_reason="tool_calls",
                tool_calls=[_tool_call("terminal", "call_1")],
            ),
        ]

        result, _ = _run(agent, _ok_handler)

        assert result["completed"] is False
        assert result.get("error")
        assert "執行前預告" in result["final_response"]

    def test_retry_budget_resets_after_a_qualifying_batch(self):
        """The bound guards each stall, not the whole run: a blank turn, a
        good turn, then another blank turn must still get its own retries."""
        agent = _build_agent(enabled=True, max_retries=1)
        agent.client.chat.completions.create.side_effect = [
            _response(
                content="",
                finish_reason="tool_calls",
                tool_calls=[_tool_call("terminal", "blank_1")],
            ),
            _response(
                content=GOOD_NOTICE,
                finish_reason="tool_calls",
                tool_calls=[_tool_call("terminal", "good_1")],
            ),
            _response(
                content="",
                finish_reason="tool_calls",
                tool_calls=[_tool_call("terminal", "blank_2")],
            ),
            _response(
                content=GOOD_NOTICE,
                finish_reason="tool_calls",
                tool_calls=[_tool_call("terminal", "good_2")],
            ),
            _response(content="兩個步驟都完成了。", finish_reason="stop"),
        ]

        result, mock_hfc = _run(agent, _ok_handler)

        assert mock_hfc.call_count == 2
        assert result["final_response"] == "兩個步驟都完成了。"
        assert agent._pre_action_notice_retries == 0


class TestNoCostWhenCompliantOrDisabled:
    def test_qualifying_first_turn_costs_no_extra_round_trip(self):
        agent = _build_agent(enabled=True)
        agent.client.chat.completions.create.side_effect = [
            _response(
                content=GOOD_NOTICE,
                finish_reason="tool_calls",
                tool_calls=[_tool_call("terminal", "call_1")],
            ),
            _response(content="完成。", finish_reason="stop"),
        ]

        result, mock_hfc = _run(agent, _ok_handler)

        assert mock_hfc.call_count == 1
        assert agent.client.chat.completions.create.call_count == 2, (
            "A compliant turn must not pay for a gate re-prompt."
        )
        assert result["final_response"] == "完成。"

    def test_disabled_gate_preserves_todays_behaviour(self):
        """Gate off: a blank tool turn executes exactly as it does today."""
        agent = _build_agent(enabled=False)
        agent.client.chat.completions.create.side_effect = [
            _response(
                content="",
                finish_reason="tool_calls",
                tool_calls=[_tool_call("terminal", "call_1")],
            ),
            _response(content="done", finish_reason="stop"),
        ]

        result, mock_hfc = _run(agent, _ok_handler)

        assert mock_hfc.call_count == 1
        assert agent.client.chat.completions.create.call_count == 2
        assert result["final_response"] == "done"


class TestScaffoldingIsEphemeral:
    def test_builder_flags_both_halves_and_carries_no_tool_calls(self):
        """The flag is applied where the pair is built, so no caller can
        produce an unflagged — and therefore persistable — recovery pair."""
        from agent.pre_action_notice import build_pre_action_notice_scaffolding
        from run_agent import _is_ephemeral_scaffolding

        scaffold_assistant, nudge = build_pre_action_notice_scaffolding("")

        assert scaffold_assistant["role"] == "assistant"
        assert scaffold_assistant["_pre_action_notice_synthetic"] is True
        assert "tool_calls" not in scaffold_assistant
        assert nudge["role"] == "user"
        assert nudge["_pre_action_notice_synthetic"] is True
        assert _is_ephemeral_scaffolding(scaffold_assistant)
        assert _is_ephemeral_scaffolding(nudge)

    def test_flag_is_classified_as_ephemeral_scaffolding(self):
        from run_agent import _EPHEMERAL_SCAFFOLDING_FLAGS, _is_ephemeral_scaffolding

        assert "_pre_action_notice_synthetic" in _EPHEMERAL_SCAFFOLDING_FLAGS
        assert _is_ephemeral_scaffolding(
            {"role": "user", "content": "nudge", "_pre_action_notice_synthetic": True}
        )
        assert _is_ephemeral_scaffolding(
            {
                "role": "assistant",
                "content": "text",
                "_pre_action_notice_synthetic": True,
            }
        )
        assert not _is_ephemeral_scaffolding({"role": "user", "content": "hi"})

    def test_nudge_is_not_a_real_user_message_for_compression(self):
        from agent.conversation_compression import (
            _SYNTHETIC_USER_FLAGS,
            _is_real_user_message,
        )

        assert "_pre_action_notice_synthetic" in _SYNTHETIC_USER_FLAGS
        assert not _is_real_user_message(
            {
                "role": "user",
                "content": "請先輸出執行目標與預估時間",
                "_pre_action_notice_synthetic": True,
            }
        )
        assert _is_real_user_message({"role": "user", "content": "請執行測試命令"})

    def test_recovered_pair_is_flagged_so_persistence_skips_it(self):
        """A recovered pair stays buried mid-list in live memory — removing it
        would change the prompt prefix and throw away the conversation's cache
        — but every surviving piece must carry the flag, so both the SQLite
        flush and the JSON log skip it regardless of position."""
        from run_agent import _is_ephemeral_scaffolding

        agent = _build_agent(enabled=True)
        agent.client.chat.completions.create.side_effect = [
            _response(
                content="",
                finish_reason="tool_calls",
                tool_calls=[_tool_call("terminal", "call_blank")],
            ),
            _response(
                content=GOOD_NOTICE,
                finish_reason="tool_calls",
                tool_calls=[_tool_call("terminal", "call_announced")],
            ),
            _response(content="完成。", finish_reason="stop"),
        ]

        result, _ = _run(agent, _ok_handler)

        durable = [
            m for m in result["messages"]
            if isinstance(m, dict) and not _is_ephemeral_scaffolding(m)
        ]
        assert not [m for m in durable if m.get("_pre_action_notice_synthetic")]
        # The nudge text must not reach the durable transcript under any role.
        assert not [
            m for m in durable
            if isinstance(m.get("content"), str) and "重新發出" in m["content"]
        ]

    def test_invalid_notice_exhaustion_strips_trailing_scaffolding(self):
        """Invalid notice-only replies consume the bounded retry budget; once
        exhausted the turn stops cleanly and no synthetic pair is persisted."""
        agent = _build_agent(enabled=True)
        agent.client.chat.completions.create.side_effect = [
            _response(
                content="",
                finish_reason="tool_calls",
                tool_calls=[_tool_call("terminal", "call_blank")],
            ),
            _response(content="改用文字回答。", finish_reason="stop"),
            _response(content="仍然沒有合格預告。", finish_reason="stop"),
        ]

        result, mock_hfc = _run(agent, _ok_handler)

        assert mock_hfc.call_count == 0
        assert result["completed"] is False
        assert "已停止" in result["final_response"]
        assert not [
            m for m in result["messages"]
            if isinstance(m, dict) and m.get("_pre_action_notice_synthetic")
        ]


# ── Two-stage notice (2026-08-05 E2E revision) ────────────────────────
# Two real openai-codex/gpt-5.6-sol sessions proved the provider will not
# put visible content in the same turn as a tool call, even when the user
# asks for it verbatim. The single-stage gate above then correctly stops
# every turn — correct, but unusable. The revision splits the requirement
# in two: the model says what it is about to do in a text-only turn, Hermes
# shows that text, and the *next* tool batch is authorised by it.
#
# What must not weaken: the batch still reaches the handler only after the
# user has seen a model-authored notice.

STAGE_TWO_NOTICE = "執行目標：執行 sentinel 命令並回報輸出。預估約 30 秒。"


def _two_stage_responses(*, tail="完成。"):
    """blank+tools → notice-only text → blank+tools → final answer."""
    return [
        _response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[_tool_call("terminal", "call_blank")],
        ),
        _response(content=STAGE_TWO_NOTICE, finish_reason="stop"),
        _response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[_tool_call("terminal", "call_armed")],
        ),
        _response(content=tail, finish_reason="stop"),
    ]


class TestTwoStageNotice:
    def test_nudge_asks_for_a_text_only_notice(self):
        """Stage 1's nudge must ask for the notice ALONE.

        The old wording asked for "notice + tool calls in the same turn" —
        exactly what the provider demonstrably cannot do. Asking for a
        text-only turn is the whole point of the revision.
        """
        from agent.pre_action_notice import PRE_ACTION_NOTICE_NUDGE

        assert "執行目標" in PRE_ACTION_NOTICE_NUDGE
        assert "預估" in PRE_ACTION_NOTICE_NUDGE
        assert "不要呼叫" in PRE_ACTION_NOTICE_NUDGE, (
            "The nudge must forbid tool calls in the notice turn."
        )

    def test_notice_only_turn_is_shown_but_runs_no_tools(self):
        """The qualifying text-only turn reaches the user immediately, and
        still nothing has executed at that point."""
        agent = _build_agent(enabled=True)
        seen = _capture_interim(agent)
        seen_when_dispatched = []

        agent.client.chat.completions.create.side_effect = _two_stage_responses()

        def _handler(*_args, **_kwargs):
            seen_when_dispatched.append(list(seen))
            return "PREACTION_GATE_E2E_OK"

        _result, mock_hfc = _run(agent, _handler)

        assert mock_hfc.call_count == 1
        assert seen_when_dispatched[0] == [STAGE_TWO_NOTICE], (
            "The notice must already be on screen when the first tool runs."
        )

    def test_armed_notice_authorizes_the_next_blank_tool_batch(self):
        agent = _build_agent(enabled=True)
        agent.client.chat.completions.create.side_effect = _two_stage_responses()

        result, mock_hfc = _run(agent, _ok_handler)

        assert mock_hfc.call_count == 1, (
            "The armed notice must let exactly one batch through."
        )
        executed = {
            m.get("tool_call_id")
            for m in result["messages"]
            if isinstance(m, dict) and m.get("role") == "tool"
        }
        assert executed == {"call_armed"}
        assert result["final_response"] == "完成。"

    def test_durable_transcript_is_one_assistant_then_tool_result(self):
        """The session DB must end up with the shape a provider accepts on
        resume: assistant(content=the model's own notice, tool_calls=...)
        immediately followed by its tool result — no notice-only assistant,
        no synthetic continue, no adjacent assistant pair."""
        agent = _build_agent(enabled=True)
        agent.client.chat.completions.create.side_effect = _two_stage_responses()

        result, _ = _run(agent, _ok_handler)

        durable = _durable(result["messages"])
        tool_turns = [m for m in durable if m.get("tool_calls")]
        assert len(tool_turns) == 1
        assert tool_turns[0]["role"] == "assistant"
        assert tool_turns[0]["content"] == STAGE_TWO_NOTICE, (
            "The persisted tool-call turn must carry the model's own notice, "
            "not a Hermes-authored stand-in."
        )

        idx = durable.index(tool_turns[0])
        assert durable[idx + 1]["role"] == "tool"

        # No standalone notice-only assistant and no hidden continue survive.
        assert not [
            m for m in durable
            if m.get("role") == "assistant"
            and not m.get("tool_calls")
            and m.get("content") == STAGE_TWO_NOTICE
        ]
        roles = [m.get("role") for m in durable]
        assert not any(
            roles[i] == "assistant" and roles[i + 1] == "assistant"
            for i in range(len(roles) - 1)
        ), f"adjacent assistant messages in the durable transcript: {roles}"

    def test_notice_is_displayed_exactly_once(self):
        """Re-attaching the notice to the persisted tool-call turn must not
        make the CLI/gateway print it a second time."""
        agent = _build_agent(enabled=True)
        seen = _capture_interim(agent)
        agent.client.chat.completions.create.side_effect = _two_stage_responses()

        _run(agent, _ok_handler)

        assert seen.count(STAGE_TWO_NOTICE) == 1, seen


class TestArmedStateFailsClosed:
    def test_armed_state_is_consumed_by_a_single_batch(self):
        """A second blank batch after the authorised one is gated again."""
        agent = _build_agent(enabled=True, max_retries=1)
        agent.client.chat.completions.create.side_effect = [
            *_two_stage_responses()[:3],
            _response(
                content="",
                finish_reason="tool_calls",
                tool_calls=[_tool_call("terminal", "call_second")],
            ),
            _response(content="仍然沒有合格預告。", finish_reason="stop"),
        ]

        result, mock_hfc = _run(agent, _ok_handler)

        assert mock_hfc.call_count == 1, (
            "The stashed notice must not authorise a second batch."
        )
        assert result["completed"] is False
        assert "已停止" in result["final_response"]

    def test_state_is_cleared_before_the_batch_is_dispatched(self):
        """Cleared *before* dispatch, so a tool that crashes, restarts Hermes
        or re-enters the loop cannot find a live authorisation."""
        agent = _build_agent(enabled=True)
        observed = {}

        agent.client.chat.completions.create.side_effect = _two_stage_responses()

        def _handler(*_args, **_kwargs):
            observed["phase"] = agent._pre_action_notice_phase
            observed["text"] = agent._pre_action_notice_text
            return "ok"

        _run(agent, _handler)

        assert observed["phase"] == "idle"
        assert observed["text"] is None

    def test_ordinary_text_answer_clears_the_armed_notice(self):
        agent = _build_agent(enabled=True)
        agent.client.chat.completions.create.side_effect = [
            _response(
                content="",
                finish_reason="tool_calls",
                tool_calls=[_tool_call("terminal", "call_blank")],
            ),
            _response(content=STAGE_TWO_NOTICE, finish_reason="stop"),
            _response(content="其實不需要工具，直接回答你。", finish_reason="stop"),
        ]

        result, mock_hfc = _run(agent, _ok_handler)

        assert mock_hfc.call_count == 0
        assert result["final_response"] == "其實不需要工具，直接回答你。"
        assert agent._pre_action_notice_phase == "idle"
        assert agent._pre_action_notice_text is None

    def test_retry_exhaustion_clears_the_armed_notice(self):
        agent = _build_agent(enabled=True, max_retries=1)
        agent.client.chat.completions.create.side_effect = [
            _response(
                content="",
                finish_reason="tool_calls",
                tool_calls=[_tool_call("terminal", "call_blank")],
            ),
            _response(content=STAGE_TWO_NOTICE, finish_reason="stop"),
            _response(
                content="",
                finish_reason="tool_calls",
                tool_calls=[_tool_call("terminal", "call_armed")],
            ),
            # Armed batch ran; this one has to earn its own notice.
            _response(
                content="",
                finish_reason="tool_calls",
                tool_calls=[_tool_call("terminal", "call_x")],
            ),
            _response(content="仍然沒有預告。", finish_reason="stop"),
            _response(
                content="",
                finish_reason="tool_calls",
                tool_calls=[_tool_call("terminal", "call_y")],
            ),
        ]

        result, mock_hfc = _run(agent, _ok_handler)

        assert mock_hfc.call_count == 1
        assert result["completed"] is False
        assert agent._pre_action_notice_phase == "idle"
        assert agent._pre_action_notice_text is None

    def test_a_new_user_turn_does_not_inherit_an_armed_notice(self):
        """State left over from a previous turn (gateway agents are cached,
        and a crash can leave the flag set) must not authorise anything."""
        agent = _build_agent(enabled=True, max_retries=0)
        agent._pre_action_notice_phase = "armed"
        agent._pre_action_notice_text = STAGE_TWO_NOTICE
        agent.client.chat.completions.create.side_effect = [
            _response(
                content="",
                finish_reason="tool_calls",
                tool_calls=[_tool_call("terminal", "call_stale")],
            ),
        ]

        result, mock_hfc = _run(agent, _ok_handler)

        assert mock_hfc.call_count == 0, (
            "A stale notice from an earlier turn must never authorise tools."
        )
        assert result["completed"] is False

    def test_notice_state_starts_idle_from_agent_init(self):
        agent = _build_agent(enabled=False)

        assert agent._pre_action_notice_phase == "idle"
        assert agent._pre_action_notice_text is None


class TestTwoStageScaffoldingIsEphemeral:
    def test_notice_turn_and_continue_are_flagged_and_stripped(self):
        from agent.pre_action_notice import build_pre_action_notice_continuation
        from run_agent import _is_ephemeral_scaffolding

        notice_msg, continue_msg = build_pre_action_notice_continuation(
            STAGE_TWO_NOTICE
        )

        assert notice_msg["role"] == "assistant"
        assert notice_msg["content"] == STAGE_TWO_NOTICE
        assert "tool_calls" not in notice_msg
        assert continue_msg["role"] == "user"
        assert _is_ephemeral_scaffolding(notice_msg)
        assert _is_ephemeral_scaffolding(continue_msg)

    def test_internal_fields_never_reach_the_provider(self):
        """The two-stage scaffolding rides in live memory only; strict
        gateways reject unknown message fields."""
        agent = _build_agent(enabled=True)
        agent.client.chat.completions.create.side_effect = _two_stage_responses()

        _run(agent, _ok_handler)

        for call in agent.client.chat.completions.create.call_args_list:
            msgs = call.kwargs.get("messages") or call.args[0].get("messages")
            for msg in msgs:
                if not isinstance(msg, dict):
                    continue
                assert not [
                    k for k in msg if isinstance(k, str) and k.startswith("_")
                ], f"internal field leaked to the provider: {msg!r}"

    def test_armed_request_keeps_role_alternation(self):
        """The request that carries the tool batch must still alternate:
        ... assistant(notice) → user(continue)."""
        agent = _build_agent(enabled=True)
        agent.client.chat.completions.create.side_effect = _two_stage_responses()

        _run(agent, _ok_handler)

        third = agent.client.chat.completions.create.call_args_list[2]
        msgs = third.kwargs.get("messages") or third.args[0].get("messages")

        assert msgs[-1]["role"] == "user"
        assert msgs[-2]["role"] == "assistant"
        assert msgs[-2]["content"] == STAGE_TWO_NOTICE
        assert not msgs[-2].get("tool_calls")
        roles = [m["role"] for m in msgs if isinstance(m, dict)]
        assert not any(
            roles[i] == roles[i + 1] == "assistant" for i in range(len(roles) - 1)
        ), roles
