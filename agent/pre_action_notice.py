"""Pure validator for the plain-language pre-action notice.

When ``agent.require_pre_action_notice`` is on, a turn that carries tool calls
must first tell the user, in Traditional Chinese and in plain language, *what*
it is about to do and *roughly how long* it will take.  The conversation loop
refuses to dispatch the batch until this predicate returns ``True``.

The check is deliberately structural, not semantic: it looks for a goal label
with substance behind it and a rough-time label with substance behind it.  It
never calls a model and never tries to judge whether the stated goal is a good
one — that would make the gate non-deterministic and untestable.
"""

from __future__ import annotations

_GOAL_LABEL = "執行目標"
_TIME_LABELS = ("預估", "概估")

# Stripped from a field's value before asking "is there anything here?".  A
# label followed only by punctuation ("執行目標：。。。") states nothing.
_FILLER = " \t\r\n　:：-—–~～、,，.。!！?？*_`\"'「」『』()（）[]【】"


def _field_value(text: str, start: int) -> str:
    """Return the text following a label, trimmed of separators/punctuation."""
    return text[start:].strip(_FILLER)


def _first_time_label(text: str, start: int = 0) -> int:
    """Index of the earliest rough-time label at or after ``start`` (-1 if none)."""
    hits = [i for i in (text.find(label, start) for label in _TIME_LABELS) if i >= 0]
    return min(hits) if hits else -1


def has_valid_pre_action_notice(content: object) -> bool:
    """Return True only for a substantive goal plus rough-time notice."""
    if not isinstance(content, str):
        return False

    text = content.strip()
    if not text:
        return False

    goal_at = text.find(_GOAL_LABEL)
    if goal_at < 0:
        return False

    time_at = _first_time_label(text)
    if time_at < 0:
        return False

    # The rough-time label must be followed by something (a duration), not sit
    # at the very end of the turn or be trailed only by punctuation.
    time_end = time_at + len(
        next(label for label in _TIME_LABELS if text.startswith(label, time_at))
    )
    if not _field_value(text, time_end):
        return False

    # The goal's value is everything after its label, cut short at the next
    # rough-time label so "執行目標：預估約 1 分鐘。" cannot borrow the estimate
    # as its goal.
    goal_start = goal_at + len(_GOAL_LABEL)
    goal_stop = _first_time_label(text, goal_start)
    goal_text = text[goal_start:goal_stop] if goal_stop >= 0 else text[goal_start:]
    return bool(goal_text.strip(_FILLER))


# ── Gate messaging ────────────────────────────────────────────────────
# The gate discards a non-conforming tool batch, so the model has to be told
# what to do instead. Kept here, next to the predicate, so the wording and the
# rule it enforces cannot drift apart.

# Stage 1 of the two-stage handshake. The original wording asked for a
# notice and the tool calls in the SAME turn — two real
# openai-codex/gpt-5.6-sol sessions showed that provider never emits visible
# content alongside a tool call, even when the user dictates the exact text,
# so that instruction could only ever fail. This asks for the notice alone;
# the tool calls are requested back in stage 2, once the notice is on screen.
PRE_ACTION_NOTICE_NUDGE = (
    "你剛才的回合直接發出工具呼叫，但沒有先向使用者說明。系統已捨棄那批工具呼叫，"
    "沒有執行任何工具。\n"
    "這個回合請「只」輸出一段繁體中文白話預告，不要呼叫任何工具，必須同時包含：\n"
    "1. 「執行目標：」後面接這次要完成的具體事情。\n"
    "2. 「預估」或「概估」後面接大概需要的時間。\n"
    "系統會把這段預告顯示給使用者，然後在下一個回合請你重新發出原本需要的工具呼叫。"
)

# Stage 2. Sent immediately after a qualifying notice-only turn has been
# shown to the user. The notice text itself is already on screen and is
# re-attached to the upcoming tool-call turn for persistence, so the model is
# told not to repeat it.
PRE_ACTION_NOTICE_CONTINUE = (
    "你的預告已經顯示給使用者了。現在請直接發出你原本需要的工具呼叫，"
    "不需要再重複那段預告文字。"
)

PRE_ACTION_NOTICE_STOP = (
    "已停止：代理未提供必要的執行前預告（「執行目標：」與「預估」時間），"
    "因此沒有執行工具。"
)

# Stands in for the discarded turn so the nudge below it has an assistant turn
# to alternate with. Deliberately states that nothing was announced rather
# than inventing a goal — a placebo notice here would let the model believe it
# had already explained itself.
PRE_ACTION_NOTICE_PLACEHOLDER = "（本回合未提供執行前預告，工具呼叫已捨棄。）"


def build_pre_action_notice_scaffolding(content: object) -> tuple[dict, dict]:
    """Return the ephemeral (assistant, user) pair that drives one gate retry.

    The assistant half carries **no** ``tool_calls``: the discarded batch must
    never re-enter the conversation, or the provider would expect tool results
    that will never exist. Both halves are flagged so persistence and
    compression treat them as internal retry state, not transcript.
    """
    visible = content.strip() if isinstance(content, str) else ""
    return (
        {
            "role": "assistant",
            "content": visible or PRE_ACTION_NOTICE_PLACEHOLDER,
            "_pre_action_notice_synthetic": True,
        },
        {
            "role": "user",
            "content": PRE_ACTION_NOTICE_NUDGE,
            "_pre_action_notice_synthetic": True,
        },
    )


def build_pre_action_notice_continuation(notice: str) -> tuple[dict, dict]:
    """Return the ephemeral (notice turn, continue) pair that arms the gate.

    ``notice`` is the model's own qualifying text, already shown to the user.
    It is replayed verbatim so the follow-up request still reads as a normal
    assistant → user exchange, and both halves are flagged: the notice is
    re-attached to the tool-call turn that follows, and persisting it here as
    well would leave two adjacent assistant rows saying the same thing.
    """
    return (
        {
            "role": "assistant",
            "content": notice,
            "_pre_action_notice_synthetic": True,
        },
        {
            "role": "user",
            "content": PRE_ACTION_NOTICE_CONTINUE,
            "_pre_action_notice_synthetic": True,
        },
    )


__all__ = [
    "has_valid_pre_action_notice",
    "build_pre_action_notice_scaffolding",
    "build_pre_action_notice_continuation",
    "PRE_ACTION_NOTICE_NUDGE",
    "PRE_ACTION_NOTICE_CONTINUE",
    "PRE_ACTION_NOTICE_STOP",
    "PRE_ACTION_NOTICE_PLACEHOLDER",
]
