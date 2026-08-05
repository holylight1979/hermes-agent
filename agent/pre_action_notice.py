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


__all__ = ["has_valid_pre_action_notice"]
