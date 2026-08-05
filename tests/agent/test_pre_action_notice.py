"""Unit tests for the pure pre-action-notice validator.

The gate that blocks tool dispatch (``agent/conversation_loop.py``) delegates
its entire "is this notice acceptable?" decision to
``has_valid_pre_action_notice``.  Keeping the predicate pure and separately
tested means the loop test only has to prove the *wiring*, not the parsing.
"""

from __future__ import annotations

import pytest

from agent.pre_action_notice import has_valid_pre_action_notice


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   \n  ",
        # Goal label present but its content is only the time label.
        "執行目標：預估約 1 分鐘。",
        # Goal but no rough-time estimate at all.
        "執行目標：讀取設定。",
        # Estimate but no goal.
        "預估約 1 分鐘。",
        # Both labels present, neither has substance.
        "執行目標：   預估：   ",
        # Time label with nothing after it.
        "執行目標：讀取設定。概估",
        # Generic filler that explains nothing.
        "準備執行工具，請稍候。",
    ],
)
def test_rejects_missing_or_empty_fields(text):
    assert has_valid_pre_action_notice(text) is False


@pytest.mark.parametrize(
    "text",
    [
        "執行目標：讀取設定。預估約 1 分鐘。",
        "執行目標: 執行測試。\n概估時間：2–4 分鐘。",
        "執行目標：讀取目前設定與近期錯誤紀錄，確認服務無法啟動的原因。預估約 2–4 分鐘。",
    ],
)
def test_accepts_goal_and_estimate(text):
    assert has_valid_pre_action_notice(text) is True


@pytest.mark.parametrize(
    "value",
    [None, 123, b"\xe5\x9f\xb7", ["執行目標：讀取設定。預估約 1 分鐘。"], {"content": "x"}],
)
def test_non_string_input_is_rejected(value):
    """Multimodal/None content must never satisfy the gate."""
    assert has_valid_pre_action_notice(value) is False


def test_goal_content_of_only_punctuation_is_not_substantive():
    assert has_valid_pre_action_notice("執行目標：。。。預估約 1 分鐘。") is False


def test_notice_embedded_in_a_longer_turn_is_accepted():
    """The notice may be one line inside a longer visible answer."""
    text = (
        "好的，我先說明接下來的動作。\n"
        "執行目標：檢查 git 狀態並列出未追蹤檔案。\n"
        "預估：少於 1 分鐘。\n"
        "完成後我會回報結果。"
    )
    assert has_valid_pre_action_notice(text) is True
