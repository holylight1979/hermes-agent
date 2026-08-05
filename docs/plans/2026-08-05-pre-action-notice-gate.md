# 執行前白話預告強制閘門實作計畫

> **For Hermes:** 由 Claude Code Opus 依此計畫逐項實作；Hermes 負責獨立檢查差異與重新執行測試。

**Goal:** 新增可設定的工具執行前強制閘門；缺少繁中「執行目標＋時間概估」時，不得派送任何工具。

**Architecture:** 在獨立小模組中實作純文字驗證，於 `agent/conversation_loop.py` 的工具派送之前套用。失敗時沿用既有 ephemeral 恢復鷹架，捨棄尚未執行的工具呼叫並重新提示；重試耗盡則停止回合。設定預設關閉，目前 `default` profile 明確開啟。

**Tech Stack:** Python 3.11、pytest、Hermes conversation loop、YAML config、SQLite session persistence。

**Authority:** `docs/superpowers/specs/2026-08-05-pre-action-notice-gate-design.md`

---

## 執行界線

- 不修改 JARVIS 專案。
- 不新增模型工具或工具 schema。
- 不推送遠端。
- 保留既有本機提交與未追蹤 `.claude/`。
- 每個程式行為先建立失敗測試，再做最小修正。
- 若實際程式結構與本計畫不同，立即停止擴張範圍；只允許調整檔案位置，不允許降低「無預告即零工具副作用」的不變量。

## Task 1：建立純預告驗證器

**Objective:** 用可單獨測試的純函式判斷預告是否同時具有實質目標與時間內容。

**Files:**
- Create: `agent/pre_action_notice.py`
- Create: `tests/agent/test_pre_action_notice.py`

**Step 1: 寫入失敗測試**

測試至少涵蓋：

```python
import pytest

from agent.pre_action_notice import has_valid_pre_action_notice


@pytest.mark.parametrize(
    "text",
    [
        "",
        "執行目標：預估約 1 分鐘。",
        "執行目標：讀取設定。",
        "預估約 1 分鐘。",
        "執行目標：   預估：   ",
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
    ],
)
def test_accepts_goal_and_estimate(text):
    assert has_valid_pre_action_notice(text) is True
```

**Step 2: 驗證測試先失敗**

Run:

```bash
python -m pytest tests/agent/test_pre_action_notice.py -o 'addopts=' -q
```

Expected: FAIL，原因為模組或函式尚不存在。

**Step 3: 實作最小驗證器**

介面固定為：

```python
def has_valid_pre_action_notice(content: object) -> bool:
    """Return True only for a substantive goal plus rough-time notice."""
```

實作要求：

- 非字串一律 `False`。
- 接受全形或半形冒號。
- `執行目標` 後必須有非空內容，且內容不能只由時間標籤構成。
- 必須出現 `預估` 或 `概估`，其後必須有非空內容。
- 不嘗試理解任務語意，不呼叫 LLM。

**Step 4: 驗證通過**

Run 同 Step 2。

Expected: 所有測試通過。

**Step 5: 提交**

```bash
git add agent/pre_action_notice.py tests/agent/test_pre_action_notice.py
git commit -m "feat(agent): validate pre-action notices"
```

## Task 2：加入正式設定鍵與代理狀態

**Objective:** 讓功能具有正式、可驗證、預設關閉的設定，並在每個代理實例初始化獨立重試狀態。

**Files:**
- Modify: `hermes_cli/config_defaults.py` 的 `DEFAULT_CONFIG["agent"]`
- Modify: `run_agent.py` 的 `AIAgent` 初始化區
- Test: 現有 config defaults 測試檔；若沒有適合位置，Create: `tests/hermes_cli/test_pre_action_notice_config.py`

**Step 1: 寫入失敗測試**

驗證：

```python
assert DEFAULT_CONFIG["agent"]["require_pre_action_notice"] is False
assert DEFAULT_CONFIG["agent"]["pre_action_notice_max_retries"] == 2
```

並建立最小 agent／測試替身，確認：

```python
assert agent.require_pre_action_notice is False
assert agent.pre_action_notice_max_retries == 2
assert agent._pre_action_notice_retries == 0
```

**Step 2: 執行目標測試並確認失敗**

Run 新測試檔，Expected: 缺少鍵或屬性。

**Step 3: 最小實作**

- 在正式 defaults 中新增兩個設定鍵。
- 使用既有 `CLI_CONFIG`／agent config 載入模式初始化屬性。
- `max_retries` 必須轉為非負整數；異常值回退為 2，不得造成啟動失敗。
- 不使用新環境變數。

**Step 4: 驗證正式鍵**

Run:

```bash
hermes config set agent.require_pre_action_notice true
hermes config set agent.pre_action_notice_max_retries 2
```

Expected: 不再出現「not a recognized config key」警告。

**Step 5: 執行測試並提交**

```bash
python -m pytest tests/hermes_cli/test_pre_action_notice_config.py -o 'addopts=' -q
git add hermes_cli/config_defaults.py run_agent.py tests/hermes_cli/test_pre_action_notice_config.py
git commit -m "feat(config): add pre-action notice gate settings"
```

## Task 3：在工具派送前加入強制閘門

**Objective:** 保證任何不合格工具回合都在副作用之前被攔截。

**Files:**
- Modify: `agent/conversation_loop.py` 工具呼叫分支（目前約 6200–6300 行，必須在 `_execute_tool_calls` 之前）
- Modify: `run_agent.py` 的 `_EPHEMERAL_SCAFFOLDING_FLAGS`
- Modify: `agent/conversation_compression.py` 的 `_SYNTHETIC_USER_FLAGS`
- Create: `tests/run_agent/test_pre_action_notice_gate.py`

**Step 1: 建立失敗測試：空白工具回合不得執行**

沿用 `tests/run_agent/test_dropped_tool_call_recovery.py` 的 mock response／loop agent 夾具。第一個模型回應包含真正的 `terminal` tool call，但 `content=""`；第二個回應包含合格預告與同一工具呼叫。

斷言：

- 第一個工具呼叫從未交給 handler。
- 第二個回合只執行一次。
- API 呼叫兩次。
- 正式 messages 不含第一個未執行的 assistant(tool_calls)。

**Step 2: 建立失敗測試：重試耗盡零副作用**

所有模型回應都提供空白內容加工具呼叫；斷言：

- handler 呼叫次數為 0。
- 回應包含「已停止」與「沒有執行工具」。
- session messages 不含未配對的 tool call。

**Step 3: 建立失敗測試：合格首回合無額外成本**

第一回合即為：

```text
執行目標：執行無害測試命令。預估少於 1 分鐘。
```

斷言 API 只呼叫一次、handler 只呼叫一次。

**Step 4: 實作最小閘門**

邏輯順序：

```python
if assistant_message.tool_calls and agent.require_pre_action_notice:
    if not has_valid_pre_action_notice(final_response):
        # Never dispatch or persist these tool calls.
        # Add ephemeral assistant-without-tools + synthetic user nudge.
        # Retry up to configured limit; otherwise return visible stop result.
```

具體要求：

- 使用新的 `_pre_action_notice_synthetic` 標記。
- 恢復用 assistant 訊息不得攜帶 `tool_calls`。
- user nudge 明確要求先輸出 `執行目標：... 預估...`，再重新發出工具呼叫。
- 透過 `_emit_status` 顯示「尚未執行任何工具」。
- 重試耗盡時建立一般可見 assistant 停止訊息，不呼叫工具。
- 成功進入工具派送後將 `_pre_action_notice_retries` 歸零。
- 不更動既有 dropped-tool-call retry 計數。

**Step 5: ephemeral 與壓縮防污染測試**

斷言：

```python
assert _is_ephemeral_scaffolding(
    {"role": "user", "_pre_action_notice_synthetic": True}
)
```

並確認 `_is_real_user_message` 對同標記回傳 `False`。

**Step 6: 執行目標測試**

```bash
python -m pytest \
  tests/agent/test_pre_action_notice.py \
  tests/run_agent/test_pre_action_notice_gate.py \
  tests/run_agent/test_dropped_tool_call_recovery.py \
  -o 'addopts=' -q
```

Expected: 全部通過。

**Step 7: 提交**

```bash
git add agent/conversation_loop.py run_agent.py agent/conversation_compression.py tests/run_agent/test_pre_action_notice_gate.py
git commit -m "feat(agent): enforce pre-action notices before tools"
```

## Task 4：回歸測試與差異審查

**Objective:** 證明沒有破壞工具派送、角色交替、session persistence 與設定寫入。

**Files:** 不新增功能檔；必要時只修正本功能測試揭露的問題。

**Step 1: 執行相關測試群**

```bash
python -m pytest tests/run_agent/ tests/agent/ tests/hermes_cli/ -o 'addopts=' -q
```

若整組耗時或環境依賴阻塞，至少完整執行：

```bash
python -m pytest \
  tests/run_agent/test_pre_action_notice_gate.py \
  tests/run_agent/test_dropped_tool_call_recovery.py \
  tests/run_agent/test_tool_execution.py \
  tests/agent/test_pre_action_notice.py \
  -o 'addopts=' -q
```

不得把缺少測試檔當作通過；應先定位實際相鄰測試檔。

**Step 2: 靜態檢查**

```bash
python -m compileall agent run_agent.py hermes_cli/config_defaults.py
git diff --check
git status --short
git diff HEAD~3..HEAD --stat
git diff HEAD~3..HEAD
```

**Step 3: 檢查不相關變更**

確認 `.claude/`、JARVIS、使用者其他設定與既有本機提交未被納入。

**Step 4: 修正協定**

- 任何測試失敗先找根因，不放寬不變量。
- 若角色交替或 tool-result 配對失敗，停止並回到 ephemeral 設計，不使用偽造 tool result。
- 若 CLI 看不到有效預告，不能只以 session DB 通過宣稱完成。

## Task 5：啟用目前 profile 並做端到端驗證

**Objective:** 在真實全新 session 中證明可見順序與零副作用失敗路徑。

**Files:**
- Modify through official CLI: `C:\Users\holylight\AppData\Local\hermes\config.yaml`

**Step 1: 啟用設定**

```bash
hermes config set agent.require_pre_action_notice true
hermes config set agent.pre_action_notice_max_retries 2
```

Expected: exit 0 且無未知鍵警告。

**Step 2: 正向端到端測試**

```bash
hermes chat -q '請使用 terminal 工具執行 python -c "print(\"PREACTION_SMOKE_OK\")"，然後簡短回報。' --source pre-action-gate-smoke
```

Expected terminal order:

1. `執行目標：...預估...`
2. 工具執行／輸出 `PREACTION_SMOKE_OK`
3. 最終回報

**Step 3: session DB 順序驗證**

用 `session_search` 或 SQLite 查該 session：

1. assistant content 非空並包含必要標籤，且同一訊息含 tool call。
2. 下一則才是 tool result。
3. 不存在第一版空白、未執行的 tool call。

**Step 4: 負向端到端／整合測試**

使用 deterministic mock provider 或單元整合夾具連續回傳空白工具回合，並使用會寫 sentinel 檔的 handler；斷言 sentinel 檔不存在、handler 次數為 0、停止訊息可見。

**Step 5: 最終提交**

只在有必要的測試／文件修正時提交：

```bash
git add <本功能相關檔案>
git commit -m "test(agent): verify pre-action notice gate end to end"
```

## Task 6：Hermes 獨立驗證與清理

**Objective:** 不依賴 Claude 自述，重新核對實際結果。

1. Hermes 執行 `git status --short` 與完整相關 diff。
2. Hermes 重新跑 Task 4 的測試命令。
3. Hermes 親自跑 Task 5 的全新 session smoke test。
4. Hermes 用 session DB 驗證訊息順序。
5. 刪除暫存腳本 `C:\Users\holylight\hermes-pre-action-config.py`。
6. 保留已驗證的設定備份，除非使用者要求刪除。
7. 不推送遠端。

## 完成回報格式

```text
實際狀態：完成／未完成
程式差異：<實際檔案與提交>
測試：<命令、通過數、失敗數>
端到端：<終端順序與 session DB 證據>
負向保證：<未產生副作用的證據>
未完成或限制：<如實列出>
回復方式：<設定關閉與備份位置>
```
