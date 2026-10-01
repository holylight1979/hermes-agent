# llm-cr bundle — 可重複套用的安裝包

把「精確純文字別名 → 原生 session 繞道（detour）→ 路由限定的 prompt 注入」這一整套功能，
打包成一份可以重複套用、可驗證、可回滾的安裝包。

本安裝包**不內建任何 provider、model、endpoint 或指令檔內容**。這些全部必須由你在命令列上
明確給出（base URL 也可以從目標 config 既有的 `providers` 區段讀取）。

---

## 1. 這個 bundle 做了什麼

三層，彼此獨立但互相配合：

三層功能，全部由 **HOME 底下的三個 plugin** 提供；core patch 只留下它們需要的通用接縫：

| 層 | 內容 | 安裝位置 |
|---|---|---|
| 精確文字別名 | 整句（去除首尾空白後）完全等於 `llm-cr` / `llm-cr-end` 時，在任何 LLM 看到訊息**之前**改寫成對應的 slash command | `text-command-aliases` plugin（Gateway 端）＋ core patch（共用 matcher 與 CLI 入口） |
| 原生 session 繞道 | `/detour` 開一個全新的子 session（不帶入母 session 的歷史），`/detour-end` 回到母 session 並還原它自己的路由 | **`session-detours` plugin**（回程記錄、CLI leg、Gateway leg 全在 plugin 內） |
| Prompt 注入 | 只有「這一個請求真的要送往設定的 provider + model + base URL + api_mode」時，才把本機指令檔附加到送出請求**副本**的第一條 system message | `llm-cr-prompt` plugin（`llm_execution` middleware） |

2.x 的核心決定是**「功能全在 plugin，core 只有通用接縫」**：`/detour` 不再是 core 的內建指令，
而是 `session-detours` plugin 用 `ctx.register_command()` 註冊的 plugin command。沒有啟用這個
plugin，`/detour` 就**完全不存在**（補完清單裡沒有、Gateway dispatch 不認、CLI 也不認），
而所有原生指令 handler 一行都沒被改。

core patch 為此加入的接縫都是**與 detour 無關、任何 plugin 都能用**的通用能力（見第 6 節）：
plugin command 可以拿到呼叫端的 host context、可以宣告 `busy_policy="reject"`、execution
middleware 可以 fail-closed 中止。

幾個刻意的設計決定：

* 改寫出來的就是一個普通的 slash command。授權、各平台的 slash 權限閘、破壞性確認流程、busy policy
  都在之後照常執行。別名不會繞過任何一道關卡。
* `/detour` 與 `/detour-end` 由**原生** handler 組成（Gateway 的 `_handle_reset_command` /
  `_handle_resume_command`，CLI 的 `new_session` / `_handle_resume_command`），不是另寫一套 session 輪替。
  plugin 透過 host context 拿到活著的 `HermesCLI` / `GatewayRunner`，再用**每個 host 一個 adapter**
  把原本的 mixin 綁上去 —— 不 monkey-patch 任何 host class，也不複製 transcript。
* `/detour` 與 `/detour-end` 宣告 `busy_policy="reject"`：它們會輪替 session，所以跟會輪替的內建指令
  （`/model`、`/resume`）一樣，在 agent 執行中直接拒絕，而不是打斷那一回合。
* prompt 注入採「路由即身分」：沒有第二個 mode 旗標。唯一的判斷是「這個請求是否正要送往設定的那條路由」。
  因此手動用 `/model` 選到同一條路由，會得到同一個 persona —— 這是設計，不是漏洞。
* 注入是 fail-closed：路由命中但指令檔遺失／空白／無法讀取／非 UTF-8，會在**任何網路呼叫之前**
  中止請求（`MiddlewareAbort`）。放行才是錯的 —— 那等於安靜地送出一個沒有 prompt 的請求。
* 這是**對話隔離，不是沙箱**。子 session 的工具、權限、檔案系統可及範圍與任何其他 session 相同。

---

## 2. 前置條件

* 目標是一個 **hermes-agent 的 git checkout**，且 `core.patch` 能乾淨套用（見第 6 節的基準 commit）。
* 一個**已初始化的 Hermes home / profile**（裡面要有 `config.yaml`）。
* Python 3.11+。安裝程式本身只用標準庫，**唯一的外部依賴是 PyYAML**（config 合併用）。
  找不到時會明確報錯，不會丟 ImportError traceback。
* 若要啟用 prompt 注入 plugin（`--enable-prompt-plugin`）：指令檔必須**已經存在**。
  安裝程式只檢查它存在，**永遠不讀、不寫、不建立**它。
  預設路徑為 `<home>/skills/productivity/llm-crack-talk/llm-cr-instruction.md`，
  也可以用 `--instruction-path` 指定別的絕對路徑（只當 metadata 使用）。

---

## 3. 指令

四個子命令。`--repo` 與 `--home` **一律必填且明確**，所以絕不會誤觸其他 profile。

```bash
# 只看計畫，不寫任何東西（dry-run 是別名）
python install.py check \
  --repo /path/to/hermes-agent \
  --home /path/to/hermes-home \
  --provider <cr-provider> --model <cr-model> \
  --return-provider <exit-provider> --return-model <exit-model>

# 實際套用
python install.py apply   ... （同上參數）

# 驗證（patch 在位、別名精確、plugin 真的被 Hermes 載入並註冊 seam）
python install.py verify  ... （同上參數）

# 回滾單一次 apply
python install.py rollback --receipt <backup-dir>/<run-id>/receipt.json \
  --repo /path/to/hermes-agent --home /path/to/hermes-home
```

主要參數：

| 參數 | 說明 |
|---|---|
| `--provider` / `--model` | CR 路由的 provider 名稱與 model id（必填，無預設） |
| `--return-provider` / `--return-model` | `llm-cr-end` 要還原成哪條路由（必填，無預設） |
| `--cr-base-url` | CR endpoint。省略時從目標 config 的 `providers.<provider>.base_url` 讀取；讀不到就**拒絕**，不會猜 |
| `--api-mode` | CR 路由的 api_mode（預設 `chat_completions`） |
| `--enable-prompt-plugin` | 啟用 `llm_execution` prompt 注入（需要指令檔已存在） |
| `--instruction-path` | 明確指定指令檔路徑（只檢查存在性） |
| `--no-aliases` | 只裝 core + payload，不啟用文字別名 |
| `--no-detour-plugin` | 裝進 `session-detours` payload 但**不啟用**它；`/detour` 與 `/detour-end` 於是在任何介面上都不存在（`verify` 會反向確認「沒有任何東西註冊這兩個指令」） |
| `--backup-dir` | 備份目錄。預設 `<home>/../llm-cr-bundle-backups`，且**必須在 repo 與 home 之外** |

退出碼：`0` 成功 / `1` 驗證失敗 / `2` 拒絕執行（fail-closed，未寫入任何東西）。

---

## 4. 安全性與可逆性保證

**Fail closed —— 下列任一情況會在寫入任何檔案之前中止，退出碼 2：**

* bundle 自身的 checksum 與 manifest 不符（patch 或 payload 被動過）。
* `core.patch` 既無法正向套用、也無法反向套用 —— 代表目標已經偏離，而且不是「已安裝」狀態。
  core 原始碼**絕不會**被整檔覆寫：唯一的寫入者是 `git apply`，它放不下的 hunk 就會自己拒絕。
* 目標 repo 少了任何一個本 bundle 程式碼實際呼叫的 API（manifest 的 `required_apis`，每一條都附上理由）。
* 必填的 provider / model 參數缺漏，或 base URL 無法解析。
* 啟用 prompt plugin 但指令檔不存在。
* `--backup-dir` 落在 repo 或 home 之內；或 home 位於 repo 內、兩者相同，或實際寫入路徑互相重疊。
  支援標準的 `<home>/hermes-agent` 目錄結構（repo 位於 home 內）。

**可重複套用（idempotent）：** 對已安裝的目標再跑一次 `apply`，什麼都不會改，並明確告知。
不會產生第二份備份，也不會寫第二張 receipt。

**備份：** 在第一次寫入**之前**，把這次可能動到的每一個路徑都複製到備份目錄，
並記錄它**當下的確切存在狀態、sha256 與權限模式**（不存在的也會記錄為「不存在」）。
receipt 同時記錄實際做了哪些 config 變更。

**回滾：** 只還原自己造成的變更。
* 任何一個被記錄的檔案在 apply 之後被別人改過 → **整個回滾拒絕執行，一個字都不寫**。
* receipt 被拿去對不同的 repo/home 重播 → 拒絕（scope guard）。
* 沒有 `git reset`、沒有 `git clean`、沒有用模板整檔取代 config。

**秘密：** 從不寫入、從不複製進 receipt、從不印到 stdout。
config 合併只會設定本 bundle 自己的 key，既有的 `providers`（含其憑證）原封不動。

**不自動重啟任何服務。** apply 完成後只會把需要做的重啟步驟印出來。

---

## 5. 安裝後必須自己做的事

1. **重啟 Gateway** —— 否則不會載入新的 plugin 與 core 程式碼。
2. **重啟任何已在執行的互動式 CLI** —— 執行中的 CLI 無法被 `/reload` 熱修補，需要全新的 process。
   重啟 Gateway **不會**順帶重啟 CLI。
3. 跑 `verify`，再跑測試（見第 7 節）。

---

## 6. 內容與基準 commit

```
install.py        安裝程式（標準庫 + PyYAML）
build.py          重新產生 manifest.json 與可重現的 ZIP
manifest.json     基準 commit、所有 checksum、API 前置檢查表、測試清單
core.patch        core 變更（git apply 相容，含 --reverse --check）
payload/          要裝到 <hermes-home> 底下的三個 plugin
                    plugins/text-command-aliases/   Gateway 端別名改寫
                    plugins/session-detours/        /detour 與 /detour-end（含回程記錄與兩邊 surface）
                    plugins/llm-cr-prompt/          路由限定的 prompt 注入
tests/            安裝程式自己的測試
docs/SKILL.md     去識別化的技能說明文件（僅供參考，不會被安裝）
```

`core.patch` 是針對 manifest 裡的 `base_commit` 產生的，涵蓋 **17 個路徑：7 個 core 檔案 +
10 個測試檔案**，其中 11 個是新檔案（1 個 core 新模組 + 10 個測試）。

### core patch 實際改了哪些接縫

這 7 個 core 檔案裡，與 llm-cr **無關**的通用能力共 4 項 —— 它們是「讓 plugin 有能力做這件事」，
不是「把這件事寫進 core」：

| 檔案 | 改動 | 為什麼是通用的 |
|---|---|---|
| `hermes_cli/plugins.py` | `bind_plugin_command_context()`：handler 若宣告 `context` 參數，就收到 `{surface, command, host, session_key, event, source}`；`register_command(..., busy_policy=...)` 與 `get_plugin_command_busy_policy()` | 任何需要組合原生 session 行為（開 session、resume、換路由）的 plugin command 都適用。沒宣告 `context` 的既有 handler 照舊呼叫，完全不受影響 |
| `cli.py` | `_run_plugin_slash_command` 綁上 CLI 的 host context；`_tui_process_one_input` 前置呼叫別名 matcher | 同上；別名入口是 CLI 這一側的 matcher 掛載點 |
| `gateway/run_inbound.py` | plugin command dispatch 綁上 Gateway 的 host context、**用正規化後的名稱**再過一次 slash 權限閘、busy 快路徑認得 `busy_policy="reject"` 的 plugin command | 正規化名稱的閘門修掉的是通用漏洞：`/detour_end` 這種底線拼法原本會繞過以輸入原文為準的冷路徑檢查 |
| `gateway/run_busy.py` | 把內建的 busy 拒絕用語抽成 `_busy_reject_text()` | 讓 plugin command 的拒絕**一字不差**等於內建指令的拒絕，而不是另寫一句 |
| `hermes_cli/middleware.py` | 新增 `MiddlewareAbort`，execution chain 遇到它直接往上拋 | execution middleware 原本 fail-open；對「工作就是改 payload」的 callback 來說，跳過它等於安靜送出另一個請求 |
| `hermes_cli/text_command_aliases.py` | 新檔案：共用的整句比對器（大小寫敏感、只去首尾空白） | 讓每個 surface 用同一條規則 |
| `hermes_cli/config.py` | 把 `text_command_aliases` 加進 `_OPEN_DICT_TOP_LEVEL_KEYS` | config 驗證才會接受這個區段 |

**v1 → v2 搬走了什麼：** v1 把 detour 寫在 core（`hermes_cli/session_detour.py`、
`hermes_cli/cli_detour_mixin.py`、`gateway/slash_commands_detour.py`，加上 `hermes_cli/commands.py`
與 `gateway/slash_commands.py` 的指令註冊）。2.x 把這些整批移到 `payload/plugins/session-detours/`
（`detour_records.py` / `cli_surface.py` / `gateway_surface.py`），core 的指令註冊表因此**一行都沒動**。
`verify` 會主動檢查這件事：若目標的內建指令表裡還找得到 `/detour`，就是舊版殘留，直接判失敗。

**關於依賴的實話：** 原始開發分支裡有不少與 llm-cr 無關的改動
（session hygiene、process registry、Windows 暫存檔修正、Discord/LINE adapter、model library 路由等），
**這些都沒有被包進來**。另外，`pre_gateway_dispatch` 的 rewrite directive、
`register_middleware`、`register_command`、`atomic_json_write`、`parse_model_switch_args` 等等
在基準 commit 上**本來就存在**，所以 Gateway 的 hook 路徑完全不需要 patch —— 只需要那個 plugin。
`gateway/run.py`、`hermes_cli/plugins_dispatch.py` 在原始分支雖有改動，
但那些 hunk 屬於上述無關的功能，因此**不在**本 bundle 內。

---

## 7. 測試

在只有套用 patch、沒有整份 bundle 原始目錄的乾淨 checkout 上，先將已安裝的 `session-detours` 外掛複製到可拋棄 checkout 的 `contrib/llm-cr-bundle/payload/plugins/session-detours/`。測試執行器會隔離 HERMES_HOME，因此不能依赖正式 HOME 的外掛作為測試來源；此副本只供測試，驗證 rollback 前須刪除。實際執行環境仍從 HOME/plugins 載入外掛。

安裝程式自己的測試（不需要目標 repo）：

```bash
scripts/run_tests.sh contrib/llm-cr-bundle/tests/test_bundle_installer.py
```

套用之後，功能測試（路徑見 `manifest.json` 的 `tests`）：

```bash
scripts/run_tests.sh \
  tests/hermes_cli/test_text_command_aliases.py \
  tests/hermes_cli/test_plugin_command_context.py \
  tests/hermes_cli/test_session_detour_records.py \
  tests/hermes_cli/test_cli_detour.py \
  tests/gateway/test_text_command_alias_dispatch.py \
  tests/gateway/test_text_command_alias_model_switch.py \
  tests/gateway/test_text_command_alias_deployed_plugin.py \
  tests/gateway/test_detour_commands.py \
  tests/gateway/test_detour_plugin_dispatch.py
```

detour 相關測試**不複製 plugin 實作**：`tests/fakes/session_detours_plugin.py` 會用 Hermes 自己的
載入方式載入真正的 plugin 檔案。優先找本 checkout 的 `contrib/llm-cr-bundle/payload/plugins/session-detours/`；
在只套了 patch、沒有 `contrib/` 的目標上，改用**已安裝**的 `<home>/plugins/session-detours/`。
也就是說在那種目標上，要先 `apply` 再跑這些測試。

```bash，複製整個 plugin 目錄到 tests/_bundle_prompt 後執行：
scripts/run_tests.sh tests/_bundle_prompt/tests/test_llm_cr_prompt_injection.py
# 完成後移除該測試副本，再做 rollback 驗證。
```

Windows 獨立驗證發現：直接指定 checkout 外的 plugin 測試路徑，雖然 36 項已跑完，pytest 仍可能在結束階段停住；放在 checkout 的 tests/ 下可載入該專案的測試清理流程，已實測正常結束。只在可拋棄 worktree 複製，勿改正式 plugin。

這些測試只使用**暫時的 home 與 sentinel 檔案**，不會碰到真正的指令檔，也不會切換任何真實 session。

在沒有本機 `.venv` 的 worktree 裡跑測試時，用 `HERMES_PYTHON` 明確指定一個裝了 pytest 的解譯器：

```bash
HERMES_PYTHON=/path/to/.venv/Scripts/python.exe bash scripts/run_tests.sh <paths>
```

---

## 8. 已知限制

* **config 註解會流失。** 合併是用 PyYAML 做 load → 改 → dump，所以 `config.yaml` 裡原有的註解與
  自訂排版不會保留（key 與值都會保留）。原始檔在備份目錄裡，回滾可完整還原。
* **`__pycache__` 不在管轄範圍內。** 那是 Python 執行時產生的，不是安裝程式寫的，
  所以回滾也不會刪它。孤立的 `.pyc`（來源 `.py` 已被移除）不會被 import，無害。
* **執行 `verify` 會真的載入 Hermes。** 它會跑真正的 plugin discovery，而那條路徑可能在目標 home
  產生 Hermes 自己的 bootstrap 檔案（例如 `SOUL.md`）。這是 Hermes 的行為，不是安裝程式寫的。
* **測試過的 CLI 介面是互動式輸入。** 以 `-q` 餵進去的一次性 query 依舊是普通 prompt。
  Desktop / ACP 等使用自己 ingress 的前端未經驗證。
* **core patch 是 checkout 層級的改動**，不是純 plugin 安裝。`hermes update` 之後可能需要重新調解；
  請保留備份並重跑測試。
* detour 記錄不會自動回收。`returned` 的記錄會留作歷史；卡在 `entering` 的記錄代表 enter leg
  在輪替前就死了 —— `/detour-end` 會清掉它，且不動任何 session。
