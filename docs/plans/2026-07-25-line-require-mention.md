# LINE Require Mention Implementation Plan

> **For Hermes:** Execute task-by-task with tests and review.

**Goal:** Enforce native LINE @mention gating for group and room messages while preserving DM behavior and allowing replies to Hermes' own messages.

**Architecture:** Parse a canonical `require_mention` platform-extra setting with compatibility fallback to `LINE_REQUIRE_MENTION`. Native mentions pass directly. LINE `quotedMessageId` values pass only when the quoted ID exists in Hermes' profile-scoped persisted outbound-message index, preventing replies between human group members from waking the bot.

**Tech Stack:** Python, asyncio, pytest, LINE Messaging API webhook payloads.

---

### Task 1: Add regression tests

**Files:**
- Modify: `tests/gateway/test_line_plugin.py`

1. Add setting-parsing tests for config and environment fallback.
2. Add strict native-mention and reply-to-own-message tests for group/room and DM behavior.
3. Run targeted tests and confirm they fail because gating is absent.

### Task 2: Implement minimal native mention gate

**Files:**
- Modify: `plugins/platforms/line/adapter.py`
- Modify: `plugins/platforms/line/plugin.yaml`

1. Parse `require_mention` from platform extra, falling back to `LINE_REQUIRE_MENTION`.
2. Add a helper that recognizes `isSelf=true` or the current BOT userId in `message.mention.mentionees`.
3. Persist `sentMessages[].id` from LINE Reply/Push responses in the existing profile-scoped rich-message index.
4. Treat an inbound quote as bot-directed only when `quotedMessageId` resolves in that index, and attach reply context to `MessageEvent`.
5. Apply the gate before storing reply tokens and dispatching group/room messages.
6. Document the compatibility environment variable and reply behavior.

### Task 3: Configure and verify

**Files:**
- Modify: active `config.yaml` only through `hermes config set` if supported; otherwise update the existing `gateway.platforms.line.extra` mapping safely.

1. Enable `gateway.platforms.line.extra.require_mention`.
2. Run the full LINE plugin test file.
3. Run neighboring Gateway tests affected by platform loading.
4. Inspect the diff for unrelated changes and secrets.

### Task 4: Independent review and runtime validation

1. Ask Claude Code/Opus for a read-only review of the diff and tests.
2. Correct every material finding and rerun tests.
3. Restart Gateway.
4. Confirm Gateway process status, LINE health endpoint, and startup log.
5. Report that automated verification is complete and request one real LINE group test for both unmentioned and mentioned messages.
