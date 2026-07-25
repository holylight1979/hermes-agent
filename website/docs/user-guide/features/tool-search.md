---
title: Tool Search
sidebar_position: 95
---

# Tool Search

When you have many MCP servers or non-core plugin tools attached to a
session, their JSON schemas can consume a substantial fraction of the
context window on every turn — even when only a few of them are relevant
to what the user actually asked for.

**Tool Search** is Hermes' opt-in progressive-disclosure layer for that
problem. When activated, MCP and plugin tools are replaced in the
model-visible tools array by three bridge tools, and the model loads each
specific tool's schema on demand.

:::info Core deferral is explicit opt-in
Hermes core tools remain directly visible by default. Setting
`defer_core: true` moves registered core tools behind the bridge except for
`always_visible` (default: `clarify` and `skill_view`). The bridge tools are
always direct, and unknown/unregistered names remain visible as a fail-safe.
Use core deferral together with the hybrid pre-router described below.
:::

## How it works

When Tool Search activates for a turn, the model sees three new tools in
place of the deferred ones:

```
tool_search(query, limit?)     — search the deferred-tool catalog
tool_describe(name)            — load the full schema for one tool
tool_call(name, arguments)     — invoke a deferred tool
```

A typical interaction looks like:

```
Model: tool_search("create a github issue")
  → { matches: [{ name: "mcp_github_create_issue", ... }, ...] }
Model: tool_describe("mcp_github_create_issue")
  → { parameters: { type: "object", properties: { ... } } }
Model: tool_call("mcp_github_create_issue", { title: "...", body: "..." })
  → { ok: true, issue_number: 42 }
```

When the model invokes `tool_call`, Hermes **unwraps the bridge** and
dispatches the underlying tool exactly as if the model had called it
directly. Pre-tool-call hooks, guardrails, approval prompts, and
post-tool-call hooks all run against the real tool name — not against
`tool_call`. The activity feed in the CLI and gateway also unwraps so you
see the underlying tool, not the bridge.

## When does it activate?

By default Tool Search runs in `auto` mode: it activates only when the
deferrable tool schemas would consume at least 10% of the active model's
context window. Below that, the tools-array assembly is a pure
pass-through and you pay no overhead.

This decision is evaluated when the session tools array is built, so:

- A session with just a few MCP tools and a long context model never
  activates Tool Search.
- A session with many MCP servers attached (15+ tools typically) starts
  activating it.
- Configuration changes take effect in a new session. The session freezes its
  bridge policy so a mid-session config edit cannot make an already-deferred
  tool silently unreachable.

## Configuration

```yaml
tools:
  tool_search:
    enabled: auto       # auto (default), on, or off
    threshold_pct: 10   # percentage of context — only used in auto mode
    defer_core: false   # opt in to a fixed small main-model tool surface
    always_visible: [clarify, skill_view]
    search_default_limit: 5
    max_search_limit: 20
```

| Key | Default | Meaning |
| --- | --- | --- |
| `enabled` | `auto` | `auto` activates above threshold; `on` always activates if there's at least one deferrable tool; `off` disables entirely. |
| `threshold_pct` | `10` | Percentage of context length at which `auto` mode kicks in. Range 0–100. |
| `defer_core` | `false` | When true, registered core tools may be deferred; unknown names still stay visible. |
| `always_visible` | `[clarify, skill_view]` | Direct tools retained when core deferral is enabled. |
| `search_default_limit` | `5` | Hits returned when the model calls `tool_search` without a `limit`. |
| `max_search_limit` | `20` | Hard upper bound the model can request via `limit`. Range 1–50. |

You can also flip the legacy boolean shape:

```yaml
tools:
  tool_search: true   # equivalent to {enabled: auto}
```

## Hybrid pre-router

For a stable small main-model surface, enable core deferral and the optional
hybrid pre-router together:

```yaml
tools:
  tool_search:
    enabled: on
    defer_core: true
    always_visible: [clarify, skill_view]
  tool_router:
    enabled: true
    mode: hybrid
    provider: rdchat-direct
    model: gemma4:e4b-64k
    timeout_seconds: 4
    confidence_threshold: 0.75
    max_candidates: 6
    max_packet_tokens: 1800
    include_compact_schemas: true
    fail_open: true
    telemetry: true
```

The model-facing `tools` array is then fixed to `clarify`, `skill_view`, and
the three bridge tools. Once per user turn, high-confidence local rules—or,
for ambiguous text, the configured low-cost classifier—append a bounded
capability/candidate packet to the API-only user-message sidecar. The packet
is a recommendation, not an authority decision:

- candidates come only from that session's already-authorized catalog;
- `tool_search` can still search the complete authorized catalog if the route
  is wrong or incomplete;
- classifier timeout, malformed output, low confidence, provider outage, and
  internal router exceptions all fail open;
- the underlying real tool name still passes through existing tool hooks,
  guardrails, approval prompts, and command safety checks;
- telemetry logs source, capability names, counts, estimated schema tokens,
  and latency only—never user text, tool arguments, or secrets.

:::warning Classifier data egress
For ambiguous text, the classifier receives the complete raw user message.
Use only a provider you trust with that content (for example, a local/private
`rdchat-direct` endpoint). Telemetry omits the message, but that does not remove
the classifier network request itself.
:::

The router runs once per user turn in the shared `AIAgent` prologue, so CLI and
gateway sessions use the same implementation. It is skipped for multimodal
turns; those retain full bridge-catalog recovery.

## When NOT to use it

Tool Search trades a fixed per-turn token cost (the three bridge tool
schemas, ~300 tokens) and at least one extra round trip (search →
describe → call) for the savings on the deferred schemas. It's a clear
win when you have many tools and use few per turn; it's overhead when
you have few tools total.

The `auto` default handles this for you. If you set `enabled: on`
unconditionally, expect a slight per-turn cost on small toolsets.

## Trade-offs that don't go away

These come from the prompt-cache integrity invariant — they are inherent
to any progressive-disclosure design, not specific to this implementation:

- **One extra round trip on cold tools.** The first time the model needs
  a deferred tool, it spends one or two extra model calls to find and
  load the schema. The token savings on the static side are real, but a
  portion is paid back at runtime.
- **No cache benefit on deferred schemas.** A loaded `tool_describe`
  result enters the conversation history (so it does get cached on
  subsequent turns) but it never benefits from the system-prompt cache
  prefix.
- **Model-quality dependence.** Tool Search assumes the model can write a
  reasonable search query for the tool it wants. Smaller models do this
  less well; the published Anthropic numbers (49% → 74% on Opus 4 with
  vs. without tool search) show the upside but also that ~26 points of
  accuracy is still retrieval failure.
- **Toolset edits invalidate cache.** Adding or removing a tool mid-
  session changes the bridge tools' descriptions (which include the
  count of deferred tools) and the catalog, so the prompt cache is
  invalidated. This is the same trade-off as any toolset edit.

## Implementation details

- **Retrieval:** BM25 over tokenized tool name + description + parameter
  names. Falls back to a literal substring match on the tool name when
  BM25 returns no positive-score hits, which protects against
  zero-IDF degenerate cases (e.g. searching `"github"` against a
  catalog where every tool name contains "github").
- **Catalog is stateless across turns.** It rebuilds from the current
  tool-defs list every assembly — no session-keyed `Map`. This avoids
  the class of bug where a stored catalog drifts out of sync with the
  live tool registry.
- **The catalog is scoped to the session's toolsets.** `tool_search`,
  `tool_describe`, and `tool_call` only ever see and invoke tools the
  session was actually granted. A subagent, kanban worker, or gateway
  session restricted to a subset of toolsets cannot use the bridge to
  discover or call a tool outside that subset — the deferred catalog is
  the deferrable slice of the session's own enabled/disabled toolsets,
  not the whole process registry.
- **No JS sandbox.** Hermes uses the simpler "structured tools" mode
  (search / describe / call as plain functions). The JS-sandbox "code
  mode" some other implementations offer is a large surface area; we
  skip it.

## See also

- `tools/tool_search.py` — the implementation
- `tests/tools/test_tool_search.py` — the regression suite
- The `openclaw-tool-search-report` PDF in the original implementation
  PR for the research that shaped the design
