---
name: llm-crack-talk
description: "Use when maintaining llm-cr native session model switching. Runtime intercepts the exact bare text llm-cr / llm-cr-end before any LLM, uses native session-scoped model commands and program-generated confirmations. There is no model-to-script relay; never invoke one as a fallback."
version: 4.0.0
metadata:
  hermes:
    tags: [native-model-switch, session-detour, prompt-injection]
---

# LLM CR — direct native model switching

> Sanitized reference copy shipped with `contrib/llm-cr-bundle`. The installer does **not** write
> this file anywhere — it is documentation only. Every provider, model, endpoint and path below is
> a placeholder: substitute the values you passed to `install.py`.
>
> Placeholders: `<cr-provider>` / `<cr-model>` / `<cr-base-url>` = the route `llm-cr` switches to;
> `<exit-provider>` / `<exit-model>` = the route `llm-cr-end` restores.

## Contract

Only the exact whole-message text (surrounding whitespace allowed) is a control command:

- `llm-cr`: create a fresh child session via `/detour` (a plugin command composed of the surface's
  own native session commands), with a durable return record, then switch to `<cr-provider>` /
  `<cr-model>`. The parent transcript is never carried into the child.
- `llm-cr-end`: `/detour-end` resumes the recorded parent transcript and restores
  `<exit-provider>` / `<exit-model>`. The child transcript is never merged into the parent; the
  child remains archived, never deleted.

Both controls execute before any LLM sees the message. The native handler reports the committed
model/provider directly. **A model's answer to "which model are you?" is not verification.** On
error, preserve the native error; never claim success and never fall back to another route.

The exit route is explicit on purpose: if you deliberately change the configured default later,
update the exit alias too. Do not infer a replacement model or normalize ids silently.

Neither alias changes global model settings — the detour route parser refuses `--global` and
`--once`. Ordinary `/new`, `/reset`, `/resume` and `/model` keep their native semantics, and a
manual `/new` or `/resume` must not silently consume an outstanding return record.

## Runtime implementation

- **Shared matcher**: `hermes_cli/text_command_aliases.py` — case-sensitive, whole-message, strips
  outer whitespace only. One rule, so every surface agrees.
- **CLI**: `cli.py::_tui_process_one_input` calls the matcher before ordinary slash dispatch.
- **Gateway**: the `text-command-aliases` plugin registers the existing `pre_gateway_dispatch` hook
  and rewrites only exact triggers on events that already carry `allow_gateway_control`. Auth,
  mention gates, slash authorization and busy handling all stay authoritative and run afterwards.
- **`/detour` is a plugin command, not a built-in.** The whole feature lives in the
  `session-detours` plugin at `$HERMES_HOME/plugins/session-detours/`: `detour_records.py` (atomic
  scoped UTF-8 JSON records under `$HERMES_HOME/session-detours/`, a per-lane transition lock in two
  layers), `cli_surface.py` and `gateway_surface.py`, which compose the native reset/resume handlers
  rather than reimplementing session rotation. Core's command registry does not know these names; if
  `/detour` ever appears there, that is a stale pre-plugin install.
- **How a plugin reaches native session behavior**: `register_command` handlers that declare a
  `context` parameter receive the dispatching surface's host (`hermes_cli.plugins.
  bind_plugin_command_context` — `{surface, command, host, session_key, event, source}`). Each
  surface module is bound to the live `HermesCLI` / `GatewayRunner` through a per-host adapter that
  delegates unknown attributes to the host, so no host class is patched and the mixins keep calling
  native commands. No host context ⇒ the command refuses; it never guesses a session.
- Both commands register `busy_policy="reject"`, so mid-run they are refused in the same words as a
  built-in `CommandDef(busy_policy="reject")`, on both the busy fast path and the access gate (which
  is checked against the normalized name, so `/detour_end` is gated identically).
- **Disabled plugin ⇒ no command at all**: `/detour` disappears from completion and from gateway and
  CLI dispatch, the `llm-cr` alias resolves to a command nothing handles, and every native handler is
  untouched.
- **Record lookup** validates lane, owner and current child; repeated entry is refused; a return
  record is consumed only after the restoration and route are verified against live runtime state.
- Record contents: `{model, provider}` per side, the two session ids, status, profile label,
  timestamps and a per-transition ownership token. Never credentials, never transcript content.
- Gateway needs a restart to load plugin/core changes; a running CLI needs a fresh process. A live
  conversation cannot be hot-patched by skill text or `/reload`.
- Seeded one-shot CLI `-q` prompts stay literal prompts by the CLI's native design. The tested CLI
  surface is interactive input — do not claim other frontends work without testing their ingress.

This is conversation isolation, **not** a security sandbox: tools, permissions and filesystem reach
are unchanged in the child session.

If a bare trigger ever reaches an LLM, the intercept is stale, disabled or unsupported on that
surface. Report that limitation and use the native command path; never fake a confirmation.

## Non-triggers

Discussion, debugging, text merely containing the phrase, `llm-cr <anything else>`, `LLM-CR`,
`llm-crack-talk`, `/skill llm-cr` — none of these switch anything.

## Prompt injection and privacy

The dedicated prompt lives at `$HERMES_HOME/skills/productivity/llm-crack-talk/llm-cr-instruction.md`
(or wherever `--instruction-path` points). **Its content must never be read, quoted, modified or
loaded into the orchestrating main model.** Only `$HERMES_HOME/plugins/llm-cr-prompt/injector.py`
loads it, and only to append it to the first system message of a COPY of the outgoing request
(inserting a system message if there is none). Hermes' existing instructions are preserved and
stored/shared conversation history is never modified.

The plugin registers `llm_execution` middleware — the last stage before the provider call, after
request hooks, request middleware and debug dumps have all run on the un-injected payload. The gate
checks the exact effective provider, model, base URL and api_mode **and** the outbound payload's own
`model`, because an upstream request middleware can rewrite the latter after the turn's context was
built. Only that exact route opens the file. Other routes neither read nor receive it. Each
request / tool round / retry injects once from the pristine original, so text cannot accumulate.

Reads are UTF-8 with BOM tolerated. Missing, unreadable, empty or non-UTF-8 content aborts before
the provider call with a generic `MiddlewareAbort` carrying a fixed reason code. Never silently
continue without the requested prompt. Errors and logs must never contain prompt content. There is
deliberately no checkout-relative or other-profile fallback: a profile whose own prompt is missing
must fail, not send somebody else's.

For proof, use harmless temporary sentinel files in tests. A real-file smoke may let the script read
and send the file to the approved endpoint, but must return only booleans / status / model metadata —
never prompt or response text.

## Verification checklist

1. Confirm the model id exists on the live provider and the native resolver returns the intended
   provider / base URL / api_mode.
2. Exercise the exact aliases through real input handling and native new/resume against a real
   temporary session database. Model commands must not invoke inference.
3. Assert: fresh child history; unchanged parent transcript; true parent replay on exit; committed
   runtime model/provider; restart recovery; independent concurrent lanes; refusal on failed lock;
   recovery on failed route. Assert other sessions and global defaults are unchanged.
4. Test near matches, unauthorized senders and failed switches — no fabricated confirmations, no
   silent fallback to another route.
5. Run everything with `scripts/run_tests.sh` against temporary homes. Deployed-plugin tests must
   load the actual installed plugin, not a duplicated test implementation. No test reads the real
   instruction file.
