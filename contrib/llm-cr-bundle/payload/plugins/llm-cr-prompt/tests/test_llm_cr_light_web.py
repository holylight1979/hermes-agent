"""Behaviour tests for the lightweight chat + web projection of the llm-cr route.

Same real seams as the sibling module: the plugin is discovered from a temp ``$HERMES_HOME`` by a
real ``PluginManager``, LLM requests run through ``run_llm_execution_middleware`` with
``agent/turn_api_call.py``'s keyword set, and tool calls run through
``run_tool_execution_middleware`` with the keyword set the three production call sites pass
(``agent/inline_tool_executors.py:tool_hook_ids``).

The instruction file is a throwaway sentinel written by the fixture; the real skill file is never
read and no assertion message can carry its text.

Run with:  scripts/run_tests.sh <abs path to this file>
"""

from __future__ import annotations

import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

PLUGIN_DIR = Path(__file__).resolve().parents[1]

if "hermes_cli" not in sys.modules:
    try:
        import hermes_cli  # noqa: F401
    except ImportError:  # pragma: no cover - only when run outside the checkout's rootdir
        sys.path.insert(0, str(PLUGIN_DIR.parents[1] / "hermes-agent"))

SENTINEL = "SENTINEL-LLM-CR-LIGHT-TEST\nharmless 測試 instruction line\n"
BASE_SYSTEM = (
    "Hermes base system prompt. SENTINEL-HERMES-SYSTEM\n"
    "# Memory\nSENTINEL-MEMORY-BLOCK\n# Skills\nSENTINEL-SKILL-INDEX\n# Identity\nSENTINEL-IDENTITY\n"
)

CR_PROVIDER = "test-cr-provider"
CR_MODEL = "test/cr-model:Q4_K_M"
CR_BASE_URL = "http://127.0.0.1:65535/v1"
CR_API_MODE = "chat_completions"

MAIN_PROVIDER = "test-main-provider"
MAIN_MODEL = "test/main-model"
MAIN_BASE_URL = "http://127.0.0.1:65534/v1"

# A plausible fat Hermes toolset: none of these may survive into the CR payload.
FAT_TOOLS = [
    {"type": "function", "function": {"name": n, "description": f"SENTINEL-TOOLDESC-{n}",
                                      "parameters": {"type": "object", "properties": {}}}}
    for n in ("terminal", "read_file", "write_file", "execute_code", "tool_call", "browser",
              "memory_search", "task_create", "web_search", "web_extract")
]


class Recorder:
    def __init__(self) -> None:
        self.payloads: list = []

    def __call__(self, payload):
        self.payloads.append(payload)
        return {"ok": True, "n": len(self.payloads)}

    @property
    def calls(self) -> int:
        return len(self.payloads)


@pytest.fixture
def cr_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "plugins").mkdir(parents=True)
    shutil.copytree(
        PLUGIN_DIR, home / "plugins" / "llm-cr-prompt",
        ignore=shutil.ignore_patterns("__pycache__", "tests", "test_*.py", "*.pyc"),
    )
    instruction = tmp_path / "sentinel-instruction.md"
    instruction.write_text(SENTINEL, encoding="utf-8", newline="")
    (home / "config.yaml").write_text(yaml.safe_dump({
        "plugins": {
            "enabled": ["llm-cr-prompt"],
            "entries": {"llm-cr-prompt": {"settings": {
                "enabled": True, "route_provider": CR_PROVIDER, "route_model": CR_MODEL,
                "route_base_url": CR_BASE_URL, "route_api_mode": CR_API_MODE,
                "instruction_path": str(instruction),
                # light_web deliberately ABSENT: these tests also prove it defaults to on.
            }}},
        },
    }), encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli import plugins as plugins_mod

    manager = plugins_mod.PluginManager()
    manager.discover_and_load()
    loaded = manager._plugins.get("llm-cr-prompt")
    assert loaded is not None and loaded.enabled, f"plugin failed to load: {loaded and loaded.error}"
    monkeypatch.setattr(plugins_mod, "_plugin_managers_by_home", {})
    monkeypatch.setattr(plugins_mod, "_plugin_manager", manager)

    _gate(home).reset()
    yield {"home": home, "instruction": instruction, "manager": manager}
    _gate(home).reset()


def _gate(home=None):
    """The loaded plugin's own ``tool_gate`` module (its stamp table is process-global).

    Found through ``sys.modules`` rather than imported directly: the loader owns the module name, and
    reaching for it by hand would import a *second* copy with its own, empty table. Earlier tests in
    the same process leave their own copies behind under the same module name, so the one loaded from
    THIS test's home wins; that is the copy the live middleware is using.
    """
    candidates = [
        module for name, module in list(sys.modules.items())
        if name.rpartition(".")[2] == "tool_gate" and getattr(module, "set_session_light", None)
    ]
    if home is not None:
        wanted = home / "plugins" / "llm-cr-prompt"
        for module in candidates:
            if Path(getattr(module, "__file__", "")).parent == wanted:
                return module
    if candidates:
        return candidates[-1]
    raise AssertionError("tool_gate was never imported by the plugin loader")


def _set_settings(home, **changes):
    from hermes_cli import config as config_mod

    cfg = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))
    settings = cfg["plugins"]["entries"]["llm-cr-prompt"]["settings"]
    for key, value in changes.items():
        if value is None:
            settings.pop(key, None)
        else:
            settings[key] = value
    (home / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    config_mod._LOAD_CONFIG_CACHE.pop(str(config_mod.get_config_path()), None)


def _request(messages=None, *, model=CR_MODEL, tools=None):
    return {
        "model": model,
        "messages": messages if messages is not None else [
            {"role": "system", "content": BASE_SYSTEM},
            {"role": "user", "content": "今天天氣如何？"},
        ],
        "tools": FAT_TOOLS if tools is None else tools,
        "temperature": 0.7,
        "max_tokens": 4096,
    }


def _run(request, *, recorder=None, provider=CR_PROVIDER, model=CR_MODEL,
         base_url=CR_BASE_URL, api_mode=CR_API_MODE, session_id="session-a"):
    from hermes_cli.middleware import run_llm_execution_middleware

    rec = recorder or Recorder()
    result = run_llm_execution_middleware(
        request, rec, original_request=request, task_id="", turn_id="turn-1",
        api_request_id="req-1", session_id=session_id, platform="cli", model=model,
        provider=provider, base_url=base_url, api_mode=api_mode, api_call_count=1,
        middleware_trace=[],
    )
    return rec, result


def _run_tool(tool_name, args=None, *, session_id="session-a"):
    """Production tool-execution seam; returns ``(result, dispatched_args_or_None)``."""
    from hermes_cli.middleware import run_tool_execution_middleware

    dispatched: list = []

    def _dispatch(next_args):
        dispatched.append(next_args)
        return json.dumps({"ok": True, "tool": tool_name})

    result = run_tool_execution_middleware(
        tool_name, args if args is not None else {}, _dispatch,
        original_args=args or {}, task_id="", session_id=session_id,
        tool_call_id="call-1", turn_id="turn-1", api_request_id="req-1",
    )
    return result, (dispatched[0] if dispatched else None)


def _tool_names(payload):
    out = []
    for entry in payload["tools"]:
        function = entry.get("function") if isinstance(entry, dict) else None
        out.append(function["name"] if isinstance(function, dict) else entry.get("name"))
    return out


def _dump(payload) -> str:
    return json.dumps(payload, ensure_ascii=False)


# --- the projected payload ------------------------------------------------------------------------


def test_light_mode_is_the_default_and_sends_only_guidance_instruction_and_user_text(cr_home):
    request = _request()
    pristine = deepcopy(request)

    rec, result = _run(request)

    assert rec.calls == 1
    sent = rec.payloads[0]
    system = sent["messages"][0]
    assert system["role"] == "system"
    # The private instruction is there, verbatim and exactly once.
    assert SENTINEL in system["content"] and system["content"].count("SENTINEL-LLM-CR-LIGHT-TEST") == 1
    # The Hermes prompt, memory, identity and skill index are not.
    body = _dump(sent)
    for leak in ("SENTINEL-HERMES-SYSTEM", "SENTINEL-MEMORY-BLOCK", "SENTINEL-SKILL-INDEX",
                 "SENTINEL-IDENTITY"):
        assert leak not in body, f"{leak} leaked into the crack-talk payload"
    # Guidance is small and names both web tools.
    assert "web_search" in system["content"] and "web_extract" in system["content"]
    assert len(system["content"]) - len(SENTINEL) < 600
    # The user's own words survive untouched.
    assert sent["messages"][1] == {"role": "user", "content": "今天天氣如何？"}
    assert len(sent["messages"]) == 2
    # Routing params the user did not ask to change are carried over as-is.
    assert sent["model"] == CR_MODEL and sent["temperature"] == 0.7 and sent["max_tokens"] == 4096
    # The caller's request (and the stored history it points at) is untouched.
    assert request == pristine and sent is not request and sent["messages"] is not request["messages"]
    assert result == {"ok": True, "n": 1}


def test_exactly_two_tool_schemas_are_advertised(cr_home):
    rec, _ = _run(_request())

    sent = rec.payloads[0]
    assert _tool_names(sent) == ["web_search", "web_extract"]
    for leak in ("terminal", "execute_code", "tool_call", "browser", "memory_search", "task_create",
                 "read_file", "write_file"):
        assert leak not in _dump(sent["tools"])


def test_a_web_tool_the_session_does_not_have_is_never_resurrected(cr_home):
    """No web tool and no Tool Search bridge: the host withheld them, so neither is advertised."""
    rec, _ = _run(_request(tools=[{"type": "function", "function": {"name": "terminal",
                                                                   "parameters": {}}}]))

    sent = rec.payloads[0]
    assert "tools" not in sent and "tool_choice" not in sent
    system = sent["messages"][0]["content"]
    # The guidance must not promise a tool that cannot be called.
    assert "web_search" not in system and "web_extract" not in system
    assert "沒有任何可用工具" in system
    assert SENTINEL in system


def test_tool_search_deferred_web_tools_are_restored_from_the_registry(cr_home, monkeypatch):
    """Deferred ≠ withdrawn: with the bridge advertised, the registry's own schemas come back."""
    from tools.registry import registry
    from tools.web_tools import WEB_EXTRACT_SCHEMA, WEB_SEARCH_SCHEMA

    asked: list = []

    def _definitions(names, quiet=False):
        asked.append(set(names))
        return [{"type": "function", "function": WEB_SEARCH_SCHEMA},
                {"type": "function", "function": WEB_EXTRACT_SCHEMA}]

    monkeypatch.setattr(registry, "get_definitions", _definitions)
    rec, _ = _run(_request(tools=[
        {"type": "function", "function": {"name": "tool_search", "parameters": {}}},
        {"type": "function", "function": {"name": "tool_call", "parameters": {}}},
    ]))

    sent = rec.payloads[0]
    assert _tool_names(sent) == ["web_search", "web_extract"]
    assert sent["tools"][0]["function"] == WEB_SEARCH_SCHEMA
    assert asked == [{"web_search", "web_extract"}], "availability is asked of the host, not assumed"
    assert "tool_search" not in _dump(sent["tools"]) and "tool_call" not in _dump(sent["tools"])


def test_a_deferred_tool_the_registry_cannot_run_stays_absent(cr_home, monkeypatch):
    """check_fn failing (no API key) / not registered: the bridge does not make it runnable."""
    from tools.registry import registry
    from tools.web_tools import WEB_SEARCH_SCHEMA

    monkeypatch.setattr(registry, "get_definitions",
                        lambda names, quiet=False: [{"type": "function", "function": WEB_SEARCH_SCHEMA}])
    rec, _ = _run(_request(tools=[{"type": "function", "function": {"name": "tool_search",
                                                                   "parameters": {}}}]))

    sent = rec.payloads[0]
    assert _tool_names(sent) == ["web_search"]
    system = sent["messages"][0]["content"]
    assert "web_search" in system and "web_extract" not in system


def test_a_registry_failure_on_a_deferred_session_aborts_rather_than_guessing(cr_home, monkeypatch):
    from hermes_cli.middleware import MiddlewareAbort
    from tools.registry import registry

    def _boom(names, quiet=False):
        raise RuntimeError("SENTINEL-REGISTRY-BOOM")

    monkeypatch.setattr(registry, "get_definitions", _boom)
    rec = Recorder()
    with pytest.raises(MiddlewareAbort) as excinfo:
        _run(_request(tools=[{"type": "function", "function": {"name": "tool_search",
                                                              "parameters": {}}}]), recorder=rec)

    assert rec.calls == 0
    assert "SENTINEL-REGISTRY-BOOM" not in str(excinfo.value)
    assert "RuntimeError" in str(excinfo.value)


def test_an_already_advertised_web_tool_is_reused_without_asking_the_registry(cr_home, monkeypatch):
    from tools.registry import registry

    def _never(names, quiet=False):  # pragma: no cover - must not be reached
        raise AssertionError("the registry must not be consulted when the request already has both")

    monkeypatch.setattr(registry, "get_definitions", _never)
    rec, _ = _run(_request())

    assert _tool_names(rec.payloads[0]) == ["web_search", "web_extract"]
    assert rec.payloads[0]["tools"][0] is FAT_TOOLS[8], "the request's own wire form is reused"


def test_repeated_calls_do_not_stack_the_prompt(cr_home):
    request = _request()
    pristine = deepcopy(request)
    rec = Recorder()

    _run(request, recorder=rec)
    _run(request, recorder=rec)
    _run(request, recorder=rec)

    assert rec.calls == 3
    for payload in rec.payloads:
        assert payload["messages"][0]["content"].count("SENTINEL-LLM-CR-LIGHT-TEST") == 1
        assert payload["messages"][0]["content"].count("web_search") == 1
        assert _tool_names(payload) == ["web_search", "web_extract"]
        assert len(payload["messages"]) == 2
    assert request == pristine


def test_web_tool_rounds_keep_their_call_result_alternation(cr_home):
    messages = [
        {"role": "system", "content": BASE_SYSTEM},
        {"role": "user", "content": "查一下"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "web_search", "arguments": '{"query":"天氣"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "web_search", "content": "SENTINEL-WEBRESULT"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c2", "type": "function",
             "function": {"name": "web_extract", "arguments": '{"urls":["https://e.com"]}'}}]},
        {"role": "tool", "tool_call_id": "c2", "content": "SENTINEL-PAGE"},
        {"role": "assistant", "content": "好的"},
        {"role": "user", "content": "再查一次"},
    ]
    rec, _ = _run(_request(messages))

    sent = rec.payloads[0]
    roles = [m["role"] for m in sent["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "tool", "assistant", "user"]
    assert sent["messages"][3]["content"] == "SENTINEL-WEBRESULT"
    assert sent["messages"][2]["tool_calls"][0]["function"]["name"] == "web_search"
    assert sent["messages"][5]["tool_call_id"] == "c2"


def test_legacy_non_web_tool_artifacts_leave_in_matched_pairs(cr_home):
    """An older CR session may hold terminal/memory rounds. Neither half may be sent."""
    messages = [
        {"role": "system", "content": BASE_SYSTEM},
        {"role": "user", "content": "之前的對話"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "t1", "type": "function",
             "function": {"name": "terminal", "arguments": '{"command":"SENTINEL-OLDCMD"}'}}]},
        {"role": "tool", "tool_call_id": "t1", "name": "terminal", "content": "SENTINEL-OLDOUT"},
        {"role": "assistant", "content": "說明", "tool_calls": [
            {"id": "t2", "type": "function",
             "function": {"name": "memory_search", "arguments": '{"q":"SENTINEL-OLDMEM"}'}},
            {"id": "w1", "type": "function",
             "function": {"name": "web_search", "arguments": '{"query":"ok"}'}}]},
        {"role": "tool", "tool_call_id": "t2", "content": "SENTINEL-OLDMEMOUT"},
        {"role": "tool", "tool_call_id": "w1", "content": "SENTINEL-WEBOK"},
    ]
    rec, _ = _run(_request(messages))

    sent = rec.payloads[0]
    body = _dump(sent)
    for leak in ("SENTINEL-OLDCMD", "SENTINEL-OLDOUT", "SENTINEL-OLDMEM", "SENTINEL-OLDMEMOUT",
                 "terminal", "memory_search"):
        assert leak not in body, f"{leak} leaked from a legacy tool round"
    assert "SENTINEL-WEBOK" in body
    roles = [m["role"] for m in sent["messages"]]
    assert roles == ["system", "user", "assistant", "tool"]
    assert [c["id"] for c in sent["messages"][2]["tool_calls"]] == ["w1"]
    assert sent["messages"][2]["content"] == "說明"


def test_unanswered_web_call_is_dropped_so_the_wire_stays_valid(cr_home):
    messages = [
        {"role": "user", "content": "問題"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c9", "type": "function",
             "function": {"name": "web_search", "arguments": "{}"}}]},
    ]
    rec, _ = _run(_request(messages))

    sent = rec.payloads[0]
    assert [m["role"] for m in sent["messages"]] == ["system", "user"]


def test_model_switch_boilerplate_is_stripped_but_the_user_text_is_not(cr_home):
    note = ("[Note: model was just switched from test/main-model to CR via test-cr-provider. "
            "Adjust your self-identification accordingly.]")
    messages = [
        {"role": "system", "content": BASE_SYSTEM},
        {"role": "user", "content": f"{note}\n\n真正的使用者問題 [Note: 我自己寫的括號]"},
    ]
    rec, _ = _run(_request(messages))

    user = rec.payloads[0]["messages"][1]
    assert user["content"] == "真正的使用者問題 [Note: 我自己寫的括號]"
    assert "model was just switched" not in _dump(rec.payloads[0])


def test_a_user_message_that_is_only_boilerplate_is_kept_rather_than_emptied(cr_home):
    note = "[Note: model was just switched from A to B via p.]"
    rec, _ = _run(_request([{"role": "user", "content": note}]))

    assert rec.payloads[0]["messages"][1]["content"] == note


def test_named_tool_choice_is_dropped_and_a_plain_one_is_kept(cr_home):
    request = _request()
    request["tool_choice"] = {"type": "function", "function": {"name": "terminal"}}
    rec, _ = _run(request)
    assert "tool_choice" not in rec.payloads[0]

    request2 = _request()
    request2["tool_choice"] = "auto"
    rec2, _ = _run(request2)
    assert rec2.payloads[0]["tool_choice"] == "auto"


def test_projection_with_no_surviving_turns_aborts_before_the_provider(cr_home):
    from hermes_cli.middleware import MiddlewareAbort

    rec = Recorder()
    with pytest.raises(MiddlewareAbort):
        _run(_request([{"role": "system", "content": BASE_SYSTEM}]), recorder=rec)
    assert rec.calls == 0


def test_missing_instruction_file_still_aborts_in_light_mode(cr_home):
    from hermes_cli.middleware import MiddlewareAbort

    cr_home["instruction"].unlink()
    rec = Recorder()
    with pytest.raises(MiddlewareAbort) as excinfo:
        _run(_request(), recorder=rec)

    assert rec.calls == 0
    assert "unavailable" in str(excinfo.value) and "SENTINEL" not in str(excinfo.value)


def test_default_route_payload_is_byte_for_byte_unaffected(cr_home):
    request = _request(model=MAIN_MODEL)
    pristine = deepcopy(request)

    rec, _ = _run(request, provider=MAIN_PROVIDER, model=MAIN_MODEL, base_url=MAIN_BASE_URL,
                  api_mode="anthropic_messages", session_id="session-main")

    assert rec.calls == 1
    assert rec.payloads[0] is request
    assert _dump(request) == _dump(pristine)
    assert _tool_names(request) == [t["function"]["name"] for t in FAT_TOOLS]


def test_light_web_can_be_turned_off_without_touching_the_route_gate(cr_home):
    _set_settings(cr_home["home"], light_web=False)

    rec, _ = _run(_request())

    sent = rec.payloads[0]
    assert sent["messages"][0]["content"] == f"{BASE_SYSTEM}\n\n{SENTINEL}"
    assert _tool_names(sent) == [t["function"]["name"] for t in FAT_TOOLS]


# --- execution-seam enforcement -------------------------------------------------------------------


@pytest.mark.parametrize("tool_name", [
    "terminal", "execute_code", "tool_call", "read_file", "write_file", "browser",
    "memory_search", "task_create", "delegate", "skill", "WEB_SEARCH", "web_search_v2", "",
])
def test_unadvertised_tools_are_refused_at_the_execution_seam(cr_home, tool_name):
    _run(_request())  # the turn that stamps the session as lightweight

    result, dispatched = _run_tool(tool_name, {"SENTINEL-ARG": 1})

    assert dispatched is None, "the handler must never be dispatched"
    body = json.loads(result)
    assert "error" in body and "llm-cr chat mode" in body["error"]


@pytest.mark.parametrize("tool_name", ["web_search", "web_extract"])
def test_the_two_web_tools_are_dispatched_normally(cr_home, tool_name):
    _run(_request())

    result, dispatched = _run_tool(tool_name, {"query": "x"})

    assert dispatched == {"query": "x"}
    assert json.loads(result) == {"ok": True, "tool": tool_name}


def test_tools_on_a_session_that_never_went_to_the_cr_route_are_untouched(cr_home):
    _run(_request(model=MAIN_MODEL), provider=MAIN_PROVIDER, model=MAIN_MODEL,
         base_url=MAIN_BASE_URL, api_mode="anthropic_messages", session_id="session-main")

    result, dispatched = _run_tool("terminal", {"command": "ls"}, session_id="session-main")

    assert dispatched == {"command": "ls"}
    assert json.loads(result) == {"ok": True, "tool": "terminal"}


def test_leaving_the_cr_route_restores_the_full_toolset(cr_home):
    _run(_request(), session_id="session-x")
    denied, _ = _run_tool("terminal", session_id="session-x")
    assert "error" in json.loads(denied)

    # llm-cr-end: the same session's next request goes to the default route.
    _run(_request(model=MAIN_MODEL), provider=MAIN_PROVIDER, model=MAIN_MODEL,
         base_url=MAIN_BASE_URL, api_mode="anthropic_messages", session_id="session-x")

    result, dispatched = _run_tool("terminal", {"command": "ls"}, session_id="session-x")
    assert dispatched == {"command": "ls"}


def test_a_request_that_aborted_never_stamps_the_session(cr_home):
    from hermes_cli.middleware import MiddlewareAbort

    cr_home["instruction"].unlink()
    with pytest.raises(MiddlewareAbort):
        _run(_request(), session_id="session-abort")

    result, dispatched = _run_tool("terminal", {"command": "ls"}, session_id="session-abort")
    assert dispatched == {"command": "ls"}, "a failed CR request must not gate the session"


def test_gating_is_per_session_in_one_process(cr_home):
    _run(_request(), session_id="session-cr")
    _run(_request(model=MAIN_MODEL), provider=MAIN_PROVIDER, model=MAIN_MODEL,
         base_url=MAIN_BASE_URL, api_mode="anthropic_messages", session_id="session-main")

    assert _run_tool("terminal", session_id="session-main")[1] is not None
    assert _run_tool("terminal", session_id="session-cr")[1] is None


def test_light_web_off_does_not_gate_tools(cr_home):
    _set_settings(cr_home["home"], light_web=False)
    _run(_request(), session_id="session-full")

    result, dispatched = _run_tool("terminal", {"command": "ls"}, session_id="session-full")
    assert dispatched == {"command": "ls"}


def test_many_live_sessions_never_lose_their_restriction(cr_home):
    """The stamp table is not an LRU cache: an older session stays gated, whatever the table size.

    Evicting a stamp would re-enable terminal/file/delegation tools on a crack-talk conversation
    that is still running — enforcement silently disappearing with age is worse than a table that
    only shrinks when a session actually leaves the route.
    """
    gate = _gate(cr_home["home"])
    first = "session-oldest"
    _run(_request(), session_id=first)

    for index in range(gate._GROWTH_WARN_AT + 40):
        _run(_request(), session_id=f"session-{index}")

    assert gate.tracked_session_count() == gate._GROWTH_WARN_AT + 41
    assert _run_tool("terminal", {"command": "ls"}, session_id=first)[1] is None
    assert _run_tool("web_search", {"query": "x"}, session_id=first)[1] == {"query": "x"}


def test_a_matched_request_without_a_session_id_fails_closed(cr_home):
    """No session id means the gate has nothing to key on, so the route must not run at all."""
    from hermes_cli.middleware import MiddlewareAbort

    rec = Recorder()
    for empty in ("", None):
        with pytest.raises(MiddlewareAbort) as excinfo:
            _run(_request(), recorder=rec, session_id=empty)
        assert "session id" in str(excinfo.value) and "SENTINEL" not in str(excinfo.value)
    assert rec.calls == 0


def test_light_web_off_without_a_session_id_still_injects(cr_home):
    """The fail-closed rule belongs to the tool gate, so it must not break plain injection mode."""
    _set_settings(cr_home["home"], light_web=False)

    rec, _ = _run(_request(), session_id="")

    assert rec.payloads[0]["messages"][0]["content"] == f"{BASE_SYSTEM}\n\n{SENTINEL}"


# --- the policy-hook layer ------------------------------------------------------------------------


@pytest.mark.parametrize("tool_name", ["terminal", "execute_code", "tool_call", "read_file"])
def test_the_pre_tool_call_hook_blocks_the_same_tools(cr_home, tool_name):
    """A second, independent layer on the host's generic policy seam (before dispatch)."""
    from hermes_cli.plugins import get_pre_tool_call_block_message

    _run(_request(), session_id="session-hook")

    message = get_pre_tool_call_block_message(tool_name, {"x": 1}, session_id="session-hook")
    assert message and "llm-cr chat mode" in message
    assert get_pre_tool_call_block_message("web_search", {"query": "x"}, session_id="session-hook") is None
    assert get_pre_tool_call_block_message(tool_name, {"x": 1}, session_id="session-other") is None


def test_agent_inline_tools_are_refused_at_the_execution_seam(cr_home):
    """todo / memory / plan-style tools do not reach the registry, but they DO reach this seam.

    ``agent/tool_executor.py`` (sequential) and ``agent/agent_runtime_helpers.py`` (concurrent) both
    wrap their inline dispatch in the tool-execution middleware, so the allowlist covers names that
    never appear in the registry. Parametrising over the host's real inline table means a newly added
    inline tool is covered the moment it exists.
    """
    from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS

    _run(_request(), session_id="session-inline")
    assert INLINE_TOOL_EXECUTORS, "the host's inline tool table is unexpectedly empty"
    for name in sorted(INLINE_TOOL_EXECUTORS):
        assert name not in ("web_search", "web_extract")
        result, dispatched = _run_tool(name, {"SENTINEL-ARG": 1}, session_id="session-inline")
        assert dispatched is None, f"inline tool {name} was dispatched on a lightweight session"
        assert "llm-cr chat mode" in json.loads(result)["error"]


def test_inline_and_registry_dispatch_still_run_through_the_execution_seam(cr_home):
    """Guard the host-side assumption this plugin's enforcement rests on.

    If a refactor ever dispatches an inline tool without the middleware wrapper, hiding schemas
    would be the only thing left on that path — which is presentation, not enforcement. These are
    the two wrappers that must keep existing.
    """
    import agent.agent_runtime_helpers as helpers
    import agent.tool_executor as tool_executor

    sequential = Path(tool_executor.__file__).read_text(encoding="utf-8")
    assert "_run_sequential_tool_execution_middleware(" in sequential
    assert "_resolve_sequential_dispatch(" in sequential
    concurrent = Path(helpers.__file__).read_text(encoding="utf-8")
    assert "run_tool_execution_middleware(" in concurrent
    assert "resolve_invoke_tool_executor(" in concurrent
