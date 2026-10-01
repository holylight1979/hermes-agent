"""Behaviour tests for the llm-cr-prompt plugin.

Everything runs through the REAL seams: the plugin is discovered from a temp ``$HERMES_HOME`` by a
real ``PluginManager``, and requests go through the real
``hermes_cli.middleware.run_llm_execution_middleware`` with the exact keyword set
``agent/turn_api_call.py`` passes at the production call site.

The instruction file used here is a throwaway sentinel written by the fixture — the real skill file
is never read, and no assertion message can carry its text.

Run with:  scripts/run_tests.sh <abs path to this file>
"""

from __future__ import annotations

import shutil
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

# This file deliberately lives in a directory WITHOUT ``__init__.py``: the plugin directory itself
# is a package, so collecting from inside it would make pytest import the plugin's ``__init__``
# outside the plugin loader, where its relative imports cannot resolve.
PLUGIN_DIR = Path(__file__).resolve().parents[1]

if "hermes_cli" not in sys.modules:
    try:
        import hermes_cli  # noqa: F401
    except ImportError:  # pragma: no cover - only when run outside the checkout's rootdir
        sys.path.insert(0, str(PLUGIN_DIR.parents[1] / "hermes-agent"))

SENTINEL = "SENTINEL-LLM-CR-PROMPT-TEST\nharmless 測試 instruction line\n"
BASE_SYSTEM = "Hermes base system prompt."

INSTRUCTION_FILENAME = "llm-cr-instruction.md"
DEFAULT_RELPATH = Path("skills") / "productivity" / "llm-crack-talk" / INSTRUCTION_FILENAME

CR_PROVIDER = "test-cr-provider"
CR_MODEL = "test/cr-model:Q4_K_M"
CR_BASE_URL = "http://127.0.0.1:65535/v1"
CR_API_MODE = "chat_completions"

GPT_PROVIDER = "exit-direct"
GPT_MODEL = "exit-model-900k"
GPT_BASE_URL = "https://exit.example.invalid/api"


class Recorder:
    """Fake downstream provider call: records payloads, never touches the network."""

    def __init__(self) -> None:
        self.payloads: list = []

    def __call__(self, payload):
        self.payloads.append(payload)
        return {"ok": True, "n": len(self.payloads)}

    @property
    def calls(self) -> int:
        return len(self.payloads)


def _request(system: str | None = BASE_SYSTEM, model: str | None = CR_MODEL) -> dict:
    messages = [{"role": "user", "content": "hi"}]
    if system is not None:
        messages.insert(0, {"role": "system", "content": system})
    request = {"messages": messages, "temperature": 0.7, "tools": [{"name": "terminal"}]}
    if model is not None:
        request["model"] = model
    return request


@pytest.fixture
def cr_home(tmp_path, monkeypatch):
    """Temp HOME with this plugin installed+enabled and a sentinel instruction file."""
    home = tmp_path / ".hermes"
    (home / "plugins").mkdir(parents=True)
    shutil.copytree(
        PLUGIN_DIR, home / "plugins" / "llm-cr-prompt",
        ignore=shutil.ignore_patterns("__pycache__", "tests", "test_*.py", "*.pyc"),
    )
    instruction = tmp_path / "sentinel-instruction.md"
    # newline="" keeps the sentinel byte-exact on Windows: the injector sends the file as stored.
    instruction.write_text(SENTINEL, encoding="utf-8", newline="")
    (home / "config.yaml").write_text(yaml.safe_dump({
        "plugins": {
            "enabled": ["llm-cr-prompt"],
            "entries": {"llm-cr-prompt": {"settings": {
                "enabled": True, "route_provider": CR_PROVIDER, "route_model": CR_MODEL,
                "route_base_url": CR_BASE_URL, "route_api_mode": CR_API_MODE,
                "instruction_path": str(instruction),
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
    # Throwaway registry so the test's manager never leaks into the per-home cache.
    monkeypatch.setattr(plugins_mod, "_plugin_managers_by_home", {})
    monkeypatch.setattr(plugins_mod, "_plugin_manager", manager)
    return {"home": home, "instruction": instruction, "manager": manager}


@pytest.fixture
def read_spy(monkeypatch):
    """Record every ``Path.read_bytes`` target, so "the file was never opened" is an assertion.

    Only the paths are kept — never the bytes. ``load_instruction`` reads exclusively through
    ``read_bytes``, so an empty (instruction-free) record is proof the gate ran first.
    """
    seen: list[str] = []
    real = Path.read_bytes

    def spy(self):
        seen.append(str(self))
        return real(self)

    monkeypatch.setattr(Path, "read_bytes", spy)
    return seen


def _instruction_reads(seen) -> list:
    return [p for p in seen if p.endswith(("sentinel-instruction.md", INSTRUCTION_FILENAME))]


def _set_settings(home, **changes):
    """Edit this plugin's settings in the temp config; ``None`` removes a key. Busts the cache."""
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


def _run(request, *, recorder=None, provider=CR_PROVIDER, model=CR_MODEL,
         base_url=CR_BASE_URL, api_mode=CR_API_MODE, session_id="session-a"):
    """Call the production middleware entry point with turn_api_call.py's keyword set."""
    from hermes_cli.middleware import run_llm_execution_middleware

    rec = recorder or Recorder()
    result = run_llm_execution_middleware(
        request, rec, original_request=request, task_id="", turn_id="turn-1",
        api_request_id="req-1", session_id=session_id, platform="cli", model=model,
        provider=provider, base_url=base_url, api_mode=api_mode, api_call_count=1,
        middleware_trace=[],
    )
    return rec, result


def _system_text(payload) -> str:
    return payload["messages"][0]["content"]


def test_matched_route_injects_exact_instruction_and_preserves_everything_else(cr_home):
    request = _request()
    pristine = deepcopy(request)

    rec, result = _run(request)

    assert rec.calls == 1
    sent = rec.payloads[0]
    assert _system_text(sent) == f"{BASE_SYSTEM}\n\n{SENTINEL}"
    # Original Hermes system text, user turns, tools and request kwargs all survive.
    assert sent["messages"][1] == {"role": "user", "content": "hi"}
    assert sent["tools"] == [{"name": "terminal"}] and sent["temperature"] == 0.7
    assert sent["model"] == CR_MODEL
    # The caller's request (and therefore the stored history it points at) is untouched.
    assert request == pristine
    assert request["messages"][0] is pristine["messages"][0] or request["messages"][0] == pristine["messages"][0]
    assert sent is not request and sent["messages"] is not request["messages"]
    assert result == {"ok": True, "n": 1}


def test_repeated_calls_inject_once_each_without_accumulating(cr_home):
    """A retry / tool round re-enters with the same original: exactly one copy of the text."""
    request = _request()
    pristine = deepcopy(request)
    rec = Recorder()

    _run(request, recorder=rec)
    _run(request, recorder=rec)
    _run(request, recorder=rec)

    assert rec.calls == 3
    for payload in rec.payloads:
        assert _system_text(payload) == f"{BASE_SYSTEM}\n\n{SENTINEL}"
        assert _system_text(payload).count("SENTINEL-LLM-CR-PROMPT-TEST") == 1
    assert request == pristine


def test_request_without_system_message_gets_one_inserted(cr_home):
    request = _request(system=None)
    pristine = deepcopy(request)

    rec, _ = _run(request)

    sent = rec.payloads[0]
    assert sent["messages"][0] == {"role": "system", "content": SENTINEL}
    assert sent["messages"][1] == {"role": "user", "content": "hi"}
    assert request == pristine


def test_utf8_bom_file_is_decoded_without_the_bom(cr_home):
    cr_home["instruction"].write_text(SENTINEL, encoding="utf-8-sig", newline="")

    rec, _ = _run(_request())

    assert _system_text(rec.payloads[0]) == f"{BASE_SYSTEM}\n\n{SENTINEL}"
    assert "﻿" not in _system_text(rec.payloads[0])


@pytest.mark.parametrize("override", [
    {"provider": "other-direct"},
    {"model": "gemma4:e4b"},
    {"base_url": "http://127.0.0.1:11434/v1"},
    {"api_mode": "codex_responses"},
])
def test_mismatched_route_passes_request_through_untouched_and_never_reads_the_file(cr_home, override):
    # Deleting the file proves the gate runs BEFORE any read: a read would abort the request.
    cr_home["instruction"].unlink()
    request = _request()

    rec, _ = _run(request, **override)

    assert rec.calls == 1
    assert rec.payloads[0] is request


def test_gpt_route_after_exit_is_untouched(cr_home):
    cr_home["instruction"].unlink()
    request = _request()

    rec, _ = _run(request, provider=GPT_PROVIDER, model=GPT_MODEL,
                  base_url=GPT_BASE_URL, api_mode="codex_responses")

    assert rec.calls == 1 and rec.payloads[0] is request


def test_sessions_are_independent_same_process(cr_home):
    cr_request, gpt_request = _request(), _request()
    rec = Recorder()

    _run(cr_request, recorder=rec, session_id="session-cr")
    _run(gpt_request, recorder=rec, session_id="session-gpt", provider=GPT_PROVIDER,
         model=GPT_MODEL, base_url=GPT_BASE_URL, api_mode="codex_responses")
    _run(cr_request, recorder=rec, session_id="session-cr")

    assert [_system_text(p) for p in rec.payloads] == [
        f"{BASE_SYSTEM}\n\n{SENTINEL}", BASE_SYSTEM, f"{BASE_SYSTEM}\n\n{SENTINEL}",
    ]


@pytest.mark.parametrize("breakage", ["missing", "empty", "whitespace", "bad-encoding"])
def test_unusable_instruction_file_aborts_before_any_provider_call(cr_home, breakage):
    from hermes_cli.middleware import MiddlewareAbort

    path = cr_home["instruction"]
    if breakage == "missing":
        path.unlink()
    elif breakage == "empty":
        path.write_bytes(b"")
    elif breakage == "whitespace":
        path.write_text("   \n\t\n", encoding="utf-8")
    else:
        path.write_bytes(b"\xff\xfe\x00bad utf-8 \x81\x8f")

    request = _request()
    pristine = deepcopy(request)
    rec = Recorder()

    with pytest.raises(MiddlewareAbort) as excinfo:
        _run(request, recorder=rec)

    assert rec.calls == 0, "fail-closed: the provider must not be called"
    assert request == pristine
    # Content-free error: no instruction text, no file path.
    message = str(excinfo.value)
    assert "SENTINEL" not in message and str(path) not in message
    assert "llm-cr-prompt" in message


def test_abort_bubbles_through_an_outer_middleware_frame(cr_home):
    """An outer middleware's own fail-open handling must not swallow the abort."""
    from hermes_cli.middleware import LLM_EXECUTION_MIDDLEWARE, MiddlewareAbort

    cr_home["instruction"].unlink()
    seen = {"entered": 0, "returned": 0}

    def outer(request=None, next_call=None, **_kwargs):
        seen["entered"] += 1
        result = next_call(request)
        seen["returned"] += 1
        return result

    chain = cr_home["manager"]._middleware[LLM_EXECUTION_MIDDLEWARE]
    chain.insert(0, outer)
    rec = Recorder()
    try:
        with pytest.raises(MiddlewareAbort):
            _run(_request(), recorder=rec)
    finally:
        chain.remove(outer)

    assert seen == {"entered": 1, "returned": 0}
    assert rec.calls == 0


def test_unrelated_middleware_failure_still_fails_open_to_the_native_call(cr_home):
    """The new abort contract does not change the fail-open default for ordinary failures."""
    from hermes_cli.middleware import LLM_EXECUTION_MIDDLEWARE

    def broken(request=None, next_call=None, **_kwargs):
        raise ValueError("unrelated middleware bug")

    chain = cr_home["manager"]._middleware[LLM_EXECUTION_MIDDLEWARE]
    chain.insert(0, broken)
    request = _request()
    rec = Recorder()
    try:
        _, result = _run(request, recorder=rec, provider=GPT_PROVIDER, model=GPT_MODEL,
                         base_url=GPT_BASE_URL, api_mode="codex_responses")
    finally:
        chain.remove(broken)

    assert rec.calls == 1 and rec.payloads[0] is request
    assert result == {"ok": True, "n": 1}


def test_disabled_setting_turns_injection_off(cr_home):
    _set_settings(cr_home["home"], enabled=False)
    cr_home["instruction"].unlink()

    request = _request()
    rec, _ = _run(request)

    assert rec.calls == 1 and rec.payloads[0] is request


# --- default instruction path: the ACTIVE home only -----------------------------------------------


def test_unset_instruction_path_reads_the_active_homes_own_skill_file(cr_home):
    """With no explicit ``instruction_path``, the file under the active ``$HERMES_HOME`` is used."""
    default_file = cr_home["home"] / DEFAULT_RELPATH
    default_file.parent.mkdir(parents=True)
    default_file.write_text(SENTINEL, encoding="utf-8", newline="")
    _set_settings(cr_home["home"], instruction_path=None)
    # Removing the fixture's explicit file proves the default path is what actually got read.
    cr_home["instruction"].unlink()

    rec, _ = _run(_request())

    assert rec.calls == 1
    assert _system_text(rec.payloads[0]) == f"{BASE_SYSTEM}\n\n{SENTINEL}"


def test_missing_default_file_aborts_instead_of_falling_back_to_another_tree(cr_home, read_spy):
    """No checkout-/other-profile fallback: this profile's own prompt missing means abort.

    The active home here is a temp dir with no skill file at all. Any read of an instruction file
    outside that home would be a fallback — the spy makes that visible without touching contents.
    """
    from hermes_cli.middleware import MiddlewareAbort

    _set_settings(cr_home["home"], instruction_path=None)
    cr_home["instruction"].unlink()
    request = _request()
    pristine = deepcopy(request)
    rec = Recorder()

    with pytest.raises(MiddlewareAbort) as excinfo:
        _run(request, recorder=rec)

    assert rec.calls == 0
    assert request == pristine
    home_prefix = str(cr_home["home"])
    assert [p for p in _instruction_reads(read_spy) if not p.startswith(home_prefix)] == []
    assert "unavailable" in str(excinfo.value)


def test_default_path_points_at_the_active_home(cr_home):
    """Unit-level check of the one remaining candidate, independent of the middleware chain."""
    sys.path.insert(0, str(cr_home["home"] / "plugins" / "llm-cr-prompt"))
    try:
        sys.modules.pop("injector", None)
        import injector

        assert injector.default_instruction_path() == cr_home["home"] / DEFAULT_RELPATH
    finally:
        sys.modules.pop("injector", None)
        sys.path.remove(str(cr_home["home"] / "plugins" / "llm-cr-prompt"))


# --- the payload's model must agree, not just the turn context ------------------------------------


def test_payload_model_rewritten_after_route_selection_aborts_without_reading_or_calling(
    cr_home, read_spy
):
    """Context still says crack-talk, but a request middleware re-pointed the payload at GPT.

    The instruction must not be read, the provider must not be called, and nothing about either may
    appear in the abort. The sentinel file is left in place on purpose: deletion alone could not
    distinguish "gate ran first" from "read failed".
    """
    from hermes_cli.middleware import MiddlewareAbort

    request = _request(model=GPT_MODEL)
    pristine = deepcopy(request)
    rec = Recorder()

    with pytest.raises(MiddlewareAbort) as excinfo:
        _run(request, recorder=rec)

    assert rec.calls == 0, "fail-closed: the provider must not be called"
    assert _instruction_reads(read_spy) == [], "the instruction file must not be opened"
    assert request == pristine
    message = str(excinfo.value)
    assert "SENTINEL" not in message and SENTINEL.strip() not in message
    assert str(cr_home["instruction"]) not in message and GPT_MODEL not in message
    assert "llm-cr-prompt" in message


@pytest.mark.parametrize("payload_model", [None, "", "  ", CR_MODEL.upper(), f"{CR_MODEL}-draft"])
def test_payload_model_missing_or_merely_similar_also_aborts(cr_home, read_spy, payload_model):
    from hermes_cli.middleware import MiddlewareAbort

    rec = Recorder()
    with pytest.raises(MiddlewareAbort):
        _run(_request(model=payload_model), recorder=rec)

    assert rec.calls == 0
    assert _instruction_reads(read_spy) == []


def test_non_mapping_request_on_a_matched_route_aborts_before_any_read(cr_home, read_spy):
    from hermes_cli.middleware import MiddlewareAbort

    rec = Recorder()
    with pytest.raises(MiddlewareAbort):
        _run(["not", "a", "mapping"], recorder=rec)

    assert rec.calls == 0
    assert _instruction_reads(read_spy) == []


def test_payload_model_agreeing_still_injects(cr_home):
    """The new check is a guard, not a second gate: the normal matched route is unaffected."""
    rec, _ = _run(_request())

    assert rec.calls == 1
    assert _system_text(rec.payloads[0]) == f"{BASE_SYSTEM}\n\n{SENTINEL}"


# --- base URL comparison: scheme/host casing only, never the path --------------------------------


@pytest.mark.parametrize("base_url", [
    "http://127.0.0.1:65535/V1",
    "http://127.0.0.1:65535/v1/V1",
    "http://127.0.0.1:65535/V1/",
])
def test_base_url_path_case_is_significant_and_never_reads_the_file(cr_home, read_spy, base_url):
    """``/V1`` is a different endpoint from the configured ``/v1`` — so it is not this route."""
    request = _request()

    rec, _ = _run(request, base_url=base_url)

    assert rec.calls == 1 and rec.payloads[0] is request
    assert _instruction_reads(read_spy) == []


@pytest.mark.parametrize("base_url", [
    CR_BASE_URL,
    f"{CR_BASE_URL}/",
    "HTTP://127.0.0.1:65535/v1",
    "Http://127.0.0.1:65535/v1/",
])
def test_scheme_and_host_casing_and_trailing_slash_still_match(cr_home, base_url):
    rec, _ = _run(_request(), base_url=base_url)

    assert _system_text(rec.payloads[0]) == f"{BASE_SYSTEM}\n\n{SENTINEL}"


def test_hostname_casing_matches_but_path_casing_does_not(cr_home):
    """Same assertion pair against a named host, where the host half really has letters."""
    _set_settings(cr_home["home"], route_base_url="http://Local-CR-Host:65535/Api/V1")
    request = _request()

    rec, _ = _run(request, base_url="http://local-cr-host:65535/Api/V1")
    assert _system_text(rec.payloads[0]) == f"{BASE_SYSTEM}\n\n{SENTINEL}"

    rec2, _ = _run(request, base_url="http://local-cr-host:65535/api/v1")
    assert rec2.calls == 1 and rec2.payloads[0] is request
