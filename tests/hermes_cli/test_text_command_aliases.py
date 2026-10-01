"""Exact bare-text → slash-command aliases: matcher contract and the CLI ingress seam.

The contract under test: an exact bare trigger becomes the configured slash command and is
dispatched locally, so no LLM call happens for the trigger itself; anything that merely *contains*
the trigger stays an ordinary prompt and reaches the agent.
"""

from types import SimpleNamespace

import pytest

from hermes_cli.text_command_aliases import alias_table, resolve_text_command_alias

SWITCH_IN = "/model hf.co/vendor/Some-Model-GGUF:Q4_K_M --provider local-direct --session"
SWITCH_OUT = "/model gpt-6-astra-900k --provider openai-codex --session"

CONFIG = {
    "text_command_aliases": {
        "enabled": True,
        "aliases": {"llm-cr": SWITCH_IN, "llm-cr-end": SWITCH_OUT},
    }
}


class TestAliasMatching:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("llm-cr", SWITCH_IN),
            ("  llm-cr  ", SWITCH_IN),
            ("llm-cr\n", SWITCH_IN),
            ("llm-cr-end", SWITCH_OUT),
            (" llm-cr-end ", SWITCH_OUT),
        ],
    )
    def test_exact_bare_trigger_resolves(self, text, expected):
        assert resolve_text_command_alias(text, CONFIG) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "llm-cr 你好嗎",                    # trigger plus content
            "請幫我看 llm-cr 的腳本",            # trigger as a substring
            "llm-cr-talk",                      # trigger as a prefix
            "llm-crack-talk",                   # old chat phrase
            "debug llm-cr-end please",          # meta discussion
            "/skill llm-cr",                    # skill invocation
            "/llm-cr",                          # slash input is never alias-rewritten
            "LLM-CR",                           # case-sensitive: not the trigger
            "小晴模式",                          # removed legacy trigger
            "",
            "   ",
        ],
    )
    def test_near_miss_text_is_left_alone(self, text):
        assert resolve_text_command_alias(text, CONFIG) is None

    @pytest.mark.parametrize("text", [None, 123, ["llm-cr"]])
    def test_non_string_input_is_left_alone(self, text):
        assert resolve_text_command_alias(text, CONFIG) is None

    def test_disabled_section_resolves_nothing(self):
        cfg = {"text_command_aliases": {"enabled": False, "aliases": {"llm-cr": SWITCH_IN}}}
        assert resolve_text_command_alias("llm-cr", cfg) is None

    @pytest.mark.parametrize("cfg", [{}, None, {"text_command_aliases": None},
                                     {"text_command_aliases": {"aliases": "nope"}}])
    def test_missing_or_malformed_section_resolves_nothing(self, cfg):
        assert resolve_text_command_alias("llm-cr", cfg) is None

    def test_invalid_entries_are_dropped_and_valid_ones_survive(self):
        cfg = {"text_command_aliases": {"enabled": True, "aliases": {
            "llm-cr": SWITCH_IN,
            "/shadow": "/status",          # a slash trigger would shadow a real command
            "": "/status",                 # empty trigger
            "multi\nline": "/status",      # multi-line trigger
            "not-a-command": "echo hi",    # target is not a slash command
            "chained": "/status\n/reset",  # target must be a single command
            "wrong-type": 42,
        }}}
        assert alias_table(cfg) == {"llm-cr": SWITCH_IN}


class TestConfigSchema:
    def test_alias_key_paths_validate(self):
        """``hermes config set text_command_aliases.aliases.<trigger>`` must be accepted."""
        from hermes_cli.config import _validate_config_key

        for key in ("text_command_aliases", "text_command_aliases.enabled",
                    "text_command_aliases.aliases.llm-cr"):
            is_known, _suggestion = _validate_config_key(key)
            assert is_known, key


def _make_cli(config):
    """A real ``HermesCLI`` with only the collaborators ``_tui_process_one_input`` touches."""
    import threading

    from cli import HermesCLI

    cli = HermesCLI.__new__(HermesCLI)
    cli.config = config
    cli._voice_lock = threading.Lock()
    cli._voice_mode = False
    cli._voice_continuous = False
    cli._pending_resume_sessions = None
    cli._pending_agent_seed = None
    cli._app = SimpleNamespace(invalidate=lambda: None, is_running=False)
    cli.calls = {"commands": [], "chats": []}
    cli.handle_bang_shell = lambda text: False
    cli.process_command = lambda cmd: cli.calls["commands"].append(cmd) or True
    cli.chat = lambda text, **kw: cli.calls["chats"].append(text)
    cli._print_user_message_preview = lambda *_a, **_k: None
    cli._turn_summary_begin = lambda *_a, **_k: None
    cli._tui_after_turn = lambda *_a, **_k: None
    return cli


class TestCLIIngress:
    def test_exact_trigger_dispatches_the_slash_command_without_a_turn(self):
        """The trigger never becomes a chat turn: no agent call, so no LLM call for it."""
        cli = _make_cli(CONFIG)

        cli._tui_process_one_input("llm-cr")

        assert cli.calls["commands"] == [SWITCH_IN]
        assert cli.calls["chats"] == []

    def test_exit_trigger_dispatches_the_restore_command(self):
        cli = _make_cli(CONFIG)

        cli._tui_process_one_input("  llm-cr-end  ")

        assert cli.calls["commands"] == [SWITCH_OUT]
        assert cli.calls["chats"] == []

    def test_near_miss_text_reaches_the_agent(self):
        cli = _make_cli(CONFIG)

        cli._tui_process_one_input("llm-cr 你好嗎")

        assert cli.calls["commands"] == []
        assert cli.calls["chats"] == ["llm-cr 你好嗎"]

    def test_no_aliases_configured_leaves_the_trigger_as_a_prompt(self):
        cli = _make_cli({})

        cli._tui_process_one_input("llm-cr")

        assert cli.calls["commands"] == []
        assert cli.calls["chats"] == ["llm-cr"]
