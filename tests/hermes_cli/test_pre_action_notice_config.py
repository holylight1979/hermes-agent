"""Config surface for the pre-action-notice gate.

Two contracts:

1. The keys are part of the official schema, so ``hermes config set
   agent.require_pre_action_notice true`` does not print the "not a recognized
   config key" notice.
2. The agent resolves them into instance state at init, defaulting to OFF so
   existing users see no behaviour change.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from hermes_cli.config_defaults import DEFAULT_CONFIG


def test_defaults_are_off_with_two_retries():
    agent_defaults = DEFAULT_CONFIG["agent"]
    assert agent_defaults["require_pre_action_notice"] is False
    assert agent_defaults["pre_action_notice_max_retries"] == 2


@pytest.mark.parametrize(
    "key",
    ["agent.require_pre_action_notice", "agent.pre_action_notice_max_retries"],
)
def test_keys_are_recognized_by_config_validation(key):
    from hermes_cli.config import _validate_config_key

    is_known, suggestion = _validate_config_key(key)
    assert is_known is True, f"{key} must be an official config key"
    assert suggestion is None


def _build_agent(agent_section):
    """Build a real AIAgent with a stubbed config, no network, no tools."""
    import copy

    from run_agent import AIAgent

    config = copy.deepcopy(DEFAULT_CONFIG)
    config["agent"] = {**config.get("agent", {}), **agent_section}
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("hermes_cli.config.load_config_readonly", return_value=config),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    return agent


def test_agent_defaults_to_disabled_gate():
    agent = _build_agent({})
    assert agent.require_pre_action_notice is False
    assert agent.pre_action_notice_max_retries == 2
    assert agent._pre_action_notice_retries == 0


def test_agent_reads_enabled_gate_from_config():
    agent = _build_agent(
        {"require_pre_action_notice": True, "pre_action_notice_max_retries": 5}
    )
    assert agent.require_pre_action_notice is True
    assert agent.pre_action_notice_max_retries == 5


@pytest.mark.parametrize("bad", ["not-a-number", None, -3, 7.5, [], {"a": 1}])
def test_bad_retry_values_fall_back_to_default_without_crashing(bad):
    agent = _build_agent(
        {"require_pre_action_notice": True, "pre_action_notice_max_retries": bad}
    )
    assert agent.pre_action_notice_max_retries == 2


def test_zero_retries_is_honoured():
    """0 is a legitimate setting: fail the turn on the first bad notice."""
    agent = _build_agent(
        {"require_pre_action_notice": True, "pre_action_notice_max_retries": 0}
    )
    assert agent.pre_action_notice_max_retries == 0
