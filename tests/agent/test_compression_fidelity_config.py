"""compression.fidelity_guard / fidelity_max_retries — config wire.

``resolve_fidelity_settings()`` and the ``ContextCompressor`` constructor both
understood these keys, but ``agent_init`` never read them: every built-in
compressor was constructed with the constructor defaults, so an operator who
set ``compression.fidelity_guard: true`` in ``config.yaml`` silently got no
guard at all.

These tests pin the parse/attach seam end to end — the value has to survive
from the config dict onto ``agent.context_compressor``, since that object is
what the compaction path actually consults.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

from agent.context_compressor import FIDELITY_MAX_RETRIES_CAP
from hermes_state import SessionDB
from run_agent import AIAgent

_UNSET = object()


def _config(*, fidelity_guard=_UNSET, fidelity_max_retries=_UNSET) -> dict:
    compression = {
        "enabled": True,
        "threshold": 0.50,
        "target_ratio": 0.20,
        "protect_first_n": 3,
        "protect_last_n": 20,
    }
    if fidelity_guard is not _UNSET:
        compression["fidelity_guard"] = fidelity_guard
    if fidelity_max_retries is not _UNSET:
        compression["fidelity_max_retries"] = fidelity_max_retries
    return {
        "compression": compression,
        "prompt_caching": {"cache_ttl": "5m"},
        "sessions": {},
        "bedrock": {},
    }


def _make_agent(
    monkeypatch,
    tmp_path: Path,
    *,
    fidelity_guard=_UNSET,
    fidelity_max_retries=_UNSET,
):
    from hermes_cli import config as config_mod

    # ``agent_init`` reads the compression section through
    # ``load_config_readonly`` (the no-deepcopy hot-path variant), so both
    # entry points have to be patched for the config to actually reach the
    # compressor — same shape as tests/run_agent/test_preflight_compression_cap_e2e.py.
    def _load():
        return _config(
            fidelity_guard=fidelity_guard,
            fidelity_max_retries=fidelity_max_retries,
        )

    monkeypatch.setattr(config_mod, "load_config", _load)
    monkeypatch.setattr(config_mod, "load_config_readonly", _load)
    db = SessionDB(db_path=tmp_path / "state.db")
    with contextlib.redirect_stdout(io.StringIO()):
        agent = AIAgent(
            base_url="https://chatgpt.com/backend-api/codex",
            api_key="test-key",
            provider="openai-codex",
            model="gpt-5.5",
            enabled_toolsets=[],
            disabled_toolsets=[],
            quiet_mode=True,
            skip_memory=True,
            session_db=db,
            session_id="fidelity-config-test",
        )
    return agent


class TestCompressionFidelityConfigWire:
    def test_defaults_are_off_with_one_retry(self, monkeypatch, tmp_path):
        # Unset config must behave exactly as before the guard existed.
        agent = _make_agent(monkeypatch, tmp_path)
        assert agent.context_compressor.fidelity_guard is False
        assert agent.context_compressor.fidelity_max_retries == 1

    def test_yaml_bool_and_int_reach_the_compressor(self, monkeypatch, tmp_path):
        # `fidelity_guard: true` / `fidelity_max_retries: 2` in config.yaml
        # arrive as a real bool and int.
        agent = _make_agent(
            monkeypatch, tmp_path, fidelity_guard=True, fidelity_max_retries=2
        )
        assert agent.context_compressor.fidelity_guard is True
        assert agent.context_compressor.fidelity_max_retries == 2

    def test_string_values_are_coerced_and_retries_clamped(
        self, monkeypatch, tmp_path
    ):
        # `hermes config set` writes strings; an absurd retry budget is capped
        # rather than turning one compaction into a multi-minute stall.
        agent = _make_agent(
            monkeypatch, tmp_path, fidelity_guard="true", fidelity_max_retries="99"
        )
        assert agent.context_compressor.fidelity_guard is True
        assert agent.context_compressor.fidelity_max_retries == FIDELITY_MAX_RETRIES_CAP

    def test_guard_off_string_and_negative_retries_floor_at_zero(
        self, monkeypatch, tmp_path
    ):
        agent = _make_agent(
            monkeypatch, tmp_path, fidelity_guard="false", fidelity_max_retries=-4
        )
        assert agent.context_compressor.fidelity_guard is False
        assert agent.context_compressor.fidelity_max_retries == 0

    def test_cli_schema_recognizes_both_fidelity_keys(self):
        from hermes_cli.config import DEFAULT_CONFIG, _validate_config_key

        assert DEFAULT_CONFIG["compression"]["fidelity_guard"] is False
        assert DEFAULT_CONFIG["compression"]["fidelity_max_retries"] == 1
        assert _validate_config_key("compression.fidelity_guard") == (True, None)
        assert _validate_config_key("compression.fidelity_max_retries") == (
            True,
            None,
        )
