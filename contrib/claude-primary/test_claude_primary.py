"""Focused tests for the update-safe Claude Code primary wrapper.

Run with:
    python -m pytest contrib/claude-primary/test_claude_primary.py -q

The wrapper is standard-library only and lives outside the Hermes package, so
these tests import it by path rather than by package name.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import stat
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parent / "claude_primary.py"
_SPEC = importlib.util.spec_from_file_location("claude_primary", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
cp = importlib.util.module_from_spec(_SPEC)
# Register before exec: dataclasses resolves annotations via sys.modules.
sys.modules["claude_primary"] = cp
_SPEC.loader.exec_module(cp)


# --------------------------------------------------------------------------
# Test doubles
# --------------------------------------------------------------------------


@dataclass
class FakeCall:
    cmd: list[str]
    cwd: str | None
    stdin_bytes: bytes | None
    stdin_path: str | None
    timeout: float
    stdin_mode: int | None = None
    stdin_existed: bool = True


@dataclass
class FakeRunner:
    """Records invocations and replays scripted results in order."""

    results: list[cp.ProcResult] = field(default_factory=list)
    calls: list[FakeCall] = field(default_factory=list)

    def __call__(self, cmd, *, cwd, stdin_bytes=None, stdin_path=None, timeout):
        mode = None
        existed = True
        if stdin_path is not None:
            existed = os.path.isfile(stdin_path)
            if existed:
                mode = stat.S_IMODE(os.stat(stdin_path).st_mode)
        self.calls.append(
            FakeCall(
                cmd=list(cmd),
                cwd=cwd,
                stdin_bytes=stdin_bytes,
                stdin_path=stdin_path,
                timeout=timeout,
                stdin_mode=mode,
                stdin_existed=existed,
            )
        )
        if not self.results:
            raise AssertionError(f"unexpected extra process spawn: {cmd}")
        return self.results.pop(0)


def _version_ok() -> cp.ProcResult:
    return cp.ProcResult(returncode=0, stdout="2.1.220 (Claude Code)\n", stderr="")


def _result_json(session_id: str = "sess-1", text: str = "MODEL_OUTPUT_TEXT") -> str:
    return json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "result": text,
            "session_id": session_id,
            "num_turns": 3,
            "total_cost_usd": 0.01,
        }
    )


def _run_ok(session_id: str = "sess-1", text: str = "MODEL_OUTPUT_TEXT") -> cp.ProcResult:
    return cp.ProcResult(returncode=0, stdout=_result_json(session_id, text), stderr="")


def _code_mentions(module_path: Path, needle: str) -> bool:
    """True if executable code (not comments/docstrings) mentions ``needle``."""
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                docstrings.add(doc)
    tokens = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value not in docstrings:
                tokens.append(node.value)
        elif isinstance(node, ast.Name):
            tokens.append(node.id)
        elif isinstance(node, ast.Attribute):
            tokens.append(node.attr)
    return any(needle in token.lower() for token in tokens)


@pytest.fixture()
def workdir(tmp_path: Path) -> Path:
    d = tmp_path / "project"
    d.mkdir()
    return d


@pytest.fixture()
def audit_log(tmp_path: Path) -> Path:
    return tmp_path / "audit" / "claude-primary.jsonl"


def _fake_exe(tmp_root: Path) -> Path:
    """A real file on disk — resolution requires a configured executable to exist."""
    exe = tmp_root / "claude-bin"
    if not exe.exists():
        exe.write_text("", encoding="utf-8")
    return exe


def _argv(workdir: Path, audit_log: Path, *extra: str) -> list[str]:
    return [
        "--workdir",
        str(workdir),
        "--audit-log",
        str(audit_log),
        "--executable",
        str(_fake_exe(workdir.parent)),
        *extra,
    ]


def _audit_events(audit_log: Path) -> list[dict]:
    text = audit_log.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


# --------------------------------------------------------------------------
# Executable resolution
# --------------------------------------------------------------------------


class TestExecutableResolution:
    def test_configured_executable_wins_over_path(self):
        res = cp.resolve_executable(
            "/explicit/claude",
            env={},
            platform="linux",
            which=lambda name: "/usr/bin/claude",
            is_file=lambda p: True,
        )
        assert res.path == "/explicit/claude"
        assert res.source == "configured"

    def test_configured_missing_has_its_own_failure_reason(self):
        with pytest.raises(cp.WrapperError) as excinfo:
            cp.resolve_executable(
                "/explicit/claude",
                env={},
                platform="linux",
                which=lambda name: "/usr/bin/claude",
                is_file=lambda p: False,
            )
        assert excinfo.value.reason == "configured_executable_missing"
        assert excinfo.value.exit_code == cp.EXIT_NO_EXECUTABLE

    def test_env_override_is_treated_as_configured(self):
        res = cp.resolve_executable(
            None,
            env={"CLAUDE_PRIMARY_EXECUTABLE": "/env/claude"},
            platform="linux",
            which=lambda name: None,
            is_file=lambda p: p == "/env/claude",
        )
        assert res.path == "/env/claude"
        assert res.source == "configured"

    def test_path_lookup_when_nothing_configured(self):
        res = cp.resolve_executable(
            None,
            env={},
            platform="linux",
            which=lambda name: "/usr/local/bin/claude" if name == "claude" else None,
            is_file=lambda p: True,
        )
        assert res.path == "/usr/local/bin/claude"
        assert res.source == "path"

    def test_windows_known_location_used_when_path_misses(self):
        env = {
            "LOCALAPPDATA": r"C:\Users\dev\AppData\Local",
            "APPDATA": r"C:\Users\dev\AppData\Roaming",
            "USERPROFILE": r"C:\Users\dev",
        }
        expected = os.path.join(env["APPDATA"], "npm", "claude.cmd")
        res = cp.resolve_executable(
            None,
            env=env,
            platform="win32",
            which=lambda name: None,
            is_file=lambda p: p == expected,
        )
        assert res.path == expected
        assert res.source == "known_location"

    def test_windows_native_install_under_userprofile_local_bin_is_found(self):
        env = {"USERPROFILE": r"C:\Users\dev"}
        expected = os.path.join(env["USERPROFILE"], ".local", "bin", "claude.exe")
        res = cp.resolve_executable(
            None,
            env=env,
            platform="win32",
            which=lambda name: None,
            is_file=lambda p: p == expected,
        )
        assert res.path == expected
        assert res.source == "known_location"

    def test_windows_local_bin_does_not_outrank_existing_candidates(self):
        env = {
            "LOCALAPPDATA": r"C:\Users\dev\AppData\Local",
            "APPDATA": r"C:\Users\dev\AppData\Roaming",
            "USERPROFILE": r"C:\Users\dev",
        }
        candidates = cp.known_locations("win32", env)
        local_bin = os.path.join(env["USERPROFILE"], ".local", "bin", "claude.exe")
        assert local_bin in candidates
        # every pre-existing candidate still resolves ahead of the native install path
        for earlier in (
            os.path.join(env["LOCALAPPDATA"], "Programs", "claude", "claude.exe"),
            os.path.join(env["LOCALAPPDATA"], "claude", "claude.exe"),
            os.path.join(env["APPDATA"], "npm", "claude.cmd"),
            os.path.join(env["USERPROFILE"], ".claude", "local", "claude.exe"),
            os.path.join(env["USERPROFILE"], ".claude", "local", "claude.cmd"),
        ):
            assert candidates.index(earlier) < candidates.index(local_bin)
        assert candidates.index(local_bin) < candidates.index(
            os.path.join("C:\\", "Program Files", "nodejs", "claude.cmd")
        )

    def test_windows_path_still_wins_over_local_bin(self):
        env = {"USERPROFILE": r"C:\Users\dev"}
        res = cp.resolve_executable(
            None,
            env=env,
            platform="win32",
            which=lambda name: r"C:\shim\claude.cmd" if name == "claude.cmd" else None,
            is_file=lambda p: True,
        )
        assert res.path == r"C:\shim\claude.cmd"
        assert res.source == "path"

    def test_posix_known_locations_include_local_and_npm_installs(self):
        env = {"HOME": "/home/dev"}
        candidates = cp.known_locations("linux", env)
        assert "/usr/local/bin/claude" in candidates
        assert "/home/dev/.claude/local/claude" in candidates

    def test_not_found_anywhere_is_nonzero_with_distinct_reason(self):
        with pytest.raises(cp.WrapperError) as excinfo:
            cp.resolve_executable(
                None,
                env={},
                platform="linux",
                which=lambda name: None,
                is_file=lambda p: False,
            )
        assert excinfo.value.reason == "executable_not_found"
        assert excinfo.value.exit_code == cp.EXIT_NO_EXECUTABLE


# --------------------------------------------------------------------------
# Model selection
# --------------------------------------------------------------------------


class TestModelSelection:
    def test_high_risk_forces_opus_over_requested_model(self):
        model, reason = cp.select_model("sonnet", high_risk=True)
        assert model == "opus"
        assert reason == "high_risk_requires_opus"

    def test_high_risk_domain_marks_run_high_risk(self):
        assert cp.is_high_risk_domain("Auth") is True
        assert cp.is_high_risk_domain("compression") is True
        assert cp.is_high_risk_domain("docs") is False

    def test_non_high_risk_keeps_requested_model(self):
        model, reason = cp.select_model("haiku", high_risk=False)
        assert model == "haiku"
        assert reason == "requested"

    def test_default_model_when_none_requested(self):
        model, reason = cp.select_model(None, high_risk=False)
        assert model == cp.DEFAULT_MODEL
        assert reason == "default"

    def test_unknown_model_is_rejected(self):
        with pytest.raises(cp.WrapperError) as excinfo:
            cp.select_model("gpt-5", high_risk=False)
        assert excinfo.value.reason == "unsupported_model"
        assert excinfo.value.exit_code == cp.EXIT_USAGE

    def test_high_risk_flag_in_cli_selects_opus(self, workdir, audit_log):
        runner = FakeRunner(results=[_version_ok(), _run_ok()])
        code = cp.main(
            _argv(workdir, audit_log, "--model", "sonnet", "--domain", "auth"),
            stdin_text="do the risky thing",
            runner=runner,
            platform="linux",
            env={},
        )
        assert code == cp.EXIT_OK
        exec_cmd = runner.calls[-1].cmd
        assert "--model" in exec_cmd
        assert exec_cmd[exec_cmd.index("--model") + 1] == "opus"
        event = _audit_events(audit_log)[0]
        assert event["requested_model"] == "sonnet"
        assert event["effective_model"] == "opus"
        assert event["model_reason"] == "high_risk_requires_opus"
        assert event["high_risk"] is True


# --------------------------------------------------------------------------
# Command construction
# --------------------------------------------------------------------------


class TestCommandConstruction:
    def test_prompt_text_never_appears_in_argv(self, workdir, audit_log):
        runner = FakeRunner(results=[_version_ok(), _run_ok()])
        secret = "SUPER_SECRET_PROMPT_BODY"
        code = cp.main(
            _argv(workdir, audit_log),
            stdin_text=secret,
            runner=runner,
            platform="linux",
            env={},
        )
        assert code == cp.EXIT_OK
        for call in runner.calls:
            assert all(secret not in part for part in call.cmd)

    def test_command_pins_print_mode_json_output_and_turn_cap(self, workdir, audit_log):
        runner = FakeRunner(results=[_version_ok(), _run_ok()])
        cp.main(
            _argv(workdir, audit_log, "--max-turns", "7"),
            stdin_text="task",
            runner=runner,
            platform="linux",
            env={},
        )
        cmd = runner.calls[-1].cmd
        assert cmd[0] == str(_fake_exe(workdir.parent))
        assert "-p" in cmd
        assert cmd[cmd.index("--output-format") + 1] == "json"
        assert cmd[cmd.index("--max-turns") + 1] == "7"

    def test_windows_batch_shim_is_wrapped_in_comspec(self):
        cmd = cp.build_command(
            r"C:\Users\dev\AppData\Roaming\npm\claude.cmd",
            ["--version"],
            platform="win32",
            env={"COMSPEC": r"C:\Windows\System32\cmd.exe"},
        )
        assert cmd[:2] == [r"C:\Windows\System32\cmd.exe", "/c"]
        assert cmd[2].endswith("claude.cmd")
        assert cmd[-1] == "--version"

    def test_posix_executable_is_not_shell_wrapped(self):
        cmd = cp.build_command("/opt/claude/claude", ["--version"], platform="linux", env={})
        assert cmd == ["/opt/claude/claude", "--version"]


# --------------------------------------------------------------------------
# Mandatory sandbox flags
# --------------------------------------------------------------------------


class TestSandboxFlags:
    """Every task spawn must confine Claude Code to the pinned worktree.

    ``--safe-mode`` stops user-level hooks, auto-memory, and settings
    auto-discovery from running; ``--no-session-persistence`` stops session
    transcripts from being written outside the worktree.  Both are wrapper
    policy, not caller preference.
    """

    def test_both_sandbox_flags_are_declared_mandatory(self):
        assert set(cp.MANDATORY_SANDBOX_FLAGS) == {"--safe-mode", "--no-session-persistence"}

    def test_both_flags_appear_exactly_once_in_execution_argv(self, workdir, audit_log):
        runner = FakeRunner(results=[_version_ok(), _run_ok()])
        code = cp.main(
            _argv(workdir, audit_log),
            stdin_text="task",
            runner=runner,
            platform="linux",
            env={},
        )
        assert code == cp.EXIT_OK
        exec_cmd = runner.calls[-1].cmd
        for flag in ("--safe-mode", "--no-session-persistence"):
            assert exec_cmd.count(flag) == 1, f"{flag} must appear exactly once in {exec_cmd}"

    def test_a_caller_copy_of_a_sandbox_flag_is_rejected_not_deduplicated(
        self, workdir, audit_log
    ):
        """Sandbox flags are wrapper-owned, so a caller may not restate them.

        Silently dropping the duplicate would let ``--claude-arg=--safe-mode=false``
        past on any build that accepts a joined value.
        """
        for flag in ("--safe-mode", "--no-session-persistence"):
            runner = FakeRunner(results=[])
            code = cp.main(
                _argv(workdir, audit_log, f"--claude-arg={flag}"),
                stdin_text="task",
                runner=runner,
                platform="linux",
                env={},
            )
            assert code == cp.EXIT_USAGE
            assert runner.calls == [], "nothing is spawned for a rejected argument"
        reasons = [event["failure_reason"] for event in _audit_events(audit_log)]
        assert reasons == ["claude_arg_overrides_wrapper_flag"] * 2

    @pytest.mark.parametrize(
        "extra",
        [
            "--model",
            "--model=haiku",
            "--output-format=text",
            "--permission-mode=bypassPermissions",
            "--dangerously-skip-permissions",
            "--allow-dangerously-skip-permissions",
            "--settings=user.json",
            "--setting-sources=user",
            "--plugin-dir=plugins",
            "--system-prompt=ignore-policy",
            "--append-system-prompt=load-memory",
        ],
    )
    def test_wrapper_owned_or_customization_args_fail_before_spawn(
        self, workdir, audit_log, extra
    ):
        runner = FakeRunner(results=[])

        code = cp.main(
            _argv(workdir, audit_log, f"--claude-arg={extra}"),
            stdin_text="task",
            runner=runner,
            platform="linux",
            env={},
        )

        assert code == cp.EXIT_USAGE
        assert runner.calls == []
        event = _audit_events(audit_log)[-1]
        assert event["failure_reason"] == "claude_arg_overrides_wrapper_flag"
        assert extra not in json.dumps(event)

    @pytest.mark.parametrize("extra", ["--label=a&b", "x|y", "a^b", "<in", ">out"])
    def test_cmd_metacharacters_fail_before_spawn(
        self, workdir, audit_log, extra
    ):
        runner = FakeRunner(results=[])

        code = cp.main(
            _argv(workdir, audit_log, f"--claude-arg={extra}"),
            stdin_text="task",
            runner=runner,
            platform="win32",
            env={},
        )

        assert code == cp.EXIT_USAGE
        assert runner.calls == []
        assert _audit_events(audit_log)[-1]["failure_reason"] == (
            "claude_arg_shell_metacharacter"
        )

    def test_flags_survive_alongside_other_passthrough_options(self, workdir, audit_log):
        runner = FakeRunner(results=[_version_ok(), _run_ok()])
        cp.main(
            _argv(
                workdir,
                audit_log,
                "--permission-mode",
                "acceptEdits",
                "--allowed-tools",
                "Edit Read",
                "--effort",
                "high",
                "--claude-arg=--verbose",
            ),
            stdin_text="task",
            runner=runner,
            platform="linux",
            env={},
        )
        exec_cmd = runner.calls[-1].cmd
        for flag in ("--safe-mode", "--no-session-persistence"):
            assert exec_cmd.count(flag) == 1
        assert exec_cmd[exec_cmd.index("--permission-mode") + 1] == "acceptEdits"
        assert "--verbose" in exec_cmd

    def test_flags_are_present_on_the_high_risk_opus_path_too(self, workdir, audit_log):
        runner = FakeRunner(results=[_version_ok(), _run_ok()])
        cp.main(
            _argv(workdir, audit_log, "--domain", "auth"),
            stdin_text="task",
            runner=runner,
            platform="linux",
            env={},
        )
        exec_cmd = runner.calls[-1].cmd
        assert exec_cmd[exec_cmd.index("--model") + 1] == "opus"
        for flag in ("--safe-mode", "--no-session-persistence"):
            assert exec_cmd.count(flag) == 1

    def test_version_probe_argv_stays_a_bare_version_call(self, workdir, audit_log):
        runner = FakeRunner(results=[_version_ok(), _run_ok()])
        cp.main(
            _argv(workdir, audit_log),
            stdin_text="task",
            runner=runner,
            platform="linux",
            env={},
        )
        probe_cmd = runner.calls[0].cmd
        assert probe_cmd[-1] == "--version"
        for flag in ("--safe-mode", "--no-session-persistence"):
            assert flag not in probe_cmd

    def test_flags_do_not_disturb_oauth_auth_probe_support(self, workdir, audit_log):
        """--probe-auth still runs its own bare `auth status` spawn and passes."""
        runner = FakeRunner(
            results=[
                _version_ok(),
                cp.ProcResult(returncode=0, stdout="", stderr=""),
                _run_ok(),
            ]
        )
        code = cp.main(
            _argv(workdir, audit_log, "--probe-auth"),
            stdin_text="task",
            runner=runner,
            platform="linux",
            env={},
        )
        assert code == cp.EXIT_OK
        auth_cmd = runner.calls[1].cmd
        assert auth_cmd[1:] == ["auth", "status", "--text"]
        assert _audit_events(audit_log)[0]["auth_state"] == "ok"
        exec_cmd = runner.calls[-1].cmd
        for flag in ("--safe-mode", "--no-session-persistence"):
            assert exec_cmd.count(flag) == 1

    def test_a_build_rejecting_a_mandatory_flag_fails_closed(self, workdir, audit_log):
        """No silent downgrade: if the flag is unsupported the run must not proceed."""
        runner = FakeRunner(
            results=[
                _version_ok(),
                cp.ProcResult(
                    returncode=1, stdout="", stderr="error: unknown option '--safe-mode'\n"
                ),
            ]
        )
        code = cp.main(
            _argv(workdir, audit_log),
            stdin_text="task",
            runner=runner,
            platform="linux",
            env={},
        )
        assert code == cp.EXIT_RUN_FAILED
        assert len(runner.calls) == 2, "there is no unsandboxed retry"
        assert _audit_events(audit_log)[0]["failure_reason"] == "sandbox_flag_unsupported"

    def test_sandbox_argv_and_audit_stay_content_free(self, workdir, audit_log):
        runner = FakeRunner(results=[_version_ok(), _run_ok(text="MODEL_OUTPUT_SECRET")])
        code = cp.main(
            _argv(workdir, audit_log),
            stdin_text="PROMPT_SECRET_BODY",
            runner=runner,
            platform="linux",
            env={},
        )
        assert code == cp.EXIT_OK
        exec_call = runner.calls[-1]
        for flag in ("--safe-mode", "--no-session-persistence"):
            assert exec_call.cmd.count(flag) == 1
        # the prompt still travels on stdin, never in the sandboxed argv
        assert exec_call.stdin_bytes == b"PROMPT_SECRET_BODY"
        assert all("PROMPT_SECRET_BODY" not in part for part in exec_call.cmd)
        raw = audit_log.read_text(encoding="utf-8")
        assert "PROMPT_SECRET_BODY" not in raw
        assert "MODEL_OUTPUT_SECRET" not in raw
        assert str(workdir) not in raw
        assert set(json.loads(raw)) == set(cp.AUDIT_KEYS)


# --------------------------------------------------------------------------
# --claude-arg passthrough allowlist
# --------------------------------------------------------------------------


class TestClaudeArgAllowlist:
    """Passthrough is fail-closed: only an exact allowlist match reaches the child.

    A denylist of dangerous flags goes stale the moment Claude Code ships a new
    one, and the wrapper's whole job is to stay correct across upgrades it does
    not control.  So the assertions below are about the *default* for anything
    unrecognized, not about any particular flag being enumerated somewhere.
    """

    def test_allowlist_is_exactly_verbose(self):
        assert set(cp.CLAUDE_ARG_ALLOWLIST) == {"--verbose"}

    @pytest.mark.parametrize(
        "extra",
        [
            # flags that reach outside the pinned workdir or resume other state
            "--fallback-model",
            "--fallback-model=opus",
            "--add-dir",
            "--add-dir=/etc",
            "--permission-prompt-tool",
            "--permission-prompt-tool=mcp__approver__approve",
            "--mcp-config",
            "--mcp-config=servers.json",
            "--agents",
            "--agents=agents.json",
            "--continue",
            "-c",
            "--resume",
            "--resume=sess-someone-else",
            "-r",
            "--session-id",
            "--session-id=11111111-2222-3333-4444-555555555555",
            "--fork-session",
            # a flag this wrapper has never heard of: the case a denylist misses
            "--teleport-to-prod",
            "--some-flag-shipped-after-this-wrapper",
            # exact match only -- no joined value, no prefix, no casing variant
            "--verbose=false",
            "--verbose=--dangerously-skip-permissions",
            "--verbosely",
            "--VERBOSE",
            "--Verbose",
            "-v",
            # a bare value token, e.g. the tail of a split `--fallback-model opus`
            "opus",
        ],
    )
    def test_everything_outside_the_allowlist_fails_before_spawn(
        self, workdir, audit_log, extra
    ):
        runner = FakeRunner(results=[])

        code = cp.main(
            _argv(workdir, audit_log, f"--claude-arg={extra}"),
            stdin_text="task",
            runner=runner,
            platform="linux",
            env={},
        )

        assert code == cp.EXIT_USAGE
        assert runner.calls == [], "nothing is spawned for a rejected argument"
        event = _audit_events(audit_log)[-1]
        assert event["failure_reason"] == "claude_arg_not_allowlisted"
        assert extra not in json.dumps(event)

    @pytest.mark.parametrize(
        "extra",
        ["--MODEL=opus", "--Model=opus", "--Safe-Mode", "--AllowedTools=Bash"],
    )
    def test_casing_variants_of_wrapper_owned_flags_keep_their_own_reason(
        self, workdir, audit_log, extra
    ):
        """Case folding is what stops a variant from landing in the generic bucket."""
        runner = FakeRunner(results=[])

        code = cp.main(
            _argv(workdir, audit_log, f"--claude-arg={extra}"),
            stdin_text="task",
            runner=runner,
            platform="linux",
            env={},
        )

        assert code == cp.EXIT_USAGE
        assert runner.calls == []
        assert _audit_events(audit_log)[-1]["failure_reason"] == (
            "claude_arg_overrides_wrapper_flag"
        )

    def test_percent_and_double_quote_are_cmd_metacharacters(self):
        for char in ("%", '"'):
            assert char in cp.CMD_METACHARACTERS

    @pytest.mark.parametrize(
        "extra",
        ["%PATH%", "--label=%COMSPEC%", '--label="x"', '"&whoami"', "50%"],
    )
    def test_percent_and_quote_fail_before_spawn(self, workdir, audit_log, extra):
        """`%` expands and `"` breaks out of Python's quoting when a .cmd shim reparses."""
        runner = FakeRunner(results=[])

        code = cp.main(
            _argv(workdir, audit_log, f"--claude-arg={extra}"),
            stdin_text="task",
            runner=runner,
            platform="win32",
            env={},
        )

        assert code == cp.EXIT_USAGE
        assert runner.calls == []
        assert _audit_events(audit_log)[-1]["failure_reason"] == (
            "claude_arg_shell_metacharacter"
        )

    def test_rejection_records_the_reason_and_never_the_token(
        self, workdir, audit_log, capsys
    ):
        """The refused token is caller text: it belongs in neither audit nor stderr."""
        payload = "--resume=SECRET_SESSION_TOKEN_VALUE"
        code = cp.main(
            _argv(workdir, audit_log, f"--claude-arg={payload}"),
            stdin_text="task",
            runner=FakeRunner(results=[]),
            platform="linux",
            env={},
        )
        assert code == cp.EXIT_USAGE
        raw = audit_log.read_text(encoding="utf-8")
        assert "SECRET_SESSION_TOKEN_VALUE" not in raw
        assert "SECRET_SESSION_TOKEN_VALUE" not in capsys.readouterr().err
        event = json.loads(raw)
        assert set(event) == set(cp.AUDIT_KEYS)
        assert event["failure_reason"] == "claude_arg_not_allowlisted"
        assert event["status"] == "failure"

    def test_allowlisted_verbose_reaches_the_child_argv(self, workdir, audit_log):
        """The allowlist is fail-closed, not closed: its one entry still passes through."""
        runner = FakeRunner(results=[_version_ok(), _run_ok()])

        code = cp.main(
            _argv(workdir, audit_log, "--claude-arg=--verbose"),
            stdin_text="task",
            runner=runner,
            platform="linux",
            env={},
        )

        assert code == cp.EXIT_OK
        exec_cmd = runner.calls[-1].cmd
        assert exec_cmd.count("--verbose") == 1
        # appended after the wrapper's own flags, which still stand
        assert exec_cmd.index("--verbose") > exec_cmd.index("--model")
        for flag in cp.MANDATORY_SANDBOX_FLAGS:
            assert exec_cmd.count(flag) == 1
        assert _audit_events(audit_log)[0]["status"] == "success"

    def test_a_rejected_token_alongside_an_allowlisted_one_still_blocks_the_run(
        self, workdir, audit_log
    ):
        runner = FakeRunner(results=[])

        code = cp.main(
            _argv(
                workdir,
                audit_log,
                "--claude-arg=--verbose",
                "--claude-arg=--add-dir=/etc",
            ),
            stdin_text="task",
            runner=runner,
            platform="linux",
            env={},
        )

        assert code == cp.EXIT_USAGE
        assert runner.calls == []
        assert _audit_events(audit_log)[-1]["failure_reason"] == "claude_arg_not_allowlisted"


# --------------------------------------------------------------------------
# Bounded probes
# --------------------------------------------------------------------------


class TestProbes:
    def test_version_probe_is_bounded_and_blocks_execution_on_timeout(self, workdir, audit_log):
        runner = FakeRunner(
            results=[cp.ProcResult(returncode=None, stdout="", stderr="", timed_out=True)]
        )
        code = cp.main(
            _argv(workdir, audit_log, "--probe-timeout", "3"),
            stdin_text="task",
            runner=runner,
            platform="linux",
            env={},
        )
        assert code == cp.EXIT_PROBE_FAILED
        assert len(runner.calls) == 1, "no execution may follow a failed probe"
        assert runner.calls[0].timeout == 3
        event = _audit_events(audit_log)[0]
        assert event["failure_reason"] == "version_probe_timeout"
        assert event["status"] == "failure"

    def test_version_probe_failure_has_distinct_reason(self, workdir, audit_log):
        runner = FakeRunner(
            results=[cp.ProcResult(returncode=1, stdout="", stderr="not installed")]
        )
        code = cp.main(
            _argv(workdir, audit_log),
            stdin_text="task",
            runner=runner,
            platform="linux",
            env={},
        )
        assert code == cp.EXIT_PROBE_FAILED
        assert _audit_events(audit_log)[0]["failure_reason"] == "version_probe_failed"

    def test_auth_probe_failure_is_a_separate_reason_and_stores_no_identity(
        self, workdir, audit_log
    ):
        runner = FakeRunner(
            results=[
                _version_ok(),
                cp.ProcResult(
                    returncode=1,
                    stdout='{"loggedIn": false, "account": "dev@example.com"}',
                    stderr="",
                ),
            ]
        )
        code = cp.main(
            _argv(workdir, audit_log, "--probe-auth"),
            stdin_text="task",
            runner=runner,
            platform="linux",
            env={},
        )
        assert code == cp.EXIT_PROBE_FAILED
        event = _audit_events(audit_log)[0]
        assert event["failure_reason"] == "auth_probe_failed"
        assert event["auth_state"] == "failed"
        assert "dev@example.com" not in audit_log.read_text(encoding="utf-8")

    def test_version_string_is_recorded_and_truncated(self, workdir, audit_log):
        runner = FakeRunner(
            results=[
                cp.ProcResult(returncode=0, stdout="9" * 300 + "\n", stderr=""),
                _run_ok(),
            ]
        )
        cp.main(
            _argv(workdir, audit_log),
            stdin_text="task",
            runner=runner,
            platform="linux",
            env={},
        )
        version = _audit_events(audit_log)[0]["executable_version"]
        assert len(version) <= cp.MAX_VERSION_CHARS


# --------------------------------------------------------------------------
# Execution, workdir pinning, failure modes
# --------------------------------------------------------------------------


class TestExecution:
    def test_workdir_is_pinned_for_every_spawn(self, workdir, audit_log):
        runner = FakeRunner(results=[_version_ok(), _run_ok()])
        cp.main(
            _argv(workdir, audit_log),
            stdin_text="task",
            runner=runner,
            platform="linux",
            env={},
        )
        assert [c.cwd for c in runner.calls] == [str(workdir.resolve())] * 2

    def test_missing_workdir_is_a_usage_failure_before_any_spawn(self, tmp_path, audit_log):
        runner = FakeRunner(results=[])
        code = cp.main(
            [
                "--workdir",
                str(tmp_path / "nope"),
                "--audit-log",
                str(audit_log),
                "--executable",
                str(_fake_exe(tmp_path)),
            ],
            stdin_text="task",
            runner=runner,
            platform="linux",
            env={},
        )
        assert code == cp.EXIT_USAGE
        assert runner.calls == []
        assert _audit_events(audit_log)[0]["failure_reason"] == "invalid_workdir"

    def test_workdir_is_required(self, audit_log):
        code = cp.main(
            ["--audit-log", str(audit_log)],
            stdin_text="task",
            runner=FakeRunner(results=[]),
            platform="linux",
            env={},
        )
        assert code == cp.EXIT_USAGE

    def test_success_returns_zero_and_records_session_id(self, workdir, audit_log, capsys):
        runner = FakeRunner(results=[_version_ok(), _run_ok(session_id="sess-abc")])
        code = cp.main(
            _argv(workdir, audit_log),
            stdin_text="task",
            runner=runner,
            platform="linux",
            env={},
        )
        assert code == cp.EXIT_OK
        event = _audit_events(audit_log)[0]
        assert event["status"] == "success"
        assert event["claude_session_id"] == "sess-abc"
        assert event["failure_reason"] is None
        assert "MODEL_OUTPUT_TEXT" in capsys.readouterr().out

    def test_execution_timeout_is_nonzero_with_its_own_reason(self, workdir, audit_log):
        runner = FakeRunner(
            results=[
                _version_ok(),
                cp.ProcResult(returncode=None, stdout="", stderr="", timed_out=True),
            ]
        )
        code = cp.main(
            _argv(workdir, audit_log, "--timeout", "30"),
            stdin_text="task",
            runner=runner,
            platform="linux",
            env={},
        )
        assert code == cp.EXIT_TIMEOUT
        assert runner.calls[-1].timeout == 30
        assert _audit_events(audit_log)[0]["failure_reason"] == "execution_timeout"

    def test_nonzero_child_exit_is_reported_as_failure(self, workdir, audit_log):
        runner = FakeRunner(
            results=[_version_ok(), cp.ProcResult(returncode=3, stdout="", stderr="boom")]
        )
        code = cp.main(
            _argv(workdir, audit_log),
            stdin_text="task",
            runner=runner,
            platform="linux",
            env={},
        )
        assert code == cp.EXIT_RUN_FAILED
        event = _audit_events(audit_log)[0]
        assert event["failure_reason"] == "nonzero_exit"
        assert event["exit_code"] == 3
        assert event["status"] == "failure"

    def test_error_result_payload_is_a_failure_not_a_success_claim(self, workdir, audit_log):
        payload = json.dumps(
            {
                "type": "result",
                "subtype": "error_max_turns",
                "result": "",
                "session_id": "sess-err",
            }
        )
        runner = FakeRunner(
            results=[_version_ok(), cp.ProcResult(returncode=0, stdout=payload, stderr="")]
        )
        code = cp.main(
            _argv(workdir, audit_log),
            stdin_text="task",
            runner=runner,
            platform="linux",
            env={},
        )
        assert code == cp.EXIT_RUN_FAILED
        event = _audit_events(audit_log)[0]
        assert event["failure_reason"] == "result_subtype_error_max_turns"
        assert event["claude_session_id"] == "sess-err"

    def test_no_automatic_codex_fallback_after_failure(self, workdir, audit_log, capsys):
        runner = FakeRunner(
            results=[_version_ok(), cp.ProcResult(returncode=1, stdout="", stderr="fail")]
        )
        code = cp.main(
            _argv(workdir, audit_log),
            stdin_text="task",
            runner=runner,
            platform="linux",
            env={},
        )
        assert code != cp.EXIT_OK
        assert len(runner.calls) == 2, "failure must not trigger another agent spawn"
        # Every spawn is the resolved Claude executable; nothing else is ever run.
        expected_exe = str(_fake_exe(workdir.parent))
        for call in runner.calls:
            assert call.cmd[0] == expected_exe
            assert not any(part.lower() == "codex" for part in call.cmd[1:])
        # No executable statement anywhere in the wrapper references another agent
        # (the module docstring may document the rule, so scan code only).
        assert not _code_mentions(_MODULE_PATH, "codex")
        captured = capsys.readouterr()
        assert "no automatic fallback" in captured.err.lower()

    def test_dry_run_probes_but_does_not_execute(self, workdir, audit_log):
        runner = FakeRunner(results=[_version_ok()])
        code = cp.main(
            _argv(workdir, audit_log, "--dry-run"),
            stdin_text="task",
            runner=runner,
            platform="linux",
            env={},
        )
        assert code == cp.EXIT_OK
        assert len(runner.calls) == 1
        assert _audit_events(audit_log)[0]["status"] == "dry_run"

    def test_dry_run_without_probe_auth_leaves_auth_state_unknown(self, workdir, audit_log):
        runner = FakeRunner(results=[_version_ok()])
        code = cp.main(
            _argv(workdir, audit_log, "--dry-run"),
            stdin_text="task",
            runner=runner,
            platform="linux",
            env={},
        )
        assert code == cp.EXIT_OK
        assert [c.cmd[-1] for c in runner.calls] == ["--version"]
        assert _audit_events(audit_log)[0]["auth_state"] == "unknown"

    def test_dry_run_with_probe_auth_records_auth_state_without_executing(
        self, workdir, audit_log
    ):
        """The auth probe is preflight: a dry run must not report auth as unknown."""
        runner = FakeRunner(
            results=[_version_ok(), cp.ProcResult(returncode=0, stdout="", stderr="")]
        )
        code = cp.main(
            _argv(workdir, audit_log, "--dry-run", "--probe-auth"),
            stdin_text="task",
            runner=runner,
            platform="linux",
            env={},
        )
        assert code == cp.EXIT_OK
        # exactly the two probes: version, then auth — and never the coding task
        assert len(runner.calls) == 2, "a dry run must not spawn the coding task"
        assert runner.calls[0].cmd[-1] == "--version"
        assert runner.calls[1].cmd[1:] == ["auth", "status", "--text"]
        for call in runner.calls:
            assert "-p" not in call.cmd
        event = _audit_events(audit_log)[0]
        assert event["status"] == "dry_run"
        assert event["auth_state"] == "ok"
        assert event["failure_reason"] is None

    def test_dry_run_auth_probe_failure_blocks_and_is_audited(self, workdir, audit_log):
        runner = FakeRunner(
            results=[_version_ok(), cp.ProcResult(returncode=1, stdout="", stderr="not logged in")]
        )
        code = cp.main(
            _argv(workdir, audit_log, "--dry-run", "--probe-auth"),
            stdin_text="task",
            runner=runner,
            platform="linux",
            env={},
        )
        assert code == cp.EXIT_PROBE_FAILED
        assert len(runner.calls) == 2, "no execution may follow a failed probe"
        event = _audit_events(audit_log)[0]
        assert event["failure_reason"] == "auth_probe_failed"
        assert event["auth_state"] == "failed"
        assert event["status"] == "failure"

    def test_dry_run_probes_carry_no_mandatory_sandbox_flags(self, workdir, audit_log):
        """Probe argv stays bare; the sandbox flags belong to the task spawn only."""
        runner = FakeRunner(
            results=[_version_ok(), cp.ProcResult(returncode=0, stdout="", stderr="")]
        )
        cp.main(
            _argv(workdir, audit_log, "--dry-run", "--probe-auth"),
            stdin_text="task",
            runner=runner,
            platform="linux",
            env={},
        )
        for call in runner.calls:
            for flag in cp.MANDATORY_SANDBOX_FLAGS:
                assert flag not in call.cmd


# --------------------------------------------------------------------------
# Prompt carriers
# --------------------------------------------------------------------------


class TestPromptCarrier:
    def test_stdin_carrier_delivers_prompt_on_child_stdin(self, workdir, audit_log):
        runner = FakeRunner(results=[_version_ok(), _run_ok()])
        cp.main(
            _argv(workdir, audit_log),
            stdin_text="the task body",
            runner=runner,
            platform="linux",
            env={},
        )
        exec_call = runner.calls[-1]
        assert exec_call.stdin_bytes == b"the task body"
        assert exec_call.stdin_path is None

    def test_file_carrier_uses_restricted_temp_file_and_removes_it(self, workdir, audit_log):
        runner = FakeRunner(results=[_version_ok(), _run_ok()])
        cp.main(
            _argv(workdir, audit_log, "--prompt-carrier", "file"),
            stdin_text="the task body",
            runner=runner,
            platform="linux",
            env={},
        )
        exec_call = runner.calls[-1]
        assert exec_call.stdin_path is not None
        assert exec_call.stdin_existed is True
        if os.name == "posix":
            assert exec_call.stdin_mode == 0o600
        assert not os.path.exists(exec_call.stdin_path), "prompt file must not survive the run"
        assert _audit_events(audit_log)[0]["prompt_carrier"] == "file"

    def test_prompt_file_input_is_read_not_passed_as_argument(
        self, workdir, audit_log, tmp_path
    ):
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("PROMPT_FROM_FILE", encoding="utf-8")
        runner = FakeRunner(results=[_version_ok(), _run_ok()])
        cp.main(
            _argv(workdir, audit_log, "--prompt-file", str(prompt_file)),
            stdin_text="",
            runner=runner,
            platform="linux",
            env={},
        )
        exec_call = runner.calls[-1]
        assert exec_call.stdin_bytes == b"PROMPT_FROM_FILE"
        assert str(prompt_file) not in exec_call.cmd

    def test_empty_prompt_is_a_usage_failure(self, workdir, audit_log):
        runner = FakeRunner(results=[])
        code = cp.main(
            _argv(workdir, audit_log),
            stdin_text="   \n",
            runner=runner,
            platform="linux",
            env={},
        )
        assert code == cp.EXIT_USAGE
        assert runner.calls == []
        assert _audit_events(audit_log)[0]["failure_reason"] == "missing_prompt"


# --------------------------------------------------------------------------
# Audit record
# --------------------------------------------------------------------------


class TestAudit:
    def test_audit_is_single_jsonl_line_with_allowlisted_keys_only(self, workdir, audit_log):
        runner = FakeRunner(results=[_version_ok(), _run_ok()])
        cp.main(
            _argv(workdir, audit_log),
            stdin_text="task",
            runner=runner,
            platform="linux",
            env={},
        )
        lines = audit_log.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        event = json.loads(lines[0])
        assert set(event) == set(cp.AUDIT_KEYS)
        assert event["event"] == "claude_primary_run"
        assert event["timestamp"].endswith("+00:00")

    def test_audit_stores_workdir_hash_not_path_and_no_content(self, workdir, audit_log):
        runner = FakeRunner(
            results=[_version_ok(), _run_ok(text="MODEL_OUTPUT_SECRET")]
        )
        cp.main(
            _argv(workdir, audit_log),
            stdin_text="PROMPT_SECRET_BODY",
            runner=runner,
            platform="linux",
            env={},
        )
        raw = audit_log.read_text(encoding="utf-8")
        assert "PROMPT_SECRET_BODY" not in raw
        assert "MODEL_OUTPUT_SECRET" not in raw
        assert str(workdir) not in raw
        event = json.loads(raw)
        assert event["workdir_hash"] == cp.hash_workdir(str(workdir.resolve()))
        assert event["prompt_bytes"] == len(b"PROMPT_SECRET_BODY")

    def test_audit_builder_rejects_unknown_keys(self):
        with pytest.raises(cp.WrapperError) as excinfo:
            cp.build_audit_event({"executable": "/opt/claude/claude", "prompt": "leak"})
        assert excinfo.value.reason == "audit_schema_violation"

    def test_audit_appends_rather_than_truncates(self, workdir, audit_log):
        for session in ("s1", "s2"):
            cp.main(
                _argv(workdir, audit_log),
                stdin_text="task",
                runner=FakeRunner(results=[_version_ok(), _run_ok(session_id=session)]),
                platform="linux",
                env={},
            )
        events = _audit_events(audit_log)
        assert [e["claude_session_id"] for e in events] == ["s1", "s2"]

    def test_unwritable_audit_target_fails_loudly(self, workdir, tmp_path, capsys):
        blocked = tmp_path / "blocked"
        blocked.mkdir()
        code = cp.main(
            [
                "--workdir",
                str(workdir),
                "--audit-log",
                str(blocked),
                "--executable",
                str(_fake_exe(tmp_path)),
            ],
            stdin_text="task",
            runner=FakeRunner(results=[_version_ok(), _run_ok()]),
            platform="linux",
            env={},
        )
        assert code == cp.EXIT_AUDIT_FAILED
        assert "audit" in capsys.readouterr().err.lower()

    @pytest.mark.skipif(os.name != "posix", reason="POSIX file modes only")
    def test_audit_file_is_owner_only(self, workdir, audit_log):
        cp.main(
            _argv(workdir, audit_log),
            stdin_text="task",
            runner=FakeRunner(results=[_version_ok(), _run_ok()]),
            platform="linux",
            env={},
        )
        assert stat.S_IMODE(audit_log.stat().st_mode) == 0o600


# --------------------------------------------------------------------------
# End-to-end against a real child process
# --------------------------------------------------------------------------


def _write_shim(tmp_path: Path, record_dir: Path) -> Path:
    recorder = tmp_path / "recorder.py"
    recorder.write_text(
        "import json, os, sys\n"
        "args = sys.argv[1:]\n"
        "if args and args[0] == '--version':\n"
        "    print('2.1.220 (Claude Code)')\n"
        "    raise SystemExit(0)\n"
        "data = sys.stdin.read()\n"
        "rec = {'argv': args, 'cwd': os.getcwd(), 'stdin': data}\n"
        f"with open(r'{record_dir / 'record.json'}', 'w', encoding='utf-8') as fh:\n"
        "    json.dump(rec, fh)\n"
        "print(json.dumps({'type': 'result', 'subtype': 'success',\n"
        "                  'result': 'SHIM_RESULT', 'session_id': 'sess-real'}))\n",
        encoding="utf-8",
    )
    if os.name == "nt":
        shim = tmp_path / "claude.cmd"
        shim.write_text(
            f'@echo off\r\n"{sys.executable}" "{recorder}" %*\r\nexit /b %ERRORLEVEL%\r\n',
            encoding="utf-8",
        )
    else:
        shim = tmp_path / "claude"
        shim.write_text(
            f'#!/bin/sh\nexec "{sys.executable}" "{recorder}" "$@"\n', encoding="utf-8"
        )
        shim.chmod(0o700)
    return shim


class TestRealChildProcess:
    def test_real_run_pins_workdir_and_pipes_prompt_on_stdin(
        self, tmp_path, workdir, audit_log
    ):
        record_dir = tmp_path / "records"
        record_dir.mkdir()
        shim = _write_shim(tmp_path, record_dir)
        code = cp.main(
            [
                "--workdir",
                str(workdir),
                "--audit-log",
                str(audit_log),
                "--executable",
                str(shim),
                "--timeout",
                "60",
            ],
            stdin_text="REAL_PROMPT_BODY",
        )
        assert code == cp.EXIT_OK
        record = json.loads((record_dir / "record.json").read_text(encoding="utf-8"))
        assert record["stdin"].strip() == "REAL_PROMPT_BODY"
        assert Path(record["cwd"]).resolve() == workdir.resolve()
        assert "REAL_PROMPT_BODY" not in " ".join(record["argv"])
        assert "-p" in record["argv"]
        event = _audit_events(audit_log)[0]
        assert event["claude_session_id"] == "sess-real"
        assert "REAL_PROMPT_BODY" not in audit_log.read_text(encoding="utf-8")

    def test_real_run_argv_carries_both_sandbox_flags_exactly_once(
        self, tmp_path, workdir, audit_log
    ):
        record_dir = tmp_path / "records"
        record_dir.mkdir()
        shim = _write_shim(tmp_path, record_dir)
        code = cp.main(
            [
                "--workdir",
                str(workdir),
                "--audit-log",
                str(audit_log),
                "--executable",
                str(shim),
                "--timeout",
                "60",
                "--claude-arg=--verbose",
            ],
            stdin_text="REAL_PROMPT_BODY",
        )
        assert code == cp.EXIT_OK
        argv = json.loads((record_dir / "record.json").read_text(encoding="utf-8"))["argv"]
        for flag in ("--safe-mode", "--no-session-persistence"):
            assert argv.count(flag) == 1, f"{flag} must appear exactly once in {argv}"
        assert "REAL_PROMPT_BODY" not in " ".join(argv)
