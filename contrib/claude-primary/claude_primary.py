#!/usr/bin/env python3
"""Update-safe Claude Code primary wrapper.

Runs one bounded, auditable Claude Code print-mode task and exits.  Standard
library only: this file is deliberately importable and runnable outside the
Hermes package so a Hermes/Claude Code upgrade cannot break it and so it never
becomes a permanent core tool.

Contract
--------
* The working directory is explicit and required; nothing is inferred from the
  caller's cwd.
* The executable is resolved from configuration, then PATH, then known
  Windows/WSL/macOS install locations.
* The version probe (and the optional auth probe) are bounded, single-shot, and
  block execution when they fail.
* The prompt travels on the child's stdin, or through a permission-restricted
  temporary file that is deleted afterwards.  It is never placed on the command
  line and never written to the audit log.
* High-risk work is forced onto Opus regardless of the requested model, and
  ``--claude-arg`` passthrough is fail-closed: a token reaches the child only
  if it matches the wrapper's allowlist exactly.
* The audit log is content-free redacted JSONL: no prompt, no model output, no
  credentials, no raw workdir path.
* Every failure mode has its own reason string and a nonzero exit code.  There
  is no automatic Codex fallback and success is never claimed on failure.

Usage::

    echo "task text" | python claude_primary.py --workdir /path/to/repo \
        --domain auth --max-turns 10 --audit-log /path/to/audit.jsonl
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone

SCHEMA_VERSION = 1
AUDIT_EVENT = "claude_primary_run"

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_NO_EXECUTABLE = 3
EXIT_PROBE_FAILED = 4
EXIT_RUN_FAILED = 5
EXIT_TIMEOUT = 6
EXIT_AUDIT_FAILED = 7

DEFAULT_MODEL = "sonnet"
HIGH_RISK_MODEL = "opus"
MODEL_ALIASES = frozenset({"opus", "sonnet", "haiku", "fable"})

DEFAULT_TIMEOUT_S = 900.0
DEFAULT_PROBE_TIMEOUT_S = 20.0
DEFAULT_MAX_TURNS = 12
MAX_VERSION_CHARS = 120

EXECUTABLE_ENV_VAR = "CLAUDE_PRIMARY_EXECUTABLE"
AUDIT_ENV_VAR = "CLAUDE_PRIMARY_AUDIT_LOG"

# Wrapper policy, not caller preference: every task spawn is confined to the
# pinned workdir.  `--safe-mode` keeps user-level hooks, auto-memory, and
# settings auto-discovery out of the run; `--no-session-persistence` keeps
# session transcripts from being written outside it.  Because auto-discovery is
# off, any authority file the task depends on must be named in the prompt.
MANDATORY_SANDBOX_FLAGS = ("--safe-mode", "--no-session-persistence")

# The passthrough policy, and the only one that decides what reaches the child:
# a `--claude-arg` token is forwarded if and only if it equals an entry here.
#
# A denylist cannot hold this line.  Claude Code gains flags on its own release
# cadence, and a wrapper pinned to a list of the dangerous ones is wrong the day
# a new flag ships that reaches outside the pinned workdir, resumes somebody
# else's session, or attaches another MCP server -- it passes through unreviewed
# until a human notices.  Failing closed inverts that: an unknown flag costs a
# rejected run and a one-line allowlist edit, not a silent escape from the
# sandbox.
#
# Entries are whole tokens compared verbatim, so `--verbose=false`, `--VERBOSE`,
# and a bare value token following a split flag are all rejected: there is no
# prefix or case-folded match for a joined payload to ride in on.  Adding an
# entry means accepting that the flag can neither widen the run's authority nor
# contradict what the audit event attests.
CLAUDE_ARG_ALLOWLIST = frozenset({"--verbose"})

# Flags whose value the wrapper decides and the audit log reports.  These are
# already excluded by the allowlist above; naming them separately only buys a
# more specific failure reason, so a caller who restates a wrapper decision is
# told that rather than being left to guess which token was refused.  Extra
# `--claude-arg` tokens are appended after the wrapper's own, so Claude Code's
# last-wins parsing would otherwise let a caller quietly swap the forced Opus
# model, drop the sandbox, widen the tool allowlist, or lift the turn cap while
# the audit event still attests to the wrapper's choice.  Comparison is
# lowercased so casing variants cannot pick up the generic reason instead.
WRAPPER_OWNED_FLAGS = frozenset(
    {
        "--model",
        "-p",
        "--print",
        "--output-format",
        "--safe-mode",
        "--no-session-persistence",
        "--permission-mode",
        "--dangerously-skip-permissions",
        "--allow-dangerously-skip-permissions",
        "--max-turns",
        "--allowedtools",
        "--allowed-tools",
        "--disallowedtools",
        "--disallowed-tools",
        "--settings",
        "--setting-sources",
        "--plugin-dir",
        "--system-prompt",
        "--system-prompt-file",
        "--append-system-prompt",
        "--append-system-prompt-file",
    }
)

# `cmd.exe /c` reparses its entire command line, so these survive the quoting
# Python applies to argv and can chain a second command whenever the resolved
# executable is a `.cmd`/`.bat` shim (the normal Windows npm install).  `%`
# expands environment variables during that reparse, and `"` closes the quoting
# Python added around the argument, which is what lets the rest of the token be
# read as command syntax in the first place.
CMD_METACHARACTERS = ("&", "|", "^", "<", ">", "%", '"')

# Substrings a build uses when it rejects an argument outright.
_UNSUPPORTED_FLAG_MARKERS = (
    "unknown option",
    "unknown argument",
    "unknown flag",
    "unrecognized option",
    "unrecognized argument",
    "unsupported option",
)

# Domains where a wrong edit is expensive or hard to detect in review.  Work
# tagged with any of these runs on Opus no matter what the caller asked for.
HIGH_RISK_DOMAINS = frozenset(
    {
        "auth",
        "authorization",
        "credentials",
        "crypto",
        "security",
        "payments",
        "billing",
        "persistence",
        "state",
        "migration",
        "concurrency",
        "compression",
        "context",
        "release",
        "deployment",
    }
)

# The complete audit schema.  Every emitted event has exactly these keys, and
# nothing outside this list may ever be written.
AUDIT_KEYS = (
    "schema_version",
    "timestamp",
    "event",
    "status",
    "failure_reason",
    "exit_code",
    "wrapper_exit_code",
    "executable",
    "executable_source",
    "executable_version",
    "auth_state",
    "requested_model",
    "effective_model",
    "model_reason",
    "high_risk",
    "high_risk_domains",
    "workdir_hash",
    "claude_session_id",
    "prompt_bytes",
    "prompt_carrier",
    "timeout_s",
    "probe_timeout_s",
    "max_turns",
    "duration_ms",
)

_AUDIT_VALUE_TYPES = (str, int, float, bool, type(None))


class WrapperError(Exception):
    """A wrapper-level failure carrying its own reason and exit code."""

    def __init__(self, reason: str, exit_code: int, detail: str = ""):
        super().__init__(detail or reason)
        self.reason = reason
        self.exit_code = exit_code
        self.detail = detail


@dataclass(frozen=True)
class Resolution:
    path: str
    source: str  # "configured" | "path" | "known_location"


@dataclass
class ProcResult:
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool = False


# ---------------------------------------------------------------------------
# Executable resolution
# ---------------------------------------------------------------------------


def known_locations(platform: str, env: dict) -> list[str]:
    """Install locations to probe after PATH, in priority order."""
    home = env.get("HOME") or env.get("USERPROFILE") or ""
    if platform == "win32":
        local = env.get("LOCALAPPDATA", "")
        roaming = env.get("APPDATA", "")
        candidates = []
        if local:
            candidates.append(os.path.join(local, "Programs", "claude", "claude.exe"))
            candidates.append(os.path.join(local, "claude", "claude.exe"))
        if roaming:
            candidates.append(os.path.join(roaming, "npm", "claude.cmd"))
        if home:
            candidates.append(os.path.join(home, ".claude", "local", "claude.exe"))
            candidates.append(os.path.join(home, ".claude", "local", "claude.cmd"))
            # native installer target on Windows
            candidates.append(os.path.join(home, ".local", "bin", "claude.exe"))
        candidates.append(os.path.join("C:\\", "Program Files", "nodejs", "claude.cmd"))
        return candidates

    if platform == "darwin":
        candidates = ["/opt/homebrew/bin/claude", "/usr/local/bin/claude"]
    else:  # linux, WSL, and anything else POSIX-shaped
        candidates = ["/usr/local/bin/claude", "/usr/bin/claude", "/snap/bin/claude"]
    if home:
        candidates += [
            f"{home}/.claude/local/claude",
            f"{home}/.local/bin/claude",
            f"{home}/.npm-global/bin/claude",
        ]
    return candidates


def resolve_executable(
    configured: str | None = None,
    *,
    env: dict | None = None,
    platform: str | None = None,
    which=None,
    is_file=None,
) -> Resolution:
    env = os.environ if env is None else env
    platform = sys.platform if platform is None else platform
    which = shutil.which if which is None else which
    is_file = os.path.isfile if is_file is None else is_file

    explicit = configured or env.get(EXECUTABLE_ENV_VAR) or ""
    if explicit:
        if not is_file(explicit):
            raise WrapperError(
                "configured_executable_missing",
                EXIT_NO_EXECUTABLE,
                "configured Claude Code executable does not exist",
            )
        return Resolution(path=explicit, source="configured")

    names = ["claude.cmd", "claude.exe", "claude"] if platform == "win32" else ["claude"]
    for name in names:
        found = which(name)
        if found:
            return Resolution(path=found, source="path")

    for candidate in known_locations(platform, env):
        if is_file(candidate):
            return Resolution(path=candidate, source="known_location")

    raise WrapperError(
        "executable_not_found",
        EXIT_NO_EXECUTABLE,
        "Claude Code was not found via configuration, PATH, or known locations",
    )


def build_command(
    executable: str,
    args: list[str],
    *,
    platform: str | None = None,
    env: dict | None = None,
) -> list[str]:
    """Wrap Windows batch shims in the command processor; pass through elsewhere."""
    platform = sys.platform if platform is None else platform
    env = os.environ if env is None else env
    if platform == "win32" and executable.lower().endswith((".cmd", ".bat")):
        comspec = env.get("COMSPEC") or "cmd.exe"
        return [comspec, "/c", executable, *args]
    return [executable, *args]


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------


def is_high_risk_domain(domain: str) -> bool:
    return domain.strip().lower() in HIGH_RISK_DOMAINS


def _validate_model(model: str) -> str:
    normalized = model.strip()
    if normalized in MODEL_ALIASES or normalized.startswith("claude-"):
        return normalized
    raise WrapperError(
        "unsupported_model",
        EXIT_USAGE,
        "model must be a Claude alias (opus/sonnet/haiku/fable) or a claude-* id",
    )


def select_model(requested: str | None, *, high_risk: bool = False) -> tuple[str, str]:
    """Return ``(effective_model, reason)``.  High-risk work always gets Opus."""
    if requested is not None:
        requested = _validate_model(requested)
    if high_risk:
        return HIGH_RISK_MODEL, "high_risk_requires_opus"
    if requested is None:
        return DEFAULT_MODEL, "default"
    return requested, "requested"


# ---------------------------------------------------------------------------
# Process execution
# ---------------------------------------------------------------------------


def subprocess_runner(cmd, *, cwd, stdin_bytes=None, stdin_path=None, timeout):
    """Default bounded runner.  Never uses a shell; always kills on timeout."""
    try:
        if stdin_path is not None:
            with open(stdin_path, "rb") as handle:
                completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
                    cmd,
                    cwd=cwd,
                    stdin=handle,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=timeout,
                    check=False,
                )
        else:
            completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
                cmd,
                cwd=cwd,
                input=stdin_bytes or b"",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
            )
    except subprocess.TimeoutExpired:
        return ProcResult(returncode=None, stdout="", stderr="", timed_out=True)
    except OSError as exc:
        raise WrapperError("spawn_failed", EXIT_RUN_FAILED, exc.strerror or "spawn failed")
    return ProcResult(
        returncode=completed.returncode,
        stdout=(completed.stdout or b"").decode("utf-8", errors="replace"),
        stderr=(completed.stderr or b"").decode("utf-8", errors="replace"),
    )


def _clean_version(raw: str) -> str:
    first_line = raw.strip().splitlines()[0] if raw.strip() else ""
    printable = "".join(ch for ch in first_line if ch.isprintable())
    return printable[:MAX_VERSION_CHARS]


def probe_version(runner, executable, *, cwd, timeout, platform, env) -> str:
    """Single bounded `--version` probe.  Raises on timeout or failure."""
    cmd = build_command(executable, ["--version"], platform=platform, env=env)
    result = runner(cmd, cwd=cwd, stdin_bytes=b"", timeout=timeout)
    if result.timed_out:
        raise WrapperError(
            "version_probe_timeout", EXIT_PROBE_FAILED, "claude --version did not return in time"
        )
    if result.returncode != 0:
        raise WrapperError(
            "version_probe_failed", EXIT_PROBE_FAILED, "claude --version exited nonzero"
        )
    return _clean_version(result.stdout or result.stderr)


def probe_auth(runner, executable, *, cwd, timeout, platform, env) -> str:
    """Single bounded auth probe.  Returns "ok"; never records account identity."""
    cmd = build_command(executable, ["auth", "status", "--text"], platform=platform, env=env)
    result = runner(cmd, cwd=cwd, stdin_bytes=b"", timeout=timeout)
    if result.timed_out:
        raise WrapperError(
            "auth_probe_timeout", EXIT_PROBE_FAILED, "claude auth status did not return in time"
        )
    if result.returncode != 0:
        raise WrapperError(
            "auth_probe_failed", EXIT_PROBE_FAILED, "claude auth status reports not authenticated"
        )
    return "ok"


@contextlib.contextmanager
def protected_prompt_file(data: bytes):
    """Yield a 0600 prompt file inside a private directory; always remove it."""
    tmpdir = tempfile.mkdtemp(prefix="claude-primary-")
    with contextlib.suppress(OSError):
        os.chmod(tmpdir, 0o700)
    path = os.path.join(tmpdir, "prompt.txt")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        yield path
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


def hash_workdir(path: str) -> str:
    digest = hashlib.sha256(os.path.abspath(path).encode("utf-8")).hexdigest()
    return f"sha256:{digest[:32]}"


def build_audit_event(payload: dict) -> dict:
    """Return a complete, key-allowlisted, content-free audit event."""
    unknown = sorted(set(payload) - set(AUDIT_KEYS))
    if unknown:
        raise WrapperError(
            "audit_schema_violation",
            EXIT_AUDIT_FAILED,
            f"audit fields not in the allowlist: {', '.join(unknown)}",
        )
    event = {}
    for key in AUDIT_KEYS:
        value = payload.get(key)
        if key == "high_risk_domains":
            value = sorted(value or ())
        elif not isinstance(value, _AUDIT_VALUE_TYPES):
            raise WrapperError(
                "audit_schema_violation",
                EXIT_AUDIT_FAILED,
                f"audit field {key} has a non-scalar value",
            )
        event[key] = value
    event["schema_version"] = SCHEMA_VERSION
    event["event"] = AUDIT_EVENT
    return event


def append_audit(audit_path: str, event: dict) -> None:
    """Append one JSONL line.  Any failure is loud, never swallowed."""
    try:
        parent = os.path.dirname(os.path.abspath(audit_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        fd = os.open(audit_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=False) + "\n")
    except OSError as exc:
        raise WrapperError(
            "audit_write_failed", EXIT_AUDIT_FAILED, exc.strerror or "audit log is not writable"
        )


def default_audit_path(env: dict) -> str:
    configured = env.get(AUDIT_ENV_VAR)
    if configured:
        return configured
    home = env.get("HOME") or env.get("USERPROFILE") or os.path.expanduser("~")
    return os.path.join(home, ".hermes", "claude-primary", "audit.jsonl")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class _Parser(argparse.ArgumentParser):
    def error(self, message):  # pragma: no cover - exercised via main()
        raise WrapperError("invalid_arguments", EXIT_USAGE, message)


def _build_parser() -> _Parser:
    parser = _Parser(prog="claude_primary", add_help=True)
    parser.add_argument("--workdir", help="explicit working directory for the run (required)")
    parser.add_argument("--audit-log", help="JSONL audit destination")
    parser.add_argument("--executable", help="explicit Claude Code executable path")
    parser.add_argument("--model", help="requested model alias or claude-* id")
    parser.add_argument(
        "--domain",
        action="append",
        default=[],
        help="task domain tag; high-risk domains force Opus (repeatable)",
    )
    parser.add_argument("--high-risk", action="store_true", help="force the high-risk model")
    parser.add_argument("--prompt-file", help="read the prompt from this file instead of stdin")
    parser.add_argument(
        "--prompt-carrier",
        choices=("stdin", "file"),
        default="stdin",
        help="deliver the prompt on the child's stdin or via a 0600 temp file",
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--probe-timeout", type=float, default=DEFAULT_PROBE_TIMEOUT_S)
    parser.add_argument(
        "--probe-auth",
        action="store_true",
        help="additionally run a bounded auth-status probe before executing",
    )
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument("--effort", help="reasoning effort passed through to Claude Code")
    parser.add_argument("--allowed-tools", help="value for --allowedTools")
    parser.add_argument("--permission-mode", help="value for --permission-mode")
    parser.add_argument(
        "--claude-arg",
        action="append",
        default=[],
        help="extra literal Claude Code argument (repeatable); never the prompt. "
        "Fail-closed: only an exact match on the wrapper's allowlist is forwarded",
    )
    parser.add_argument("--dry-run", action="store_true", help="probe and resolve, do not execute")
    return parser


def _read_prompt(args, stdin_text: str | None) -> bytes:
    if args.prompt_file:
        try:
            with open(args.prompt_file, "rb") as handle:
                raw = handle.read()
        except OSError as exc:
            raise WrapperError(
                "prompt_file_unreadable", EXIT_USAGE, exc.strerror or "prompt file unreadable"
            )
    else:
        if stdin_text is None:
            stdin_text = sys.stdin.read()
        raw = stdin_text.encode("utf-8")
    if not raw.strip():
        raise WrapperError("missing_prompt", EXIT_USAGE, "no prompt supplied on stdin or --prompt-file")
    return raw


def _resolve_workdir(raw: str | None) -> str:
    if not raw:
        raise WrapperError("missing_workdir", EXIT_USAGE, "--workdir is required")
    resolved = os.path.abspath(raw)
    if not os.path.isdir(resolved):
        raise WrapperError("invalid_workdir", EXIT_USAGE, "--workdir is not an existing directory")
    return resolved


def validate_claude_args(claude_args) -> None:
    """Fail closed on every ``--claude-arg`` token outside the allowlist.

    ``CLAUDE_ARG_ALLOWLIST`` is authoritative: a token is forwarded only if it
    matches an entry exactly, so unknown newer flags, joined values
    (``--verbose=false``), casing variants (``--VERBOSE``), and the bare value
    token that follows a split flag are all refused.  The wrapper-owned check
    runs first purely to give a restated wrapper decision its own reason.

    Failure is content-free — the offending token is caller text and reaches
    neither the audit log nor stderr.
    """
    for arg in claude_args:
        if any(char in arg for char in CMD_METACHARACTERS):
            raise WrapperError(
                "claude_arg_shell_metacharacter",
                EXIT_USAGE,
                "--claude-arg may not contain the cmd.exe metacharacters "
                f"({' '.join(CMD_METACHARACTERS)}); they would be reparsed by a .cmd shim",
            )
        name = arg.split("=", 1)[0].strip().lower()
        if name in WRAPPER_OWNED_FLAGS:
            raise WrapperError(
                "claude_arg_overrides_wrapper_flag",
                EXIT_USAGE,
                "--claude-arg may not set a flag the wrapper owns and audits "
                "(model, print/output-format, sandbox, permission-mode, max-turns, "
                "tool allowlists); use the wrapper's own option instead",
            )
        if arg not in CLAUDE_ARG_ALLOWLIST:
            raise WrapperError(
                "claude_arg_not_allowlisted",
                EXIT_USAGE,
                "--claude-arg forwards only the tokens on the wrapper's allowlist "
                f"({', '.join(sorted(CLAUDE_ARG_ALLOWLIST))}), matched exactly; every "
                "other token -- including flags newer than this wrapper, joined "
                "values, and casing variants -- is refused before the run starts",
            )


def _task_args(args, model: str) -> list[str]:
    task = [
        "-p",
        "--output-format",
        "json",
        "--model",
        model,
        "--max-turns",
        str(args.max_turns),
    ]
    if args.effort:
        task += ["--effort", args.effort]
    if args.allowed_tools:
        task += ["--allowedTools", args.allowed_tools]
    if args.permission_mode:
        task += ["--permission-mode", args.permission_mode]
    task += list(args.claude_arg)
    # Injected right after the executable.  `validate_claude_args` has already
    # refused any caller copy, so this stays a guard against a future edit to
    # the fixed prefix rather than a deduplicator for caller input.
    missing = [flag for flag in MANDATORY_SANDBOX_FLAGS if flag not in task]
    return missing + task


def classify_run_failure(stderr: str, stdout: str = "") -> str:
    """Reason string for a nonzero task exit.

    A build that rejects a mandatory sandbox flag gets its own reason: the run
    fails closed there rather than being retried without the sandbox.
    """
    haystack = f"{stderr}\n{stdout}".lower()
    if any(marker in haystack for marker in _UNSUPPORTED_FLAG_MARKERS) and any(
        flag in haystack for flag in MANDATORY_SANDBOX_FLAGS
    ):
        return "sandbox_flag_unsupported"
    return "nonzero_exit"


def _parse_result(stdout: str) -> tuple[str | None, str | None, str]:
    """Return ``(session_id, failure_reason, result_text)`` from a JSON result."""
    try:
        payload = json.loads(stdout.strip().splitlines()[-1]) if stdout.strip() else {}
    except (ValueError, IndexError):
        return None, "unparsable_result", ""
    if not isinstance(payload, dict):
        return None, "unparsable_result", ""
    session_id = payload.get("session_id")
    subtype = payload.get("subtype")
    text = payload.get("result") or ""
    if subtype and subtype != "success":
        return session_id, f"result_subtype_{subtype}", text
    return session_id, None, text


def main(argv=None, *, stdin_text=None, runner=None, platform=None, env=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    env = dict(os.environ) if env is None else env
    platform = sys.platform if platform is None else platform
    runner = subprocess_runner if runner is None else runner

    try:
        args = _build_parser().parse_args(argv)
    except WrapperError as exc:
        print(f"claude-primary: {exc.reason}: {exc.detail}", file=sys.stderr)
        return exc.exit_code

    audit_path = args.audit_log or default_audit_path(env)
    started = time.monotonic()
    record = {
        "status": "failure",
        "failure_reason": None,
        "exit_code": None,
        "wrapper_exit_code": None,
        "executable": None,
        "executable_source": None,
        "executable_version": None,
        "auth_state": "unknown",
        "requested_model": args.model,
        "effective_model": None,
        "model_reason": None,
        "high_risk": False,
        "high_risk_domains": [],
        "workdir_hash": None,
        "claude_session_id": None,
        "prompt_bytes": None,
        "prompt_carrier": args.prompt_carrier,
        "timeout_s": args.timeout,
        "probe_timeout_s": args.probe_timeout,
        "max_turns": args.max_turns,
    }
    exit_code = EXIT_OK
    detail = ""

    try:
        workdir = _resolve_workdir(args.workdir)
        record["workdir_hash"] = hash_workdir(workdir)

        prompt = _read_prompt(args, stdin_text)
        record["prompt_bytes"] = len(prompt)

        risky_domains = [d for d in args.domain if is_high_risk_domain(d)]
        high_risk = bool(args.high_risk or risky_domains)
        record["high_risk"] = high_risk
        record["high_risk_domains"] = [d.strip().lower() for d in risky_domains]

        model, model_reason = select_model(args.model, high_risk=high_risk)
        record["effective_model"] = model
        record["model_reason"] = model_reason

        # After model selection so the rejection is audited against the choice
        # it tried to overturn, and before any spawn so nothing is launched.
        validate_claude_args(args.claude_arg)

        resolution = resolve_executable(
            args.executable, env=env, platform=platform, which=shutil.which
        )
        record["executable"] = resolution.path
        record["executable_source"] = resolution.source

        record["executable_version"] = probe_version(
            runner,
            resolution.path,
            cwd=workdir,
            timeout=args.probe_timeout,
            platform=platform,
            env=env,
        )
        # The auth probe belongs to preflight, not to execution: a dry run that
        # asked for it must still report a real auth_state instead of "unknown".
        if args.probe_auth:
            try:
                record["auth_state"] = probe_auth(
                    runner,
                    resolution.path,
                    cwd=workdir,
                    timeout=args.probe_timeout,
                    platform=platform,
                    env=env,
                )
            except WrapperError:
                record["auth_state"] = "failed"
                raise

        if args.dry_run:
            record["status"] = "dry_run"
        else:
            cmd = build_command(
                resolution.path, _task_args(args, model), platform=platform, env=env
            )
            if args.prompt_carrier == "file":
                with protected_prompt_file(prompt) as prompt_path:
                    result = runner(cmd, cwd=workdir, stdin_path=prompt_path, timeout=args.timeout)
            else:
                result = runner(cmd, cwd=workdir, stdin_bytes=prompt, timeout=args.timeout)

            if result.timed_out:
                raise WrapperError(
                    "execution_timeout", EXIT_TIMEOUT, "claude did not finish within --timeout"
                )
            record["exit_code"] = result.returncode
            if result.returncode != 0:
                session_id, _, _ = _parse_result(result.stdout)
                record["claude_session_id"] = session_id
                reason = classify_run_failure(result.stderr, result.stdout)
                if reason == "sandbox_flag_unsupported":
                    raise WrapperError(
                        reason,
                        EXIT_RUN_FAILED,
                        "this Claude Code build rejected a mandatory sandbox flag "
                        f"({', '.join(MANDATORY_SANDBOX_FLAGS)}); the run is not retried unsandboxed",
                    )
                raise WrapperError(reason, EXIT_RUN_FAILED, "claude exited nonzero")

            session_id, failure_reason, text = _parse_result(result.stdout)
            record["claude_session_id"] = session_id
            if failure_reason:
                raise WrapperError(failure_reason, EXIT_RUN_FAILED, "claude did not complete the task")
            record["status"] = "success"
            if text:
                print(text)
    except WrapperError as exc:
        record["failure_reason"] = exc.reason
        exit_code = exc.exit_code
        detail = exc.detail
    except Exception as exc:  # unexpected: still audited, still nonzero
        record["failure_reason"] = "internal_error"
        exit_code = EXIT_RUN_FAILED
        detail = type(exc).__name__

    record["duration_ms"] = int((time.monotonic() - started) * 1000)
    record["timestamp"] = datetime.now(timezone.utc).isoformat()
    record["wrapper_exit_code"] = exit_code

    if exit_code != EXIT_OK:
        print(
            f"claude-primary: {record['failure_reason']}: {detail}\n"
            "claude-primary: Claude Code run did not succeed. There is no automatic fallback "
            "to another coding agent; rerun deliberately or escalate to the operator.",
            file=sys.stderr,
        )

    try:
        append_audit(audit_path, build_audit_event(record))
    except WrapperError as exc:
        print(
            f"claude-primary: {exc.reason}: {exc.detail} (audit path not writable; "
            "the run is reported as failed because it cannot be audited)",
            file=sys.stderr,
        )
        return EXIT_AUDIT_FAILED

    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
