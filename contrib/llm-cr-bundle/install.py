#!/usr/bin/env python3
"""Reapplicable installer for the llm-cr bundle (exact text aliases + the session-detours plugin +
route-gated prompt injection).

The feature itself lives in PLUGINS (``payload/plugins/``); ``core.patch`` only adds the generic
seams those plugins need — a host context for plugin slash commands, a busy policy for them, the
alias matcher and config key, and the fail-closed middleware abort.

Standard library only, except PyYAML for the config merge — the one non-stdlib dependency, probed
explicitly in :func:`_require_yaml` with an actionable message instead of an ImportError traceback.

Commands
--------
``check``     Report exactly what ``apply`` would do. Writes nothing. (``dry-run`` is an alias.)
``apply``     Back up, apply ``core.patch``, install the plugin payload, merge config, write receipt.
``verify``    Prove the install: exact aliases resolve, plugins import and load, patch is present.
``rollback``  Undo one receipt's own changes, refusing if the user has since edited those files.

Design rules this file holds to
-------------------------------
* **Fail closed.** A conflicting patch hunk, a missing/incompatible target API, an unresolvable
  provider base URL or a missing instruction-file prerequisite aborts BEFORE any write. The core
  source is never overwritten wholesale: the only writer is ``git apply``, which refuses a hunk it
  cannot place.
* **Idempotent.** A second ``apply`` over an already-installed target changes nothing and says so.
* **Backed up.** Every path the run would touch is copied to a backup directory outside both the
  repo and the Hermes home first, together with its exact prior existence, hash and mode.
* **Scoped.** ``--repo`` and ``--home`` are mandatory and explicit. Nothing outside those two trees
  (plus the backup dir) is read or written, so sibling profiles are untouched. The two trees MAY
  nest the way a standard install nests them (the checkout at ``<home>/hermes-agent``); what is
  refused is the overlap that would actually make ownership ambiguous — see
  :meth:`Options.validate` and :func:`scope_problems`.
* **Never destructive.** No ``git reset``, no ``git clean``, no full config replacement, no service
  restart. Restart instructions are reported for the operator to run.
* **No secrets.** Secrets are never written, never copied into the receipt, and never printed.
  Existing ``providers`` entries (including their credentials) are preserved byte-for-byte by the
  merge, which only ever sets this bundle's own keys.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BUNDLE_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = BUNDLE_DIR / "manifest.json"
CORE_PATCH_PATH = BUNDLE_DIR / "core.patch"
PAYLOAD_DIR = BUNDLE_DIR / "payload"

ALIAS_SECTION = "text_command_aliases"
PROMPT_PLUGIN = "llm-cr-prompt"
ALIAS_PLUGIN = "text-command-aliases"
#: Owns /detour and /detour-end. Enabled by default because the aliases this bundle configures
#: resolve to exactly those commands: leaving it off would wire a text alias to a command nothing
#: handles. ``--no-detour-plugin`` is the deliberate opt-out.
DETOUR_PLUGIN = "session-detours"
#: The slash commands the detour plugin must register once it is enabled, with the busy policy it
#: must declare for each. Read off the live plugin manager in ``verify`` — never assumed, and never
#: looked for in the built-in command registry, which (by design) no longer knows these names.
DETOUR_COMMANDS = {"detour": "reject", "detour-end": "reject"}
DEFAULT_API_MODE = "chat_completions"
INSTRUCTION_RELPATH = Path("skills") / "productivity" / "llm-crack-talk" / "llm-cr-instruction.md"

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_REFUSED = 2

# Every git invocation is bounded. A git that hangs (stale index.lock, a network-backed filter, a
# half-mounted worktree) must not hang the installer, and must never be read as a verdict.
GIT_TIMEOUT_SECONDS = 180


class Refused(Exception):
    """A fail-closed refusal: nothing was written, and the message says why."""


# ── small helpers ───────────────────────────────────────────────────────────────


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require_yaml():
    """PyYAML is the single non-stdlib dependency; probe it with an actionable message."""
    try:
        import yaml  # noqa: PLC0415
    except ImportError:
        raise Refused(
            "PyYAML is required for the config merge but is not importable by "
            f"{sys.executable}. Install it (pip install PyYAML) or run this installer with the "
            "interpreter Hermes itself uses, then retry."
        ) from None
    return yaml


def _is_within(child: Path, parent: Path) -> bool:
    """True when ``child`` is ``parent`` or lives underneath it (both resolved)."""
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except (ValueError, OSError):
        return False


def _is_strictly_within(child: Path, parent: Path) -> bool:
    """:func:`_is_within` minus the equal-paths case."""
    return _is_within(child, parent) and not _same_path(child, parent)


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return left == right


def _module_is_from(module: Any, root: Path) -> bool:
    """True when an imported module's code came from under ``root`` (file or namespace package)."""
    origin = getattr(module, "__file__", None)
    if origin:
        return _is_within(Path(str(origin)), root)
    for entry in list(getattr(module, "__path__", None) or []):
        with _suppress():
            if _is_within(Path(str(entry)), root):
                return True
    return False


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Temp file in the target directory + fsync + ``os.replace``, preserving an existing mode."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mode: Optional[int] = None
    if path.exists():
        mode = path.stat().st_mode & 0o777
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        with _suppress():
            tmp.unlink()
        raise


class _suppress:
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return True


def _run_git(repo: Path, args: List[str], *, stdin: Optional[bytes] = None,
             hint: str = "") -> subprocess.CompletedProcess:
    """Run git in ``repo`` with a hard timeout; bytes in, bytes out.

    The patch must reach ``git apply`` byte-for-byte, so no ``text=True`` here — callers decode the
    streams afterwards. Anything that is not an exit code (a timeout, a missing git, an OS error) is
    a refusal, never an implicit ``"absent"``/``"applied"`` verdict: silently guessing the patch
    state is exactly how an installer ends up applying a patch twice.
    """
    try:
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            input=stdin if stdin is not None else b"",
            capture_output=True, timeout=GIT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        raise Refused(
            f"git {' '.join(args)} in {repo} did not finish within {GIT_TIMEOUT_SECONDS}s and was "
            "killed; the target may hold a stale index.lock." + (f" {hint}" if hint else "")
        ) from None
    except FileNotFoundError:
        raise Refused(
            "git was not found on PATH; this installer needs it to check and apply core.patch"
        ) from None
    except OSError as exc:
        raise Refused(
            f"git could not be run in {repo} ({type(exc).__name__})" + (f". {hint}" if hint else "")
        ) from None


def _load_manifest() -> Dict[str, Any]:
    if not MANIFEST_PATH.is_file():
        raise Refused(f"bundle is incomplete: {MANIFEST_PATH} is missing")
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise Refused(f"bundle manifest is unreadable: {type(exc).__name__}") from None


# ── bundle self-integrity ───────────────────────────────────────────────────────


def verify_bundle_integrity(manifest: Dict[str, Any]) -> List[str]:
    """Every shipped artifact hashed against the manifest, before the target is even looked at."""
    problems: List[str] = []
    expected_patch = manifest.get("core_patch", {}).get("sha256")
    if not CORE_PATCH_PATH.is_file():
        problems.append("core.patch is missing")
    elif expected_patch and _sha256_file(CORE_PATCH_PATH) != expected_patch:
        problems.append("core.patch checksum does not match manifest")
    for rel, expected in sorted(manifest.get("payload", {}).items()):
        candidate = PAYLOAD_DIR / rel
        if not candidate.is_file():
            problems.append(f"payload file missing: {rel}")
        elif _sha256_file(candidate) != expected:
            problems.append(f"payload checksum mismatch: {rel}")
    extra = {
        str(p.relative_to(PAYLOAD_DIR)).replace(os.sep, "/")
        for p in PAYLOAD_DIR.rglob("*")
        # ``__pycache__`` is written by the interpreter, never shipped and never installed, so it is
        # not a manifest entry and must not read as a tampered bundle either.
        if p.is_file() and "__pycache__" not in p.parts
    } - set(manifest.get("payload", {}))
    for rel in sorted(extra):
        problems.append(f"unexpected payload file not in manifest: {rel}")
    return problems


# ── target compatibility preflight ──────────────────────────────────────────────


def preflight_apis(repo: Path, manifest: Dict[str, Any]) -> List[str]:
    """Prove the target still exposes every API this bundle's code calls into.

    The patch itself only proves the *lines* fit. These probes prove the *seams* exist: a repo that
    renamed ``run_llm_execution_middleware``'s context kwargs, dropped the ``pre_gateway_dispatch``
    rewrite directive or moved ``atomic_json_write`` would take the patch and then fail at runtime.
    Each probe is a required substring in a required file; an absent file or substring is a refusal,
    never a warning.
    """
    problems: List[str] = []
    for probe in manifest.get("required_apis", []):
        rel = probe.get("file", "")
        target = repo / rel
        if not target.is_file():
            problems.append(f"{rel}: file not found in target repo ({probe.get('why', '')})")
            continue
        try:
            text = target.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            problems.append(f"{rel}: unreadable ({type(exc).__name__})")
            continue
        for needle in probe.get("contains", []):
            if needle not in text:
                problems.append(
                    f"{rel}: required API not found: {needle!r} — {probe.get('why', '')}"
                )
    return problems


def core_patch_state(repo: Path) -> str:
    """``"absent"`` (applies cleanly), ``"applied"`` (reverses cleanly) or ``"conflict"``."""
    patch = CORE_PATCH_PATH.read_bytes()
    forward = _run_git(repo, ["apply", "--check", "--whitespace=nowarn", "-"], stdin=patch)
    if forward.returncode == 0:
        return "absent"
    reverse = _run_git(
        repo, ["apply", "--reverse", "--check", "--whitespace=nowarn", "-"], stdin=patch)
    if reverse.returncode == 0:
        return "applied"
    return "conflict"


def core_patch_conflict_detail(repo: Path) -> str:
    proc = _run_git(repo, ["apply", "--check", "--whitespace=nowarn", "-"],
                    stdin=CORE_PATCH_PATH.read_bytes())
    out = (proc.stderr or b"") + (proc.stdout or b"")
    return out.decode("utf-8", errors="replace").strip()


def patch_paths(manifest: Dict[str, Any]) -> List[str]:
    return list(manifest.get("core_patch", {}).get("paths", []))


# ── config merge ────────────────────────────────────────────────────────────────


def _alias_commands(opts: "Options") -> Dict[str, str]:
    enter = f"/detour {opts.model} --provider {opts.provider}"
    leave = f"/detour-end {opts.return_model} --provider {opts.return_provider}"
    return {"llm-cr": enter, "llm-cr-end": leave}


def read_config(config_path: Path) -> Dict[str, Any]:
    yaml = _require_yaml()
    if not config_path.is_file():
        raise Refused(
            f"no config.yaml at {config_path}. Point --home at an initialised Hermes home "
            "(the installer never creates one)."
        )
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise Refused(f"{config_path} is not loadable YAML ({type(exc).__name__})") from None
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise Refused(f"{config_path} does not contain a YAML mapping at the top level")
    return loaded


def resolve_cr_base_url(config: Dict[str, Any], opts: "Options") -> str:
    """``--cr-base-url`` if given, else the target's OWN ``providers.<provider>.base_url``.

    Deliberately no built-in default: this bundle ships no endpoint. If the target config does not
    already describe the provider, the operator has to name the URL, because guessing one would
    point the route — and therefore the instruction read — at an endpoint nobody chose.
    """
    if opts.cr_base_url:
        return opts.cr_base_url
    providers = config.get("providers")
    entry = providers.get(opts.provider) if isinstance(providers, dict) else None
    existing = entry.get("base_url") if isinstance(entry, dict) else None
    if isinstance(existing, str) and existing.strip():
        return existing.strip()
    raise Refused(
        f"cannot resolve a base URL for provider {opts.provider!r}: it is not configured under "
        f"'providers' in the target config, and --cr-base-url was not given. Pass "
        "--cr-base-url explicitly (this bundle ships no endpoint default)."
    )


def plan_config_merge(config: Dict[str, Any], opts: "Options", cr_base_url: str,
                      ) -> Tuple[Dict[str, Any], List[str]]:
    """Return ``(new_config, changes)``: a deep-ish copy with ONLY this bundle's keys set.

    Unrelated top-level keys, unrelated plugin entries, the existing ``plugins.enabled`` list and
    every ``providers`` entry (with its secrets) are carried through untouched. ``changes`` is a
    human-readable, secret-free list of exactly what differs — an empty list means already applied.
    """
    import copy

    new = copy.deepcopy(config)
    changes: List[str] = []

    def _note(text: str) -> None:
        changes.append(text)

    if opts.enable_aliases:
        aliases = _alias_commands(opts)
        section = new.get(ALIAS_SECTION)
        if not isinstance(section, dict):
            section = {}
        else:
            section = copy.deepcopy(section)
        if section.get("enabled") is not True:
            section["enabled"] = True
            _note(f"set {ALIAS_SECTION}.enabled = true")
        table = section.get("aliases")
        table = copy.deepcopy(table) if isinstance(table, dict) else {}
        for trigger, command in aliases.items():
            if table.get(trigger) != command:
                table[trigger] = command
                _note(f"set {ALIAS_SECTION}.aliases[{trigger!r}] = {command!r}")
        section["aliases"] = table
        new[ALIAS_SECTION] = section

    plugins = new.get("plugins")
    plugins = copy.deepcopy(plugins) if isinstance(plugins, dict) else {}
    enabled = plugins.get("enabled")
    enabled = list(enabled) if isinstance(enabled, list) else []
    wanted = ([ALIAS_PLUGIN] if opts.enable_aliases else []) + (
        [DETOUR_PLUGIN] if opts.enable_detours else []
    ) + (
        [PROMPT_PLUGIN] if opts.enable_prompt_plugin else []
    )
    for name in wanted:
        if name not in enabled:
            enabled.append(name)  # append, never replace: the operator's list is preserved
            _note(f"append {name!r} to plugins.enabled")
    plugins["enabled"] = enabled

    if opts.enable_prompt_plugin:
        entries = plugins.get("entries")
        entries = copy.deepcopy(entries) if isinstance(entries, dict) else {}
        entry = entries.get(PROMPT_PLUGIN)
        entry = copy.deepcopy(entry) if isinstance(entry, dict) else {}
        settings = entry.get("settings")
        settings = copy.deepcopy(settings) if isinstance(settings, dict) else {}
        desired = {
            "enabled": True,
            "route_provider": opts.provider,
            "route_model": opts.model,
            "route_base_url": cr_base_url,
            "route_api_mode": opts.api_mode,
        }
        if opts.instruction_path:
            desired["instruction_path"] = str(opts.instruction_path)
        for key, value in desired.items():
            if settings.get(key) != value:
                settings[key] = value
                _note(f"set plugins.entries.{PROMPT_PLUGIN}.settings.{key}")
        entry["settings"] = settings
        entries[PROMPT_PLUGIN] = entry
        plugins["entries"] = entries

    new["plugins"] = plugins
    return new, changes


def dump_config(config: Dict[str, Any]) -> bytes:
    yaml = _require_yaml()
    return yaml.safe_dump(
        config, allow_unicode=True, sort_keys=False, default_flow_style=False, width=100,
    ).encode("utf-8")


# ── options ─────────────────────────────────────────────────────────────────────


class Options:
    def __init__(self, args: argparse.Namespace):
        self.repo = Path(args.repo).resolve()
        self.home = Path(args.home).resolve()
        self.provider = (args.provider or "").strip()
        self.model = (args.model or "").strip()
        self.return_provider = (args.return_provider or "").strip()
        self.return_model = (args.return_model or "").strip()
        self.cr_base_url = (args.cr_base_url or "").strip().rstrip("/")
        self.api_mode = (args.api_mode or DEFAULT_API_MODE).strip() or DEFAULT_API_MODE
        self.instruction_path = (
            Path(args.instruction_path).resolve() if args.instruction_path else None
        )
        self.enable_prompt_plugin = bool(args.enable_prompt_plugin)
        self.enable_aliases = not bool(args.no_aliases)
        self.enable_detours = not bool(getattr(args, "no_detour_plugin", False))
        self.backup_dir = Path(args.backup_dir).resolve() if args.backup_dir else (
            self.home.parent / "llm-cr-bundle-backups"
        )

    @property
    def config_path(self) -> Path:
        return self.home / "config.yaml"

    @property
    def plugins_root(self) -> Path:
        return self.home / "plugins"

    @property
    def instruction_file(self) -> Path:
        return self.instruction_path or (self.home / INSTRUCTION_RELPATH)

    def validate(self) -> None:
        # ``.git`` is a directory in a normal clone and a FILE holding ``gitdir: ...`` in a
        # ``git worktree`` checkout. Both are real checkouts that ``git apply`` works in, so this
        # probe asks only whether the entry exists.
        if not (self.repo / ".git").exists():
            raise Refused(f"--repo {self.repo} is not a git repository checkout")
        if not self.home.is_dir():
            raise Refused(f"--home {self.home} is not an existing directory")
        # The standard install nests the checkout INSIDE the home (``<home>/hermes-agent``), so
        # "not nested" would reject the ordinary layout. What actually has to hold is that each
        # tree keeps its own identity and no file ends up with two owners:
        #   * repo inside home  -> supported (the standard layout).
        #   * home inside repo  -> refused: the Hermes home would be part of the git checkout, so
        #     `git apply` and the payload/config writer would both own the same subtree.
        #   * repo == home      -> refused: there would be one tree, not two, and a receipt could
        #     no longer say which root a restored file belongs to.
        # The per-file half of this rule is enforced by :func:`scope_problems`, which looks at the
        # paths the run would really touch rather than at the roots alone.
        if _same_path(self.repo, self.home):
            raise Refused(
                f"--repo and --home are the same directory ({self.repo}); they must be two "
                "distinct trees (a checkout at <home>/hermes-agent is fine)"
            )
        if _is_strictly_within(self.home, self.repo):
            raise Refused(
                f"--home {self.home} is inside --repo {self.repo}. A Hermes home nested in the git "
                "checkout would be written by both `git apply` and the payload installer, so "
                "'restore only my own changes' could not be honoured. (The reverse nesting — the "
                "checkout inside the home, as in <home>/hermes-agent — is supported.)"
            )
        if (
            _is_within(self.backup_dir, self.repo)
            or _is_within(self.backup_dir, self.home)
            or _is_within(self.repo, self.backup_dir)
            or _is_within(self.home, self.backup_dir)
        ):
            raise Refused(
                f"--backup-dir {self.backup_dir} overlaps the repo or the Hermes home; it must be "
                "outside both (neither inside them nor containing them) so a rollback source "
                "cannot be clobbered by the install itself"
            )
        if self.enable_aliases:
            missing = [
                flag for flag, value in (
                    ("--provider", self.provider), ("--model", self.model),
                    ("--return-provider", self.return_provider),
                    ("--return-model", self.return_model),
                ) if not value
            ]
            if missing:
                raise Refused(
                    "the alias commands name a concrete route, so these are required: "
                    + ", ".join(missing)
                    + ". This bundle ships no provider/model defaults."
                )
        if self.enable_prompt_plugin and not self.instruction_file.is_file():
            raise Refused(
                f"--enable-prompt-plugin requires an existing instruction file at "
                f"{self.instruction_file}, which the installer only checks for existence — it "
                "never reads, writes or creates it. Create it first, or pass "
                "--instruction-path, or install without --enable-prompt-plugin."
            )


# ── file-state bookkeeping ──────────────────────────────────────────────────────


def _file_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"existed": False, "sha256": None, "mode": None}
    return {
        "existed": True,
        "sha256": _sha256_file(path),
        "mode": path.stat().st_mode & 0o777,
    }


def _touched_paths(opts: Options, manifest: Dict[str, Any]) -> List[Tuple[str, Path]]:
    """Every path a run may write, as ``(kind, absolute path)``."""
    touched: List[Tuple[str, Path]] = [("core", opts.repo / rel) for rel in patch_paths(manifest)]
    for rel in sorted(manifest.get("payload", {})):
        touched.append(("payload", opts.home / rel))
    touched.append(("config", opts.config_path))
    return touched


def _touched_root(opts: Options, kind: str) -> Path:
    """The one tree a given layer is allowed to write in."""
    return opts.repo if kind == "core" else opts.home


def scope_problems(opts: Options, manifest: Dict[str, Any]) -> List[str]:
    """Per-file overlap checks over the paths this run would ACTUALLY write.

    Nesting the two trees is legitimate (``<home>/hermes-agent``), so the roots alone cannot decide
    safety — the files can. Three things have to hold, and each maps to a concrete way a rollback
    could otherwise destroy work it does not own:

    * **One owner per file.** A path claimed by two layers would be backed up twice, written twice
      and restored to whichever of the two copies the receipt happened to list last.
    * **Every path inside its own root.** A manifest entry that escaped (``../``, an absolute path)
      would write outside the two explicit trees, which the scoping promise forbids outright.
    * **No payload/config file inside the checkout.** ``git apply`` is the only writer allowed in
      the repo. A payload or config file landing there would be a second, untracked writer in a tree
      whose "did the user change this?" answer the patch state depends on.

    Returns a list of human-readable problems; empty means the layout is safe to write.
    """
    problems: List[str] = []
    owners: Dict[Path, str] = {}
    slots: Dict[Path, str] = {}
    run_dir = _backup_run_dir(opts, "preflight")
    for kind, path in _touched_paths(opts, manifest):
        root = _touched_root(opts, kind)
        if not _is_within(path, root):
            problems.append(f"{kind}: {path} is not inside its own root {root}")
            continue
        if _same_path(path, root):
            problems.append(f"{kind}: {path} IS its own root, not a file inside it")
            continue
        try:
            resolved = path.resolve()
        except OSError:
            problems.append(f"{kind}: {path} cannot be resolved")
            continue
        previous = owners.get(resolved)
        if previous is not None:
            problems.append(
                f"{path} would be written by both the {previous} and {kind} layers — one file "
                "cannot have two owners"
            )
        owners[resolved] = kind
        if kind != "core" and _is_within(path, opts.repo):
            problems.append(
                f"{kind}: {path} is inside the git checkout {opts.repo}; only core.patch may write "
                "there"
            )
        slot = _backup_slot(run_dir, kind, root, path)
        if slot is None:
            problems.append(f"{kind}: {path} has no representable backup slot under {run_dir}")
            continue
        clash = slots.get(slot)
        if clash is not None:
            problems.append(
                f"backup slot collision: {path} and {clash} would both be saved as {slot}"
            )
        else:
            slots[slot] = str(path)
    return problems


def _backup_run_dir(opts: Options, receipt_id: str) -> Path:
    return opts.backup_dir / receipt_id


def _backup_slot(run_dir: Path, kind: str, root: Path, path: Path) -> Optional[Path]:
    """Where ``path`` is saved inside one run's backup dir, or ``None`` if it escapes ``root``.

    The layer name is part of the slot, which is what keeps nested trees apart: with the checkout at
    ``<home>/hermes-agent`` a core path and a payload path can share a prefix, but never a slot.
    """
    try:
        relative = Path(os.path.relpath(path, root))
    except ValueError:  # different drives on Windows
        return None
    if relative.is_absolute() or ".." in relative.parts:
        return None
    return run_dir / kind / relative.as_posix()


# ── commands ────────────────────────────────────────────────────────────────────


def cmd_check(opts: Options, manifest: Dict[str, Any], *, quiet: bool = False) -> int:
    """Everything ``apply`` validates, with no writes. Returns an exit code."""
    lines: List[str] = []
    problems = verify_bundle_integrity(manifest)
    if problems:
        for text in problems:
            print(f"REFUSED bundle integrity: {text}", file=sys.stderr)
        return EXIT_REFUSED

    opts.validate()
    overlaps = scope_problems(opts, manifest)
    if overlaps:
        print("REFUSED: the files this run would write overlap in a way that makes rollback "
              "ambiguous. Nothing was written.", file=sys.stderr)
        for text in overlaps:
            print(f"  - {text}", file=sys.stderr)
        return EXIT_REFUSED

    api_problems = preflight_apis(opts.repo, manifest)
    if api_problems:
        print("REFUSED: the target repo does not expose the APIs this bundle needs. Nothing was "
              "written.", file=sys.stderr)
        for text in api_problems:
            print(f"  - {text}", file=sys.stderr)
        return EXIT_REFUSED

    state = core_patch_state(opts.repo)
    if state == "conflict":
        print("REFUSED: core.patch neither applies nor reverses cleanly against this checkout — "
              "the target has diverged. Nothing was written.", file=sys.stderr)
        detail = core_patch_conflict_detail(opts.repo)
        for text in detail.splitlines()[:20]:
            print(f"  {text}", file=sys.stderr)
        print(f"  (bundle base commit: {manifest.get('base_commit', '?')})", file=sys.stderr)
        return EXIT_REFUSED

    config = read_config(opts.config_path)
    cr_base_url = resolve_cr_base_url(config, opts) if opts.enable_prompt_plugin else ""
    _new_config, changes = plan_config_merge(config, opts, cr_base_url)

    payload_pending = [
        rel for rel, digest in sorted(manifest.get("payload", {}).items())
        if _file_state(opts.home / rel)["sha256"] != digest
    ]

    lines.append(f"bundle         : {manifest.get('name')} {manifest.get('version')}")
    lines.append(f"base commit    : {manifest.get('base_commit')}")
    lines.append(f"repo           : {opts.repo}")
    lines.append(f"hermes home    : {opts.home}")
    lines.append(f"backup dir     : {opts.backup_dir}")
    lines.append(f"core patch     : {'already applied' if state == 'applied' else 'to apply'}"
                 f" ({len(patch_paths(manifest))} paths)")
    lines.append(f"payload        : {len(payload_pending)} of "
                 f"{len(manifest.get('payload', {}))} file(s) to write")
    for rel in payload_pending:
        lines.append(f"  + {rel}")
    lines.append(f"config merge   : {len(changes)} change(s) to {opts.config_path}")
    for text in changes:
        lines.append(f"  + {text}")
    if opts.enable_detours:
        lines.append(f"detour plugin  : {DETOUR_PLUGIN} enabled; it registers "
                     + ", ".join(f"/{name}" for name in sorted(DETOUR_COMMANDS))
                     + " (core registers neither)")
    else:
        lines.append(f"detour plugin  : {DETOUR_PLUGIN} installed but NOT enabled "
                     "(--no-detour-plugin): /detour and /detour-end will not exist")
    if opts.enable_prompt_plugin:
        lines.append(f"prompt plugin  : enabled; instruction file present at "
                     f"{opts.instruction_file} (existence only — never read)")
    else:
        lines.append("prompt plugin  : not enabled (pass --enable-prompt-plugin to wire it)")
    nothing_to_do = state == "applied" and not payload_pending and not changes
    lines.append("")
    lines.append("RESULT: already fully applied — apply would change nothing"
                 if nothing_to_do else "RESULT: ready to apply")
    if not quiet:
        print("\n".join(lines))
    return EXIT_OK


def cmd_apply(opts: Options, manifest: Dict[str, Any]) -> int:
    # The same validation `check` runs, printed once: a refusal here means nothing was written.
    rc = cmd_check(opts, manifest)
    if rc != EXIT_OK:
        return rc
    print("")

    state = core_patch_state(opts.repo)
    config = read_config(opts.config_path)
    cr_base_url = resolve_cr_base_url(config, opts) if opts.enable_prompt_plugin else ""
    new_config, changes = plan_config_merge(config, opts, cr_base_url)
    payload_pending = [
        rel for rel, digest in sorted(manifest.get("payload", {}).items())
        if _file_state(opts.home / rel)["sha256"] != digest
    ]

    if state == "applied" and not payload_pending and not changes:
        print("Already applied — nothing to change. (Idempotent: no backup and no receipt "
              "written, because no file was touched.)")
        return EXIT_OK

    receipt_id = f"llm-cr-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    run_dir = _backup_run_dir(opts, receipt_id)
    run_dir.mkdir(parents=True, exist_ok=False)

    # ── back up BEFORE the first write, recording exact prior existence/hash/mode ──
    # One file at a time, by exact path: never a directory tree copy, so nothing that this run does
    # not write can end up in the backup (or be restored out of it).
    entries: List[Dict[str, Any]] = []
    for kind, path in _touched_paths(opts, manifest):
        root = _touched_root(opts, kind)
        before = _file_state(path)
        slot = _backup_slot(run_dir, kind, root, path)
        if slot is None:  # already reported by scope_problems; refuse rather than skip a backup
            raise Refused(f"{kind} path {path} has no backup slot under {run_dir}")
        if before["existed"]:
            if path.is_dir():
                raise Refused(f"{kind} path {path} is a directory; this installer only owns files")
            slot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, slot)
            if _sha256_file(slot) != before["sha256"]:
                raise Refused(
                    f"the backup copy of {path} does not match the file it was taken from; "
                    "refusing to go on without a trustworthy rollback source"
                )
        entries.append({
            "kind": kind,
            "path": str(path),
            "root": str(root),
            "backup": str(slot) if before["existed"] else None,
            "existed_before": before["existed"],
            "sha256_before": before["sha256"],
            "mode_before": before["mode"],
            "sha256_after": None,
        })
    print(f"Backed up {sum(1 for e in entries if e['existed_before'])} existing "
          f"(and recorded {sum(1 for e in entries if not e['existed_before'])} absent) "
          f"path(s) to {run_dir}")

    # ── core patch ──
    if state == "absent":
        proc = _run_git(
            opts.repo, ["apply", "--whitespace=nowarn", "-"],
            stdin=CORE_PATCH_PATH.read_bytes(),
            hint=f"Nothing else has been written yet; the pre-apply copies are in {run_dir}. "
                 f"Check `git -C {opts.repo} status` before retrying.",
        )
        if proc.returncode != 0:
            detail = ((proc.stderr or b"") + (proc.stdout or b"")).decode("utf-8", "replace")
            raise Refused(
                "git apply failed after its own --check passed; the checkout changed underneath "
                f"this run. Restore from {run_dir} if anything looks wrong.\n{detail.strip()}"
            )
        print(f"Applied core.patch ({len(patch_paths(manifest))} paths)")
    else:
        print("Core patch already present — left untouched")

    # ── payload ──
    for rel in payload_pending:
        target = opts.home / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_bytes(target, (PAYLOAD_DIR / rel).read_bytes())
    print(f"Installed {len(payload_pending)} payload file(s) under {opts.home}")

    # ── config (atomic, bundle keys only) ──
    if changes:
        _atomic_write_bytes(opts.config_path, dump_config(new_config))
        print(f"Merged {len(changes)} config change(s) into {opts.config_path}")
    else:
        print("Config already carries this bundle's keys — left untouched")

    for entry in entries:
        entry["sha256_after"] = _file_state(Path(entry["path"]))["sha256"]

    receipt = {
        "receipt_version": 1,
        "receipt_id": receipt_id,
        "applied_at_utc": _now(),
        "bundle": {
            "name": manifest.get("name"),
            "version": manifest.get("version"),
            "base_commit": manifest.get("base_commit"),
            "core_patch_sha256": manifest.get("core_patch", {}).get("sha256"),
        },
        "repo": str(opts.repo),
        "home": str(opts.home),
        "backup_dir": str(run_dir),
        "core_patch_applied_by_this_run": state == "absent",
        "config_changes": changes,
        "prompt_plugin_enabled": opts.enable_prompt_plugin,
        "aliases_enabled": opts.enable_aliases,
        # Route identifiers are config, not credentials; no api_key/secret is ever recorded.
        "route": {
            "provider": opts.provider, "model": opts.model, "api_mode": opts.api_mode,
            "return_provider": opts.return_provider, "return_model": opts.return_model,
            "base_url_configured": bool(cr_base_url),
        },
        "instruction_file_checked": str(opts.instruction_file) if opts.enable_prompt_plugin else None,
        "files": entries,
    }
    receipt_path = run_dir / "receipt.json"
    _atomic_write_bytes(receipt_path, json.dumps(receipt, indent=2, ensure_ascii=False).encode("utf-8"))

    print("")
    print(f"APPLIED. Receipt: {receipt_path}")
    print("")
    print("Next steps (this installer deliberately restarts nothing):")
    print("  1. Restart the Hermes Gateway so it loads the new plugins and core code.")
    print("  2. Restart any already-running interactive CLI process — a running CLI cannot be")
    print("     hot-patched by /reload; it needs a fresh process to pick up code and aliases.")
    print(f"  3. Verify:  python {Path(__file__).name} verify --repo <repo> --home <home>"
          + (" --enable-prompt-plugin" if opts.enable_prompt_plugin else ""))
    return EXIT_OK


def cmd_verify(opts: Options, manifest: Dict[str, Any]) -> int:
    """Prove the install actually works: patch present, aliases exact, plugins importable/loaded."""
    failures: List[str] = []
    checks: List[str] = []

    overlaps = scope_problems(opts, manifest)
    if overlaps:
        raise Refused(
            "the installed layout overlaps in a way that makes ownership ambiguous: "
            + "; ".join(overlaps)
        )

    state = core_patch_state(opts.repo)
    if state == "applied":
        checks.append("core patch present (reverse-check passes)")
    else:
        failures.append(f"core patch is not applied (state={state})")

    for rel, digest in sorted(manifest.get("payload", {}).items()):
        got = _file_state(opts.home / rel)["sha256"]
        if got == digest:
            checks.append(f"payload matches: {rel}")
        else:
            failures.append(f"payload missing or modified: {rel}")

    config = read_config(opts.config_path)
    if opts.enable_aliases:
        expected = _alias_commands(opts)
        section = config.get(ALIAS_SECTION) if isinstance(config.get(ALIAS_SECTION), dict) else {}
        table = section.get("aliases") if isinstance(section.get("aliases"), dict) else {}
        if section.get("enabled") is not True:
            failures.append(f"{ALIAS_SECTION}.enabled is not true")
        for trigger, command in expected.items():
            if table.get(trigger) != command:
                failures.append(f"{ALIAS_SECTION}.aliases[{trigger!r}] is {table.get(trigger)!r}, "
                                f"expected {command!r}")
            else:
                checks.append(f"alias configured: {trigger!r} -> {command!r}")

    # The matcher is the thing the aliases actually depend on: run it in-process against the real
    # target checkout, so a config the CLI would reject never reads as "verified".
    #
    # Everything this import does to the interpreter is undone in the ``finally`` below — sys.path,
    # the modules loaded out of the target checkout, and HERMES_HOME. ``verify`` is read-only on
    # disk, and it has to be read-only on process state too: a leaked HERMES_HOME would point a
    # later command at the wrong profile, and leaked modules would make a second verify re-use the
    # first run's objects instead of the files now on disk.
    repo_entry = str(opts.repo)
    sys.path.insert(0, repo_entry)
    hermes_home_before = os.environ.get("HERMES_HOME")
    modules_before = set(sys.modules)
    try:
        try:
            from hermes_cli.text_command_aliases import resolve_text_command_alias
        except Exception as exc:
            failures.append(f"hermes_cli.text_command_aliases is not importable from the target "
                            f"repo ({type(exc).__name__}: {exc})")
            resolve_text_command_alias = None
        if resolve_text_command_alias is not None and opts.enable_aliases:
            for trigger, command in _alias_commands(opts).items():
                got = resolve_text_command_alias(trigger, config)
                if got == command:
                    checks.append(f"matcher resolves {trigger!r} -> {command!r}")
                else:
                    failures.append(f"matcher resolved {trigger!r} to {got!r}, expected {command!r}")
            for near in ("llm-cr 你好", "LLM-CR", "llm-crack-talk", "x llm-cr"):
                got = resolve_text_command_alias(near, config)
                if got is None:
                    checks.append(f"near-miss correctly not an alias: {near!r}")
                else:
                    failures.append(f"near-miss {near!r} wrongly resolved to {got!r}")

        # Importability of the installed plugin modules, loaded from the TARGET home by path, so a
        # syntactically broken or truncated payload cannot pass as installed.
        import importlib.util

        for name, rel in (
            ("llm_cr_prompt_injector", "plugins/llm-cr-prompt/injector.py"),
        ):
            target = opts.home / rel
            try:
                spec = importlib.util.spec_from_file_location(name, target)
                if spec is None or spec.loader is None:
                    raise ImportError(f"no loader for {target}")
                module = importlib.util.module_from_spec(spec)
                # Register BEFORE exec: ``@dataclass`` resolves ``sys.modules[cls.__module__]`` to
                # validate annotations, and an unregistered module makes that lookup None.
                sys.modules[name] = module
                try:
                    spec.loader.exec_module(module)
                finally:
                    sys.modules.pop(name, None)
                for symbol in ("make_middleware", "route_matches", "load_instruction",
                               "inject_instruction", "resolve_settings"):
                    if not hasattr(module, symbol):
                        failures.append(f"{rel}: missing {symbol}")
                checks.append(f"plugin module imports: {rel}")
            except Exception as exc:
                failures.append(f"{rel} is not importable ({type(exc).__name__}: {exc})")

        # Production plugin discovery: does Hermes ITSELF load these, enable them, and register
        # the seams they claim? A payload that merely sits on disk is not an install.
        try:
            os.environ["HERMES_HOME"] = str(opts.home)
            from hermes_cli.plugins import PluginManager

            manager = PluginManager()
            manager.discover_and_load(force=True)
            loaded = getattr(manager, "_plugins", {}) or {}
            hooks = getattr(manager, "_hooks", {}) or {}
            middleware = getattr(manager, "_middleware", {}) or {}
            commands = getattr(manager, "_plugin_commands", {}) or {}
            wanted = (
                [(ALIAS_PLUGIN, "hook", "pre_gateway_dispatch")] if opts.enable_aliases else []
            ) + (
                [(DETOUR_PLUGIN, "command", tuple(sorted(DETOUR_COMMANDS)))]
                if opts.enable_detours else []
            ) + (
                [(PROMPT_PLUGIN, "middleware", "llm_execution")] if opts.enable_prompt_plugin else []
            )
            for name, kind, seam in wanted:
                entry = loaded.get(name)
                if entry is None:
                    failures.append(f"plugin NOT loaded by production discovery: {name} "
                                    f"(saw {len(loaded)} plugin(s))")
                    continue
                if not getattr(entry, "enabled", False):
                    failures.append(f"plugin {name} was discovered but is not enabled")
                if getattr(entry, "error", None):
                    failures.append(f"plugin {name} loaded with an error: {entry.error}")
                if kind == "command":
                    # The feature IS these registrations: read them off the live manager rather
                    # than trusting that the payload is on disk.
                    for command in seam:
                        registration = commands.get(command)
                        if not isinstance(registration, dict) or not registration.get("handler"):
                            failures.append(f"plugin {name} did not register /{command}")
                            continue
                        policy, expected = registration.get("busy_policy"), DETOUR_COMMANDS[command]
                        if policy != expected:
                            failures.append(
                                f"/{command} registered with busy_policy {policy!r}, expected "
                                f"{expected!r} (it rotates the session, so it must refuse mid-turn)"
                            )
                        else:
                            checks.append(f"plugin loaded + enabled by production discovery with "
                                          f"command /{command} (busy_policy={policy!r}): {name}")
                    continue
                registered = hooks if kind == "hook" else middleware
                if registered.get(seam):
                    checks.append(f"plugin loaded + enabled by production discovery with "
                                  f"{kind} {seam!r}: {name}")
                else:
                    failures.append(f"plugin {name} did not register {kind} {seam!r}")

            # Disabled means gone, not merely inert: with the plugin unconsented nothing may claim
            # /detour on any surface. (This is the whole point of the plugin-first layout.)
            if not opts.enable_detours:
                stray = sorted(name for name in DETOUR_COMMANDS if name in commands)
                if stray:
                    failures.append("detour plugin is not enabled, yet these commands are "
                                    "registered: " + ", ".join(f"/{n}" for n in stray))
                else:
                    checks.append("detour plugin not enabled: "
                                  + ", ".join(f"/{n}" for n in sorted(DETOUR_COMMANDS))
                                  + " are registered by nothing")

            # ...and the patched core must not be providing them either. Checked against the
            # target's OWN built-in registry, with no assumption that the registry knows the
            # names: finding them there would mean this is a pre-2.0 (core-resident) install.
            try:
                from hermes_cli.commands import COMMAND_REGISTRY
            except Exception:
                checks.append("built-in command registry not importable from this target "
                              "(skipped the core-residue check)")
            else:
                builtin = {getattr(c, "name", None) for c in COMMAND_REGISTRY}
                residue = sorted(name for name in DETOUR_COMMANDS if name in builtin)
                if residue:
                    failures.append("the patched core still registers " +
                                    ", ".join(f"/{n}" for n in residue) +
                                    " as a built-in command; this bundle expects the plugin to "
                                    "own them (stale core.patch, or an older install on top)")
                else:
                    checks.append("core registers no detour command: the feature comes only "
                                  "from the plugin")
            # A pre-existing third-party plugin must survive the install untouched.
            for name, entry in loaded.items():
                if name not in {n for n, _k, _s in wanted} and getattr(entry, "enabled", False):
                    checks.append(f"pre-existing enabled plugin still loads: {name}")
        except Exception as exc:
            failures.append(f"plugin discovery could not be exercised ({type(exc).__name__}: {exc})")
    finally:
        with _suppress():
            sys.path.remove(repo_entry)
        for name in sorted(set(sys.modules) - modules_before, key=len, reverse=True):
            module = sys.modules.get(name)
            if module is not None and _module_is_from(module, opts.repo):
                sys.modules.pop(name, None)
        if hermes_home_before is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = hermes_home_before

    for text in checks:
        print(f"  ok   {text}")
    for text in failures:
        print(f"  FAIL {text}", file=sys.stderr)
    print("")
    if failures:
        print(f"VERIFY FAILED: {len(failures)} failure(s), {len(checks)} check(s) passed",
              file=sys.stderr)
        return EXIT_FAIL
    print(f"VERIFY OK: {len(checks)} check(s) passed")
    return EXIT_OK


def cmd_rollback(receipt_path: Path, *, force_scope: Optional[Tuple[Path, Path]] = None) -> int:
    """Undo exactly one receipt's own changes, refusing if the user edited those files since.

    No ``git reset``, no ``git clean``, no config replacement from a template: each recorded path is
    either restored byte-for-byte from its backup copy, or removed because the receipt proves it did
    not exist before. A path whose current hash differs from the post-apply hash was changed by
    somebody else after the install, so the whole rollback refuses up front rather than silently
    discarding that work.
    """
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise Refused(f"receipt {receipt_path} is unreadable ({type(exc).__name__})") from None
    if receipt.get("receipt_version") != 1:
        raise Refused(f"unsupported receipt_version {receipt.get('receipt_version')!r}")

    entries = receipt.get("files") or []
    if force_scope is not None:
        repo, home = force_scope
        if Path(receipt.get("repo", "")) != repo or Path(receipt.get("home", "")) != home:
            raise Refused(
                f"receipt scope mismatch: it records repo={receipt.get('repo')} "
                f"home={receipt.get('home')}, but --repo/--home name {repo} / {home}. "
                "Refusing so a receipt can never be replayed against a different profile."
            )

    if not receipt.get("repo") or not receipt.get("home"):
        raise Refused(f"receipt {receipt_path} does not record its own repo/home scope")
    repo_recorded = Path(receipt["repo"])
    home_recorded = Path(receipt["home"])

    # ── exact file ownership, checked before anything is read out of the backup ──
    # Each entry has to name ONE file inside ONE of this receipt's own roots, with its own backup
    # copy. The roots may nest (the standard layout puts the checkout at ``<home>/hermes-agent``),
    # so the entry's own recorded root is what decides ownership — never a directory walk, and never
    # a recursive copy of either tree.
    for entry in entries:
        path = Path(entry.get("path", ""))
        root = Path(entry.get("root", ""))
        if not entry.get("path") or not entry.get("root"):
            raise Refused(f"receipt {receipt_path} has an entry without a path or a root")
        if not (_same_path(root, repo_recorded) or _same_path(root, home_recorded)):
            raise Refused(
                f"receipt entry for {path} names root {root}, which is neither the receipt's repo "
                f"({repo_recorded}) nor its home ({home_recorded}). Refusing to restore outside the "
                "two trees the install was scoped to."
            )
        if not _is_strictly_within(path, root):
            raise Refused(f"receipt entry {path} is not a file inside its recorded root {root}")
        if path.is_dir():
            raise Refused(
                f"receipt entry {path} is a directory; this installer owns individual files only, "
                "so a rollback never copies or removes a directory tree"
            )
        if entry.get("existed_before") and not entry.get("backup"):
            raise Refused(f"receipt says {path} existed before the install but records no backup copy")
        if entry.get("backup") and Path(entry["backup"]).is_dir():
            raise Refused(f"backup for {path} is a directory, not a file copy")

    drifted: List[str] = []
    missing_backup: List[str] = []
    stale_backup: List[str] = []
    for entry in entries:
        path = Path(entry["path"])
        now = _file_state(path)
        if now["sha256"] == entry.get("sha256_after"):
            continue  # untouched since apply
        if not now["existed"] and entry.get("sha256_after") is None:
            continue  # was absent after apply and still is
        if now["existed"] and now["sha256"] == entry.get("sha256_before"):
            continue  # already back at its pre-apply content
        drifted.append(str(path))
    for entry in entries:
        if entry.get("existed_before") and entry.get("backup"):
            backup = Path(entry["backup"])
            if not backup.is_file():
                missing_backup.append(entry["path"])
            elif entry.get("sha256_before") and _sha256_file(backup) != entry["sha256_before"]:
                # The backup is the only thing a restore trusts, so it is hashed against what the
                # receipt recorded at apply time rather than taken on faith.
                stale_backup.append(entry["path"])

    if drifted or missing_backup or stale_backup:
        print("REFUSED: rollback would discard changes it did not make. Nothing was written.",
              file=sys.stderr)
        for path in drifted:
            print(f"  - modified since apply: {path}", file=sys.stderr)
        for path in missing_backup:
            print(f"  - backup copy missing for: {path}", file=sys.stderr)
        for path in stale_backup:
            print(f"  - backup copy no longer matches the hash recorded at apply time, so it is "
                  f"not a trustworthy restore source for: {path}", file=sys.stderr)
        return EXIT_REFUSED

    # Directories that must survive a rollback whatever the entries say: the two roots themselves
    # (and, with the standard nested layout, the checkout sitting inside the home).
    boundaries = {p.resolve() for p in (repo_recorded, home_recorded)}

    restored = removed = 0
    for entry in entries:
        path = Path(entry["path"])
        root = Path(entry["root"])
        if entry.get("existed_before"):
            # One file, copied back by exact path from its own backup copy.
            shutil.copy2(entry["backup"], path)
            if entry.get("mode_before") is not None:
                with _suppress():
                    os.chmod(path, entry["mode_before"])
            restored += 1
        elif path.exists():
            path.unlink()
            removed += 1
            # Prune directories this install created, but only while they are empty, only inside the
            # entry's own root, and never a root (or the nested checkout) itself.
            parent = path.parent
            while _is_strictly_within(parent, root) and parent.resolve() not in boundaries:
                try:
                    parent.rmdir()
                except OSError:
                    break
                parent = parent.parent

    print(f"ROLLED BACK receipt {receipt.get('receipt_id')}: restored {restored} file(s), "
          f"removed {removed} file(s) that did not exist before.")
    print("Restart the Gateway and any running CLI process to drop the removed code/plugins.")
    return EXIT_OK


# ── CLI ─────────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="install.py",
        description="Reapplicable installer for the llm-cr bundle (aliases + the session-detours "
                    "plugin + prompt injection). Ships no provider, model or endpoint defaults.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def _common(p: argparse.ArgumentParser, *, route: bool = True) -> None:
        p.add_argument("--repo", required=True, help="hermes-agent checkout to patch (explicit)")
        p.add_argument("--home", required=True, help="Hermes home / profile to install into (explicit)")
        p.add_argument("--backup-dir", default=None,
                       help="where to back up touched files (default: <home>/../llm-cr-bundle-backups; "
                            "must be outside both --repo and --home)")
        if route:
            p.add_argument("--provider", default="", help="CR provider name as configured in the target")
            p.add_argument("--model", default="", help="CR model id")
            p.add_argument("--return-provider", default="", help="provider to restore on llm-cr-end")
            p.add_argument("--return-model", default="", help="model to restore on llm-cr-end")
            p.add_argument("--cr-base-url", default="",
                           help="CR endpoint; omitted = read providers.<provider>.base_url from the "
                                "target config, and refuse if it is not there")
            p.add_argument("--api-mode", default=DEFAULT_API_MODE, help="CR route api_mode")
            p.add_argument("--instruction-path", default=None,
                           help="absolute path to the existing instruction file (metadata only: "
                                "checked for existence, never read, written or created)")
            p.add_argument("--enable-prompt-plugin", action="store_true",
                           help="wire the llm_execution prompt-injection plugin (requires an "
                                "existing instruction file)")
            p.add_argument("--no-aliases", action="store_true",
                           help="install core + payload without enabling the text aliases")
            p.add_argument("--no-detour-plugin", action="store_true",
                           help="install the session-detours payload without enabling it "
                                "(/detour and /detour-end then do not exist on any surface)")

    for name in ("check", "dry-run", "apply", "verify"):
        _common(sub.add_parser(name, help=f"{name} the bundle"))
    roll = sub.add_parser("rollback", help="undo one receipt's own changes")
    roll.add_argument("--receipt", required=True, help="path to receipt.json written by apply")
    roll.add_argument("--repo", required=True, help="must match the receipt (scope guard)")
    roll.add_argument("--home", required=True, help="must match the receipt (scope guard)")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "rollback":
            return cmd_rollback(
                Path(args.receipt).resolve(),
                force_scope=(Path(args.repo).resolve(), Path(args.home).resolve()),
            )
        manifest = _load_manifest()
        opts = Options(args)
        if args.command in ("check", "dry-run"):
            return cmd_check(opts, manifest)
        if args.command == "apply":
            return cmd_apply(opts, manifest)
        if args.command == "verify":
            opts.validate()
            return cmd_verify(opts, manifest)
        return EXIT_FAIL
    except Refused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
