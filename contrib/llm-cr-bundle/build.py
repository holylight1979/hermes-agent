#!/usr/bin/env python3
"""Deterministically build ``manifest.json`` and the distributable ZIP for this bundle.

Determinism matters because the manifest's checksums are what ``install.py`` fails closed on, and
because a reviewer has to be able to rebuild the archive and get the same bytes. So: sorted entries,
a fixed ZIP timestamp, fixed compression, no mtimes from the filesystem.

Usage::

    python build.py                       # rewrite manifest.json in place
    python build.py --zip <path>          # also write the archive (never inside the repo)

``required_apis`` below is the real dependency analysis: each probe is a symbol or seam the shipped
code genuinely calls into, with the reason recorded next to it. They were each confirmed present in
the bundle's base commit; the installer refuses when a target no longer has one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path

BUNDLE_DIR = Path(__file__).resolve().parent
NAME = "llm-cr-bundle"
VERSION = "2.0.0"  # 2.x is plugin-first: the detour feature moved out of core into a payload plugin
BASE_COMMIT = "45a6101f36576367359c171cd5820ee76a3d047b"

# Files the bundle itself consists of, in archive order. ``manifest.json`` is excluded on purpose:
# it is the thing being generated, and hashing it into itself is not possible.
BUNDLE_FILES = ["README.md", "install.py", "build.py", "core.patch", "docs/SKILL.md"]

# Tests the operator should run after apply, as paths inside the patched repo plus the payload's own
# suite inside the Hermes home.
TESTS_IN_REPO = [
    "tests/hermes_cli/test_text_command_aliases.py",
    "tests/hermes_cli/test_plugin_command_context.py",
    "tests/hermes_cli/test_session_detour_records.py",
    "tests/hermes_cli/test_cli_detour.py",
    "tests/gateway/test_text_command_alias_dispatch.py",
    "tests/gateway/test_text_command_alias_model_switch.py",
    "tests/gateway/test_text_command_alias_deployed_plugin.py",
    "tests/gateway/test_detour_commands.py",
    "tests/gateway/test_detour_plugin_dispatch.py",
]
TESTS_IN_HOME = ["plugins/llm-cr-prompt/tests/test_llm_cr_prompt_injection.py"]

REQUIRED_APIS = [
    {
        "file": "gateway/run_inbound.py",
        "contains": ["def _hm_pre_gateway_dispatch_hook", '_action == "rewrite"',
                     "allow_gateway_control", "_hm_dispatch_quick_and_plugin_commands",
                     "get_plugin_command_handler"],
        "why": "the text-command-aliases plugin rewrites event.text through this existing hook "
               "seam; without the rewrite directive the alias would silently never fire. The last "
               "two are the plugin-command dispatch sink the patch extends with a host context and "
               "a normalized-name access gate",
    },
    {
        "file": "hermes_cli/plugins.py",
        "contains": ['"pre_gateway_dispatch"', "def register_middleware", "def register_hook",
                     "def get_config", "def register_command", "_plugin_commands"],
        "why": "the three plugins register through these loader APIs and read their own settings "
               "subtree; session-detours registers slash commands, so the patch extends "
               "register_command (busy_policy) and the command table the surfaces read",
    },
    {
        "file": "cli.py",
        "contains": ["def _run_plugin_slash_command", "resolve_plugin_command_result"],
        "why": "the CLI leg of a plugin slash command is dispatched here; the patch binds the host "
               "context onto this call, which is what lets session-detours compose native session "
               "commands from outside core",
    },
    {
        "file": "gateway/run_busy.py",
        "contains": ["def _dispatch_busy_slash_command", "def _check_slash_access",
                     "can't run "],
        "why": "the patch factors the built-in busy refusal text out of this mixin so a plugin "
               "command that declared busy_policy=\"reject\" is refused mid-turn in exactly the "
               "same words, and reuses this access gate for it",
    },
    {
        "file": "hermes_cli/middleware.py",
        "contains": ['LLM_EXECUTION_MIDDLEWARE = "llm_execution"', "def _run_execution_chain",
                     "def run_llm_execution_middleware"],
        "why": "llm-cr-prompt is llm_execution middleware; the patch adds MiddlewareAbort to this "
               "chain, so the chain has to be the one the patch expects",
    },
    {
        "file": "agent/turn_api_call.py",
        "contains": ["run_llm_execution_middleware(", "provider=agent.provider",
                     "base_url=agent.base_url", "api_mode=agent.api_mode"],
        "why": "the injector's route gate reads exactly these context kwargs; a renamed or dropped "
               "kwarg would make the gate never match and the instruction never load",
    },
    {
        "file": "hermes_cli/config.py",
        "contains": ["def read_raw_config_readonly", "_OPEN_DICT_TOP_LEVEL_KEYS"],
        "why": "the gateway plugin reads raw config through the first; the patch adds the alias "
               "section to the second so config validation accepts it",
    },
    {
        "file": "utils.py",
        "contains": ["def atomic_json_write"],
        "why": "the session-detours plugin writes its durable return records through this",
    },
    {
        "file": "hermes_cli/profiles.py",
        "contains": ["def get_active_profile_name"],
        "why": "the session-detours plugin labels a record with its profile",
    },
    {
        "file": "hermes_constants.py",
        "contains": ["def get_hermes_home"],
        "why": "both the record directory and the injector's default instruction path derive from it",
    },
    {
        "file": "hermes_cli/model_switch.py",
        "contains": ["def parse_model_switch_args"],
        "why": "the plugin's detour route parser reuses the one /model parser, including its flag conflicts",
    },
    {
        "file": "gateway/slash_commands_session.py",
        "contains": ["def _handle_reset_command", "def _handle_resume_command"],
        "why": "gateway /detour and /detour-end are composed from these native handlers (the "
               "resume handler's IDOR guard included) rather than reimplementing session rotation",
    },
    {
        "file": "gateway/slash_commands_model.py",
        "contains": ["def _handle_model_command"],
        "why": "the gateway detour applies its route through the native /model handler",
    },
    {
        "file": "gateway/run.py",
        "contains": ["def _session_key_for_source", "def _normalize_source_for_session_key"],
        "why": "the gateway detour lane key is the native session key for the source, resolved "
               "through the same source normalization the turn itself uses",
    },
    {
        "file": "hermes_cli/cli_session_mixin.py",
        "contains": ["def new_session"],
        "why": "CLI /detour runs the native fresh-session path, including its memory-boundary flush",
    },
    {
        "file": "hermes_cli/cli_commands_mixin.py",
        "contains": ["def _handle_resume_command"],
        "why": "CLI /detour-end returns through the native resume handler",
    },
    {
        "file": "hermes_cli/cli_model_switch_mixin.py",
        "contains": ["def _handle_model_switch"],
        "why": "the CLI detour applies its route through the native model switch",
    },
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_index() -> dict:
    root = BUNDLE_DIR / "payload"
    index = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        index[str(path.relative_to(root)).replace(os.sep, "/")] = sha256_file(path)
    return index


def patch_paths() -> list:
    patch = (BUNDLE_DIR / "core.patch").read_text(encoding="utf-8", errors="replace")
    paths = []
    for line in patch.splitlines():
        if line.startswith("+++ b/"):
            paths.append(line[6:].strip())
    return sorted(paths)


def build_manifest() -> dict:
    core = BUNDLE_DIR / "core.patch"
    return {
        "name": NAME,
        "version": VERSION,
        "manifest_version": 1,
        "base_commit": BASE_COMMIT,
        "base_commit_note": "core.patch is generated against this upstream commit and is checked "
                            "with `git apply --check` (and `--reverse --check` for the "
                            "already-applied case) before anything is written.",
        "ships_no_defaults": [
            "provider", "model", "return_provider", "return_model", "base_url",
            "instruction file content",
        ],
        "core_patch": {
            "file": "core.patch",
            "sha256": sha256_file(core),
            "paths": patch_paths(),
        },
        "payload": payload_index(),
        "payload_install_root": "<hermes-home>",
        "required_apis": REQUIRED_APIS,
        "tests": {"in_repo": TESTS_IN_REPO, "in_home": TESTS_IN_HOME},
        "bundle_files": {
            rel: sha256_file(BUNDLE_DIR / rel)
            for rel in BUNDLE_FILES if (BUNDLE_DIR / rel).is_file()
        },
    }


def write_zip(target: Path, manifest: dict) -> None:
    """Deterministic archive: sorted names, fixed timestamp, fixed compression, no mtimes."""
    # The archive must land outside the checkout that holds this bundle, so building never dirties
    # the repo it ships from.
    repo_root = BUNDLE_DIR
    while repo_root != repo_root.parent and not (repo_root / ".git").exists():
        repo_root = repo_root.parent
    if (repo_root / ".git").exists() and target.resolve().is_relative_to(repo_root.resolve()):
        raise SystemExit(f"refusing to write the archive inside the checkout {repo_root}: {target}")
    names = sorted(
        [*manifest["bundle_files"], "manifest.json",
         *(f"payload/{rel}" for rel in manifest["payload"]),
         *(f"tests/{p.name}" for p in sorted((BUNDLE_DIR / "tests").glob("*.py")))]
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in names:
            source = BUNDLE_DIR / name
            if not source.is_file():
                continue
            info = zipfile.ZipInfo(f"{NAME}/{name}", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, source.read_bytes())
    print(f"wrote {target} sha256={sha256_file(target)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", default=None, help="also write a deterministic archive here")
    args = parser.parse_args()
    manifest = build_manifest()
    (BUNDLE_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(f"manifest.json: {len(manifest['payload'])} payload file(s), "
          f"{len(manifest['core_patch']['paths'])} patched path(s), "
          f"{len(manifest['required_apis'])} API probe(s)")
    if args.zip:
        manifest = build_manifest()  # re-hash: manifest.json just changed on disk
        write_zip(Path(args.zip).resolve(), manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
