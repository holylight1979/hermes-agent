"""Tests for the llm-cr bundle installer itself.

These cover the parts that decide whether a write happens at all — the config merge, the base-URL
resolution, the alias wording, the bundle's self-integrity and the rollback drift guard — so a
regression in them is caught without needing a repo to patch.

The final section goes end to end — check, apply, a second apply, verify, rollback — against a
throwaway fixture laid out the way a real install is laid out: a real ``git init`` checkout nested
at ``<home>/hermes-agent``. That nested layout is the one the installer used to refuse outright, so
it is a fixture here rather than a manual procedure. No real profile, repo or instruction file is
read or written by any test in this file.

Run with the repo's canonical runner::

    scripts/run_tests.sh contrib/llm-cr-bundle/tests/test_bundle_installer.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

BUNDLE_DIR = Path(__file__).resolve().parents[1]


def _load_installer():
    spec = importlib.util.spec_from_file_location(
        "llm_cr_bundle_install", BUNDLE_DIR / "install.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


install = _load_installer()


def _opts(**overrides) -> "install.Options":
    base = dict(
        repo=str(BUNDLE_DIR), home=str(BUNDLE_DIR), backup_dir=None,
        provider="cr-local", model="test/cr-model:Q4_K_M",
        return_provider="exit-direct", return_model="exit-model-900k",
        cr_base_url="", api_mode="chat_completions", instruction_path=None,
        enable_prompt_plugin=False, no_aliases=False,
    )
    base.update(overrides)
    return install.Options(Namespace(**base))


# ── bundle self-integrity ───────────────────────────────────────────────────────


def test_the_shipped_manifest_matches_every_shipped_artifact():
    """The manifest is what the installer fails closed on, so a stale one must be caught here."""
    manifest = json.loads((BUNDLE_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert install.verify_bundle_integrity(manifest) == []


def test_manifest_patch_paths_match_the_patch_itself():
    manifest = json.loads((BUNDLE_DIR / "manifest.json").read_text(encoding="utf-8"))
    from_patch = sorted(
        line[6:].strip()
        for line in (BUNDLE_DIR / "core.patch").read_text(encoding="utf-8").splitlines()
        if line.startswith("+++ b/")
    )
    assert sorted(manifest["core_patch"]["paths"]) == from_patch


def test_integrity_check_reports_a_tampered_payload(tmp_path, monkeypatch):
    manifest = json.loads((BUNDLE_DIR / "manifest.json").read_text(encoding="utf-8"))
    tampered = dict(manifest)
    tampered["payload"] = dict(manifest["payload"])
    first = sorted(tampered["payload"])[0]
    tampered["payload"][first] = "0" * 64
    problems = install.verify_bundle_integrity(tampered)
    assert any("checksum mismatch" in p for p in problems)


def test_the_bundle_ships_no_provider_model_or_endpoint_default():
    """A public bundle must not carry the author's route. The only place these can come from is
    the command line (or, for the base URL, the target's own config)."""
    sources = [
        (BUNDLE_DIR / "install.py").read_text(encoding="utf-8"),
        *(p.read_text(encoding="utf-8", errors="replace")
          for p in (BUNDLE_DIR / "payload").rglob("*.py")),
        *(p.read_text(encoding="utf-8", errors="replace")
          for p in (BUNDLE_DIR / "payload").rglob("*.yaml")),
    ]
    # Assembled from fragments rather than written out, so this guard does not itself reintroduce
    # the identifiers it exists to keep out of the distributed bundle.
    forbidden = ["rd" + "chat", "openai" + "-codex", "gpt-6" + "-astra"]
    for text in sources:
        for token in forbidden:
            assert token not in text
    # install.py must not default any route flag to a non-empty value.
    parser_text = sources[0]
    for flag in ("--provider", "--model", "--return-provider", "--return-model", "--cr-base-url"):
        assert f'"{flag}", default=""' in parser_text


# ── route arguments are mandatory, never guessed ────────────────────────────────


@pytest.mark.parametrize("missing", ["provider", "model", "return_provider", "return_model"])
def test_a_missing_route_argument_is_refused_not_defaulted(missing):
    opts = _opts(**{missing: ""})
    with pytest.raises(install.Refused) as excinfo:
        opts.validate()
    assert "--" in str(excinfo.value)


def test_base_url_comes_from_the_targets_own_provider_entry_when_not_passed():
    config = {"providers": {"cr-local": {"base_url": "http://127.0.0.1:65500/v1",
                                         "api_key": "sk-untouched"}}}
    assert install.resolve_cr_base_url(config, _opts()) == "http://127.0.0.1:65500/v1"


def test_an_explicit_base_url_wins_over_the_config():
    config = {"providers": {"cr-local": {"base_url": "http://127.0.0.1:1/v1"}}}
    opts = _opts(cr_base_url="http://127.0.0.1:65500/v1/")
    assert install.resolve_cr_base_url(config, opts) == "http://127.0.0.1:65500/v1"


def test_an_unresolvable_base_url_is_refused_rather_than_invented():
    with pytest.raises(install.Refused) as excinfo:
        install.resolve_cr_base_url({"providers": {}}, _opts())
    assert "--cr-base-url" in str(excinfo.value)


def test_aliases_name_the_detour_commands_the_plugin_registers_for_the_given_route():
    assert install._alias_commands(_opts()) == {
        "llm-cr": "/detour test/cr-model:Q4_K_M --provider cr-local",
        "llm-cr-end": "/detour-end exit-model-900k --provider exit-direct",
    }


# ── config merge ────────────────────────────────────────────────────────────────


def test_the_merge_preserves_unrelated_keys_existing_plugins_and_secrets():
    config = {
        "model": {"default": "something-else"},
        "providers": {"cr-local": {"base_url": "http://127.0.0.1:65500/v1",
                                   "api_key": "sk-must-survive"}},
        "plugins": {"enabled": ["already-on"],
                    "entries": {"already-on": {"settings": {"keep": True}}}},
        "unrelated": {"nested": "keep-me"},
    }
    new, changes = install.plan_config_merge(
        config, _opts(enable_prompt_plugin=True), "http://127.0.0.1:65500/v1")
    assert new["unrelated"] == {"nested": "keep-me"}
    assert new["model"] == {"default": "something-else"}
    assert new["providers"]["cr-local"]["api_key"] == "sk-must-survive"
    assert new["plugins"]["enabled"][0] == "already-on"  # appended to, never replaced
    assert new["plugins"]["entries"]["already-on"] == {"settings": {"keep": True}}
    assert set(new["plugins"]["enabled"]) == {
        "already-on", "text-command-aliases", "session-detours", "llm-cr-prompt"}
    assert changes  # a fresh target has work to do
    # The input mapping is untouched, so a refusal mid-plan cannot have mutated the live config.
    assert config["plugins"]["enabled"] == ["already-on"]
    assert "text_command_aliases" not in config


def test_the_merge_is_idempotent_so_a_second_apply_changes_nothing():
    opts = _opts(enable_prompt_plugin=True)
    first, changes_one = install.plan_config_merge({}, opts, "http://127.0.0.1:65500/v1")
    second, changes_two = install.plan_config_merge(first, opts, "http://127.0.0.1:65500/v1")
    assert changes_one
    assert changes_two == []
    assert second == first


def test_changing_the_route_rewrites_only_this_bundles_keys():
    opts = _opts(enable_prompt_plugin=True)
    first, _ = install.plan_config_merge(
        {"unrelated": 1}, opts, "http://127.0.0.1:65500/v1")
    moved = _opts(enable_prompt_plugin=True, model="other/model")
    second, changes = install.plan_config_merge(first, moved, "http://127.0.0.1:65500/v1")
    assert second["unrelated"] == 1
    assert second["text_command_aliases"]["aliases"]["llm-cr"].endswith(
        "/detour other/model --provider cr-local")
    assert second["plugins"]["entries"]["llm-cr-prompt"]["settings"]["route_model"] == "other/model"
    assert any("route_model" in c for c in changes)


def test_without_the_prompt_plugin_no_prompt_entry_is_written():
    new, _ = install.plan_config_merge({}, _opts(), "")
    assert "llm-cr-prompt" not in new["plugins"]["enabled"]
    assert "llm-cr-prompt" not in new["plugins"].get("entries", {})


def test_no_aliases_leaves_the_alias_section_absent():
    new, _ = install.plan_config_merge({}, _opts(no_aliases=True, enable_prompt_plugin=True), "u")
    assert install.ALIAS_SECTION not in new
    assert "text-command-aliases" not in new["plugins"]["enabled"]


def test_the_detour_plugin_is_enabled_by_default_and_can_be_declined():
    """/detour is a plugin in 2.x, so enabling it is a config decision the operator can refuse —
    and refusing it must not drag the other two plugins out with it."""
    on, changes = install.plan_config_merge({}, _opts(), "")
    assert install.DETOUR_PLUGIN in on["plugins"]["enabled"]
    assert any(install.DETOUR_PLUGIN in c for c in changes)
    off, _ = install.plan_config_merge({}, _opts(no_detour_plugin=True), "")
    assert install.DETOUR_PLUGIN not in off["plugins"]["enabled"]
    assert "text-command-aliases" in off["plugins"]["enabled"]  # unaffected
    # The plugin carries no settings subtree of its own: there is nothing to configure, so a
    # declined detour leaves no entry behind either.
    assert install.DETOUR_PLUGIN not in off["plugins"].get("entries", {})


def test_the_change_list_never_echoes_a_settings_value_that_could_be_sensitive():
    """Change lines for the plugin settings name the key only; secrets are never written at all,
    but the receipt and stdout must not grow a habit of printing values either."""
    _new, changes = install.plan_config_merge(
        {}, _opts(enable_prompt_plugin=True), "http://127.0.0.1:65500/v1")
    settings_lines = [c for c in changes if "settings." in c]
    assert settings_lines
    for line in settings_lines:
        assert "=" not in line
        assert "127.0.0.1" not in line


# ── scope and prerequisite guards ──────────────────────────────────────────────


def test_a_backup_dir_inside_the_repo_or_home_is_refused(tmp_path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    home = tmp_path / "home"
    home.mkdir()
    opts = _opts(repo=str(repo), home=str(home), backup_dir=str(repo / "backups"))
    with pytest.raises(install.Refused) as excinfo:
        opts.validate()
    assert "outside both" in str(excinfo.value)


def test_a_backup_dir_that_contains_the_home_is_refused(tmp_path):
    """"Outside both" has to mean outside, not merely "not underneath": a backup dir that *holds*
    the home is not an independent rollback source either."""
    home = tmp_path / "hermes"
    home.mkdir()
    repo = home / "hermes-agent"
    (repo / ".git").mkdir(parents=True)
    opts = _opts(repo=str(repo), home=str(home), backup_dir=str(tmp_path))
    with pytest.raises(install.Refused) as excinfo:
        opts.validate()
    assert "outside both" in str(excinfo.value)


def test_the_standard_layout_with_the_checkout_inside_the_home_is_supported(tmp_path):
    """The REAL install puts the checkout at ``<home>/hermes-agent``. Refusing that made the whole
    bundle unusable on the layout it exists for, so it is explicitly accepted."""
    home = tmp_path / "hermes"
    home.mkdir()
    repo = home / "hermes-agent"
    (repo / ".git").mkdir(parents=True)
    opts = _opts(repo=str(repo), home=str(home), backup_dir=str(tmp_path / "backups"))
    opts.validate()  # must not raise
    assert opts.repo == repo.resolve()
    assert opts.home == home.resolve()


def test_a_home_inside_the_repo_is_refused(tmp_path):
    """The reverse nesting is the ambiguous one: the home would be part of the git checkout, so
    `git apply` and the payload writer would own the same subtree."""
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    home = repo / "home"
    home.mkdir()
    with pytest.raises(install.Refused) as excinfo:
        _opts(repo=str(repo), home=str(home), backup_dir=str(tmp_path / "b")).validate()
    assert "is inside --repo" in str(excinfo.value)


def test_repo_and_home_being_one_tree_is_refused(tmp_path):
    both = tmp_path / "one"
    (both / ".git").mkdir(parents=True)
    with pytest.raises(install.Refused) as excinfo:
        _opts(repo=str(both), home=str(both), backup_dir=str(tmp_path / "b")).validate()
    assert "same directory" in str(excinfo.value)


def test_a_checkout_whose_dot_git_is_a_worktree_file_is_accepted(tmp_path):
    """``git worktree`` checkouts carry ``.git`` as a FILE holding ``gitdir: ...``. That is a real
    checkout `git apply` works in, so the probe asks for existence, not for a directory."""
    home = tmp_path / "hermes"
    home.mkdir()
    repo = home / "hermes-agent"
    repo.mkdir()
    (repo / ".git").write_text(f"gitdir: {tmp_path / 'bare' / 'worktrees' / 'wt'}\n",
                               encoding="utf-8")
    _opts(repo=str(repo), home=str(home), backup_dir=str(tmp_path / "backups")).validate()


# ── per-file overlap guards (the half the roots alone cannot decide) ────────────


def _scope_opts(tmp_path):
    home = tmp_path / "hermes"
    home.mkdir()
    repo = home / "hermes-agent"
    (repo / ".git").mkdir(parents=True)
    return _opts(repo=str(repo), home=str(home), backup_dir=str(tmp_path / "backups"))


def test_the_nested_standard_layout_has_no_per_file_overlap(tmp_path):
    opts = _scope_opts(tmp_path)
    manifest = {"core_patch": {"paths": ["cli.py", "hermes_cli/config.py"]},
                "payload": {"plugins/llm-cr-prompt/injector.py": "x"}}
    assert install.scope_problems(opts, manifest) == []


def test_a_payload_file_landing_inside_the_checkout_is_refused(tmp_path):
    """With the checkout nested in the home, a payload path CAN point into the repo. Only
    `git apply` may write there, so that is the overlap that actually has to be refused."""
    opts = _scope_opts(tmp_path)
    manifest = {"core_patch": {"paths": []},
                "payload": {"hermes-agent/plugins/sneaky.py": "x"}}
    problems = install.scope_problems(opts, manifest)
    assert any("inside the git checkout" in p for p in problems)


def test_one_file_claimed_by_two_layers_is_refused(tmp_path):
    opts = _scope_opts(tmp_path)
    manifest = {"core_patch": {"paths": ["config.yaml"]}, "payload": {}}
    # repo/config.yaml is a core path here; make the config path collide with it.
    opts.home = opts.repo
    problems = install.scope_problems(opts, manifest)
    assert any("two owners" in p for p in problems)


def test_a_payload_path_escaping_the_home_is_refused(tmp_path):
    opts = _scope_opts(tmp_path)
    manifest = {"core_patch": {"paths": []}, "payload": {"../outside.py": "x"}}
    problems = install.scope_problems(opts, manifest)
    assert any("not inside its own root" in p for p in problems)


def test_a_core_path_escaping_the_repo_is_refused(tmp_path):
    opts = _scope_opts(tmp_path)
    manifest = {"core_patch": {"paths": ["../../elsewhere/cli.py"]}, "payload": {}}
    problems = install.scope_problems(opts, manifest)
    assert any("not inside its own root" in p for p in problems)


def test_the_prompt_plugin_requires_an_existing_instruction_file_and_never_creates_one(tmp_path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    home = tmp_path / "home"
    home.mkdir()
    opts = _opts(repo=str(repo), home=str(home), backup_dir=str(tmp_path / "b"),
                 enable_prompt_plugin=True)
    with pytest.raises(install.Refused) as excinfo:
        opts.validate()
    assert "never reads, writes or creates it" in str(excinfo.value)
    assert not (home / install.INSTRUCTION_RELPATH).exists()


def test_an_explicit_instruction_path_is_honoured_and_only_checked_for_existence(tmp_path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    home = tmp_path / "home"
    home.mkdir()
    sentinel = tmp_path / "sentinel.md"
    sentinel.write_text("SENTINEL ONLY\n", encoding="utf-8")
    opts = _opts(repo=str(repo), home=str(home), backup_dir=str(tmp_path / "b"),
                 enable_prompt_plugin=True, instruction_path=str(sentinel))
    opts.validate()
    assert opts.instruction_file == sentinel.resolve()
    assert sentinel.read_text(encoding="utf-8") == "SENTINEL ONLY\n"  # untouched


# ── API preflight ───────────────────────────────────────────────────────────────


def test_every_api_probe_names_a_file_a_needle_and_a_reason():
    manifest = json.loads((BUNDLE_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["required_apis"]
    for probe in manifest["required_apis"]:
        assert probe["file"] and probe["contains"] and probe["why"]


def test_preflight_refuses_a_target_missing_a_required_api(tmp_path):
    manifest = {"required_apis": [
        {"file": "mod.py", "contains": ["def needed"], "why": "because"},
    ]}
    (tmp_path / "mod.py").write_text("def other(): pass\n", encoding="utf-8")
    problems = install.preflight_apis(tmp_path, manifest)
    assert len(problems) == 1
    assert "def needed" in problems[0] and "because" in problems[0]


def test_preflight_refuses_a_target_missing_the_file_entirely(tmp_path):
    manifest = {"required_apis": [{"file": "gone.py", "contains": ["x"], "why": "y"}]}
    assert "not found" in install.preflight_apis(tmp_path, manifest)[0]


def test_preflight_passes_when_every_needle_is_present(tmp_path):
    manifest = {"required_apis": [
        {"file": "mod.py", "contains": ["def needed", "SOME_CONST"], "why": "z"},
    ]}
    (tmp_path / "mod.py").write_text("SOME_CONST = 1\ndef needed(): pass\n", encoding="utf-8")
    assert install.preflight_apis(tmp_path, manifest) == []


# ── rollback drift guard ────────────────────────────────────────────────────────


def _receipt(tmp_path, entries) -> Path:
    receipt = {
        "receipt_version": 1, "receipt_id": "test", "repo": str(tmp_path / "repo"),
        "home": str(tmp_path / "home"), "files": entries,
    }
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    return path


def test_rollback_restores_a_modified_file_and_removes_a_created_one(tmp_path, capsys):
    backup = tmp_path / "backup"
    backup.mkdir()
    existing = tmp_path / "home" / "existing.txt"
    existing.parent.mkdir(parents=True)
    # Bytes, not write_text: the receipt hashes bytes, and write_text would translate the newline
    # on Windows so the fixture's own hash would never match its own file.
    existing.write_bytes(b"after\n")
    (backup / "existing.txt").write_bytes(b"before\n")
    created = tmp_path / "home" / "new" / "created.txt"
    created.parent.mkdir(parents=True)
    created.write_bytes(b"installed\n")

    receipt = _receipt(tmp_path, [
        {"kind": "payload", "path": str(existing), "root": str(tmp_path / "home"),
         "backup": str(backup / "existing.txt"), "existed_before": True,
         "sha256_before": install._sha256_bytes(b"before\n"), "mode_before": None,
         "sha256_after": install._sha256_bytes(b"after\n")},
        {"kind": "payload", "path": str(created), "root": str(tmp_path / "home"),
         "backup": None, "existed_before": False, "sha256_before": None, "mode_before": None,
         "sha256_after": install._sha256_bytes(b"installed\n")},
    ])
    assert install.cmd_rollback(receipt) == install.EXIT_OK
    assert existing.read_bytes() == b"before\n"
    assert not created.exists()
    assert not created.parent.exists()  # emptied directory pruned


def test_rollback_refuses_wholesale_when_any_touched_file_changed_after_apply(tmp_path, capsys):
    backup = tmp_path / "backup"
    backup.mkdir()
    drifted = tmp_path / "home" / "drifted.txt"
    drifted.parent.mkdir(parents=True)
    drifted.write_bytes(b"user edited this later\n")
    (backup / "drifted.txt").write_bytes(b"before\n")
    untouched = tmp_path / "home" / "untouched.txt"
    untouched.write_bytes(b"installed\n")

    receipt = _receipt(tmp_path, [
        {"kind": "payload", "path": str(drifted), "root": str(tmp_path / "home"),
         "backup": str(backup / "drifted.txt"), "existed_before": True,
         "sha256_before": install._sha256_bytes(b"before\n"), "mode_before": None,
         "sha256_after": install._sha256_bytes(b"after\n")},
        {"kind": "payload", "path": str(untouched), "root": str(tmp_path / "home"),
         "backup": None, "existed_before": False, "sha256_before": None, "mode_before": None,
         "sha256_after": install._sha256_bytes(b"installed\n")},
    ])
    assert install.cmd_rollback(receipt) == install.EXIT_REFUSED
    # NOTHING was written: not even the file that had not drifted.
    assert drifted.read_bytes() == b"user edited this later\n"
    assert untouched.exists()


def test_rollback_is_a_no_op_on_a_file_already_back_at_its_pre_apply_content(tmp_path):
    backup = tmp_path / "backup"
    backup.mkdir()
    target = tmp_path / "home" / "f.txt"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"before\n")
    (backup / "f.txt").write_bytes(b"before\n")
    receipt = _receipt(tmp_path, [
        {"kind": "payload", "path": str(target), "root": str(tmp_path / "home"),
         "backup": str(backup / "f.txt"), "existed_before": True,
         "sha256_before": install._sha256_bytes(b"before\n"), "mode_before": None,
         "sha256_after": install._sha256_bytes(b"after\n")},
    ])
    assert install.cmd_rollback(receipt) == install.EXIT_OK
    assert target.read_bytes() == b"before\n"


def test_rollback_refuses_a_receipt_whose_backup_copy_is_gone(tmp_path):
    target = tmp_path / "home" / "f.txt"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"after\n")
    receipt = _receipt(tmp_path, [
        {"kind": "payload", "path": str(target), "root": str(tmp_path / "home"),
         "backup": str(tmp_path / "missing" / "f.txt"), "existed_before": True,
         "sha256_before": install._sha256_bytes(b"before\n"), "mode_before": None,
         "sha256_after": install._sha256_bytes(b"after\n")},
    ])
    assert install.cmd_rollback(receipt) == install.EXIT_REFUSED
    assert target.read_bytes() == b"after\n"


def test_rollback_refuses_an_unknown_receipt_version(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"receipt_version": 99, "files": []}), encoding="utf-8")
    with pytest.raises(install.Refused):
        install.cmd_rollback(path)


def test_rollback_refuses_a_receipt_replayed_against_another_scope(tmp_path):
    receipt = _receipt(tmp_path, [])
    with pytest.raises(install.Refused) as excinfo:
        install.cmd_rollback(receipt, force_scope=(tmp_path / "repo", tmp_path / "other-home"))
    assert "scope mismatch" in str(excinfo.value)


# ── atomic config write ────────────────────────────────────────────────────────


def test_the_config_write_is_atomic_and_leaves_no_temp_file(tmp_path):
    target = tmp_path / "config.yaml"
    target.write_text("old: 1\n", encoding="utf-8")
    install._atomic_write_bytes(target, b"new: 2\n")
    assert target.read_text(encoding="utf-8") == "new: 2\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["config.yaml"]


# ── end to end on the REAL layout: a git checkout at <home>/hermes-agent ────────
#
# Everything above tests decisions in isolation. This section runs the actual commands —
# check -> apply -> apply again -> verify -> rollback -> verify — against a throwaway fixture laid
# out the way a real install is laid out: the Hermes home on the outside, the checkout nested inside
# it. That nesting is precisely what the installer used to refuse, so a unit test on validate() is
# not enough: the backup slots, the receipt roots and the directory pruning all have to survive it
# too.
#
# The repo is a real `git init` checkout (so `git apply --check`, `--reverse --check` and the real
# patch application are exercised), but its contents are stubs written by this test. The PAYLOAD is
# the bundle's real payload, so the installed plugin files and `verify`'s import of them are the
# shipped ones. No private instruction file, no real profile and no real repo is read or written.

_CORE_BEFORE = b'''"""Stand-in for a core module this bundle patches."""

MARKER = "pre-install"
'''

_CORE_AFTER = b'''"""Stand-in for a core module this bundle patches."""

MARKER = "pre-install"

DETOUR_INSTALLED = True
'''

_STUB_REPO_FILES = {
    ".gitignore": b"__pycache__/\n",
    "src/core_module.py": _CORE_BEFORE,
    "hermes_cli/__init__.py": b"",
    "hermes_cli/middleware.py": b'''"""Stand-in for hermes_cli.middleware."""

LLM_EXECUTION_MIDDLEWARE = "llm_execution"


class MiddlewareAbort(Exception):
    """The abort type the shipped injector imports and raises."""
''',
    # The post-patch core registry: in 2.x the detour commands are NOT here, they are the plugin's.
    # `verify` reads this to catch a pre-2.0 (core-resident) install underneath, so the fixture has
    # to own it rather than leaving the probe to import this repo's real registry.
    "hermes_cli/commands.py": b'''"""Stand-in built-in command registry."""


class CommandDef:
    def __init__(self, name, busy_policy=None):
        self.name = name
        self.busy_policy = busy_policy


COMMAND_REGISTRY = [CommandDef("new"), CommandDef("resume"),
                    CommandDef("model", busy_policy="reject")]
''',
    "hermes_cli/text_command_aliases.py": b'''"""Stand-in matcher with the exact-match semantics `verify` asserts on."""


def resolve_text_command_alias(text, config):
    if not isinstance(text, str) or not isinstance(config, dict):
        return None
    section = config.get("text_command_aliases")
    if not isinstance(section, dict) or section.get("enabled") is not True:
        return None
    table = section.get("aliases")
    if not isinstance(table, dict):
        return None
    return table.get(text.strip())
''',
    "hermes_cli/config.py": b'''"""Stand-in config reader."""

import os
from pathlib import Path

_OPEN_DICT_TOP_LEVEL_KEYS = ("text_command_aliases",)


def read_raw_config_readonly():
    import yaml

    path = Path(os.environ["HERMES_HOME"]) / "config.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
''',
    # Enough of the real loader's shape for `verify` to exercise production discovery: it finds
    # plugin directories under the home, enables only those named in config, hands each a ctx with
    # get_config/register_hook/register_middleware, and records what they registered.
    "hermes_cli/plugins.py": b'''"""Stand-in for hermes_cli.plugins.PluginManager."""

import importlib.util
import os
import sys
from pathlib import Path

import yaml


class _Loaded:
    def __init__(self, name):
        self.name = name
        self.enabled = False
        self.error = None


class _Ctx:
    def __init__(self, manager, name, settings):
        self._manager = manager
        self._name = name
        self._settings = settings

    def get_config(self, key, default=None):
        return self._settings.get(key, default)

    def register_hook(self, seam, handler):
        self._manager._hooks.setdefault(seam, []).append((self._name, handler))

    def register_middleware(self, seam, handler):
        self._manager._middleware.setdefault(seam, []).append((self._name, handler))

    def register_command(self, name, handler, description="", args_hint="",
                         argument_mode=None, busy_policy=None):
        """Same shape the real loader records: a normalized name and an entry dict whose
        ``busy_policy`` is kept only when it is one the busy path actually understands."""
        clean = name.lower().strip().lstrip("/").replace(" ", "-")
        if not clean:
            return
        self._manager._plugin_commands[clean] = {
            "handler": handler, "description": description or "Plugin command",
            "plugin": self._name, "args_hint": args_hint.strip(),
            "argument_mode": argument_mode if argument_mode in {"options", "text", "mixed"}
            else ("text" if args_hint.strip() else None),
            "busy_policy": busy_policy if busy_policy in {"reject"} else None,
        }


class PluginManager:
    def __init__(self):
        self._plugins = {}
        self._hooks = {}
        self._middleware = {}
        self._plugin_commands = {}

    def discover_and_load(self, force=False):
        home = Path(os.environ["HERMES_HOME"])
        config = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8")) or {}
        plugins = config.get("plugins") or {}
        enabled = plugins.get("enabled") or []
        entries = plugins.get("entries") or {}
        for directory in sorted((home / "plugins").glob("*")):
            if not (directory / "__init__.py").is_file():
                continue
            name = directory.name
            record = _Loaded(name)
            self._plugins[name] = record
            if name not in enabled:
                continue
            record.enabled = True
            settings = (entries.get(name) or {}).get("settings") or {}
            module_name = "stub_plugin_" + name.replace("-", "_")
            try:
                spec = importlib.util.spec_from_file_location(
                    module_name, directory / "__init__.py",
                    submodule_search_locations=[str(directory)],
                )
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                try:
                    spec.loader.exec_module(module)
                    module.register(_Ctx(self, name, settings))
                finally:
                    sys.modules.pop(module_name, None)
            except Exception as exc:
                record.error = f"{type(exc).__name__}: {exc}"
''',
}

_FIXTURE_CONFIG = b"""model:
  default: fixture-main-model
providers:
  fixture-cr:
    base_url: http://127.0.0.1:65500/v1
    api_key: sk-must-survive
  fixture-main:
    base_url: http://127.0.0.1:65501/v1
unrelated:
  nested: keep-me
plugins:
  enabled: []
"""

_HERMES_MODULE_PREFIXES = ("hermes_cli", "hermes_constants", "agent", "gateway", "utils")


def _fixture_git(repo: Path, *args: str) -> bytes:
    import subprocess

    proc = subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid",
         "-c", "commit.gpgsign=false", "-c", "core.autocrlf=false", *args],
        capture_output=True,
    )
    assert proc.returncode == 0, (
        f"git {' '.join(args)} failed: {(proc.stderr or proc.stdout).decode('utf-8', 'replace')}"
    )
    return proc.stdout


class _NoHermesImports:
    """Keep the fixture's stub ``hermes_cli`` from colliding with this repo's real one.

    ``verify`` imports ``hermes_cli`` out of the target checkout, but an already-imported package of
    that name would win over the sys.path entry it inserts. So: drop those modules for the duration,
    and put the originals back afterwards.
    """

    def __enter__(self):
        self._saved = {k: v for k, v in sys.modules.items() if self._ours(k)}
        for name in self._saved:
            sys.modules.pop(name, None)
        return self

    def __exit__(self, *_exc):
        for name in [k for k in list(sys.modules) if self._ours(k)]:
            sys.modules.pop(name, None)
        sys.modules.update(self._saved)
        return False

    @staticmethod
    def _ours(name: str) -> bool:
        return name.split(".")[0] in _HERMES_MODULE_PREFIXES


@pytest.fixture
def nested_install(tmp_path, monkeypatch):
    """The real layout: home on the outside, a real git checkout at ``<home>/hermes-agent``."""
    home = tmp_path / "hermes"
    repo = home / "hermes-agent"
    repo.mkdir(parents=True)
    backups = tmp_path / "llm-cr-bundle-backups"

    for rel, data in _STUB_REPO_FILES.items():
        target = repo / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    _fixture_git(repo, "init", "-q")
    _fixture_git(repo, "add", "-A")
    _fixture_git(repo, "commit", "-q", "-m", "fixture base")

    # A synthetic bundle: this fixture's own patch and manifest, the bundle's REAL payload.
    bundle = tmp_path / "bundle"
    payload = bundle / "payload"
    shutil.copytree(BUNDLE_DIR / "payload", payload,
                    ignore=shutil.ignore_patterns("__pycache__"))

    (repo / "src" / "core_module.py").write_bytes(_CORE_AFTER)
    patch_bytes = _fixture_git(repo, "diff")
    _fixture_git(repo, "checkout", "--", "src/core_module.py")
    assert (repo / "src" / "core_module.py").read_bytes() == _CORE_BEFORE
    core_patch = bundle / "core.patch"
    core_patch.write_bytes(patch_bytes)

    payload_index = {
        str(p.relative_to(payload)).replace(os.sep, "/"): install._sha256_file(p)
        for p in sorted(payload.rglob("*")) if p.is_file()
    }
    manifest = {
        "name": "llm-cr-bundle-fixture",
        "version": "0.0.0-fixture",
        "manifest_version": 1,
        "base_commit": _fixture_git(repo, "rev-parse", "HEAD").decode().strip(),
        "core_patch": {
            "file": "core.patch",
            "sha256": install._sha256_file(core_patch),
            "paths": ["src/core_module.py"],
        },
        "payload": payload_index,
        "required_apis": [
            {"file": "hermes_cli/middleware.py",
             "contains": ['LLM_EXECUTION_MIDDLEWARE = "llm_execution"', "class MiddlewareAbort"],
             "why": "the shipped injector imports both"},
            {"file": "hermes_cli/text_command_aliases.py",
             "contains": ["def resolve_text_command_alias"],
             "why": "the aliases resolve through this matcher"},
        ],
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    monkeypatch.setattr(install, "MANIFEST_PATH", bundle / "manifest.json")
    monkeypatch.setattr(install, "CORE_PATCH_PATH", core_patch)
    monkeypatch.setattr(install, "PAYLOAD_DIR", payload)

    home.mkdir(exist_ok=True)
    (home / "config.yaml").write_bytes(_FIXTURE_CONFIG)
    instruction = home / install.INSTRUCTION_RELPATH
    instruction.parent.mkdir(parents=True, exist_ok=True)
    instruction.write_bytes(b"FIXTURE SENTINEL ONLY - not a real instruction file\n")

    opts = _opts(
        repo=str(repo), home=str(home), backup_dir=str(backups),
        provider="fixture-cr", model="fixture/cr-model:Q4_K_M",
        return_provider="fixture-main", return_model="fixture-main-model",
        enable_prompt_plugin=True,
    )
    return SimpleNamespace(
        home=home, repo=repo, backups=backups, bundle=bundle, opts=opts,
        manifest=install._load_manifest(), instruction=instruction,
        core=repo / "src" / "core_module.py",
        config=home / "config.yaml",
    )


def test_the_nested_standard_layout_installs_verifies_and_rolls_back(nested_install, capsys):
    fx = nested_install
    config_before = fx.config.read_bytes()

    # ── check: writes nothing ──
    assert install.cmd_check(fx.opts, fx.manifest) == install.EXIT_OK
    out = capsys.readouterr().out
    assert "RESULT: ready to apply" in out
    assert str(fx.repo) in out and str(fx.home) in out
    assert fx.core.read_bytes() == _CORE_BEFORE
    assert fx.config.read_bytes() == config_before
    assert not fx.backups.exists()

    # ── apply ──
    assert install.cmd_apply(fx.opts, fx.manifest) == install.EXIT_OK
    assert "APPLIED. Receipt:" in capsys.readouterr().out
    assert fx.core.read_bytes() == _CORE_AFTER
    for rel, digest in fx.manifest["payload"].items():
        assert install._sha256_file(fx.home / rel) == digest, rel

    import yaml

    merged = yaml.safe_load(fx.config.read_text(encoding="utf-8"))
    assert merged["text_command_aliases"]["aliases"] == {
        "llm-cr": "/detour fixture/cr-model:Q4_K_M --provider fixture-cr",
        "llm-cr-end": "/detour-end fixture-main-model --provider fixture-main",
    }
    assert merged["providers"]["fixture-cr"]["api_key"] == "sk-must-survive"
    assert merged["unrelated"] == {"nested": "keep-me"}
    assert merged["plugins"]["entries"]["llm-cr-prompt"]["settings"]["route_base_url"] == (
        "http://127.0.0.1:65500/v1")

    receipts = sorted(fx.backups.glob("*/receipt.json"))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert receipt["repo"] == str(fx.repo) and receipt["home"] == str(fx.home)
    # Nested trees, separate backup slots — and the config's pre-apply bytes are really there.
    kinds = {entry["kind"] for entry in receipt["files"]}
    assert kinds == {"core", "payload", "config"}
    config_entry = next(e for e in receipt["files"] if e["kind"] == "config")
    assert Path(config_entry["backup"]).read_bytes() == config_before
    core_entry = next(e for e in receipt["files"] if e["kind"] == "core")
    assert Path(core_entry["backup"]).read_bytes() == _CORE_BEFORE
    assert "api_key" not in receipts[0].read_text(encoding="utf-8")

    # ── apply again: idempotent, no second receipt, no second backup ──
    assert install.cmd_apply(fx.opts, fx.manifest) == install.EXIT_OK
    assert "Already applied" in capsys.readouterr().out
    assert sorted(fx.backups.glob("*/receipt.json")) == receipts
    assert fx.core.read_bytes() == _CORE_AFTER

    # ── verify: real patch state, real payload import, production-shaped discovery ──
    with _NoHermesImports():
        assert install.cmd_verify(fx.opts, fx.manifest) == install.EXIT_OK
    out = capsys.readouterr().out
    assert "VERIFY OK" in out
    assert "matcher resolves 'llm-cr'" in out
    assert "near-miss correctly not an alias" in out
    assert "plugin loaded + enabled by production discovery" in out
    # The detour feature IS these two plugin-command registrations, busy_policy included.
    assert "command /detour (busy_policy='reject')" in out
    assert "command /detour-end (busy_policy='reject')" in out
    assert "core registers no detour command" in out
    # verify is read-only on process state as well as on disk.
    assert "HERMES_HOME" not in os.environ
    assert str(fx.repo) not in sys.path

    # ── rollback: exact restore of the config and the source, nothing else touched ──
    assert install.cmd_rollback(receipts[0], force_scope=(fx.repo, fx.home)) == install.EXIT_OK
    assert "ROLLED BACK" in capsys.readouterr().out
    assert fx.core.read_bytes() == _CORE_BEFORE
    assert fx.config.read_bytes() == config_before
    assert _fixture_git(fx.repo, "status", "--porcelain").decode().strip() == ""
    for rel in fx.manifest["payload"]:
        assert not (fx.home / rel).exists(), rel
    # Every file the install wrote is gone and the directories it emptied are pruned. What may
    # survive is interpreter-written ``__pycache__`` from verify's own imports — documented as not
    # the installer's to remove, and harmless because its sources are gone.
    leftovers = sorted(p.relative_to(fx.home).as_posix()
                       for p in (fx.home / "plugins").rglob("*") if p.is_file())
    assert [rel for rel in leftovers if "__pycache__" not in rel] == []
    assert not (fx.home / "plugins" / "llm-cr-prompt" / "tests").exists()
    assert fx.home.is_dir() and fx.repo.is_dir()  # roots themselves survive
    assert fx.instruction.read_bytes().startswith(b"FIXTURE SENTINEL ONLY")  # never touched
    assert receipts[0].is_file()  # the receipt and its backups are kept

    # ── and the install really is gone ──
    with _NoHermesImports():
        assert install.cmd_verify(fx.opts, fx.manifest) == install.EXIT_FAIL
    assert "VERIFY FAILED" in capsys.readouterr().err


def test_a_second_check_after_apply_reports_nothing_to_do(nested_install, capsys):
    fx = nested_install
    assert install.cmd_apply(fx.opts, fx.manifest) == install.EXIT_OK
    capsys.readouterr()
    assert install.cmd_check(fx.opts, fx.manifest) == install.EXIT_OK
    assert "already fully applied" in capsys.readouterr().out


def test_rollback_refuses_after_the_operator_edits_an_installed_file(nested_install, capsys):
    """The drift guard on the real layout: one hand-edited payload file stops the whole rollback,
    including the core patch and the config, so no work is silently discarded."""
    fx = nested_install
    assert install.cmd_apply(fx.opts, fx.manifest) == install.EXIT_OK
    capsys.readouterr()
    receipt = next(iter(sorted(fx.backups.glob("*/receipt.json"))))
    edited = fx.home / "plugins" / "llm-cr-prompt" / "injector.py"
    edited.write_bytes(b"# operator edit\n")

    assert install.cmd_rollback(receipt, force_scope=(fx.repo, fx.home)) == install.EXIT_REFUSED
    err = capsys.readouterr().err
    assert "modified since apply" in err
    assert edited.read_bytes() == b"# operator edit\n"
    assert fx.core.read_bytes() == _CORE_AFTER  # nothing rolled back, not even the patch


def test_a_git_timeout_is_a_refusal_not_a_patch_state(nested_install, monkeypatch):
    """A hung git must never be read as "absent" (which would apply the patch twice) or "applied"."""
    import subprocess

    def _hang(*_args, **kwargs):
        raise subprocess.TimeoutExpired(kwargs.get("args", "git"), install.GIT_TIMEOUT_SECONDS)

    monkeypatch.setattr(subprocess, "run", _hang)
    with pytest.raises(install.Refused) as excinfo:
        install.core_patch_state(nested_install.repo)
    assert "did not finish within" in str(excinfo.value)


def test_a_missing_git_is_a_refusal(nested_install, monkeypatch):
    import subprocess

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()))
    with pytest.raises(install.Refused) as excinfo:
        install.core_patch_state(nested_install.repo)
    assert "not found on PATH" in str(excinfo.value)
