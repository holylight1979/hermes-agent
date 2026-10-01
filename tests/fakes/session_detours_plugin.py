"""Load the ``session-detours`` plugin the way Hermes loads it, for tests.

The feature no longer lives in core, so the tests that own its contracts have to import it from
the plugin payload in this checkout (``contrib/llm-cr-bundle/payload/plugins/session-detours``).
The package is imported by file location under a stable module name, which is exactly what
``hermes_cli.plugins_loader`` does, so the plugin's own relative imports resolve.

``detour_cli(cli)`` / ``detour_gateway(runner)`` return the plugin's per-host surface adapter —
the same object the registered ``/detour`` handler builds when a real CLI or gateway dispatches
the command — so a test can drive the real leg against a real host.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

MODULE_NAME = "hermes_session_detours_plugin"


def _plugin_dir() -> Path:
    """Where the plugin source lives for this checkout.

    In the checkout that *ships* the bundle it is the payload under ``contrib/``. A checkout that
    only received the bundle (patch applied, payload installed into the Hermes home) has no
    ``contrib/llm-cr-bundle``, so fall back to the installed copy — the same files, at the location
    the installer wrote them to.
    """
    in_repo = (Path(__file__).resolve().parents[2]
               / "contrib" / "llm-cr-bundle" / "payload" / "plugins" / "session-detours")
    if (in_repo / "__init__.py").is_file():
        return in_repo
    from hermes_constants import get_hermes_home

    return Path(get_hermes_home()) / "plugins" / "session-detours"


PLUGIN_DIR = _plugin_dir()


def load() -> object:
    """The imported plugin package (cached in ``sys.modules`` after the first call)."""
    existing = sys.modules.get(MODULE_NAME)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(
        MODULE_NAME, PLUGIN_DIR / "__init__.py", submodule_search_locations=[str(PLUGIN_DIR)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[MODULE_NAME]
        raise
    return module


plugin = load()
records = importlib.import_module(f"{MODULE_NAME}.detour_records")
detour_cli = plugin._cli_adapter
detour_gateway = plugin._gateway_adapter
dispatch_detour = plugin._detour_command
dispatch_detour_end = plugin._detour_end_command
register = plugin.register

# Re-export the record store's public names so a test imports them from one place
# (``from tests.fakes.session_detours_plugin import DetourScope, read_record, ...``).
globals().update({name: getattr(records, name)
                  for name in dir(records) if not name.startswith("_")})


def install_into_home(home: Path, *, enabled: bool = True):
    """Install the plugin into *home* the way the bundle installer does, then discover it.

    Copies the payload to ``<home>/plugins/session-detours`` and writes the ``plugins.enabled``
    consent into ``<home>/config.yaml`` (user plugins are opt-in), so the slash commands come
    from a real discovery pass over a real plugins directory — not from a hand-built registry.
    With ``enabled=False`` the plugin is on disk but not consented, which is how a test asserts
    that a disabled plugin means no ``/detour`` behavior at all.

    Returns the live ``PluginManager``. The caller is responsible for restoring process-global
    discovery state (see :func:`reset_discovery`).
    """
    target = Path(home) / "plugins" / "session-detours"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(PLUGIN_DIR, target, ignore=shutil.ignore_patterns("__pycache__", "tests"))

    config = Path(home) / "config.yaml"
    existing = config.read_text(encoding="utf-8") if config.exists() else ""
    entries = "\n    - session-detours" if enabled else ""
    config.write_text(f"{existing}\nplugins:\n  enabled:{entries or ' []'}\n  disabled: []\n",
                      encoding="utf-8")

    from hermes_cli import plugins as plugins_mod
    reset_discovery()
    return plugins_mod._ensure_plugins_discovered(force=True)


def reset_discovery() -> None:
    """Drop the process-global plugin manager so the next discovery starts from this home."""
    from hermes_cli import plugins as plugins_mod

    plugins_mod._reset_plugin_managers_for_tests()
