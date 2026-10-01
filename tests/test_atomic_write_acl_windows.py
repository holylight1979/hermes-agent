"""An ACL-denied directory must fail an atomic write FAST, on a real Windows ACL.

``tempfile.mkstemp`` treats every ``PermissionError`` as "a directory with that name
already exists" whenever ``os.path.isdir(dir) and os.access(dir, W_OK)``.  Windows
``os.access`` only reads the read-only attribute — it never consults the ACL — so a
directory the account is denied by ACL answers True and mkstemp retries ``TMP_MAX``
(2**31-1 on Windows) names that can never be created.  That loop ran inside the
process-results receipt write and wedged the gateway event loop for hours.

This cannot be faked: it needs a real NTFS deny ACE, because the whole bug is that
``os.access`` disagrees with the ACL.  The write runs in a child process so a
regression is a failed 5s timeout, not a hung test session.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.windows_only

REPO_ROOT = Path(__file__).resolve().parents[1]

# Generous: the real fix returns in well under a second; the bug never returns at all.
_TIMEOUT_SECONDS = 5.0

_CHILD_SOURCE = """
import json, sys, time
sys.path.insert(0, %r)
from utils import atomic_json_write

start = time.monotonic()
try:
    atomic_json_write(sys.argv[1], {"clobbered": True})
    outcome = "wrote"
except PermissionError:
    outcome = "PermissionError"
except OSError as exc:
    outcome = type(exc).__name__
print(json.dumps({"outcome": outcome, "seconds": time.monotonic() - start}))
"""


def _icacls(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["icacls", *args], capture_output=True, text=True, timeout=30)


def _current_account() -> str:
    who = subprocess.run(["whoami"], capture_output=True, text=True, timeout=30)
    account = who.stdout.strip()
    if not account:
        pytest.skip("could not resolve the current Windows account")
    return account


def test_write_denied_by_acl_fails_fast_and_leaves_the_target_intact(tmp_path: Path) -> None:
    denied_dir = tmp_path / "process-results"
    denied_dir.mkdir()
    target = denied_dir / "proc_receipt.json"
    original = {"session_id": "proc_receipt", "exit_code": 0}
    target.write_text(json.dumps(original), encoding="utf-8")

    account = _current_account()
    # No (OI)(CI): the deny covers creating entries in this folder only, so the existing
    # receipt keeps its own ACL and stays readable for the clobber check.
    denied = _icacls(str(denied_dir), "/deny", f"{account}:(W)")
    if denied.returncode != 0:
        pytest.skip(f"icacls could not apply a deny ACE here: {denied.stdout}{denied.stderr}")
    try:
        probe = denied_dir / "acl_probe.tmp"
        try:
            probe.touch()
        except PermissionError:
            pass
        else:
            probe.unlink()
            pytest.skip("the deny ACE does not bind this token (elevated or backup privilege)")

        child = subprocess.run(
            [sys.executable, "-c", _CHILD_SOURCE % str(REPO_ROOT), str(target)],
            capture_output=True, text=True, timeout=_TIMEOUT_SECONDS, cwd=str(tmp_path),
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"atomic_json_write did not return within {_TIMEOUT_SECONDS}s on an ACL-denied "
            "directory — the unbounded mkstemp retry is back"
        )
    finally:
        _icacls(str(denied_dir), "/remove:d", account)

    assert child.returncode == 0, f"child crashed: {child.stderr}"
    result = json.loads(child.stdout.strip().splitlines()[-1])
    assert result["outcome"] == "PermissionError", f"unexpected outcome: {result}"
    assert result["seconds"] < _TIMEOUT_SECONDS
    # A failed write must never destroy the receipt it was replacing.
    assert json.loads(target.read_text(encoding="utf-8")) == original
