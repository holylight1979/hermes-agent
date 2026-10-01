"""``_create_temp_file`` retry policy: a name collision earns another name, nothing else does.

``tempfile.mkstemp`` retries every ``OSError`` it can plausibly read as a collision
(``TMP_MAX`` = 2**31-1 on Windows).  The replacement retries ONLY ``FileExistsError``,
and only a bounded number of times, so an undeliverable write surfaces its real errno
instead of spinning.  Real ``os.open`` throughout — the OS is never faked; only the
random name source is pinned, to force the collision deterministically.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from utils import _TEMP_NAME_ATTEMPTS, _create_temp_file


def _pin_names(monkeypatch: pytest.MonkeyPatch, *hex_names: str) -> None:
    """Make the candidate names deterministic; the last one repeats forever."""
    queue = list(hex_names)

    def fake_urandom(n: int) -> bytes:
        return bytes.fromhex(queue.pop(0) if len(queue) > 1 else queue[0])

    monkeypatch.setattr(os, "urandom", fake_urandom)


def test_collision_moves_on_to_a_fresh_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A taken name is retried, and the squatter's content survives untouched."""
    taken = tmp_path / ("pfx" + "aa" * 8 + ".tmp")
    taken.write_text("not mine", encoding="utf-8")

    _pin_names(monkeypatch, "aa" * 8, "bb" * 8)
    fd, created = _create_temp_file(tmp_path, "pfx", ".tmp")
    os.close(fd)

    assert Path(created).name == "pfx" + "bb" * 8 + ".tmp"
    assert taken.read_text(encoding="utf-8") == "not mine"


def test_endless_collisions_give_up_instead_of_spinning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every candidate taken → a bounded number of attempts, then the real error."""
    (tmp_path / ("pfx" + "cc" * 8 + ".tmp")).write_text("squat", encoding="utf-8")
    _pin_names(monkeypatch, "cc" * 8)

    attempts = 0
    real_open = os.open

    def counting_open(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        return real_open(*args, **kwargs)

    monkeypatch.setattr(os, "open", counting_open)
    with pytest.raises(FileExistsError):
        _create_temp_file(tmp_path, "pfx", ".tmp")
    assert attempts == _TEMP_NAME_ATTEMPTS


def test_a_non_collision_error_is_raised_on_the_first_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bug class: any error that is not a collision must not be retried at all.

    A missing directory is the portable stand-in for the Windows ACL denial — same
    contract, same code path, no privileged setup.
    """
    attempts = 0
    real_open = os.open

    def counting_open(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        return real_open(*args, **kwargs)

    monkeypatch.setattr(os, "open", counting_open)
    with pytest.raises(OSError) as excinfo:
        _create_temp_file(tmp_path / "does-not-exist", "pfx", ".tmp")

    assert not isinstance(excinfo.value, FileExistsError)
    assert attempts == 1


def test_an_existing_symlink_is_never_opened_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """O_EXCL means a symlink squatting on the candidate name is a collision, not a target."""
    victim = tmp_path / "victim.txt"
    victim.write_text("precious", encoding="utf-8")
    link = tmp_path / ("pfx" + "dd" * 8 + ".tmp")
    try:
        link.symlink_to(victim)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable for this account: {exc}")

    _pin_names(monkeypatch, "dd" * 8, "ee" * 8)
    fd, created = _create_temp_file(tmp_path, "pfx", ".tmp")
    try:
        os.write(fd, b"temp payload")
    finally:
        os.close(fd)

    assert Path(created).name == "pfx" + "ee" * 8 + ".tmp"
    assert victim.read_text(encoding="utf-8") == "precious"
