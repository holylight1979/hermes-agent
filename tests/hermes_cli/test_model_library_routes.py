"""Contract tests for the model-library dashboard routes (``/api/model/library``).

Real FastAPI ``TestClient`` against the real app, so the assertions cover the
*mounted* surface (route order / auth middleware), not just the router module.
Storage is exercised against a temp ``HERMES_HOME``; nothing here touches the
network or the user's real config.

The invariants under test:

* the library file is ``models.json`` under the ACTIVE ``HERMES_HOME``;
* the configured model is prepended to ``GET`` and appears exactly once;
* ``POST`` is idempotent on (provider, model, baseUrl);
* a corrupt library degrades to "no shortcuts", never to a 500;
* every route is gated by the dashboard's single session-token scheme.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    """An isolated HERMES_HOME for this test; the library lives inside it."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    return hermes_home


@pytest.fixture
def client(home):
    from hermes_cli import web_server

    test_client = TestClient(web_server.app)
    # Same auth pattern as the other router tests: present the session token.
    test_client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    return test_client


def _library_file(home: Path) -> Path:
    return home / "models.json"


def _write_model_config(home: Path, *, provider: str, model: str, base_url: str = "") -> None:
    """Set the configured model the way config.yaml really carries it, so GET's
    "current model first" row comes from the real ``load_config()`` path."""
    import yaml

    from hermes_cli.config import _LOAD_CONFIG_CACHE

    model_cfg = {"provider": provider, "default": model}
    if base_url:
        model_cfg["base_url"] = base_url
    config_path = home / "config.yaml"
    config_path.write_text(yaml.safe_dump({"model": model_cfg}), encoding="utf-8")
    # load_config() caches per config path on an mtime/size signature; drop this path's
    # entry so the route reads what was just written rather than a same-tick stale hit.
    _LOAD_CONFIG_CACHE.pop(str(config_path), None)


# ── auth boundary ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "method,path",
    [
        ("GET", "/api/model/library"),
        ("POST", "/api/model/library"),
        ("PATCH", "/api/model/library/remote:library:abc"),
        ("DELETE", "/api/model/library/remote:library:abc"),
    ],
)
def test_every_library_route_requires_the_session_token(home, method, path):
    """No second auth scheme: the existing /api/ token gate owns all four verbs."""
    from hermes_cli import web_server

    unauth = TestClient(web_server.app)
    response = unauth.request(method, path, json={"provider": "p", "model": "m"})
    assert response.status_code == 401


# ── GET ───────────────────────────────────────────────────────


def test_get_empty_library_is_an_empty_list(client, home):
    assert not _library_file(home).exists()
    response = client.get("/api/model/library")
    assert response.status_code == 200
    assert response.json() == {"models": []}


def test_get_reads_models_json_from_hermes_home(client, home):
    _library_file(home).write_text(
        json.dumps([{"id": "remote:library:x", "name": "Saved", "provider": "openai",
                     "model": "gpt-5", "baseUrl": "", "createdAt": 7}]),
        encoding="utf-8",
    )
    rows = client.get("/api/model/library").json()["models"]
    assert [(r["id"], r["provider"], r["model"]) for r in rows] == [
        ("remote:library:x", "openai", "gpt-5")
    ]


def test_every_returned_row_carries_the_full_wire_shape(client, home):
    _library_file(home).write_text(
        json.dumps([{"provider": "openai", "model": "x-ai/grok"}]), encoding="utf-8"
    )
    (row,) = client.get("/api/model/library").json()["models"]
    # The picker reads these keys unconditionally — each must exist and be typed.
    assert set(row) == {"id", "name", "provider", "model", "baseUrl", "createdAt"}
    assert isinstance(row["id"], str) and row["id"]
    assert isinstance(row["baseUrl"], str)
    assert isinstance(row["createdAt"], (int, float))
    # A missing name falls back to the segment after the last "/".
    assert row["name"] == "grok"


def test_unusable_rows_are_dropped_rather_than_surfaced(client, home):
    _library_file(home).write_text(
        json.dumps([
            {"provider": "openai"},                      # no model
            {"model": "gpt-5"},                          # no provider
            "not-a-row",
            {"provider": "openai", "model": "gpt-5"},    # the only good one
        ]),
        encoding="utf-8",
    )
    rows = client.get("/api/model/library").json()["models"]
    assert [r["model"] for r in rows] == ["gpt-5"]


def test_duplicate_stored_rows_collapse(client, home):
    _library_file(home).write_text(
        json.dumps([
            {"provider": "OpenAI", "model": "GPT-5", "baseUrl": "https://api.x/v1/"},
            {"provider": "openai", "model": "gpt-5", "baseUrl": "https://api.x/v1"},
        ]),
        encoding="utf-8",
    )
    rows = client.get("/api/model/library").json()["models"]
    # Row identity is case- and trailing-slash-insensitive on (provider, model, baseUrl).
    assert len(rows) == 1


def test_corrupt_library_degrades_to_empty_not_500(client, home):
    _library_file(home).write_text("{not json at all", encoding="utf-8")
    response = client.get("/api/model/library")
    assert response.status_code == 200
    assert response.json() == {"models": []}


def test_configured_model_is_prepended(client, home):
    _write_model_config(home, provider="anthropic", model="claude-opus-5")
    client.post("/api/model/library", json={"provider": "openai", "model": "gpt-5"})

    rows = client.get("/api/model/library").json()["models"]
    assert rows[0]["provider"] == "anthropic"
    assert rows[0]["model"] == "claude-opus-5"
    assert rows[0]["id"] == "remote:active:anthropic:claude-opus-5"
    assert [r["model"] for r in rows[1:]] == ["gpt-5"]


def test_configured_model_appears_exactly_once_when_also_saved(client, home):
    _write_model_config(home, provider="anthropic", model="claude-opus-5")
    saved = client.post(
        "/api/model/library", json={"provider": "anthropic", "model": "claude-opus-5"}
    ).json()

    rows = client.get("/api/model/library").json()["models"]
    matches = [r for r in rows if (r["provider"], r["model"]) == ("anthropic", "claude-opus-5")]
    assert len(matches) == 1
    # The active row wins the slot, so the picker shows "what I run now" at the top.
    assert matches[0]["id"] == "remote:active:anthropic:claude-opus-5"
    assert saved["id"].startswith("remote:library:")


def test_no_configured_model_means_no_prepended_row(client, home):
    # Default config carries no model; GET must not invent a blank current row.
    client.post("/api/model/library", json={"provider": "openai", "model": "gpt-5"})
    rows = client.get("/api/model/library").json()["models"]
    assert [r["model"] for r in rows] == ["gpt-5"]
    assert not any(r["id"].startswith("remote:active:") for r in rows)


# ── POST ──────────────────────────────────────────────────────


def test_post_persists_to_models_json_and_returns_the_row(client, home):
    row = client.post(
        "/api/model/library",
        json={"provider": "openai", "model": "gpt-5", "baseUrl": "https://api.x/v1", "name": "Work"},
    ).json()
    assert row["id"].startswith("remote:library:")
    assert (row["provider"], row["model"], row["baseUrl"], row["name"]) == (
        "openai", "gpt-5", "https://api.x/v1", "Work",
    )
    assert row["createdAt"] > 0

    stored = json.loads(_library_file(home).read_text(encoding="utf-8"))
    assert [s["id"] for s in stored] == [row["id"]]


def test_post_accepts_snake_case_base_url(client):
    row = client.post(
        "/api/model/library",
        json={"provider": "openai", "model": "gpt-5", "base_url": "https://api.x/v1"},
    ).json()
    assert row["baseUrl"] == "https://api.x/v1"


def test_post_is_idempotent_on_provider_model_base_url(client, home):
    first = client.post("/api/model/library", json={"provider": "openai", "model": "gpt-5"}).json()
    second = client.post(
        "/api/model/library", json={"provider": "OpenAI", "model": "gpt-5", "name": "Other"}
    ).json()
    # A repeat save returns the existing row instead of growing the list.
    assert second["id"] == first["id"]
    assert len(json.loads(_library_file(home).read_text(encoding="utf-8"))) == 1


def test_post_treats_a_different_base_url_as_a_different_row(client, home):
    a = client.post(
        "/api/model/library", json={"provider": "openai", "model": "gpt-5", "baseUrl": "https://a/v1"}
    ).json()
    b = client.post(
        "/api/model/library", json={"provider": "openai", "model": "gpt-5", "baseUrl": "https://b/v1"}
    ).json()
    assert a["id"] != b["id"]
    assert len(json.loads(_library_file(home).read_text(encoding="utf-8"))) == 2


@pytest.mark.parametrize("body", [{}, {"provider": "openai"}, {"model": "gpt-5"},
                                  {"provider": "  ", "model": "gpt-5"}])
def test_post_requires_provider_and_model(client, home, body):
    assert client.post("/api/model/library", json=body).status_code == 400
    assert not _library_file(home).exists()


# ── PATCH ─────────────────────────────────────────────────────


def test_patch_updates_only_the_keys_present(client, home):
    row = client.post(
        "/api/model/library",
        json={"provider": "openai", "model": "gpt-5", "baseUrl": "https://a/v1", "name": "Work"},
    ).json()

    response = client.patch(f"/api/model/library/{row['id']}", json={"name": "Renamed"})
    assert response.status_code == 200
    updated = response.json()["model"]
    assert response.json()["ok"] is True
    assert updated["name"] == "Renamed"
    # Untouched keys survive the partial edit, id included.
    assert (updated["id"], updated["provider"], updated["model"], updated["baseUrl"]) == (
        row["id"], "openai", "gpt-5", "https://a/v1",
    )
    stored = json.loads(_library_file(home).read_text(encoding="utf-8"))
    assert stored[0]["name"] == "Renamed"


def test_patch_can_clear_the_base_url(client):
    row = client.post(
        "/api/model/library", json={"provider": "openai", "model": "gpt-5", "baseUrl": "https://a/v1"}
    ).json()
    updated = client.patch(f"/api/model/library/{row['id']}", json={"baseUrl": ""}).json()["model"]
    assert updated["baseUrl"] == ""


def test_patch_refuses_an_edit_that_would_strip_a_required_field(client, home):
    row = client.post("/api/model/library", json={"provider": "openai", "model": "gpt-5"}).json()
    assert client.patch(f"/api/model/library/{row['id']}", json={"provider": ""}).status_code == 400
    # Refused, not silently stored — the row is still readable afterwards.
    rows = client.get("/api/model/library").json()["models"]
    assert [r["id"] for r in rows] == [row["id"]]
    assert json.loads(_library_file(home).read_text(encoding="utf-8"))[0]["provider"] == "openai"


def test_patch_unknown_id_is_404(client):
    assert client.patch("/api/model/library/remote:library:nope", json={"name": "x"}).status_code == 404


# ── DELETE ────────────────────────────────────────────────────


def test_delete_removes_the_row(client, home):
    keep = client.post("/api/model/library", json={"provider": "openai", "model": "keep"}).json()
    drop = client.post("/api/model/library", json={"provider": "openai", "model": "drop"}).json()

    response = client.delete(f"/api/model/library/{drop['id']}")
    assert response.status_code == 200
    assert response.json() == {"ok": True}

    stored = json.loads(_library_file(home).read_text(encoding="utf-8"))
    assert [s["id"] for s in stored] == [keep["id"]]


def test_delete_unknown_id_is_404(client):
    assert client.delete("/api/model/library/remote:library:nope").status_code == 404


def test_delete_does_not_remove_the_configured_model_row(client, home):
    """The prepended active row is synthesised, not stored — deleting it is a 404
    and must not touch the saved shortcuts."""
    _write_model_config(home, provider="anthropic", model="claude-opus-5")
    saved = client.post("/api/model/library", json={"provider": "openai", "model": "gpt-5"}).json()

    assert client.delete("/api/model/library/remote:active:anthropic:claude-opus-5").status_code == 404
    stored = json.loads(_library_file(home).read_text(encoding="utf-8"))
    assert [s["id"] for s in stored] == [saved["id"]]


# ── storage location ──────────────────────────────────────────


def test_library_follows_hermes_home(client, home, tmp_path, monkeypatch):
    """The path is resolved per call, so switching profile/HERMES_HOME moves the
    library with it instead of leaking the first home's shortcuts."""
    client.post("/api/model/library", json={"provider": "openai", "model": "first-home"})

    other = tmp_path / "other-home"
    other.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(other))
    assert client.get("/api/model/library").json() == {"models": []}

    client.post("/api/model/library", json={"provider": "openai", "model": "second-home"})
    assert [r["model"] for r in client.get("/api/model/library").json()["models"]] == ["second-home"]

    monkeypatch.setenv("HERMES_HOME", str(home))
    assert [r["model"] for r in client.get("/api/model/library").json()["models"]] == ["first-home"]


def test_write_is_atomic_and_leaves_no_temp_file(client, home):
    client.post("/api/model/library", json={"provider": "openai", "model": "gpt-5"})
    assert _library_file(home).exists()
    assert not list(home.glob("models.json.tmp"))
