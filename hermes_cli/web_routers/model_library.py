"""Model *library* dashboard routes — the configured-model shortcut list at ``/api/model/library``.

Upstream exposes ``/api/model/options`` (what you *could* pick) and ``/api/model/set``
(what is assigned). This surface is the small, user-curated shortcut list a remote/SSH
model picker reads: rows the user saved by hand, stored as ``models.json`` inside this
agent's ``HERMES_HOME`` so a remote's shortcuts stay on the remote host and survive
desktop restarts. It never touches model assignment — ``/api/model/set`` owns that.

Auth is the dashboard's single session-token scheme: these paths start with ``/api/``
and are absent from ``_PUBLIC_API_PATHS``, so ``web_server.auth_middleware`` gates them
like every other route. No second scheme, no per-route auth.
"""

from __future__ import annotations

import json
import logging
import secrets
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, HTTPException

from hermes_cli.web_deps import late
from hermes_constants import get_hermes_home

_log = logging.getLogger("hermes_cli.web_server")
router = APIRouter()

# Late-bound so a test's monkeypatch on the owning module wins at call time.
load_config = late("load_config", "hermes_cli.config")


def _library_path():
    """``models.json`` under the ACTIVE profile's home — resolved per call, never cached,
    so a profile switch moves the library with it."""
    return get_hermes_home() / "models.json"


def _short_label(model: Any) -> str:
    """Display name for a model id: the part after the last ``/`` (``x-ai/grok`` → ``grok``)."""
    text = str(model or "").strip()
    return (text.rsplit("/", 1)[-1] if text else "") or text


def _row_key(row: Dict[str, Any]) -> tuple:
    """Identity of a library row for dedup: (provider, model, baseUrl), case/slash-insensitive."""
    return (
        str(row.get("provider", "")).strip().lower(),
        str(row.get("model", "")).strip().lower(),
        str(row.get("baseUrl", row.get("base_url", ""))).strip().rstrip("/").lower(),
    )


def _normalize_row(row: Any, index: int = 0) -> Optional[Dict[str, Any]]:
    """Coerce a stored/edited row to the wire shape, or ``None`` when unusable.

    ``provider`` and ``model`` are the only required fields; a row missing either is
    dropped on read rather than surfaced as a broken shortcut.
    """
    if not isinstance(row, dict):
        return None
    provider = str(row.get("provider", "")).strip()
    model = str(row.get("model", "")).strip()
    if not provider or not model:
        return None
    return {
        "id": str(row.get("id") or f"remote:library:{provider}:{index}:{model}"),
        "name": str(row.get("name") or _short_label(model) or provider),
        "provider": provider,
        "model": model,
        "baseUrl": str(row.get("baseUrl", row.get("base_url", "")) or "").strip(),
        "createdAt": row.get("createdAt") if isinstance(row.get("createdAt"), (int, float)) else 0,
    }


def _read_library() -> List[Dict[str, Any]]:
    """Stored rows, normalized and deduped. A missing/corrupt file reads as empty —
    the picker degrades to "no shortcuts", never to a 500."""
    path = _library_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    except Exception:
        _log.warning("model library at %s is unreadable; treating as empty", path, exc_info=True)
        raw = []
    rows: List[Dict[str, Any]] = []
    seen = set()
    for index, item in enumerate(raw if isinstance(raw, list) else []):
        row = _normalize_row(item, index)
        if row is None:
            continue
        key = _row_key(row)
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    return rows


def _write_library(rows: List[Dict[str, Any]]) -> None:
    path = _library_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    tmp.replace(path)  # atomic: a crash mid-write leaves the old library intact


def _current_model_row() -> Optional[Dict[str, Any]]:
    """The assigned model as a library-shaped row, so the picker can show "what I run
    now" at the top without the user having to save it first."""
    try:
        cfg = load_config()
    except Exception:
        _log.exception("model library: reading the configured model failed")
        return None
    model_cfg = cfg.get("model", {})
    if isinstance(model_cfg, dict):
        provider = str(model_cfg.get("provider", "") or "").strip()
        model = str(model_cfg.get("default", model_cfg.get("name", "")) or "").strip()
        base_url = str(model_cfg.get("base_url", "") or "").strip()
    else:
        provider, model, base_url = "", str(model_cfg or "").strip(), ""
    if not provider or not model:
        return None
    return {
        "id": f"remote:active:{provider}:{model}",
        "name": _short_label(model) or provider,
        "provider": provider,
        "model": model,
        "baseUrl": base_url,
        "createdAt": 0,
    }


@router.get("/api/model/library")
def get_model_library():
    """Saved shortcuts, with the configured model prepended (and de-duplicated against
    the saved rows, so it appears exactly once)."""
    rows = _read_library()
    current = _current_model_row()
    if current:
        current_key = _row_key(current)
        rows = [current] + [row for row in rows if _row_key(row) != current_key]
    return {"models": rows}


@router.post("/api/model/library")
def add_model_library_row(body: Dict[str, Any] = Body(...)):
    """Save a shortcut. Idempotent on (provider, model, baseUrl): a repeat POST returns
    the existing row rather than growing the list."""
    provider = str(body.get("provider", "") or "").strip()
    model = str(body.get("model", "") or "").strip()
    if not provider or not model:
        raise HTTPException(status_code=400, detail="provider and model required")
    base_url = str(body.get("baseUrl", body.get("base_url", "")) or "").strip()
    name = str(body.get("name", "") or "").strip() or _short_label(model) or provider
    rows = _read_library()
    key = (provider.lower(), model.lower(), base_url.rstrip("/").lower())
    for row in rows:
        if _row_key(row) == key:
            return row
    row = {
        "id": f"remote:library:{secrets.token_hex(8)}",
        "name": name,
        "provider": provider,
        "model": model,
        "baseUrl": base_url,
        "createdAt": int(time.time() * 1000),
    }
    rows.append(row)
    _write_library(rows)
    return row


@router.patch("/api/model/library/{model_id:path}")
def update_model_library_row(model_id: str, body: Dict[str, Any] = Body(...)):
    """Partial edit of one shortcut. Only the keys present in the body change; an edit
    that would strip ``provider``/``model`` is refused instead of silently dropping the
    row on the next read."""
    rows = _read_library()
    for index, row in enumerate(rows):
        if row.get("id") != model_id:
            continue
        next_row = dict(row)
        for key in ("name", "provider", "model"):
            if key in body:
                next_row[key] = str(body.get(key, "") or "").strip()
        if "baseUrl" in body or "base_url" in body:
            next_row["baseUrl"] = str(body.get("baseUrl", body.get("base_url", "")) or "").strip()
        normalized = _normalize_row(next_row, index)
        if normalized is None:
            raise HTTPException(status_code=400, detail="provider and model required")
        rows[index] = normalized
        _write_library(rows)
        return {"ok": True, "model": normalized}
    raise HTTPException(status_code=404, detail="model not found")


@router.delete("/api/model/library/{model_id:path}")
def delete_model_library_row(model_id: str):
    rows = _read_library()
    filtered = [row for row in rows if row.get("id") != model_id]
    if len(filtered) == len(rows):
        raise HTTPException(status_code=404, detail="model not found")
    _write_library(filtered)
    return {"ok": True}
