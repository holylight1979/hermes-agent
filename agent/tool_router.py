"""Hybrid pre-routing for Hermes' fixed progressive tool surface.

This module recommends capabilities for one user turn. It never executes a
 tool, grants permission, changes approvals, or narrows the bridge catalog.
Every candidate is selected from the session's already-authorized tool
schemas. Any classifier failure returns a fail-open recommendation that leaves
``tool_search`` able to search that complete authorized catalog.
"""

from __future__ import annotations

import json
import logging
import queue
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Bound classifier work without serializing healthy gateway turns.  Eight
# concurrent requests is deliberately above normal single-process chat load;
# excess turns fail open with a distinct observable error instead of creating
# unbounded daemon workers when an endpoint is stuck.
_AI_CLASSIFIER_SLOT = threading.BoundedSemaphore(8)


class ClassifierBusyError(TimeoutError):
    """The bounded process-wide classifier pool has no free slot."""

CAPABILITIES = frozenset({
    "none",
    "web",
    "files_read",
    "files_write",
    "terminal",
    "browser",
    "vision",
    "media",
    "desktop",
    "history",
    "memory",
    "skills",
    "delegation",
    "schedule",
    "messaging",
    "planning",
})

_PACKET_PREFIX = "[HERMES_TOOL_ROUTE]\n"
_PACKET_SUFFIX = "\n[/HERMES_TOOL_ROUTE]"
_RECOVERY = "Use tool_search when candidates are insufficient."


def _safe_bool(value: Any, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"true", "1", "yes", "on"}:
            return True
        if value in {"false", "0", "no", "off"}:
            return False
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    return fallback


def _safe_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _safe_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


@dataclass(frozen=True)
class ToolRouterConfig:
    enabled: bool = False
    mode: str = "hybrid"
    provider: str = "rdchat-direct"
    model: str = "gemma4:e4b-64k"
    timeout_seconds: float = 4.0
    confidence_threshold: float = 0.75
    max_candidates: int = 6
    max_packet_tokens: int = 1800
    include_compact_schemas: bool = True
    fail_open: bool = True
    telemetry: bool = True

    @classmethod
    def from_raw(cls, raw: Any) -> "ToolRouterConfig":
        if not isinstance(raw, dict):
            return cls()
        mode = str(raw.get("mode", "hybrid") or "hybrid").strip().lower()
        if mode not in {"hybrid", "rules", "ai"}:
            mode = "hybrid"
        provider = str(raw.get("provider") or "rdchat-direct").strip()
        model = str(raw.get("model") or "gemma4:e4b-64k").strip()
        return cls(
            enabled=_safe_bool(raw.get("enabled"), False),
            mode=mode,
            provider=provider or "rdchat-direct",
            model=model or "gemma4:e4b-64k",
            timeout_seconds=max(0.25, min(30.0, _safe_float(raw.get("timeout_seconds"), 4.0))),
            confidence_threshold=max(
                0.0, min(1.0, _safe_float(raw.get("confidence_threshold"), 0.75))
            ),
            max_candidates=max(1, min(20, _safe_int(raw.get("max_candidates"), 6))),
            max_packet_tokens=max(
                256, min(8192, _safe_int(raw.get("max_packet_tokens"), 1800))
            ),
            include_compact_schemas=_safe_bool(
                raw.get("include_compact_schemas"), True
            ),
            fail_open=_safe_bool(raw.get("fail_open"), True),
            telemetry=_safe_bool(raw.get("telemetry"), True),
        )


def load_config() -> ToolRouterConfig:
    try:
        from hermes_cli.config import load_config as _load

        config = _load() or {}
        tools = config.get("tools") if isinstance(config.get("tools"), dict) else {}
        return ToolRouterConfig.from_raw(tools.get("tool_router"))
    except Exception as exc:
        logger.debug("tool router config unavailable: %s", type(exc).__name__)
        return ToolRouterConfig()


@dataclass(frozen=True)
class RouteDecision:
    source: str
    capabilities: Tuple[str, ...]
    candidates: Tuple[Dict[str, Any], ...] = ()
    confidence: float = 1.0
    latency_ms: float = 0.0
    search_scope: str = "all_authorized"
    error_type: Optional[str] = None

    @property
    def candidate_names(self) -> Tuple[str, ...]:
        return tuple(
            str((tool.get("function") or {}).get("name") or "")
            for tool in self.candidates
            if (tool.get("function") or {}).get("name")
        )


_SOCIAL_RE = re.compile(
    r"^(?:hi|hello|hey|ok|okay|thanks|thank you|你好|嗨|哈囉|謝謝|多謝|收到|了解|好的|好)$",
    re.IGNORECASE,
)
_CONFLICT_MARKERS = (
    "如果",
    "假如",
    "若是",
    "不要",
    "不需要",
    "除非",
    "可能",
    "或許",
    "但",
    " however ",
    " if ",
    " unless ",
    " don't ",
    " do not ",
)


def _contains_any(text: str, terms: Iterable[str]) -> bool:
    return any(term in text for term in terms)


def classify_by_rule(message: str) -> Optional[Tuple[str, ...]]:
    """Return high-confidence capabilities, or None when AI should classify."""
    text = str(message or "").strip()
    if not text:
        return ("none",)
    lowered = text.lower()
    social = re.sub(r"[\s.!！?？,，。]+", "", lowered)
    if _SOCIAL_RE.fullmatch(social):
        return ("none",)
    padded = f" {lowered} "
    if _contains_any(padded, _CONFLICT_MARKERS):
        return None

    found: list[str] = []

    def add(capability: str) -> None:
        if capability not in found:
            found.append(capability)

    if re.search(r"https?://|www\.", lowered) or _contains_any(
        lowered,
        ("最新", "即時", "天氣", "新聞", "目前版本", "current version", "latest", "weather", "news"),
    ):
        add("web")

    has_path = bool(
        re.search(r"(?:[a-zA-Z]:[\\/]|(?:^|\s)/(?:[^\s]+/)*[^\s]+|[\w.-]+\.(?:py|js|ts|json|ya?ml|toml|md|txt|csv|rs|go|java|cpp|h))", text)
    )
    read_terms = ("讀取", "查看檔", "開啟檔", "檔案內容", "read file", "inspect file")
    write_terms = ("修改", "編輯", "寫入", "建立檔", "刪除檔", "patch", "edit", "write file")
    terminal_terms = (
        "pytest", "測試", "建置", "編譯", "命令", "終端機", "shell", "terminal",
        "npm ", "uv ", "git ", "cargo ", "跑 ", "執行 ",
    )
    if has_path or _contains_any(lowered, read_terms):
        add("files_read")
    if _contains_any(lowered, write_terms):
        add("files_write")
        add("files_read")
    if _contains_any(lowered, terminal_terms):
        add("terminal")
        if _contains_any(lowered, ("pytest", "測試", "建置", "編譯")):
            add("files_read")

    if _contains_any(
        lowered,
        ("登入頁", "按鈕", "表單", "瀏覽器", "網頁互動", "browser", "click button", "fill form"),
    ):
        add("browser")
    if _contains_any(lowered, ("圖片", "照片", "截圖", "image", "screenshot")):
        add("vision")
    if _contains_any(lowered, ("影片", "音訊", "語音", "video", "audio", "podcast")):
        add("media")
    if _contains_any(
        lowered,
        ("排程", "提醒", "定期", "每週", "每天", "每月", "cron", "schedule", "remind"),
    ):
        add("schedule")
    if _contains_any(
        lowered,
        ("上次", "之前", "過去對話", "discord 對話", "歷史工作階段", "previous session", "conversation history"),
    ):
        add("history")
    if _contains_any(
        lowered,
        ("桌面", "視窗", "應用程式", "滑鼠", "鍵盤", "desktop", "window", "application"),
    ) and _contains_any(lowered, ("點擊", "操作", "切換", "打開", "click", "control", "focus")):
        add("desktop")

    return tuple(found) if found else None


_CAPABILITY_MATCHERS: Dict[str, Tuple[re.Pattern[str], ...]] = {
    "web": (re.compile(r"^web_"),),
    "files_read": (re.compile(r"^(?:read_file|search_files)$"),),
    "files_write": (re.compile(r"^(?:write_file|patch)$"),),
    "terminal": (re.compile(r"^(?:terminal|process|execute_code)$"),),
    "browser": (re.compile(r"^browser_"),),
    "vision": (re.compile(r"^(?:vision_analyze|browser_vision)$"),),
    "media": (re.compile(r"^(?:video_|text_to_speech|audio_)"),),
    "desktop": (re.compile(r"(?:computer_use|desktop|focus_pane|read_terminal|open_preview)"),),
    "history": (re.compile(r"^session_search$"),),
    "memory": (re.compile(r"^(?:memory|atomic_memory_)"),),
    "skills": (re.compile(r"^(?:skill_|skills_)"),),
    "delegation": (re.compile(r"^(?:delegate_task|kanban_)"),),
    "schedule": (re.compile(r"^cronjob$"),),
    "messaging": (re.compile(r"^(?:send_message|discord|telegram|slack|feishu|yuanbao)"),),
    "planning": (re.compile(r"^(?:todo|clarify)$"),),
}


def candidates_for_capabilities(
    capabilities: Sequence[str],
    tool_defs: Sequence[Dict[str, Any]],
    *,
    limit: int,
) -> Tuple[Dict[str, Any], ...]:
    """Map capabilities to tools without ever leaving the supplied scope."""
    if "none" in capabilities:
        return ()
    selected: list[Dict[str, Any]] = []
    matchers = [
        matcher
        for capability in capabilities
        for matcher in _CAPABILITY_MATCHERS.get(capability, ())
    ]
    for tool in tool_defs:
        name = str((tool.get("function") or {}).get("name") or "")
        if name and any(matcher.search(name) for matcher in matchers):
            selected.append(tool)
            if len(selected) >= limit:
                break
    return tuple(selected)


def _parse_ai_payload(payload: Any, config: ToolRouterConfig) -> Tuple[Tuple[str, ...], float]:
    if isinstance(payload, str):
        text = payload.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("classifier response must be an object")
    capabilities = payload.get("capabilities")
    confidence = payload.get("confidence")
    if not isinstance(capabilities, list) or not capabilities:
        raise ValueError("capabilities must be a non-empty list")
    if not all(isinstance(item, str) and item in CAPABILITIES for item in capabilities):
        raise ValueError("classifier returned an unknown capability")
    confidence_value = float(confidence)
    if not 0.0 <= confidence_value <= 1.0:
        raise ValueError("confidence must be between zero and one")
    if confidence_value < config.confidence_threshold:
        raise ValueError("classifier confidence below threshold")
    return tuple(dict.fromkeys(capabilities)), confidence_value


def _call_ai_classifier(message: str, config: ToolRouterConfig) -> Any:
    """Call the configured low-cost classifier; no tools or permissions supplied."""
    from agent.auxiliary_client import call_llm, extract_content_or_reasoning

    capability_list = ", ".join(sorted(CAPABILITIES))
    response = call_llm(
        task="tool_router",
        provider=config.provider,
        model=config.model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Classify the user request into capability enums only. "
                    "Return strict JSON with keys capabilities (array) and confidence (0..1). "
                    f"Allowed capabilities: {capability_list}. Never return tool names, permissions, or approvals."
                ),
            },
            {"role": "user", "content": message},
        ],
        temperature=0,
        max_tokens=160,
        timeout=config.timeout_seconds,
    )
    return extract_content_or_reasoning(response)


def _fallback(started: float, exc: Optional[BaseException] = None) -> RouteDecision:
    return RouteDecision(
        source="fallback",
        capabilities=(),
        candidates=(),
        confidence=0.0,
        latency_ms=(time.perf_counter() - started) * 1000.0,
        search_scope="all_authorized",
        error_type=type(exc).__name__ if exc is not None else None,
    )


def _classify_with_hard_deadline(
    classifier: Callable[[str, ToolRouterConfig], Any],
    message: str,
    config: ToolRouterConfig,
) -> Any:
    """Run a classifier behind a strict wall-clock deadline.

    HTTP client timeouts typically measure socket inactivity, not total model
    generation time. A daemon worker lets the main turn fail open at the
    configured wall deadline even if an OpenAI-compatible endpoint keeps the
    connection active indefinitely. The bounded shared pool limits orphaned
    work while still allowing normal concurrent gateway turns.
    """
    if not _AI_CLASSIFIER_SLOT.acquire(blocking=False):
        raise ClassifierBusyError("tool-router classifier pool is busy")
    result_queue: "queue.Queue[Tuple[bool, Any]]" = queue.Queue(maxsize=1)

    def worker() -> None:
        try:
            result_queue.put((True, classifier(message, config)), block=False)
        except BaseException as exc:  # propagate in the caller thread
            try:
                result_queue.put((False, exc), block=False)
            except queue.Full:
                pass
        finally:
            _AI_CLASSIFIER_SLOT.release()

    try:
        threading.Thread(
            target=worker,
            name="hermes-tool-router-classifier",
            daemon=True,
        ).start()
    except BaseException:
        _AI_CLASSIFIER_SLOT.release()
        raise
    try:
        ok, value = result_queue.get(timeout=config.timeout_seconds)
    except queue.Empty as exc:
        raise TimeoutError(
            f"tool-router classifier exceeded {config.timeout_seconds:.2f}s wall deadline"
        ) from exc
    if ok:
        return value
    raise value


def route_turn(
    message: str,
    tool_defs: Sequence[Dict[str, Any]],
    *,
    config: Optional[ToolRouterConfig] = None,
    ai_classifier: Optional[Callable[[str, ToolRouterConfig], Any]] = None,
) -> RouteDecision:
    """Build one recommendation for a user turn, always preserving recovery."""
    config = config or load_config()
    started = time.perf_counter()
    if not config.enabled:
        return RouteDecision(
            source="disabled",
            capabilities=(),
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    try:
        capabilities: Optional[Tuple[str, ...]] = None
        source = "ai"
        confidence = 1.0
        if config.mode != "ai":
            capabilities = classify_by_rule(message)
            if capabilities is not None:
                source = "rule"
        if capabilities is None:
            if config.mode == "rules":
                decision = _fallback(started)
                _log_telemetry(decision, config, tool_defs)
                return decision
            classifier = ai_classifier or _call_ai_classifier
            payload = _classify_with_hard_deadline(
                classifier, str(message or ""), config
            )
            capabilities, confidence = _parse_ai_payload(payload, config)
            source = "ai"
        candidates = candidates_for_capabilities(
            capabilities, tool_defs, limit=config.max_candidates
        )
        decision = RouteDecision(
            source=source,
            capabilities=capabilities,
            candidates=candidates,
            confidence=confidence,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            search_scope="all_authorized",
        )
    except Exception as exc:
        decision = _fallback(started, exc)
    _log_telemetry(decision, config, tool_defs)
    return decision


def _compact_parameters(schema: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(schema, dict):
        return None
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        properties = {}
    compact_props = {}
    for name, value in properties.items():
        if not isinstance(value, dict):
            continue
        compact: Dict[str, Any] = {}
        if "type" in value:
            compact["type"] = value["type"]
        if "enum" in value and isinstance(value["enum"], list):
            compact["enum"] = value["enum"][:12]
        compact_props[name] = compact
    result: Dict[str, Any] = {"type": "object", "properties": compact_props}
    required = schema.get("required")
    if isinstance(required, list):
        result["required"] = [item for item in required if isinstance(item, str)]
    return result


def _packet_payload(
    decision: RouteDecision,
    config: ToolRouterConfig,
    *,
    descriptions: bool,
    schemas: bool,
    candidate_limit: int,
) -> Dict[str, Any]:
    candidates = []
    for tool in decision.candidates[:candidate_limit]:
        function = tool.get("function") or {}
        item: Dict[str, Any] = {"name": str(function.get("name") or "")}
        if descriptions:
            item["description"] = str(function.get("description") or "")[:240]
        if schemas and config.include_compact_schemas:
            params = _compact_parameters(function.get("parameters"))
            if params is not None:
                item["parameters"] = params
        candidates.append(item)
    return {
        "source": decision.source,
        "capabilities": list(decision.capabilities),
        "confidence": round(decision.confidence, 3),
        "candidates": candidates,
        "search_scope": decision.search_scope,
        "recovery": _RECOVERY,
    }


def _render_packet(payload: Dict[str, Any]) -> str:
    return _PACKET_PREFIX + json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    ) + _PACKET_SUFFIX


def build_route_packet(decision: RouteDecision, config: ToolRouterConfig) -> str:
    """Render a bounded packet, removing optional detail before candidates."""
    byte_limit = config.max_packet_tokens * 4
    limit = min(config.max_candidates, len(decision.candidates))
    for descriptions, schemas in ((True, True), (True, False), (False, False)):
        packet = _render_packet(
            _packet_payload(
                decision,
                config,
                descriptions=descriptions,
                schemas=schemas,
                candidate_limit=limit,
            )
        )
        if len(packet.encode("utf-8")) <= byte_limit:
            return packet
    while limit > 0:
        limit -= 1
        packet = _render_packet(
            _packet_payload(
                decision,
                config,
                descriptions=False,
                schemas=False,
                candidate_limit=limit,
            )
        )
        if len(packet.encode("utf-8")) <= byte_limit:
            return packet
    # The bounded config floor (256 tokens / 1024 bytes) is large enough for
    # this valid minimal JSON object, so no byte slicing can corrupt it.
    minimal = {
        "source": decision.source,
        "capabilities": [],
        "confidence": round(decision.confidence, 3),
        "candidates": [],
        "search_scope": "all_authorized",
        "recovery": _RECOVERY,
    }
    return _render_packet(minimal)


def _log_telemetry(
    decision: RouteDecision,
    config: ToolRouterConfig,
    tool_defs: Sequence[Dict[str, Any]],
) -> None:
    if not config.telemetry:
        return
    try:
        from tools.tool_search import estimate_tokens_from_schemas

        full_tokens = estimate_tokens_from_schemas(tool_defs)
        candidate_tokens = estimate_tokens_from_schemas(decision.candidates)
    except Exception:
        full_tokens = 0
        candidate_tokens = 0
    logger.info(
        "tool_router source=%s capabilities=%s latency_ms=%.3f candidates=%d "
        "candidate_tokens=%d full_schema_tokens=%d search_scope=%s error_type=%s",
        decision.source,
        ",".join(decision.capabilities),
        decision.latency_ms,
        len(decision.candidates),
        candidate_tokens,
        full_tokens,
        decision.search_scope,
        decision.error_type or "none",
    )


__all__ = [
    "CAPABILITIES",
    "ToolRouterConfig",
    "RouteDecision",
    "load_config",
    "classify_by_rule",
    "candidates_for_capabilities",
    "route_turn",
    "build_route_packet",
    "ClassifierBusyError",
]
