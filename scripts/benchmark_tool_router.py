#!/usr/bin/env python3
"""Local, non-destructive benchmark for the hybrid tool router."""

from __future__ import annotations

import argparse
import json
import statistics
import time

from agent.tool_router import ToolRouterConfig, build_route_packet, route_turn
from model_tools import get_tool_definitions
from tools.tool_search import (
    ToolSearchConfig,
    assemble_tool_defs,
    build_catalog,
    classify_tools,
    estimate_tokens_from_schemas,
)


def median_ms(fn, repetitions: int) -> float:
    samples = []
    for _ in range(repetitions):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--live-ai", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    raw = get_tool_definitions(quiet_mode=True, skip_tool_search_assembly=True)
    discovery_ms = (time.perf_counter() - started) * 1000.0
    disclosure = ToolSearchConfig.from_raw(
        {
            "enabled": "on",
            "defer_core": True,
            "always_visible": ["clarify", "skill_view"],
        }
    )
    fixed_result = assemble_tool_defs(raw, config=disclosure)
    fixed = fixed_result.tool_defs
    _, deferred = classify_tools(raw, config=disclosure)
    catalog = build_catalog(deferred)

    router_cfg = ToolRouterConfig.from_raw({"enabled": True, "telemetry": False})
    rule_decision = route_turn(
        "讀取 C:/repo/app.py，修改後跑 pytest",
        raw,
        config=router_cfg,
    )
    packet = build_route_packet(rule_decision, router_cfg)

    def timeout_classifier(*_):
        raise TimeoutError("benchmark timeout")

    def provider_failure(*_):
        raise ConnectionError("benchmark provider unavailable")

    ambiguous = "幫我處理一下那個問題"
    timeout = route_turn(
        ambiguous, raw, config=router_cfg, ai_classifier=timeout_classifier
    )
    provider_down = route_turn(
        ambiguous, raw, config=router_cfg, ai_classifier=provider_failure
    )
    low_conf = route_turn(
        ambiguous,
        raw,
        config=router_cfg,
        ai_classifier=lambda *_: {"capabilities": ["web"], "confidence": 0.1},
    )

    output = {
        "before": {
            "tool_count": len(raw),
            "schema_chars": len(json.dumps(raw, ensure_ascii=False)),
            "estimated_tokens": estimate_tokens_from_schemas(raw),
            "cold_discovery_ms": round(discovery_ms, 3),
        },
        "after_fixed_surface": {
            "tool_count": len(fixed),
            "tool_names": [t["function"]["name"] for t in fixed],
            "schema_chars": len(json.dumps(fixed, ensure_ascii=False)),
            "estimated_tokens": estimate_tokens_from_schemas(fixed),
            "deferred_catalog_count": len(catalog),
            "assembly_median_ms": round(
                median_ms(lambda: assemble_tool_defs(raw, config=disclosure), args.repetitions),
                3,
            ),
        },
        "rule_route": {
            "capabilities": list(rule_decision.capabilities),
            "candidate_names": list(rule_decision.candidate_names),
            "latency_ms": round(rule_decision.latency_ms, 3),
            "packet_bytes": len(packet.encode("utf-8")),
            "packet_estimated_tokens": (len(packet.encode("utf-8")) + 3) // 4,
        },
        "fail_open": {
            "timeout": {
                "source": timeout.source,
                "scope": timeout.search_scope,
                "error_type": timeout.error_type,
            },
            "provider_down": {
                "source": provider_down.source,
                "scope": provider_down.search_scope,
                "error_type": provider_down.error_type,
            },
            "low_confidence": {
                "source": low_conf.source,
                "scope": low_conf.search_scope,
                "error_type": low_conf.error_type,
            },
        },
    }

    if args.live_ai:
        live_start = time.perf_counter()
        live = route_turn(ambiguous, raw, config=router_cfg)
        output["live_ai"] = {
            "source": live.source,
            "capabilities": list(live.capabilities),
            "candidate_names": list(live.candidate_names),
            "confidence": live.confidence,
            "latency_ms": round(live.latency_ms, 3),
            "wall_ms": round((time.perf_counter() - live_start) * 1000.0, 3),
            "scope": live.search_scope,
            "error_type": live.error_type,
        }

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
