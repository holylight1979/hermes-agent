from __future__ import annotations

import json
import time

import pytest


def _td(name: str, description: str = "", properties=None):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
            },
        },
    }


class TestToolRouterConfig:
    def test_defaults_are_disabled_and_fail_open(self):
        from agent.tool_router import ToolRouterConfig

        cfg = ToolRouterConfig.from_raw(None)
        assert cfg.enabled is False
        assert cfg.fail_open is True
        assert cfg.provider == "rdchat-direct"
        assert cfg.model == "gemma4:e4b-64k"

    def test_invalid_values_use_safe_bounds(self):
        from agent.tool_router import ToolRouterConfig

        cfg = ToolRouterConfig.from_raw({
            "enabled": "yes",
            "timeout_seconds": -9,
            "confidence_threshold": 7,
            "max_candidates": 999,
            "max_packet_tokens": 10,
        })
        assert cfg.enabled is True
        assert cfg.timeout_seconds == 0.25
        assert cfg.confidence_threshold == 1.0
        assert cfg.max_candidates == 20
        assert cfg.max_packet_tokens == 256


class TestFastRules:
    @pytest.mark.parametrize("text", ["你好", "謝謝", "OK", "收到"])
    def test_social_messages_need_no_tools(self, text):
        from agent.tool_router import classify_by_rule

        result = classify_by_rule(text)
        assert result == ("none",)

    def test_web_rule(self):
        from agent.tool_router import classify_by_rule

        result = classify_by_rule("請查 https://example.com 的最新版本")
        assert "web" in result

    def test_code_and_test_rule_is_multi_capability(self):
        from agent.tool_router import classify_by_rule

        result = classify_by_rule("讀取 C:/repo/app.py，修改後跑 pytest")
        assert {"files_read", "files_write", "terminal"}.issubset(set(result))

    @pytest.mark.parametrize(
        ("text", "capability"),
        [
            ("打開登入頁並按下送出按鈕", "browser"),
            ("分析這張圖片", "vision"),
            ("每週一提醒我", "schedule"),
            ("找出上次 Discord 對話", "history"),
            ("點擊桌面上的 Chrome 視窗", "desktop"),
        ],
    )
    def test_explicit_capabilities(self, text, capability):
        from agent.tool_router import classify_by_rule

        assert capability in classify_by_rule(text)

    def test_negation_or_condition_defers_to_ai(self):
        from agent.tool_router import classify_by_rule

        assert classify_by_rule("如果需要，不要先查網路，看看檔案即可") is None

    def test_ambiguous_request_defers_to_ai(self):
        from agent.tool_router import classify_by_rule

        assert classify_by_rule("幫我處理一下那個問題") is None


class TestAiClassificationAndFallback:
    def _defs(self):
        return [
            _td("read_file", "Read a file", {"path": {"type": "string"}}),
            _td("write_file", "Write a file"),
            _td("terminal", "Run a command"),
            _td("session_search", "Search history"),
        ]

    def test_ai_success_accepts_only_enum_and_scoped_tools(self):
        from agent.tool_router import ToolRouterConfig, route_turn

        cfg = ToolRouterConfig.from_raw({"enabled": True})
        decision = route_turn(
            "幫我處理一下那個問題",
            self._defs(),
            config=cfg,
            ai_classifier=lambda _message, _cfg: {
                "capabilities": ["files_read", "terminal"],
                "confidence": 0.91,
            },
        )
        assert decision.source == "ai"
        assert set(decision.candidate_names) == {"read_file", "terminal"}
        assert "write_file" not in decision.candidate_names

    @pytest.mark.parametrize(
        "payload",
        [
            {"capabilities": ["web"], "confidence": 0.2},
            {"capabilities": ["root_access"], "confidence": 0.99},
            {"confidence": 0.99},
            "not-json",
        ],
    )
    def test_bad_ai_outputs_fail_open(self, payload):
        from agent.tool_router import ToolRouterConfig, route_turn

        cfg = ToolRouterConfig.from_raw({"enabled": True})
        decision = route_turn(
            "幫我處理一下那個問題",
            self._defs(),
            config=cfg,
            ai_classifier=lambda _message, _cfg: payload,
        )
        assert decision.source == "fallback"
        assert decision.search_scope == "all_authorized"
        assert decision.candidate_names == ()

    def test_timeout_fails_open_quickly(self):
        from agent.tool_router import ToolRouterConfig, route_turn

        def timeout(_message, _cfg):
            raise TimeoutError("router timeout")

        cfg = ToolRouterConfig.from_raw({"enabled": True, "timeout_seconds": 0.25})
        started = time.perf_counter()
        decision = route_turn(
            "幫我處理一下那個問題", self._defs(), config=cfg, ai_classifier=timeout
        )
        assert time.perf_counter() - started < 0.5
        assert decision.source == "fallback"
        assert decision.error_type == "TimeoutError"

    def test_rule_misclassification_can_recover_through_full_catalog(self):
        from agent.tool_router import ToolRouterConfig, route_turn

        cfg = ToolRouterConfig.from_raw({"enabled": True})
        decision = route_turn("你好", self._defs(), config=cfg)
        assert decision.source == "rule"
        assert decision.capabilities == ("none",)
        assert decision.error_type is None
        # The fixed bridge remains available and its catalog is not narrowed by
        # this recommendation, so a main-model tool_search can still recover.
        assert decision.search_scope == "all_authorized"

    def test_classifier_pool_busy_is_observable_and_fails_open(self, monkeypatch):
        import agent.tool_router as router

        class BusySlot:
            @staticmethod
            def acquire(blocking=False):
                return False

        monkeypatch.setattr(router, "_AI_CLASSIFIER_SLOT", BusySlot())
        cfg = router.ToolRouterConfig.from_raw({"enabled": True})
        decision = router.route_turn(
            "幫我處理一下那個問題",
            self._defs(),
            config=cfg,
            ai_classifier=lambda *_: {
                "capabilities": ["web"], "confidence": 0.99
            },
        )
        assert decision.source == "fallback"
        assert decision.error_type == "ClassifierBusyError"
        assert decision.search_scope == "all_authorized"

    def test_healthy_classifiers_can_run_concurrently(self):
        import threading
        from concurrent.futures import ThreadPoolExecutor
        from agent.tool_router import ToolRouterConfig, route_turn

        barrier = threading.Barrier(2)

        def concurrent_classifier(_message, _cfg):
            barrier.wait(timeout=1.0)
            return {"capabilities": ["web"], "confidence": 0.99}

        cfg = ToolRouterConfig.from_raw({
            "enabled": True, "timeout_seconds": 1.0
        })
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(
                    route_turn,
                    "幫我處理一下那個問題",
                    self._defs(),
                    config=cfg,
                    ai_classifier=concurrent_classifier,
                )
                for _ in range(2)
            ]
        assert [future.result().source for future in futures] == ["ai", "ai"]

    def test_thread_start_failure_releases_classifier_capacity(self, monkeypatch):
        import agent.tool_router as router

        real_thread = router.threading.Thread

        class FailingThread:
            def __init__(self, *args, **kwargs):
                pass

            def start(self):
                raise RuntimeError("thread start failed")

        monkeypatch.setattr(router.threading, "Thread", FailingThread)
        cfg = router.ToolRouterConfig.from_raw({"enabled": True})
        failed = router.route_turn(
            "幫我處理一下那個問題",
            self._defs(),
            config=cfg,
            ai_classifier=lambda *_: {
                "capabilities": ["web"], "confidence": 0.99
            },
        )
        assert failed.source == "fallback"
        assert failed.error_type == "RuntimeError"

        # Restore the constructor within this test and prove the acquired slot
        # was released rather than leaving a permanent fail-open degradation.
        monkeypatch.setattr(router.threading, "Thread", real_thread)
        recovered = router.route_turn(
            "幫我處理一下那個問題",
            self._defs(),
            config=cfg,
            ai_classifier=lambda *_: {
                "capabilities": ["web"], "confidence": 0.99
            },
        )
        assert recovered.source == "ai"

    def test_hard_wall_deadline_fails_open_while_classifier_is_still_running(self):
        from agent.tool_router import ToolRouterConfig, route_turn

        def slow(_message, _cfg):
            time.sleep(1.0)
            return {"capabilities": ["web"], "confidence": 0.99}

        cfg = ToolRouterConfig.from_raw({
            "enabled": True,
            "timeout_seconds": 0.25,
        })
        started = time.perf_counter()
        decision = route_turn(
            "幫我處理一下那個問題", self._defs(), config=cfg, ai_classifier=slow
        )
        elapsed = time.perf_counter() - started
        assert elapsed < 0.6
        assert decision.source == "fallback"
        assert decision.error_type == "TimeoutError"
        assert decision.search_scope == "all_authorized"


class TestCapabilityMappingAndPacket:
    def test_candidates_never_exceed_session_definitions(self):
        from agent.tool_router import candidates_for_capabilities

        defs = [_td("read_file"), _td("terminal")]
        candidates = candidates_for_capabilities(("web", "files_read"), defs, limit=6)
        assert [c["function"]["name"] for c in candidates] == ["read_file"]

    def test_packet_is_bounded_and_json_remains_valid(self):
        from agent.tool_router import RouteDecision, ToolRouterConfig, build_route_packet

        defs = [
            _td(
                f"tool_{i}",
                "long description " * 100,
                {f"arg_{j}": {"type": "string", "description": "x" * 300} for j in range(20)},
            )
            for i in range(20)
        ]
        decision = RouteDecision(
            source="ai",
            capabilities=("terminal",),
            candidates=tuple(defs),
            confidence=0.9,
            latency_ms=1.0,
        )
        cfg = ToolRouterConfig.from_raw({"enabled": True, "max_packet_tokens": 256})
        packet = build_route_packet(decision, cfg)
        assert len(packet.encode("utf-8")) <= cfg.max_packet_tokens * 4
        prefix = "[HERMES_TOOL_ROUTE]\n"
        suffix = "\n[/HERMES_TOOL_ROUTE]"
        payload = json.loads(packet[len(prefix):-len(suffix)])
        assert payload["source"] == "ai"
        assert payload["recovery"] == "Use tool_search when candidates are insufficient."

    def test_telemetry_has_no_message_or_arguments(self, caplog):
        from agent.tool_router import ToolRouterConfig, route_turn

        secret = "private-user-message-DO-NOT-LOG"
        cfg = ToolRouterConfig.from_raw({"enabled": True, "telemetry": True})
        with caplog.at_level("INFO", logger="agent.tool_router"):
            route_turn(
                secret,
                [_td("read_file")],
                config=cfg,
                ai_classifier=lambda *_: {"capabilities": ["files_read"], "confidence": 0.9},
            )
        assert secret not in caplog.text
        assert "arguments" not in caplog.text
