"""
P3 企业级部署与运维功能测试。

覆盖：
1. P3-1: X-Request-ID 中间件 - 响应头注入 + 客户端透传 + 格式校验
2. P3-2: 优雅停机 - app_state 清理钩子 LIFO 执行
3. P3-3: 就绪探针分离 - /health/ready 状态码与响应体
4. P3-4: 业务指标增强 - RAG 检索指标与会话时长 histogram
5. P3-7: 启动配置自检报告 - 脱敏函数
"""
import logging
import time
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from cayz_agent.api import app, _mask_secret
from cayz_agent import app_state
from cayz_agent.request_context import (
    set_request_id,
    get_request_id,
    new_request_id,
    RequestIdLogFilter,
)


@pytest.fixture
def client():
    """FastAPI 测试客户端"""
    return TestClient(app)


# ============================================================
# P3-1: X-Request-ID 中间件
# ============================================================

class TestRequestIdMiddleware:
    """测试请求追踪中间件"""

    def test_response_has_request_id_header(self, client):
        """响应应包含 X-Request-ID 头"""
        resp = client.get("/health")
        assert resp.status_code == 200
        assert "X-Request-ID" in resp.headers
        # 默认生成的 request_id 应为 32 字符 UUID hex
        rid = resp.headers["X-Request-ID"]
        assert len(rid) == 32
        assert all(c in "0123456789abcdef" for c in rid)

    def test_client_provided_request_id_passthrough(self, client):
        """客户端传入的合法 X-Request-ID 应被透传"""
        custom_id = "my-trace-id-12345"
        resp = client.get("/health", headers={"X-Request-ID": custom_id})
        assert resp.status_code == 200
        assert resp.headers["X-Request-ID"] == custom_id

    def test_invalid_request_id_replaced_with_generated(self, client):
        """非法格式的 X-Request-ID 应被替换为生成的 UUID（防注入）"""
        # 含特殊字符的非法 ID
        invalid_id = "'; DROP TABLE logs; --"
        resp = client.get("/health", headers={"X-Request-ID": invalid_id})
        assert resp.status_code == 200
        rid = resp.headers["X-Request-ID"]
        # 应为生成的 32 字符 UUID，而非客户端传入的非法值
        assert rid != invalid_id
        assert len(rid) == 32

    def test_oversized_request_id_replaced(self, client):
        """超长 X-Request-ID 应被拒绝并替换（防 DoS）"""
        oversized = "a" * 200  # 超过 128 字符限制
        resp = client.get("/health", headers={"X-Request-ID": oversized})
        assert resp.status_code == 200
        rid = resp.headers["X-Request-ID"]
        assert rid != oversized
        assert len(rid) == 32


# ============================================================
# P3-2: 优雅停机 - app_state 清理钩子
# ============================================================

class TestAppState:
    """测试应用状态管理"""

    def test_cleanup_hooks_executed_lifo(self):
        """清理钩子应按 LIFO 顺序执行"""
        app_state.reset()
        order = []

        app_state.register_cleanup(lambda: order.append("first"))
        app_state.register_cleanup(lambda: order.append("second"))
        app_state.register_cleanup(lambda: order.append("third"))

        executed = app_state.run_cleanups(timeout=5.0)

        assert executed == 3
        # LIFO：后注册的先执行
        assert order == ["third", "second", "first"]

    def test_cleanup_hook_failure_does_not_block_others(self):
        """单个清理钩子失败不应阻塞其他钩子"""
        app_state.reset()
        executed = []

        def failing_hook():
            raise RuntimeError("模拟清理失败")

        app_state.register_cleanup(failing_hook)
        app_state.register_cleanup(lambda: executed.append("after_failing"))

        # failing_hook 后注册，LIFO 先执行；其失败不应阻塞后续
        result = app_state.run_cleanups(timeout=5.0)

        # 失败的钩子不计入 executed，但后续钩子仍执行
        assert executed == ["after_failing"]
        assert result == 1  # 只有成功的 1 个

    def test_ready_flag_lifecycle(self):
        """就绪标志应可设置与查询"""
        app_state.reset()
        assert app_state.is_ready() is False

        app_state.set_ready(True)
        assert app_state.is_ready() is True

        app_state.set_ready(False)
        assert app_state.is_ready() is False


# ============================================================
# P3-3: 就绪探针分离
# ============================================================

class TestHealthReady:
    """测试 /health/ready 就绪探针"""

    def test_ready_endpoint_returns_503_when_not_ready(self, client):
        """未就绪时应返回 503"""
        with patch("cayz_agent.api.app_state.is_ready", return_value=False):
            resp = client.get("/health/ready")
        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "not_ready"
        assert "version" in data
        assert "uptime_seconds" in data

    def test_ready_endpoint_returns_200_when_ready(self, client):
        """就绪时应返回 200"""
        with patch("cayz_agent.api.app_state.is_ready", return_value=True):
            resp = client.get("/health/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"

    def test_ready_endpoint_is_public(self, client):
        """/health/ready 应为公开端点（无需鉴权）"""
        # 不传 API Key 也应能访问
        resp = client.get("/health/ready")
        # 无论就绪状态，都应能访问（不返回 401）
        assert resp.status_code in (200, 503)


# ============================================================
# P3-4: 业务指标增强
# ============================================================

class TestBusinessMetrics:
    """测试 P3 新增业务指标"""

    def test_rag_search_metrics_recorded(self):
        """record_rag_search 应记录检索次数/延迟/结果数"""
        from cayz_agent.monitor import get_registry, record_rag_search

        reg = get_registry()
        reg.reset()

        # 记录 3 次成功检索
        record_rag_search(success=True, latency=0.1, result_count=5)
        record_rag_search(success=True, latency=0.2, result_count=3)
        # 1 次失败检索
        record_rag_search(success=False, latency=0.05, result_count=0)

        assert reg.rag_searches_total.get() == 3
        assert reg.rag_search_errors.get() == 1

        # 延迟 histogram 应有 3 个观测值
        latency_stats = reg.rag_search_latency.get_stats()
        assert latency_stats["count"] == 3

        # 结果数 histogram 应有 3 个观测值
        results_stats = reg.rag_search_results.get_stats()
        assert results_stats["count"] == 3

    def test_session_duration_histogram(self):
        """record_session_end 应记录会话时长到 histogram"""
        from cayz_agent.monitor import get_registry, record_session_start, record_session_end

        reg = get_registry()
        reg.reset()

        record_session_start()
        # 模拟 10 秒会话
        record_session_end(duration=10.0)

        # 活跃会话应回到 0
        assert reg.active_sessions.get() == 0
        # 会话时长 histogram 应有 1 个观测值
        duration_stats = reg.session_duration.get_stats()
        assert duration_stats["count"] == 1
        assert duration_stats["sum"] == 10.0

    def test_session_end_without_duration_skips_histogram(self):
        """record_session_end 不传 duration 时不应记录 histogram"""
        from cayz_agent.monitor import get_registry, record_session_start, record_session_end

        reg = get_registry()
        reg.reset()

        record_session_start()
        record_session_end()  # 不传 duration

        assert reg.active_sessions.get() == 0
        # histogram 应为空
        duration_stats = reg.session_duration.get_stats()
        assert duration_stats["count"] == 0

    def test_rag_metrics_exported_to_prometheus(self):
        """RAG 指标应导出到 Prometheus 格式"""
        from cayz_agent.monitor import export_prometheus, record_rag_search

        record_rag_search(success=True, latency=0.1, result_count=5)
        output = export_prometheus()

        assert "cayz_rag_searches_total" in output
        assert "cayz_rag_search_errors_total" in output
        assert "cayz_rag_search_latency_seconds" in output
        assert "cayz_rag_search_results" in output


# ============================================================
# P3-7: 启动配置自检 - 脱敏函数
# ============================================================

class TestMaskSecret:
    """测试敏感信息脱敏函数"""

    def test_empty_secret(self):
        """空字符串应返回 <empty>"""
        assert _mask_secret("") == "<empty>"

    def test_short_secret_fully_masked(self):
        """短字符串（<=8）应全部替换为 ***"""
        assert _mask_secret("short") == "***"
        assert _mask_secret("12345678") == "***"

    def test_long_secret_partial_mask(self):
        """长字符串应显示前4+***+后4"""
        result = _mask_secret("sk-abcdefghijklmnop")
        assert result == "sk-a***mnop"
        # 不应包含中间部分
        assert "bcdefghijkl" not in result

    def test_api_key_format_masked(self):
        """典型 API Key 格式应正确脱敏"""
        key = "sk-proj-abcdef1234567890xyz"
        result = _mask_secret(key)
        assert result.startswith("sk-p")
        assert result.endswith("0xyz")
        assert "***" in result


# ============================================================
# P3-1: request_context 模块
# ============================================================

class TestRequestContext:
    """测试请求上下文管理"""

    def test_set_and_get_request_id(self):
        """set_request_id 后 get_request_id 应返回相同值"""
        test_id = "test-request-id-123"
        set_request_id(test_id)
        assert get_request_id() == test_id
        # 清理
        set_request_id(None)
        assert get_request_id() is None

    def test_new_request_id_format(self):
        """new_request_id 应生成 32 字符 hex"""
        rid = new_request_id()
        assert len(rid) == 32
        assert all(c in "0123456789abcdef" for c in rid)

    def test_new_request_id_unique(self):
        """连续生成的 request_id 应不同"""
        ids = {new_request_id() for _ in range(100)}
        assert len(ids) == 100  # 全部唯一

    def test_request_id_log_filter_injects_field(self):
        """RequestIdLogFilter 应将 request_id 注入 LogRecord"""
        set_request_id("trace-abc-123")
        try:
            record = logging.LogRecord(
                name="test", level=logging.INFO, pathname="", lineno=0,
                msg="test message", args=(), exc_info=None,
            )
            filter_obj = RequestIdLogFilter()
            assert filter_obj.filter(record) is True
            assert record.request_id == "trace-abc-123"
        finally:
            set_request_id(None)

    def test_request_id_log_filter_no_id_when_unset(self):
        """未设置 request_id 时不应注入字段"""
        set_request_id(None)
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="test message", args=(), exc_info=None,
        )
        filter_obj = RequestIdLogFilter()
        assert filter_obj.filter(record) is True
        # 不应有 request_id 属性（未设置时注入）
        assert not hasattr(record, "request_id")
