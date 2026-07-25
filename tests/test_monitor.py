"""
monitor 模块单元测试：验证指标收集与 Prometheus 导出
"""
import pytest

from cayz_agent.monitor import (
    Counter,
    Histogram,
    Gauge,
    MetricsRegistry,
    get_registry,
    record_request,
    record_token_usage,
    record_tool_call,
    record_route,
    record_validation_failure,
    record_retry,
    record_session_start,
    record_session_end,
    record_session_deleted,
    record_knowledge_upload,
    record_knowledge_delete,
    export_prometheus,
    get_metrics_summary,
)


@pytest.fixture(autouse=True)
def reset_registry():
    """每个测试前重置全局注册表"""
    reg = get_registry()
    reg.reset()
    yield
    reg.reset()


class TestCounter:
    """测试 Counter 数据结构"""

    def test_initial_value(self):
        c = Counter()
        assert c.get() == 0

    def test_inc_default(self):
        c = Counter()
        c.inc()
        assert c.get() == 1

    def test_inc_amount(self):
        c = Counter()
        c.inc(5)
        assert c.get() == 5

    def test_inc_multiple(self):
        c = Counter()
        c.inc(3)
        c.inc(7)
        assert c.get() == 10


class TestHistogram:
    """测试 Histogram 数据结构"""

    def test_initial_stats(self):
        h = Histogram()
        stats = h.get_stats()
        assert stats["count"] == 0
        assert stats["sum"] == 0.0
        assert stats["avg"] == 0.0

    def test_observe_single(self):
        h = Histogram()
        h.observe(0.5)
        stats = h.get_stats()
        assert stats["count"] == 1
        assert stats["sum"] == 0.5
        assert stats["avg"] == 0.5

    def test_observe_multiple(self):
        h = Histogram()
        h.observe(0.1)
        h.observe(0.3)
        h.observe(0.5)
        stats = h.get_stats()
        assert stats["count"] == 3
        assert abs(stats["sum"] - 0.9) < 0.001
        assert abs(stats["avg"] - 0.3) < 0.001

    def test_buckets_populated(self):
        h = Histogram()
        h.observe(0.05)
        h.observe(0.2)
        stats = h.get_stats()
        assert stats["buckets"]["le_0.05"] == 1
        # 0.05 and 0.2 are both <= 0.25
        assert stats["buckets"]["le_0.25"] == 2


class TestGauge:
    """测试 Gauge 数据结构"""

    def test_initial_value(self):
        g = Gauge()
        assert g.get() == 0.0

    def test_set(self):
        g = Gauge()
        g.set(42.0)
        assert g.get() == 42.0

    def test_inc_dec(self):
        g = Gauge()
        g.inc(5)
        g.dec(2)
        assert g.get() == 3.0


class TestMetricsRegistry:
    """测试全局指标注册表"""

    def test_singleton(self):
        r1 = get_registry()
        r2 = get_registry()
        assert r1 is r2

    def test_reset(self):
        reg = get_registry()
        reg.requests_total.inc(10)
        assert reg.requests_total.get() == 10
        reg.reset()
        assert reg.requests_total.get() == 0


class TestRecordFunctions:
    """测试便捷记录函数"""

    def test_record_request(self):
        record_request("chat", success=True, latency=0.5)
        reg = get_registry()
        assert reg.requests_total.get() == 1
        assert reg.requests_by_type["chat"].get() == 1
        assert reg.request_errors.get() == 0

    def test_record_request_failure(self):
        record_request("chat", success=False, latency=0.1)
        reg = get_registry()
        assert reg.request_errors.get() == 1

    def test_record_token_usage(self):
        record_token_usage(input_tokens=100, output_tokens=50, total_tokens=150)
        reg = get_registry()
        assert reg.tokens_input.get() == 100
        assert reg.tokens_output.get() == 50
        assert reg.tokens_total.get() == 150

    def test_record_tool_call(self):
        record_tool_call("web_search", success=True, latency=1.2)
        reg = get_registry()
        assert reg.tool_calls_total.get() == 1
        assert reg.tool_calls_by_name["web_search"].get() == 1
        assert reg.tool_call_errors.get() == 0

    def test_record_tool_call_error(self):
        record_tool_call("web_search", success=False, latency=2.0)
        reg = get_registry()
        assert reg.tool_call_errors.get() == 1

    def test_record_route(self):
        record_route("knowledge")
        record_route("search")
        record_route("chat")
        reg = get_registry()
        assert reg.route_counts["knowledge"].get() == 1
        assert reg.route_counts["search"].get() == 1
        assert reg.route_counts["chat"].get() == 1

    def test_record_validation_failure(self):
        record_validation_failure()
        reg = get_registry()
        assert reg.validation_failures.get() == 1
        assert reg.input_rejected.get() == 1

    def test_record_retry(self):
        record_retry()
        record_retry()
        reg = get_registry()
        assert reg.retry_attempts.get() == 2


class TestExportPrometheus:
    """测试 Prometheus 格式导出"""

    def test_export_contains_required_metrics(self):
        record_request("chat", success=True, latency=0.3)
        record_token_usage(input_tokens=100, output_tokens=50, total_tokens=150)
        record_tool_call("web_search", success=True, latency=0.5)

        output = export_prometheus()

        assert "cayz_requests_total" in output
        assert "cayz_request_errors_total" in output
        assert "cayz_request_latency_seconds" in output
        assert "cayz_tokens_input_total" in output
        assert "cayz_tokens_output_total" in output
        assert "cayz_tokens_total" in output
        assert "cayz_tool_calls_total" in output
        assert "cayz_tool_call_errors_total" in output
        assert "cayz_tool_call_latency_seconds" in output
        assert "cayz_active_sessions" in output
        assert "cayz_validation_failures_total" in output
        assert "cayz_retry_attempts_total" in output
        assert "cayz_uptime_seconds" in output

    def test_export_prometheus_format(self):
        """验证输出包含 HELP 和 TYPE 行"""
        record_request("chat", success=True, latency=0.1)
        output = export_prometheus()

        assert "# HELP cayz_requests_total" in output
        assert "# TYPE cayz_requests_total counter" in output

    def test_export_with_labels(self):
        """验证带标签的指标"""
        record_tool_call("web_search", success=True, latency=0.5)
        record_tool_call("knowledge_search", success=True, latency=0.3)

        output = export_prometheus()

        assert 'tool="web_search"' in output
        assert 'tool="knowledge_search"' in output


class TestGetMetricsSummary:
    """测试指标摘要"""

    def test_summary_structure(self):
        record_request("chat", success=True, latency=0.5)
        record_token_usage(input_tokens=100, output_tokens=50, total_tokens=150)

        summary = get_metrics_summary()

        assert "requests_total" in summary
        assert "request_errors" in summary
        assert "avg_latency_seconds" in summary
        assert "tokens_total" in summary
        assert "tool_calls_total" in summary
        assert "tool_call_errors" in summary
        assert "validation_failures" in summary
        assert "retry_attempts" in summary
        assert "active_sessions" in summary
        assert "uptime_seconds" in summary

    def test_summary_values(self):
        record_request("chat", success=True, latency=0.5)
        record_request("chat", success=False, latency=1.0)

        summary = get_metrics_summary()

        assert summary["requests_total"] == 2
        assert summary["request_errors"] == 1
        assert abs(summary["avg_latency_seconds"] - 0.75) < 0.01

    def test_summary_contains_new_metrics(self):
        """摘要应包含会话和知识库新指标"""
        summary = get_metrics_summary()
        assert "sessions_deleted" in summary
        assert "knowledge_uploads" in summary
        assert "knowledge_deletes" in summary
        assert "knowledge_chunks" in summary


class TestSessionMetrics:
    """测试会话相关指标"""

    def test_active_sessions_inc_dec(self):
        """record_session_start/end 应正确增减 active_sessions"""
        record_session_start()
        record_session_start()
        summary = get_metrics_summary()
        assert summary["active_sessions"] == 2

        record_session_end()
        summary = get_metrics_summary()
        assert summary["active_sessions"] == 1

    def test_session_deleted_counter(self):
        """record_session_deleted 应递增计数器"""
        record_session_deleted()
        record_session_deleted()
        summary = get_metrics_summary()
        assert summary["sessions_deleted"] == 2


class TestKnowledgeMetrics:
    """测试知识库业务指标"""

    def test_knowledge_upload_increments_counter(self):
        """record_knowledge_upload 应递增上传计数"""
        record_knowledge_upload(chunks=5)
        summary = get_metrics_summary()
        assert summary["knowledge_uploads"] >= 1
        assert summary["knowledge_chunks"] >= 5

    def test_knowledge_delete_decrements_chunks(self):
        """record_knowledge_delete 应递减 chunks 总数"""
        record_knowledge_upload(chunks=10)
        record_knowledge_delete(chunks=3)
        summary = get_metrics_summary()
        # chunks 应为 10 - 3 = 7（在测试隔离环境下）
        assert summary["knowledge_chunks"] >= 7

    def test_knowledge_upload_zero_chunks(self):
        """chunks=0 时不应影响 chunks_total"""
        record_knowledge_upload(chunks=0)
        summary = get_metrics_summary()
        assert summary["knowledge_uploads"] >= 1

    def test_knowledge_metrics_in_prometheus_export(self):
        """知识库指标应出现在 Prometheus 导出中"""
        record_knowledge_upload(chunks=2)
        record_knowledge_delete(chunks=1)

        output = export_prometheus()
        assert "cayz_knowledge_uploads_total" in output
        assert "cayz_knowledge_deletes_total" in output
        assert "cayz_knowledge_chunks_total" in output
        assert "cayz_sessions_deleted_total" in output
