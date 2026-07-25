"""
alerts 模块单元测试：验证告警逻辑
"""

import logging
import time

import pytest

from cayz_agent.alerts import (
    Alert,
    AlertLevel,
    AlertManager,
    AlertWatcher,
    check_alerts,
    get_alert_manager,
    start_alert_watcher,
    stop_alert_watcher,
)
from cayz_agent.monitor import (
    get_registry,
    record_request,
    record_retry,
    record_tool_call,
    record_validation_failure,
)


@pytest.fixture(autouse=True)
def reset_state():
    """每个测试前重置全局状态，并停止可能残留的 watcher 线程"""
    stop_alert_watcher()
    reg = get_registry()
    reg.reset()
    manager = get_alert_manager()
    manager.clear_alerts()
    manager.clear_suppression()
    yield
    stop_alert_watcher()
    reg.reset()
    manager.clear_alerts()
    manager.clear_suppression()


class TestAlertManager:
    """测试告警管理器"""

    def test_no_alerts_when_metrics_normal(self):
        """指标正常时不应触发告警"""
        manager = AlertManager(suppress_seconds=0)
        record_request("chat", success=True, latency=0.1)
        alerts = manager.check_and_alert()
        assert len(alerts) == 0

    def test_high_error_rate_alert(self):
        """错误率超过 50% 应触发 CRITICAL 告警"""
        manager = AlertManager(suppress_seconds=0)
        record_request("chat", success=False, latency=0.1)
        record_request("chat", success=False, latency=0.1)
        record_request("chat", success=True, latency=0.1)

        alerts = manager.check_and_alert()
        assert len(alerts) == 1
        assert alerts[0].name == "high_error_rate"
        assert alerts[0].level == AlertLevel.CRITICAL

    def test_high_latency_alert(self):
        """平均延迟超过 5s 应触发 WARNING 告警"""
        manager = AlertManager(suppress_seconds=0)
        record_request("chat", success=True, latency=6.0)
        record_request("chat", success=True, latency=7.0)

        alerts = manager.check_and_alert()
        assert len(alerts) == 1
        assert alerts[0].name == "high_latency"
        assert alerts[0].level == AlertLevel.WARNING

    def test_high_tool_error_rate_alert(self):
        """工具错误率超过 30% 应触发 WARNING 告警"""
        manager = AlertManager(suppress_seconds=0)
        record_tool_call("web_search", success=False, latency=0.1)
        record_tool_call("web_search", success=False, latency=0.1)
        record_tool_call("web_search", success=True, latency=0.1)

        alerts = manager.check_and_alert()
        assert len(alerts) == 1
        assert alerts[0].name == "high_tool_error_rate"

    def test_excessive_validation_failures_alert(self):
        """验证失败超过 10 次应触发 WARNING 告警"""
        manager = AlertManager(suppress_seconds=0)
        for _ in range(11):
            record_validation_failure()

        alerts = manager.check_and_alert()
        assert len(alerts) == 1
        assert alerts[0].name == "excessive_validation_failures"

    def test_excessive_retries_alert(self):
        """重试次数超过 5 次应触发 WARNING 告警"""
        manager = AlertManager(suppress_seconds=0)
        for _ in range(6):
            record_retry()

        alerts = manager.check_and_alert()
        assert len(alerts) == 1
        assert alerts[0].name == "excessive_retries"

    def test_alert_suppression(self):
        """同一告警在抑制窗口内不应重复触发"""
        manager = AlertManager(suppress_seconds=300)
        record_request("chat", success=False, latency=0.1)
        record_request("chat", success=False, latency=0.1)

        alerts1 = manager.check_and_alert()
        assert len(alerts1) == 1

        alerts2 = manager.check_and_alert()
        assert len(alerts2) == 0

    def test_clear_suppression(self):
        """清除抑制后应重新触发告警"""
        manager = AlertManager(suppress_seconds=300)
        record_request("chat", success=False, latency=0.1)
        record_request("chat", success=False, latency=0.1)

        manager.check_and_alert()
        manager.clear_suppression("high_error_rate")

        alerts = manager.check_and_alert()
        assert len(alerts) == 1

    def test_custom_callback(self):
        """自定义回调应被调用"""
        received_alerts = []

        def callback(alert: Alert):
            received_alerts.append(alert)

        manager = AlertManager(suppress_seconds=0, callback=callback)
        record_request("chat", success=False, latency=0.1)
        record_request("chat", success=False, latency=0.1)

        manager.check_and_alert()
        assert len(received_alerts) == 1
        assert received_alerts[0].name == "high_error_rate"

    def test_get_recent_alerts(self):
        """获取最近告警记录"""
        manager = AlertManager(suppress_seconds=0)
        record_request("chat", success=False, latency=0.1)
        record_request("chat", success=False, latency=0.1)

        manager.check_and_alert()
        recent = manager.get_recent_alerts(count=5)
        assert len(recent) == 1
        assert recent[0]["name"] == "high_error_rate"
        assert recent[0]["level"] == "CRITICAL"


class TestGlobalAlertManager:
    """测试全局告警管理器"""

    def test_singleton(self):
        m1 = get_alert_manager()
        m2 = get_alert_manager()
        assert m1 is m2

    def test_check_alerts_convenience(self):
        """便捷函数应返回告警列表"""
        record_request("chat", success=False, latency=0.1)
        record_request("chat", success=False, latency=0.1)

        alerts = check_alerts()
        assert isinstance(alerts, list)


class TestAlertWatcher:
    """测试后台告警 watcher（P2 新增）"""

    def test_watcher_starts_and_stops(self):
        """watcher 应能启动并停止"""
        watcher = AlertWatcher(interval=60.0)
        assert not watcher._started
        watcher.start()
        assert watcher._started
        assert watcher._thread is not None
        assert watcher._thread.is_alive()
        watcher.stop(timeout=2.0)
        assert not watcher._started

    def test_start_idempotent(self):
        """重复调用 start() 不会创建多个线程"""
        watcher = AlertWatcher(interval=60.0)
        watcher.start()
        thread1 = watcher._thread
        watcher.start()
        thread2 = watcher._thread
        assert thread1 is thread2
        watcher.stop(timeout=2.0)

    def test_stop_without_start_is_safe(self):
        """未启动的 watcher 调用 stop() 不应报错"""
        watcher = AlertWatcher(interval=60.0)
        watcher.stop()  # 不应抛异常

    def test_watcher_triggers_check_alerts(self):
        """watcher 运行时应实际调用 check_alerts（短间隔验证）"""
        manager = get_alert_manager()
        manager.clear_alerts()
        manager.clear_suppression()
        # 触发 high_latency 告警条件
        record_request("chat", success=True, latency=6.0)
        record_request("chat", success=True, latency=7.0)

        watcher = AlertWatcher(interval=0.1)
        watcher.start()
        # 等待 watcher 至少跑一次（间隔 0.1s + 容差）
        time.sleep(0.5)
        watcher.stop(timeout=2.0)

        # watcher 应该已经触发告警
        recent = manager.get_recent_alerts(count=10)
        assert any(a["name"] == "high_latency" for a in recent)

    def test_global_start_stop_functions(self):
        """全局 start_alert_watcher / stop_alert_watcher 应正常工作"""
        start_alert_watcher(interval=60.0)
        start_alert_watcher(interval=60.0)  # 重复调用安全
        stop_alert_watcher()
        stop_alert_watcher()  # 重复调用安全

    def test_watcher_swallows_exceptions(self):
        """watcher 内部异常不应导致线程退出"""
        from unittest.mock import patch

        watcher = AlertWatcher(interval=0.05)

        # 让 check_alerts 抛异常
        with patch("cayz_agent.alerts.check_alerts", side_effect=RuntimeError("boom")):
            watcher.start()
            time.sleep(0.2)
            # 线程应仍存活
            assert watcher._thread is not None
            assert watcher._thread.is_alive()
            watcher.stop(timeout=2.0)
