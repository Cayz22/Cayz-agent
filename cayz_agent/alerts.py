"""
告警模块

- 基于阈值的告警检查器
- 支持多种告警类型：高错误率、高延迟、Token 超限、工具失败率
- 告警回调机制（日志 + 可插拔的自定义回调）
- 告警抑制（避免短时间内重复告警）
- 后台定时 watcher 线程：避免依赖 /metrics 端点被访问才触发检查
"""

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

from .monitor import get_metrics_summary

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """告警级别"""

    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class Alert:
    """告警记录"""

    name: str
    level: AlertLevel
    message: str
    timestamp: float = field(default_factory=time.time)
    metric_value: float = 0.0
    threshold: float = 0.0


# P1/P2：限制告警历史最大保留数，避免长跑内存无界增长
_MAX_ALERTS_HISTORY = 1000


class AlertManager:
    """
    告警管理器

    定期检查指标，当超过阈值时触发告警。
    支持告警抑制（同一告警在 suppress_seconds 内不重复触发）。

    P1 线程安全：后台 watcher 线程与 /metrics 端点并发调用 check_and_alert / get_recent_alerts，
    通过 _lock 保护 _last_alert_time 与 _alerts。
    P2 内存：_alerts 改用 deque(maxlen=...)，避免长跑内存无界增长。
    """

    def __init__(
        self,
        suppress_seconds: float = 300.0,
        callback: Callable[[Alert], None] | None = None,
    ):
        self.suppress_seconds = suppress_seconds
        self.callback = callback
        self._last_alert_time: dict[str, float] = {}
        # P2：用 deque(maxlen=...) 自动裁剪旧告警，避免长跑内存增长
        self._alerts: deque = deque(maxlen=_MAX_ALERTS_HISTORY)
        # P1：保护 _last_alert_time 与 _alerts 的并发读写
        self._lock = threading.Lock()

    def check_and_alert(self) -> list[Alert]:
        """
        检查所有指标，返回新触发的告警列表。

        告警规则：
        1. 请求错误率 > 50% → CRITICAL
        2. 平均请求延迟 > 5s → WARNING
        3. 工具调用错误率 > 30% → WARNING
        4. 验证失败次数 > 10 → WARNING
        5. 重试次数 > 5 → WARNING
        """
        summary = get_metrics_summary()
        new_alerts = []

        # 规则 1：请求错误率
        requests_total = summary["requests_total"]
        if requests_total > 0:
            error_rate = summary["request_errors"] / requests_total
            if error_rate > 0.5:
                alert = self._maybe_alert(
                    name="high_error_rate",
                    level=AlertLevel.CRITICAL,
                    message=f"请求错误率过高: {error_rate:.1%} (阈值: 50%)",
                    metric_value=error_rate,
                    threshold=0.5,
                )
                if alert:
                    new_alerts.append(alert)

        # 规则 2：平均延迟
        avg_latency = summary["avg_latency_seconds"]
        if avg_latency > 5.0:
            alert = self._maybe_alert(
                name="high_latency",
                level=AlertLevel.WARNING,
                message=f"平均请求延迟过高: {avg_latency:.2f}s (阈值: 5s)",
                metric_value=avg_latency,
                threshold=5.0,
            )
            if alert:
                new_alerts.append(alert)

        # 规则 3：工具调用错误率
        tool_total = summary["tool_calls_total"]
        if tool_total > 0:
            tool_error_rate = summary["tool_call_errors"] / tool_total
            if tool_error_rate > 0.3:
                alert = self._maybe_alert(
                    name="high_tool_error_rate",
                    level=AlertLevel.WARNING,
                    message=f"工具调用错误率过高: {tool_error_rate:.1%} (阈值: 30%)",
                    metric_value=tool_error_rate,
                    threshold=0.3,
                )
                if alert:
                    new_alerts.append(alert)

        # 规则 4：验证失败
        validation_failures = summary["validation_failures"]
        if validation_failures > 10:
            alert = self._maybe_alert(
                name="excessive_validation_failures",
                level=AlertLevel.WARNING,
                message=f"输入验证失败过多: {validation_failures} 次 (阈值: 10)",
                metric_value=validation_failures,
                threshold=10.0,
            )
            if alert:
                new_alerts.append(alert)

        # 规则 5：重试次数
        retry_attempts = summary["retry_attempts"]
        if retry_attempts > 5:
            alert = self._maybe_alert(
                name="excessive_retries",
                level=AlertLevel.WARNING,
                message=f"重试次数过多: {retry_attempts} 次 (阈值: 5)",
                metric_value=retry_attempts,
                threshold=5.0,
            )
            if alert:
                new_alerts.append(alert)

        return new_alerts

    def _maybe_alert(
        self,
        name: str,
        level: AlertLevel,
        message: str,
        metric_value: float,
        threshold: float,
    ) -> Alert | None:
        """检查抑制窗口，未抑制则创建告警（P1 线程安全：在锁内读写 _last_alert_time 与 _alerts）"""
        now = time.time()
        with self._lock:
            last_time = self._last_alert_time.get(name, 0.0)
            if now - last_time < self.suppress_seconds:
                return None

            alert = Alert(
                name=name,
                level=level,
                message=message,
                metric_value=metric_value,
                threshold=threshold,
            )

            self._last_alert_time[name] = now
            self._alerts.append(alert)

        # 记录日志（锁外执行，避免日志 I/O 阻塞其他线程）
        log_level = logging.CRITICAL if level == AlertLevel.CRITICAL else logging.WARNING
        logger.log(log_level, "[告警] %s: %s", name, message)

        # 调用自定义回调（锁外执行，避免回调异常/慢调用阻塞其他线程）
        if self.callback:
            try:
                self.callback(alert)
            except Exception:
                logger.exception("告警回调执行失败")

        return alert

    def get_recent_alerts(self, count: int = 10) -> list[dict]:
        """获取最近的告警记录（P1 线程安全：在锁内切片快照）"""
        with self._lock:
            # deque 切片返回 deque 的 slice，转 list 便于序列化
            recent = list(self._alerts)[-count:] if self._alerts else []
        return [
            {
                "name": a.name,
                "level": a.level.value,
                "message": a.message,
                "timestamp": a.timestamp,
                "metric_value": a.metric_value,
                "threshold": a.threshold,
            }
            for a in recent
        ]

    def clear_suppression(self, name: str | None = None):
        """清除抑制状态（用于测试或手动重置）"""
        with self._lock:
            if name:
                self._last_alert_time.pop(name, None)
            else:
                self._last_alert_time.clear()

    def clear_alerts(self):
        """清除所有告警记录"""
        with self._lock:
            self._alerts.clear()


# 全局告警管理器实例
_global_manager: AlertManager | None = None


def get_alert_manager() -> AlertManager:
    """获取全局告警管理器"""
    global _global_manager
    if _global_manager is None:
        _global_manager = AlertManager()
    return _global_manager


def check_alerts() -> list[Alert]:
    """便捷函数：执行一次告警检查"""
    return get_alert_manager().check_and_alert()


# ============================================================
# 后台定时告警 watcher
# ============================================================


class AlertWatcher:
    """
    后台守护线程：每隔 interval 秒自动触发一次 check_alerts。

    解决"告警仅在被访问 /metrics 时才检查"的问题：
    无需外部调度器，启动后自动周期性扫描指标并触发告警。
    """

    def __init__(self, interval: float = 60.0):
        self.interval = interval
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._started = False

    def start(self) -> None:
        """启动后台 watcher（重复调用安全，仅首次实际启动线程）"""
        if self._started:
            return
        self._started = True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="alert-watcher",
            daemon=True,
        )
        self._thread.start()
        logger.info("告警 watcher 已启动，检查间隔: %ss", self.interval)

    def stop(self, timeout: float = 5.0) -> None:
        """停止后台 watcher（主要用于测试）"""
        if not self._started:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._started = False

    def _run_loop(self) -> None:
        """主循环：每隔 interval 秒执行 check_alerts，异常不退出"""
        while not self._stop_event.wait(self.interval):
            try:
                check_alerts()
            except Exception:
                # watcher 必须永不因单次异常退出
                logger.exception("告警 watcher 执行失败，将继续重试")


_watcher: AlertWatcher | None = None


def start_alert_watcher(interval: float = 60.0) -> AlertWatcher:
    """启动全局告警 watcher（重复调用安全）"""
    global _watcher
    if _watcher is None:
        _watcher = AlertWatcher(interval=interval)
    _watcher.start()
    return _watcher


def stop_alert_watcher() -> None:
    """停止全局告警 watcher（主要用于测试）"""
    global _watcher
    if _watcher is not None:
        _watcher.stop()
        _watcher = None
