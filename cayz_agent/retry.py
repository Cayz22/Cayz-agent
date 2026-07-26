"""
重试与监控工具

- retry_on_error: 对瞬时错误（网络/超时/限流/业务集成瞬时错误）自动重试
- log_execution: 记录函数调用耗时与异常（同时记录工具调用监控指标）
"""

import functools
import logging
import socket
import time
from collections.abc import Callable
from typing import TypeVar

from tenacity import (
    RetryCallState,
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from .exceptions import CayzAgentError, EmailError, LLMRateLimitError, NotifyError, RAGConnectionError
from .monitor import record_retry, record_tool_call

logger = logging.getLogger(__name__)

T = TypeVar("T")

# 视为瞬时错误、值得重试的异常类型
# P2-11 修复：移除过宽的 OSError（其包含 FileNotFoundError / PermissionError 等永久性错误），
# 改用 socket.timeout 精确捕获网络超时。ConnectionError 已覆盖网络连接重置/拒绝。
# - 网络层：ConnectionError / TimeoutError / socket.timeout
# - 业务集成瞬时错误：NotifyError（Webhook 网络抖动）/ EmailError（SMTP 瞬时失败）
# - RAG 连接错误：ChromaDB 瞬时不可用
# - LLM 限流：服务端 429
RETRYABLE_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    socket.timeout,  # P2-11：精确捕获 socket 超时，替代过宽的 OSError
    NotifyError,
    EmailError,
    RAGConnectionError,
    LLMRateLimitError,
)


def _is_retryable(exc: BaseException) -> bool:
    """P2-12：判断异常是否可重试。

    优先检查 CayzAgentError 子类的 retryable 属性：
    - retryable=False：永久性错误（如 SMTP 认证失败），不重试
    - retryable=True：瞬时错误，重试
    - 非 CayzAgentError：回退到 RETRYABLE_EXCEPTIONS 类型匹配
    """
    if isinstance(exc, CayzAgentError):
        # 显式标记 retryable=False 的永久性错误不重试
        if not exc.retryable:
            return False
        return True
    return isinstance(exc, RETRYABLE_EXCEPTIONS)


def retry_on_error(max_attempts: int = 3, min_wait: float = 1.0, max_wait: float = 10.0):
    """
    装饰器：对瞬时网络错误进行指数退避重试。

    P2-12 修复：尊重 CayzAgentError.retryable 属性，永久性错误（retryable=False）
    不重试，避免凭据错误等场景无谓重试浪费时间。

    Args:
        max_attempts: 最大重试次数（含首次）
        min_wait: 首次重试等待秒数
        max_wait: 最大等待秒数
    """

    def _on_retry(state: RetryCallState):
        """重试回调：仅在非首次尝试时记录重试指标"""
        if state.attempt_number > 1:
            record_retry()

    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=min_wait, max=max_wait),
        retry=retry_if_exception(_is_retryable),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        after=_on_retry,
        reraise=True,
    )


def log_execution(func: Callable[..., T]) -> Callable[..., T]:
    """
    装饰器：记录函数调用耗时与异常，同时记录工具调用监控指标。

    用于工具函数，便于监控调用性能。
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> T:
        func_name = getattr(func, "__name__", str(func))
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logger.info("%s 执行完成, 耗时 %.3fs", func_name, elapsed)
            record_tool_call(func_name, success=True, latency=elapsed)
            return result
        except Exception:
            elapsed = time.perf_counter() - start
            logger.exception("%s 执行失败, 耗时 %.3fs", func_name, elapsed)
            record_tool_call(func_name, success=False, latency=elapsed)
            raise

    return wrapper
