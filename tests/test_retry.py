"""
retry 模块单元测试：验证重试与监控逻辑
"""
import logging
import time

import pytest

from cayz_agent.retry import retry_on_error, log_execution, RETRYABLE_EXCEPTIONS
from cayz_agent.exceptions import (
    NotifyError,
    EmailError,
    RAGConnectionError,
    LLMRateLimitError,
    CRMError,
)


class TestRetryOnError:
    """测试 retry_on_error 装饰器"""

    def test_success_no_retry(self):
        """成功调用不应重试"""
        call_count = 0

        @retry_on_error(max_attempts=3)
        def func():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = func()
        assert result == "ok"
        assert call_count == 1

    def test_retries_on_connection_error(self):
        """ConnectionError 应触发重试"""
        call_count = 0

        @retry_on_error(max_attempts=3, min_wait=0.01, max_wait=0.05)
        def func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("网络断开")
            return "recovered"

        result = func()
        assert result == "recovered"
        assert call_count == 3

    def test_reraises_after_max_attempts(self):
        """超过最大重试次数后应抛出原异常"""
        call_count = 0

        @retry_on_error(max_attempts=2, min_wait=0.01, max_wait=0.02)
        def func():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("持续失败")

        with pytest.raises(ConnectionError):
            func()
        assert call_count == 2

    def test_non_retryable_exception_not_retried(self):
        """非瞬时异常不应重试"""
        call_count = 0

        @retry_on_error(max_attempts=3, min_wait=0.01, max_wait=0.02)
        def func():
            nonlocal call_count
            call_count += 1
            raise ValueError("参数错误，非瞬时")

        with pytest.raises(ValueError):
            func()
        assert call_count == 1

    def test_retryable_exceptions_includes_connection_error(self):
        """RETRYABLE_EXCEPTIONS 应包含常见网络错误"""
        assert ConnectionError in RETRYABLE_EXCEPTIONS
        assert TimeoutError in RETRYABLE_EXCEPTIONS

    def test_retryable_exceptions_includes_business_errors(self):
        """RETRYABLE_EXCEPTIONS 应包含业务集成瞬时错误（NotifyError / EmailError）"""
        assert NotifyError in RETRYABLE_EXCEPTIONS
        assert EmailError in RETRYABLE_EXCEPTIONS

    def test_retryable_exceptions_includes_rag_and_llm_errors(self):
        """RETRYABLE_EXCEPTIONS 应包含 RAGConnectionError 和 LLMRateLimitError"""
        assert RAGConnectionError in RETRYABLE_EXCEPTIONS
        assert LLMRateLimitError in RETRYABLE_EXCEPTIONS

    def test_retries_on_notify_error(self):
        """NotifyError 应触发重试（修复 P0：原 RETRYABLE_EXCEPTIONS 未包含 NotifyError）"""
        call_count = 0

        @retry_on_error(max_attempts=3, min_wait=0.01, max_wait=0.05)
        def func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise NotifyError("企业微信 Webhook 暂时不可用")
            return "notified"

        result = func()
        assert result == "notified"
        assert call_count == 2

    def test_retries_on_email_error(self):
        """EmailError 应触发重试（修复 P0：原 RETRYABLE_EXCEPTIONS 未包含 EmailError）"""
        call_count = 0

        @retry_on_error(max_attempts=3, min_wait=0.01, max_wait=0.05)
        def func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise EmailError("SMTP 瞬时拥堵")
            return "sent"

        result = func()
        assert result == "sent"
        assert call_count == 2

    def test_non_retryable_crm_error_not_retried(self):
        """CRMError 是业务集成永久错误，不应重试（区别于 NotifyError/EmailError）"""
        call_count = 0

        @retry_on_error(max_attempts=3, min_wait=0.01, max_wait=0.02)
        def func():
            nonlocal call_count
            call_count += 1
            raise CRMError("客户 ID 不存在")

        with pytest.raises(CRMError):
            func()
        assert call_count == 1

    def test_reraises_notify_error_after_max_attempts(self):
        """NotifyError 重试用尽后应 reraise（保证错误能传播到上层 tools.py）"""
        call_count = 0

        @retry_on_error(max_attempts=2, min_wait=0.01, max_wait=0.02)
        def func():
            nonlocal call_count
            call_count += 1
            raise NotifyError("持续失败")

        with pytest.raises(NotifyError):
            func()
        assert call_count == 2


class TestLogExecution:
    """测试 log_execution 装饰器"""

    def test_logs_success(self, caplog):
        """成功时应记录耗时"""
        @log_execution
        def func():
            return 42

        with caplog.at_level(logging.INFO):
            result = func()

        assert result == 42
        assert any("执行完成" in record.message for record in caplog.records)

    def test_logs_failure_and_reraises(self, caplog):
        """失败时应记录异常并重新抛出"""
        @log_execution
        def func():
            raise RuntimeError("boom")

        with caplog.at_level(logging.INFO):
            with pytest.raises(RuntimeError):
                func()

        assert any("执行失败" in record.message for record in caplog.records)

    def test_preserves_function_name(self):
        """装饰器应保留原函数名"""
        @log_execution
        def my_function():
            """docstring"""
            return None

        assert my_function.__name__ == "my_function"
