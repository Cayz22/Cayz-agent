"""
异常体系单元测试

验证自定义异常的继承关系、属性、字符串表示。
"""

import pytest

from cayz_agent.exceptions import (
    CayzAgentError,
    ConfigError,
    CRMError,
    EmailError,
    IntegrationError,
    LLMError,
    LLMRateLimitError,
    NotifyError,
    RAGConnectionError,
    RAGError,
    RAGIngestError,
    ToolError,
)


class TestCayzAgentError:
    """测试基础异常类"""

    def test_default_values(self):
        """默认值：空 message，无 cause，不可重试"""
        err = CayzAgentError()
        assert err.message == ""
        assert err.cause is None
        assert err.retryable is False

    def test_with_message(self):
        err = CayzAgentError("something wrong")
        assert err.message == "something wrong"
        assert str(err) == "something wrong"

    def test_with_cause(self):
        cause = ValueError("root cause")
        err = CayzAgentError("wrapper", cause=cause)
        assert err.cause is cause
        assert "root cause" in str(err)
        assert "wrapper" in str(err)

    def test_retryable_flag(self):
        err = CayzAgentError("retry me", retryable=True)
        assert err.retryable is True

    def test_is_exception_subclass(self):
        """应是 Exception 的子类，可被 except Exception 捕获"""
        err = CayzAgentError("test")
        assert isinstance(err, Exception)

    def test_reraise_preserves_chain(self):
        """异常链应在 raise ... from ... 时保留"""
        with pytest.raises(CayzAgentError) as exc_info:
            try:
                raise ValueError("original")
            except ValueError as e:
                raise CayzAgentError("wrapped", cause=e) from e
        assert exc_info.value.cause is not None


class TestExceptionHierarchy:
    """测试异常继承关系"""

    def test_config_error_is_cayz_error(self):
        assert issubclass(ConfigError, CayzAgentError)

    def test_llm_error_is_cayz_error(self):
        assert issubclass(LLMError, CayzAgentError)

    def test_llm_rate_limit_is_llm_error(self):
        assert issubclass(LLMRateLimitError, LLMError)

    def test_tool_error_is_cayz_error(self):
        assert issubclass(ToolError, CayzAgentError)

    def test_rag_error_is_cayz_error(self):
        assert issubclass(RAGError, CayzAgentError)

    def test_rag_connection_is_rag_error(self):
        assert issubclass(RAGConnectionError, RAGError)

    def test_rag_ingest_is_rag_error(self):
        assert issubclass(RAGIngestError, RAGError)

    def test_integration_error_is_cayz_error(self):
        assert issubclass(IntegrationError, CayzAgentError)

    def test_crm_error_is_integration_error(self):
        assert issubclass(CRMError, IntegrationError)

    def test_notify_error_is_integration_error(self):
        assert issubclass(NotifyError, IntegrationError)

    def test_email_error_is_integration_error(self):
        assert issubclass(EmailError, IntegrationError)

    def test_catch_by_base_class(self):
        """所有自定义异常应可被 CayzAgentError 捕获"""
        for exc_class in [
            ConfigError,
            LLMError,
            ToolError,
            RAGError,
            IntegrationError,
            CRMError,
            NotifyError,
            EmailError,
        ]:
            with pytest.raises(CayzAgentError):
                raise exc_class("test")


class TestRetryableFlag:
    """测试可重试标记"""

    def test_llm_rate_limit_is_retryable(self):
        err = LLMRateLimitError()
        assert err.retryable is True

    def test_rag_connection_is_retryable(self):
        err = RAGConnectionError()
        assert err.retryable is True

    def test_config_error_not_retryable(self):
        err = ConfigError("bad config")
        assert err.retryable is False

    def test_tool_error_not_retryable_by_default(self):
        err = ToolError("tool failed")
        assert err.retryable is False
