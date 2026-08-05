"""
config 模块单元测试：验证配置加载、单例模式和日志初始化
"""

import logging
from pathlib import Path

import pytest

from cayz_agent.config import (
    Settings,
    _JsonFormatter,
    get_settings,
    reset_settings_cache,
    setup_logging,
)


class TestGetSettings:
    """测试 get_settings() 单例模式（修复 P1：原每次调用创建新实例）"""

    def setup_method(self):
        """每个测试前重置单例缓存"""
        reset_settings_cache()

    def test_returns_settings_instance(self):
        """应返回 Settings 实例"""
        s = get_settings()
        assert isinstance(s, Settings)

    def test_singleton_returns_same_instance(self):
        """多次调用应返回同一实例（修复 P0：原每次创建新实例）"""
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_reset_cache_creates_new_instance(self):
        """reset_settings_cache 后应创建新实例"""
        s1 = get_settings()
        reset_settings_cache()
        s2 = get_settings()
        assert s1 is not s2

    def test_singleton_after_reset_returns_same_instance(self):
        """重置后再次多次调用应返回新的同一实例"""
        s1 = get_settings()
        reset_settings_cache()
        s2 = get_settings()
        s3 = get_settings()
        assert s1 is not s2
        assert s2 is s3


class TestSettingsDefaults:
    """测试 Settings 默认值"""

    def test_default_values(self):
        """应返回正确的默认值（不含被 conftest 覆盖的 AUTH_REQUIRED）"""
        reset_settings_cache()
        s = get_settings()
        assert s.llm_provider == "openai"
        assert s.log_level == "INFO"
        assert s.log_format == "text"
        assert s.api_host == "0.0.0.0"
        assert s.api_port == 8000
        assert s.rate_limit_per_minute == 60
        # P0 安全默认值：CORS 默认为 *（允许所有来源），生产环境应通过 .env 显式配置
        assert s.cors_allowed_origins == "*"
        # P0 安全默认值：关闭 API 文档（auth_required 默认 True 由 middleware 测试覆盖，
        # 此处 conftest 设了 AUTH_REQUIRED=false 环境变量故不直接断言）
        assert s.docs_enabled is False

    def test_auth_required_default_true_without_env(self):
        """未设置 AUTH_REQUIRED 环境变量时，auth_required 默认应为 True（P0 安全）"""
        import os

        saved = os.environ.pop("AUTH_REQUIRED", None)
        # 临时设置一个 dummy OPENAI_API_KEY，避免 validator 在 auth_required=True 时报错
        saved_openai = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "dummy-key-for-test"
        try:
            reset_settings_cache()
            # 临时重命名 .env 文件，避免 pydantic-settings 从中读取 AUTH_REQUIRED=false
            env_path = Path(".env")
            env_backup = Path(".env.bak_test")
            env_exists = env_path.exists()
            if env_exists:
                env_path.rename(env_backup)
            try:
                s = get_settings()
                assert s.auth_required is True
            finally:
                if env_exists:
                    env_backup.rename(env_path)
        finally:
            if saved is not None:
                os.environ["AUTH_REQUIRED"] = saved
            if saved_openai is not None:
                os.environ["OPENAI_API_KEY"] = saved_openai
            else:
                os.environ.pop("OPENAI_API_KEY", None)
            reset_settings_cache()


class TestSetupLogging:
    """测试 setup_logging() 函数"""

    def test_text_format(self):
        """text 格式应正确初始化日志"""
        setup_logging(level="INFO", log_format="text")
        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO

    def test_json_format(self):
        """json 格式应使用 _JsonFormatter"""
        setup_logging(level="DEBUG", log_format="json")
        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG
        # 验证至少有一个 handler 使用 _JsonFormatter
        has_json_formatter = any(isinstance(h.formatter, _JsonFormatter) for h in root_logger.handlers if h.formatter)
        assert has_json_formatter

    def test_invalid_level_defaults_to_info(self):
        """无效日志级别应回退到 INFO"""
        setup_logging(level="INVALID_LEVEL", log_format="text")
        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO


class TestJsonFormatter:
    """测试 _JsonFormatter"""

    def test_format_basic_record(self):
        """应正确格式化基本日志记录为 JSON"""
        import json

        formatter = _JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="测试消息",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        data = json.loads(output)
        assert data["level"] == "INFO"
        assert data["logger"] == "test"
        assert data["message"] == "测试消息"
        assert "timestamp" in data

    def test_format_with_exception(self):
        """应包含异常信息"""
        import json

        formatter = _JsonFormatter()
        try:
            raise ValueError("测试异常")
        except ValueError:
            import sys

            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="出错",
            args=(),
            exc_info=exc_info,
        )
        output = formatter.format(record)
        data = json.loads(output)
        assert "exc_info" in data
        assert "测试异常" in data["exc_info"]
