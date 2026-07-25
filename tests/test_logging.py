"""
结构化日志单元测试

验证 text 和 json 两种格式的输出。
直接测试 Formatter，避免 handler/stream 绑定问题。
"""

import json
import logging

import pytest

from cayz_agent.config import _JsonFormatter


def _make_record(msg: str, level=logging.INFO, name="test.logger", extra: dict | None = None) -> logging.LogRecord:
    """构造 LogRecord"""
    record = logging.LogRecord(
        name=name,
        level=level,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=(),
        exc_info=None,
    )
    for k, v in (extra or {}).items():
        setattr(record, k, v)
    return record


class TestJsonFormatter:
    """测试 JSON 结构化日志格式器"""

    def test_basic_output_is_valid_json(self):
        """基本输出应是合法 JSON"""
        formatter = _JsonFormatter()
        record = _make_record("hello", level=logging.INFO)

        output = formatter.format(record)
        data = json.loads(output)

        assert data["level"] == "INFO"
        assert data["logger"] == "test.logger"
        assert data["message"] == "hello"
        assert "timestamp" in data

    def test_warning_level(self):
        """WARNING 级别应正确记录"""
        formatter = _JsonFormatter()
        record = _make_record("careful", level=logging.WARNING)

        data = json.loads(formatter.format(record))
        assert data["level"] == "WARNING"
        assert data["message"] == "careful"

    def test_error_level(self):
        """ERROR 级别应正确记录"""
        formatter = _JsonFormatter()
        record = _make_record("oops", level=logging.ERROR)

        data = json.loads(formatter.format(record))
        assert data["level"] == "ERROR"

    def test_extra_fields_merged(self):
        """extra 字段应合并到 JSON 输出"""
        formatter = _JsonFormatter()
        record = _make_record(
            "with extra",
            extra={"request_id": "abc-123", "user_id": 42},
        )

        data = json.loads(formatter.format(record))
        assert data["request_id"] == "abc-123"
        assert data["user_id"] == 42

    def test_exception_info_included(self):
        """异常信息应包含在输出中"""
        formatter = _JsonFormatter()
        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test.exc",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg="failed",
            args=(),
            exc_info=exc_info,
        )

        data = json.loads(formatter.format(record))
        assert "exc_info" in data
        assert "ValueError" in data["exc_info"]
        assert "boom" in data["exc_info"]

    def test_non_serializable_extra_coerced_to_string(self):
        """不可序列化的 extra 值应转为字符串"""
        formatter = _JsonFormatter()

        class CustomObj:
            def __str__(self):
                return "<custom>"

        record = _make_record("msg", extra={"obj": CustomObj()})

        data = json.loads(formatter.format(record))
        assert data["obj"] == "<custom>"

    def test_chinese_message_preserved(self):
        """中文消息应正确输出（ensure_ascii=False）"""
        formatter = _JsonFormatter()
        record = _make_record("你好，世界")

        output = formatter.format(record)
        assert "你好，世界" in output
        data = json.loads(output)
        assert data["message"] == "你好，世界"


class TestTextFormatter:
    """测试传统文本日志格式"""

    def test_text_format_contains_required_fields(self):
        """text 格式应包含时间、级别、logger 名、消息"""
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        record = _make_record("hello world", level=logging.INFO, name="test.text")

        output = formatter.format(record)

        assert "hello world" in output
        assert "INFO" in output
        assert "test.text" in output
        assert "|" in output

    def test_text_format_with_warning(self):
        """text 格式应正确记录 WARNING"""
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        record = _make_record("warning msg", level=logging.WARNING)

        output = formatter.format(record)
        assert "WARNING" in output
        assert "warning msg" in output


class TestSetupLogging:
    """测试 setup_logging 配置函数"""

    def test_json_format_creates_json_handler(self):
        """log_format=json 应注册 JSON formatter"""
        from cayz_agent.config import _JsonFormatter, setup_logging

        setup_logging(level="INFO", log_format="json")

        root = logging.getLogger()
        # 至少有一个 handler 使用 _JsonFormatter
        has_json = any(isinstance(h.formatter, _JsonFormatter) for h in root.handlers)
        assert has_json

    def test_text_format_creates_text_handler(self):
        """log_format=text 应注册文本 formatter（非 JSON）"""
        from cayz_agent.config import _JsonFormatter, setup_logging

        setup_logging(level="INFO", log_format="text")

        root = logging.getLogger()
        has_text = any(h.formatter is not None and not isinstance(h.formatter, _JsonFormatter) for h in root.handlers)
        assert has_text

    def test_log_level_applied(self):
        """日志级别应被正确应用"""
        from cayz_agent.config import setup_logging

        setup_logging(level="DEBUG", log_format="text")
        root = logging.getLogger()
        assert root.level == logging.DEBUG

        setup_logging(level="WARNING", log_format="text")
        assert root.level == logging.WARNING
