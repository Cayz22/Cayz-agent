"""
企业微信通知集成模块单元测试
"""
from unittest.mock import patch, MagicMock

import pytest

from cayz_agent.integrations.notify import WeChatNotifier


class TestWeChatNotifier:
    """测试企业微信通知客户端"""

    def test_no_webhook_url_returns_error(self):
        """未配置 Webhook URL 时返回错误"""
        notifier = WeChatNotifier(webhook_url="")
        result = notifier.send_text("测试消息")
        assert result["errcode"] == -1
        assert "未配置" in result["errmsg"]

    @patch("cayz_agent.integrations.notify._get_http_session")
    def test_send_text_success(self, mock_get_session):
        """发送文本消息成功"""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"errcode": 0, "errmsg": "ok"}
        mock_session.post.return_value = mock_response
        mock_get_session.return_value = mock_session

        notifier = WeChatNotifier(webhook_url="https://qyapi.weixin.qq.com/test")
        result = notifier.send_text("测试消息")

        assert result["errcode"] == 0
        mock_session.post.assert_called_once()
        # 验证请求参数
        call_args = mock_session.post.call_args
        payload = call_args.kwargs["json"]
        assert payload["msgtype"] == "text"
        assert payload["text"]["content"] == "测试消息"

    @patch("cayz_agent.integrations.notify._get_http_session")
    def test_send_text_with_mentioned_list(self, mock_get_session):
        """发送文本消息带 @ 功能"""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"errcode": 0, "errmsg": "ok"}
        mock_session.post.return_value = mock_response
        mock_get_session.return_value = mock_session

        notifier = WeChatNotifier(webhook_url="https://qyapi.weixin.qq.com/test")
        notifier.send_text("通知消息", mentioned_list=["@all"])

        call_args = mock_session.post.call_args
        payload = call_args.kwargs["json"]
        assert payload["text"]["mentioned_list"] == ["@all"]

    @patch("cayz_agent.integrations.notify._get_http_session")
    def test_send_markdown_success(self, mock_get_session):
        """发送 Markdown 消息成功"""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"errcode": 0, "errmsg": "ok"}
        mock_session.post.return_value = mock_response
        mock_get_session.return_value = mock_session

        notifier = WeChatNotifier(webhook_url="https://qyapi.weixin.qq.com/test")
        result = notifier.send_markdown("# 标题\n**内容**")

        assert result["errcode"] == 0
        call_args = mock_session.post.call_args
        payload = call_args.kwargs["json"]
        assert payload["msgtype"] == "markdown"
        assert "# 标题" in payload["markdown"]["content"]

    @patch("cayz_agent.integrations.notify._get_http_session")
    def test_send_failure_returns_error(self, mock_get_session):
        """发送失败时返回错误信息"""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"errcode": 93000, "errmsg": "invalid webhook url"}
        mock_session.post.return_value = mock_response
        mock_get_session.return_value = mock_session

        notifier = WeChatNotifier(webhook_url="https://invalid-url")
        result = notifier.send_text("测试")

        assert result["errcode"] == 93000
        assert "invalid" in result["errmsg"]


class TestP2WebhookSanitize:
    """P2 Webhook 内容审核：测试发送前自动脱敏"""

    def test_text_payload_sanitized(self):
        """text 类型消息中的敏感信息应被脱敏"""
        notifier = WeChatNotifier(webhook_url="https://qyapi.weixin.qq.com")
        payload = {
            "msgtype": "text",
            "text": {"content": "API Key: sk-abcdefghij1234567890xyz"},
        }
        sanitized = notifier._sanitize_payload(payload)
        assert "sk-abcdefghij1234567890xyz" not in sanitized["text"]["content"]
        assert "敏感信息已隐藏" in sanitized["text"]["content"]

    def test_markdown_payload_sanitized(self):
        """markdown 类型消息中的敏感信息应被脱敏"""
        notifier = WeChatNotifier(webhook_url="https://qyapi.weixin.qq.com")
        payload = {
            "msgtype": "markdown",
            "markdown": {"content": "密码: secret123456"},
        }
        sanitized = notifier._sanitize_payload(payload)
        assert "secret123456" not in sanitized["markdown"]["content"]

    def test_normal_text_unchanged(self):
        """无敏感信息的文本不应被修改"""
        notifier = WeChatNotifier(webhook_url="https://qyapi.weixin.qq.com")
        payload = {
            "msgtype": "text",
            "text": {"content": "今日报告：已完成3个任务"},
        }
        sanitized = notifier._sanitize_payload(payload)
        assert sanitized["text"]["content"] == "今日报告：已完成3个任务"

    @patch("cayz_agent.integrations.notify._get_http_session")
    def test_send_text_calls_sanitize(self, mock_get_session):
        """send_text 发送时应触发脱敏"""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"errcode": 0}
        mock_session.post.return_value = mock_response
        mock_get_session.return_value = mock_session

        notifier = WeChatNotifier(webhook_url="https://qyapi.weixin.qq.com")
        notifier.send_text("Bearer abcdefghijklmnopqrstuvwxyz12")

        # 检查发送的 payload 中敏感信息已脱敏
        sent_payload = mock_session.post.call_args.kwargs["json"]
        assert "abcdefghijklmnopqrstuvwxyz12" not in sent_payload["text"]["content"]
