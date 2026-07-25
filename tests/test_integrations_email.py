"""
邮件发送集成模块单元测试
"""
from unittest.mock import patch, MagicMock

import pytest

from cayz_agent.integrations.email_sender import EmailSender
from cayz_agent.exceptions import EmailError


class TestEmailSender:
    """测试邮件发送客户端"""

    def test_not_configured_returns_error(self):
        """未配置 SMTP 时返回错误"""
        sender = EmailSender(smtp_host="", username="", password="")
        result = sender.send(["test@example.com"], "测试", "内容")
        assert result["success"] is False
        assert "未配置" in result["error"]

    def test_empty_recipients_returns_error(self):
        """收件人为空时返回错误"""
        sender = EmailSender(
            smtp_host="smtp.qq.com",
            username="user@qq.com",
            password="password",
        )
        result = sender.send([], "测试", "内容")
        assert result["success"] is False
        assert "空" in result["error"]

    @patch("cayz_agent.integrations.email_sender.smtplib.SMTP_SSL")
    def test_send_plain_text_success(self, mock_smtp):
        """发送纯文本邮件成功"""
        mock_server = MagicMock()
        mock_smtp.return_value = mock_server

        sender = EmailSender(
            smtp_host="smtp.qq.com",
            smtp_port=465,
            username="user@qq.com",
            password="password",
            from_addr="user@qq.com",
        )
        result = sender.send(
            to_addrs=["recipient@example.com"],
            subject="测试邮件",
            body="这是一封测试邮件",
        )

        assert result["success"] is True
        assert result["to"] == ["recipient@example.com"]
        assert result["subject"] == "测试邮件"
        mock_server.login.assert_called_once_with("user@qq.com", "password")
        mock_server.sendmail.assert_called_once()

    @patch("cayz_agent.integrations.email_sender.smtplib.SMTP_SSL")
    def test_send_html_email(self, mock_smtp):
        """发送 HTML 格式邮件"""
        mock_server = MagicMock()
        mock_smtp.return_value = mock_server

        sender = EmailSender(
            smtp_host="smtp.qq.com",
            username="user@qq.com",
            password="password",
        )
        result = sender.send(
            to_addrs=["recipient@example.com"],
            subject="HTML邮件",
            body="<h1>标题</h1><p>内容</p>",
            html=True,
        )

        assert result["success"] is True
        # 验证发送的邮件内容为 HTML 格式（base64 编码）
        sent_email = mock_server.sendmail.call_args[0][2]
        assert "text/html" in sent_email

    @patch("cayz_agent.integrations.email_sender.smtplib.SMTP_SSL")
    def test_send_multiple_recipients(self, mock_smtp):
        """发送给多个收件人"""
        mock_server = MagicMock()
        mock_smtp.return_value = mock_server

        sender = EmailSender(
            smtp_host="smtp.qq.com",
            username="user@qq.com",
            password="password",
        )
        result = sender.send(
            to_addrs=["a@example.com", "b@example.com"],
            subject="多收件人",
            body="内容",
        )

        assert result["success"] is True
        assert len(result["to"]) == 2

    @patch("cayz_agent.integrations.email_sender.smtplib.SMTP_SSL")
    def test_send_failure_returns_error(self, mock_smtp):
        """SMTP 连接失败时 send() 应返回错误 dict（保持 API 契约，retry 在内部 _send_via_smtp 上生效）"""
        mock_smtp.side_effect = Exception("连接超时")

        sender = EmailSender(
            smtp_host="smtp.qq.com",
            username="user@qq.com",
            password="password",
        )
        result = sender.send(["test@example.com"], "测试", "内容")

        assert result["success"] is False
        assert "连接超时" in result["error"]

    @patch("cayz_agent.integrations.email_sender.smtplib.SMTP")
    def test_send_without_ssl(self, mock_smtp):
        """不使用 SSL 时使用 SMTP + starttls（patch 必须针对 smtplib.SMTP 而非 SMTP_SSL）"""
        mock_server = MagicMock()
        mock_smtp.return_value = mock_server

        sender = EmailSender(
            smtp_host="smtp.gmail.com",
            smtp_port=587,
            username="user@gmail.com",
            password="password",
            use_ssl=False,
        )
        result = sender.send(["test@example.com"], "测试", "内容")

        assert result["success"] is True
        mock_server.starttls.assert_called_once()

    def test_from_addr_defaults_to_username(self):
        """from_addr 默认使用 username"""
        sender = EmailSender(
            smtp_host="smtp.qq.com",
            username="user@qq.com",
            password="password",
        )
        assert sender.from_addr == "user@qq.com"


class TestEmailSenderRetry:
    """测试邮件发送的 retry 行为（修复 P0：原 send() 上的 @retry 因异常被吞而失效）"""

    @patch("cayz_agent.integrations.email_sender.smtplib.SMTP_SSL")
    def test_send_via_smtp_retries_on_smtp_exception(self, mock_smtp):
        """_send_via_smtp 应在 SMTPException 时重试，最终成功后 send() 返回 success=True

        修复 P0：原 send() 上的 @retry 因 SMTPException 被 except 捕获并 return dict 而失效。
        现在 _send_via_smtp 抛出 EmailError，retry 真正生效。
        注：max_attempts=2 + min_wait=1.0，测试会等待约 1 秒。
        """
        import smtplib

        # 第 1 次抛 SMTPException，第 2 次成功（max_attempts=2 允许 2 次尝试）
        mock_server = MagicMock()
        mock_smtp.side_effect = [
            smtplib.SMTPException("瞬时拥堵"),
            mock_server,
        ]

        sender = EmailSender(
            smtp_host="smtp.qq.com",
            username="user@qq.com",
            password="password",
        )
        result = sender.send(["test@example.com"], "测试", "内容")

        assert result["success"] is True
        # 应该被调用 2 次（1 次失败 + 1 次成功），证明 retry 生效
        assert mock_smtp.call_count == 2

    @patch("cayz_agent.integrations.email_sender.smtplib.SMTP_SSL")
    def test_send_via_smtp_reraises_email_error_after_max_attempts(self, mock_smtp):
        """_send_via_smtp 重试用尽后应 raise EmailError，send() 兜底返回 dict

        注：max_attempts=2 + min_wait=1.0，测试会等待约 1 秒。
        """
        import smtplib

        mock_smtp.side_effect = smtplib.SMTPException("持续失败")

        sender = EmailSender(
            smtp_host="smtp.qq.com",
            username="user@qq.com",
            password="password",
        )

        # send() 应捕获 EmailError 返回 dict（保持 API 契约）
        result = sender.send(["test@example.com"], "测试", "内容")
        assert result["success"] is False
        assert "SMTP 发送失败" in result["error"]
        # max_attempts=2，应调用 2 次（1 次原始 + 1 次重试）
        assert mock_smtp.call_count == 2

    @patch("cayz_agent.integrations.email_sender.smtplib.SMTP_SSL")
    def test_send_via_smtp_wraps_unknown_exception_as_email_error(self, mock_smtp):
        """_send_via_smtp 应将未知异常包装为 EmailError 以触发 retry

        注：max_attempts=2 + min_wait=1.0，测试会等待约 1 秒。
        """
        mock_smtp.side_effect = RuntimeError("网络层异常")

        sender = EmailSender(
            smtp_host="smtp.qq.com",
            username="user@qq.com",
            password="password",
        )

        # send() 应捕获 EmailError 返回 dict（包装后的错误信息）
        result = sender.send(["test@example.com"], "测试", "内容")
        assert result["success"] is False
        assert "网络层异常" in result["error"]
        # max_attempts=2，应调用 2 次（证明 retry 生效）
        assert mock_smtp.call_count == 2


class TestP2SMTPWhitelist:
    """P2 SMTP 白名单：测试收件人域名白名单校验"""

    @patch("cayz_agent.config.get_settings")
    def test_whitelist_blocks_external_domain(self, mock_settings):
        """白名单配置后，外部域名收件人应被拒绝"""
        mock_settings.return_value.smtp_allowed_domains = "company.cn, internal.com"
        sender = EmailSender(
            smtp_host="smtp.qq.com",
            username="user@company.cn",
            password="password",
        )
        result = sender.send(["someone@gmail.com"], "主题", "内容")
        assert result["success"] is False
        assert "白名单" in result["error"]

    @patch("cayz_agent.config.get_settings")
    def test_whitelist_allows_internal_domain(self, mock_settings):
        """白名单配置后，内部域名收件人应放行（进入 SMTP 发送）"""
        mock_settings.return_value.smtp_allowed_domains = "company.cn"
        with patch("cayz_agent.integrations.email_sender.smtplib.SMTP_SSL") as mock_smtp:
            mock_smtp.return_value = MagicMock()
            sender = EmailSender(
                smtp_host="smtp.qq.com",
                username="user@company.cn",
                password="password",
            )
            result = sender.send(["colleague@company.cn"], "主题", "内容")
        assert result["success"] is True

    @patch("cayz_agent.config.get_settings")
    def test_no_whitelist_allows_all(self, mock_settings):
        """白名单为空时，所有域名应放行"""
        mock_settings.return_value.smtp_allowed_domains = ""
        sender = EmailSender(
            smtp_host="smtp.qq.com",
            username="user@company.cn",
            password="password",
        )
        with patch("cayz_agent.integrations.email_sender.smtplib.SMTP_SSL") as mock_smtp:
            mock_smtp.return_value = MagicMock()
            result = sender.send(["anyone@gmail.com"], "主题", "内容")
        assert result["success"] is True
