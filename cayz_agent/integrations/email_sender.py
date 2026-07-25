"""
邮件发送集成

通过 SMTP 协议发送邮件，支持 HTML 格式和附件。
适用于：报告发送、告警通知、客户跟进等场景。

使用标准库 smtplib + email，无需额外依赖。
"""

import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from ..config import get_settings
from ..exceptions import EmailError
from ..retry import retry_on_error

logger = logging.getLogger(__name__)


class EmailSender:
    """
    邮件发送客户端

    支持：
    - 纯文本和 HTML 格式
    - 多收件人
    - SMTP 认证
    """

    def __init__(
        self,
        smtp_host: str = "",
        smtp_port: int = 465,
        username: str = "",
        password: str = "",
        from_addr: str = "",
        use_ssl: bool = True,
    ):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_addr = from_addr or username
        self.use_ssl = use_ssl

    def _is_configured(self) -> bool:
        """检查是否已配置 SMTP"""
        return bool(self.smtp_host and self.username and self.password)

    @staticmethod
    def _get_allowed_domains() -> set[str]:
        """P2 SMTP 白名单：从配置读取允许的收件人域名集合。

        返回空集合表示不限制（仅开发环境推荐）。
        """
        from ..config import get_settings

        raw = get_settings().smtp_allowed_domains
        return {d.strip().lower() for d in raw.split(",") if d.strip()}

    @staticmethod
    def _is_addr_allowed(addr: str, allowed_domains: set[str]) -> bool:
        """检查收件人地址域名是否在白名单中（大小写不敏感）"""
        if "@" not in addr:
            return False
        domain = addr.rsplit("@", 1)[1].strip().lower()
        return domain in allowed_domains

    def send(
        self,
        to_addrs: list[str],
        subject: str,
        body: str,
        html: bool = False,
    ) -> dict:
        """
        发送邮件

        Args:
            to_addrs: 收件人邮箱列表
            subject: 邮件主题
            body: 邮件内容
            html: 是否为 HTML 格式

        Returns:
            发送结果字典（success=True 表示成功，False 时附带 error 字段）
        """
        if not self._is_configured():
            logger.warning("SMTP 未配置，邮件未发送")
            return {"success": False, "error": "SMTP 未配置"}

        if not to_addrs:
            return {"success": False, "error": "收件人为空"}

        # P2 SMTP 白名单校验：防止邮件外发到未授权域名导致数据泄露
        allowed_domains = self._get_allowed_domains()
        if allowed_domains:
            blocked = [a for a in to_addrs if not self._is_addr_allowed(a, allowed_domains)]
            if blocked:
                logger.warning("收件人域名不在白名单，已拒绝发送: %s", blocked)
                return {
                    "success": False,
                    "error": f"收件人域名不在白名单: {blocked}（允许: {sorted(allowed_domains)}）",
                }

        logger.info("发送邮件: to=%s, subject=%s", to_addrs, subject)

        # 构建邮件
        msg = MIMEMultipart("alternative")
        msg["From"] = self.from_addr
        msg["To"] = ", ".join(to_addrs)
        msg["Subject"] = subject

        content_type = "html" if html else "plain"
        msg.attach(MIMEText(body, content_type, "utf-8"))

        # 调用带重试的内部 SMTP 发送方法；重试用尽后抛出 EmailError
        try:
            self._send_via_smtp(to_addrs, msg.as_string())
        except EmailError as e:
            logger.error("%s", e)
            return {"success": False, "error": e.message}

        logger.info("邮件发送成功: to=%s", to_addrs)
        return {"success": True, "to": to_addrs, "subject": subject}

    @retry_on_error(max_attempts=2, min_wait=1.0, max_wait=4.0)
    def _send_via_smtp(self, to_addrs: list[str], raw_email: str) -> None:
        """
        实际执行 SMTP 发送。

        瞬时失败（连接超时、SMTP 临时错误）会抛出 retryable=True 的 EmailError，
        由上层 @retry_on_error 装饰器捕获并指数退避重试。
        永久性失败（认证错误、收件人被拒）抛出 retryable=False 的 EmailError，
        装饰器不会重试，直接 reraise 由 send() 兜底捕获返回 dict。

        P2-1 修复：使用 try/finally 保证 SMTP 连接在任何情况下都关闭，
        避免异常路径下 TCP/TLS 连接泄漏。
        P2-12 修复：区分永久性与瞬时性 SMTP 错误，永久性错误不重试。
        """
        # P2-12：永久性 SMTP 错误，重试无意义且浪费时间
        _SMTP_PERMANENT = (
            smtplib.SMTPAuthenticationError,
            smtplib.SMTPSenderRefused,
            smtplib.SMTPRecipientsRefused,
            smtplib.SMTPHeloError,
        )

        server = None
        try:
            if self.use_ssl:
                server = smtplib.SMTP_SSL(self.smtp_host, self.smtp_port, timeout=15)
            else:
                server = smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=15)
                server.starttls()

            server.login(self.username, self.password)
            server.sendmail(self.from_addr, to_addrs, raw_email)
        except _SMTP_PERMANENT as e:
            # P2-12：永久性错误标记为不可重试，避免凭据错误时无谓重试
            raise EmailError(
                f"SMTP 永久性错误（不重试）: {e}",
                cause=e,
                retryable=False,
            ) from e
        except smtplib.SMTPException as e:
            # 瞬时性 SMTP 错误（如服务暂时不可用），可重试
            raise EmailError(f"SMTP 发送失败: {e}", cause=e, retryable=True) from e
        except (ConnectionError, TimeoutError, OSError):
            # 网络层瞬时错误，直接 raise（已在 RETRYABLE_EXCEPTIONS 中）
            raise
        except Exception as e:
            # 其他未知错误包装为 EmailError，便于 retry 与上层统一处理
            raise EmailError(f"邮件发送异常: {e}", cause=e, retryable=True) from e
        finally:
            # P2-1：无论成功还是异常，都关闭 SMTP 连接，防止 TCP/TLS 连接泄漏
            if server is not None:
                try:
                    server.quit()
                except Exception:
                    try:
                        server.close()
                    except Exception:
                        pass  # 连接已断开，忽略关闭异常


# 全局单例
_email_sender: EmailSender | None = None


def get_email_sender() -> EmailSender:
    """获取邮件发送客户端单例"""
    global _email_sender
    if _email_sender is None:
        settings = get_settings()
        _email_sender = EmailSender(
            smtp_host=settings.smtp_host,
            smtp_port=settings.smtp_port,
            username=settings.smtp_username,
            password=settings.smtp_password,
            from_addr=settings.email_from_addr,
            use_ssl=settings.smtp_use_ssl,
        )
    return _email_sender
