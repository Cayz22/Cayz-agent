"""
企业通知集成

通过企业微信群机器人 Webhook 推送消息到指定群聊。
适用于：告警通知、任务完成提醒、审批提醒等场景。

使用方式：
1. 在企业微信群中添加机器人，获取 Webhook URL
2. 将 URL 配置到 .env 的 WECOM_WEBHOOK_URL 中
3. 调用 notifier.send_text() 或 notifier.send_markdown()
"""

import logging
import threading

import requests

from ..config import get_settings
from ..exceptions import NotifyError
from ..retry import retry_on_error

logger = logging.getLogger(__name__)

# P1 性能：模块级 requests.Session 复用 TCP 连接 + TLS 会话，
# 避免每次 send 都新建连接（含 DNS 解析 + TCP 握手 + TLS 协商）
# Session 线程安全（requests 文档明确说明），可被多线程并发使用
_http_session: requests.Session | None = None
_http_session_lock = threading.Lock()


def _get_http_session() -> requests.Session:
    """获取模块级 requests.Session 单例（线程安全）"""
    global _http_session
    if _http_session is None:
        with _http_session_lock:
            if _http_session is None:
                _http_session = requests.Session()
                # 配置连接池上限（默认 10 偏低）
                adapter = requests.adapters.HTTPAdapter(
                    pool_connections=10,
                    pool_maxsize=20,
                    max_retries=0,  # 重试由 retry_on_error 装饰器控制
                )
                _http_session.mount("http://", adapter)
                _http_session.mount("https://", adapter)
    return _http_session


class WeChatNotifier:
    """
    企业微信通知客户端

    支持消息类型：文本、Markdown
    文档：https://developer.work.weixin.qq.com/document/path/91770
    """

    def __init__(self, webhook_url: str = ""):
        self.webhook_url = webhook_url

    def _send_payload(self, payload: dict) -> dict:
        """发送 Webhook 请求"""
        if not self.webhook_url:
            logger.warning("企业微信 Webhook URL 未配置，消息未发送")
            return {"errcode": -1, "errmsg": "webhook URL 未配置"}

        # P2 Webhook 内容审核：发送前对消息内容脱敏，防止敏感信息外发到群聊
        payload = self._sanitize_payload(payload)

        logger.info("发送企业微信通知")
        try:
            # P1 性能：复用模块级 Session，避免每次新建 TCP+TLS 连接
            response = _get_http_session().post(
                self.webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            result = response.json()
        except requests.RequestException as e:
            # P2-1：网络瞬时错误，显式标记 retryable=True 触发重试
            raise NotifyError("企业微信 Webhook 请求失败", cause=e, retryable=True) from e
        except ValueError as e:
            raise NotifyError("企业微信响应解析失败", cause=e) from e

        if result.get("errcode") == 0:
            logger.info("企业微信通知发送成功")
        else:
            logger.warning("企业微信通知发送失败: %s", result.get("errmsg", "未知错误"))

        return result

    @staticmethod
    def _sanitize_payload(payload: dict) -> dict:
        """P2 Webhook 内容审核：对 payload 中的消息内容自动脱敏。

        对 text/markdown 类型消息的 content 字段调用 sanitize_text，
        防止 API Key / 密码 / 手机号等敏感信息外发到企业微信群聊。
        """
        from ..sanitizers import sanitize_text

        if payload.get("msgtype") == "text" and "text" in payload:
            content = payload["text"].get("content", "")
            if content:
                payload["text"]["content"] = sanitize_text(content)
        elif payload.get("msgtype") == "markdown" and "markdown" in payload:
            content = payload["markdown"].get("content", "")
            if content:
                payload["markdown"]["content"] = sanitize_text(content)
        return payload

    @retry_on_error(max_attempts=2, min_wait=1.0, max_wait=4.0)
    def send_text(self, content: str, mentioned_list: list[str] | None = None) -> dict:
        """
        发送文本消息

        Args:
            content: 消息内容
            mentioned_list: 需要 @ 的人的企业微信账号（手机号或UserID），"@all" 表示 @所有人
        """
        payload = {
            "msgtype": "text",
            "text": {
                "content": content,
                "mentioned_list": mentioned_list or [],
            },
        }
        return self._send_payload(payload)

    @retry_on_error(max_attempts=2, min_wait=1.0, max_wait=4.0)
    def send_markdown(self, content: str) -> dict:
        """
        发送 Markdown 消息

        Args:
            content: Markdown 格式的消息内容
        """
        payload = {
            "msgtype": "markdown",
            "markdown": {"content": content},
        }
        return self._send_payload(payload)


# 全局单例
_notifier: WeChatNotifier | None = None


def get_notifier() -> WeChatNotifier:
    """获取企业微信通知客户端单例"""
    global _notifier
    if _notifier is None:
        settings = get_settings()
        _notifier = WeChatNotifier(webhook_url=settings.wecom_webhook_url)
    return _notifier
