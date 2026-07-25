"""
业务系统集成模块

提供与企业系统的对接能力：
- CRM 集成：客户信息查询、订单跟踪
- 企业通知：企业微信 Webhook 消息推送
- 邮件通知：SMTP 邮件发送
"""
from .crm import CRMClient, get_crm_client
from .notify import WeChatNotifier, get_notifier
from .email_sender import EmailSender, get_email_sender

__all__ = [
    "CRMClient",
    "get_crm_client",
    "WeChatNotifier",
    "get_notifier",
    "EmailSender",
    "get_email_sender",
]
