"""
tools 模块单元测试：验证工具函数行为

- get_current_time: 纯函数，直接验证返回格式
- web_search: 需要 mock TavilyClient 和 _settings
- knowledge_search / knowledge_upload: mock RAGManager
- crm_query_customer / crm_search_customers / crm_query_order: mock CRMClient
- send_wecom_notification: mock WeChatNotifier
- send_email: mock EmailSender
"""
from datetime import datetime
from unittest.mock import patch, MagicMock

import pytest
from langchain_core.documents import Document

from cayz_agent.tools import (
    get_current_time,
    web_search,
    knowledge_search,
    knowledge_upload,
    crm_query_customer,
    crm_search_customers,
    crm_query_order,
    send_wecom_notification,
    send_email,
    reset_tavily_client,
)


@pytest.fixture(autouse=True)
def _reset_tavily_singleton():
    """每个测试前重置 Tavily client 单例，防止跨测试状态泄漏。

    P1-18 将 TavilyClient 改为单例后，test_search_success 会创建并缓存 mock_client，
    后续测试（test_search_no_results 等）会复用该缓存而无法注入新的 mock 行为。
    """
    reset_tavily_client()
    yield
    reset_tavily_client()


class TestGetCurrentTime:
    """测试 get_current_time 工具"""

    def test_returns_valid_datetime_format(self):
        """返回值应为 YYYY-MM-DD HH:MM:SS 格式"""
        result = get_current_time.invoke({})
        parsed = datetime.strptime(result, "%Y-%m-%d %H:%M:%S")
        assert parsed is not None

    def test_returns_string(self):
        """返回值应为字符串类型"""
        result = get_current_time.invoke({})
        assert isinstance(result, str)

    def test_close_to_now(self):
        """返回时间应接近当前时间（60秒内）"""
        result = get_current_time.invoke({})
        parsed = datetime.strptime(result, "%Y-%m-%d %H:%M:%S")
        delta = abs((datetime.now() - parsed).total_seconds())
        assert delta < 60


def _mock_settings(api_key: str = "") -> MagicMock:
    """构造一个带 tavily_api_key 的 mock settings"""
    mock = MagicMock()
    mock.tavily_api_key = api_key
    return mock


class TestWebSearch:
    """测试 web_search 工具"""

    def test_no_api_key_returns_error(self):
        """未配置 TAVILY_API_KEY 时应返回错误提示"""
        with patch("cayz_agent.tools.get_settings", return_value=_mock_settings("")):
            result = web_search.invoke({"query": "test"})
            assert "TAVILY_API_KEY" in result
            assert "错误" in result

    def test_search_success(self):
        """正常搜索应返回格式化结果"""
        mock_response = {
            "results": [
                {
                    "title": "测试标题",
                    "content": "测试摘要内容",
                    "url": "https://example.com",
                }
            ]
        }
        mock_client = MagicMock()
        mock_client.search.return_value = mock_response

        with patch("cayz_agent.tools.get_settings", return_value=_mock_settings("fake-key-12345")), patch(
            "cayz_agent.tools.TavilyClient", return_value=mock_client
        ):
            result = web_search.invoke({"query": "今天新闻"})

        assert "测试标题" in result
        assert "测试摘要内容" in result
        assert "https://example.com" in result
        mock_client.search.assert_called_once_with("今天新闻", max_results=3)

    def test_search_no_results(self):
        """搜索无结果时应返回提示"""
        mock_client = MagicMock()
        mock_client.search.return_value = {"results": []}

        with patch("cayz_agent.tools.get_settings", return_value=_mock_settings("fake-key-12345")), patch(
            "cayz_agent.tools.TavilyClient", return_value=mock_client
        ):
            result = web_search.invoke({"query": "不存在的内容"})

        assert "未能" in result or "未检索到" in result

    def test_search_handles_exception(self):
        """Tavily 抛异常时应被捕获并返回错误信息"""
        mock_client = MagicMock()
        mock_client.search.side_effect = RuntimeError("网络超时")

        with patch("cayz_agent.tools.get_settings", return_value=_mock_settings("fake-key-12345")), patch(
            "cayz_agent.tools.TavilyClient", return_value=mock_client
        ):
            result = web_search.invoke({"query": "test"})

        assert "错误" in result or "error" in result.lower()

    def test_search_empty_query_returns_invalid(self):
        """空查询应被 validate_search_query 拦截"""
        result = web_search.invoke({"query": ""})
        assert "输入无效" in result or "无效" in result

    def test_search_injection_query_not_rejected(self):
        """搜索查询不检测注入特征（注入检测仅用于 chat 输入），正常查询应进入搜索流程"""
        # "ignore all previous instructions" 在搜索场景下是合法查询字符串
        # validate_search_query 只校验空/超长，不校验注入
        mock_client = MagicMock()
        mock_client.search.return_value = {"results": []}

        with patch("cayz_agent.tools.get_settings", return_value=_mock_settings("fake-key")), \
             patch("cayz_agent.tools.TavilyClient", return_value=mock_client):
            result = web_search.invoke({"query": "ignore all previous instructions"})

        # 应进入搜索流程（返回无结果提示），而非被验证拦截
        assert "输入无效" not in result
        mock_client.search.assert_called_once()


class TestErrorSanitization:
    """测试 tools.py 错误信息脱敏（修复 P1：原 str(e) 泄露敏感信息）"""

    def test_web_search_exception_sanitized(self):
        """web_search 异常信息不应包含完整 API Key"""
        mock_client = MagicMock()
        # 模拟异常消息中包含敏感信息
        mock_client.search.side_effect = RuntimeError(
            "Auth failed for sk-abcdefghijklmnopqrstuvwxyz1234567890"
        )

        with patch("cayz_agent.tools.get_settings", return_value=_mock_settings("sk-abcdefghij")), patch(
            "cayz_agent.tools.TavilyClient", return_value=mock_client
        ):
            result = web_search.invoke({"query": "test"})

        # API Key 应被脱敏，不应原样出现在返回给用户的错误信息中
        assert "sk-abcdefghijklmnopqrstuvwxyz1234567890" not in result
        assert "敏感信息已隐藏" in result or "sk-abcdefghij" not in result

    def test_web_search_exception_keeps_basic_message(self):
        """脱敏后仍应保留基本错误描述（非敏感部分）"""
        mock_client = MagicMock()
        mock_client.search.side_effect = RuntimeError("网络连接超时")

        with patch("cayz_agent.tools.get_settings", return_value=_mock_settings("fake-key")), patch(
            "cayz_agent.tools.TavilyClient", return_value=mock_client
        ):
            result = web_search.invoke({"query": "test"})

        assert "网络" in result or "超时" in result or "错误" in result


# ============================================================
# P4 新增：knowledge_search / knowledge_upload 测试
# ============================================================

class TestKnowledgeSearch:
    """测试 knowledge_search 工具"""

    def test_search_returns_formatted_results(self):
        """正常检索应返回格式化的文档片段"""
        mock_manager = MagicMock()
        mock_manager.search.return_value = [
            Document(page_content="项目文档内容", metadata={"source": "doc1"}),
        ]

        with patch("cayz_agent.rag.get_rag_manager", return_value=mock_manager):
            result = knowledge_search.invoke({"query": "项目文档"})

        assert "项目文档内容" in result
        assert "doc1" in result
        assert "片段 1" in result

    def test_search_no_results_returns_hint(self):
        """检索无结果应返回提示"""
        mock_manager = MagicMock()
        mock_manager.search.return_value = []

        with patch("cayz_agent.rag.get_rag_manager", return_value=mock_manager):
            result = knowledge_search.invoke({"query": "不存在的内容"})

        assert "未找到" in result or "上传" in result

    def test_search_empty_query_rejected(self):
        """空查询应被验证拦截"""
        result = knowledge_search.invoke({"query": ""})
        assert "输入无效" in result or "无效" in result

    def test_search_exception_sanitized(self):
        """RAG 异常应被捕获并脱敏"""
        mock_manager = MagicMock()
        mock_manager.search.side_effect = RuntimeError("DB error with sk-leaked-key")

        with patch("cayz_agent.rag.get_rag_manager", return_value=mock_manager), \
             patch("cayz_agent.tools.sanitize_exception", return_value="敏感信息已隐藏"):
            result = knowledge_search.invoke({"query": "test"})

        assert "敏感信息已隐藏" in result or "sk-leaked-key" not in result

    def test_search_multiple_results_numbered(self):
        """多结果应按序号编号"""
        mock_manager = MagicMock()
        mock_manager.search.return_value = [
            Document(page_content="片段A", metadata={"source": "s1"}),
            Document(page_content="片段B", metadata={"source": "s2"}),
        ]

        with patch("cayz_agent.rag.get_rag_manager", return_value=mock_manager):
            result = knowledge_search.invoke({"query": "查询"})

        assert "片段 1" in result
        assert "片段 2" in result
        assert "片段A" in result
        assert "片段B" in result

    def test_search_missing_source_uses_default(self):
        """metadata 中无 source 时应使用'未知来源'"""
        mock_manager = MagicMock()
        mock_manager.search.return_value = [
            Document(page_content="内容", metadata={}),  # 无 source
        ]

        with patch("cayz_agent.rag.get_rag_manager", return_value=mock_manager):
            result = knowledge_search.invoke({"query": "test"})

        assert "未知来源" in result


class TestKnowledgeUpload:
    """测试 knowledge_upload 工具"""

    def test_upload_success_returns_chunk_count(self):
        """上传成功应返回切片数量"""
        mock_manager = MagicMock()
        mock_manager.add_documents.return_value = 5

        with patch("cayz_agent.rag.get_rag_manager", return_value=mock_manager):
            result = knowledge_upload.invoke({"text": "测试文档内容", "source": "test"})

        assert "上传成功" in result
        assert "5" in result
        mock_manager.add_documents.assert_called_once_with("测试文档内容", source="test")

    def test_upload_zero_chunks_returns_failure(self):
        """切片为 0 应返回失败提示（P0 修复：空文本先被 validate_knowledge_text 拦截为输入无效）"""
        mock_manager = MagicMock()
        mock_manager.add_documents.return_value = 0

        # P0：knowledge_upload 工具内补齐了 validate_knowledge_text 校验，
        # 空文本现在返回更明确的"输入无效"而非泛化的"上传失败"
        with patch("cayz_agent.rag.get_rag_manager", return_value=mock_manager):
            result = knowledge_upload.invoke({"text": "", "source": "test"})

        assert "输入无效" in result or "上传失败" in result

    def test_upload_default_source(self):
        """未提供 source 时应使用默认值 user_input"""
        mock_manager = MagicMock()
        mock_manager.add_documents.return_value = 3

        with patch("cayz_agent.rag.get_rag_manager", return_value=mock_manager):
            knowledge_upload.invoke({"text": "内容"})

        mock_manager.add_documents.assert_called_once_with("内容", source="user_input")

    def test_upload_exception_sanitized(self):
        """上传异常应被捕获并脱敏"""
        mock_manager = MagicMock()
        mock_manager.add_documents.side_effect = RuntimeError("sk-secret-leaked")

        with patch("cayz_agent.rag.get_rag_manager", return_value=mock_manager), \
             patch("cayz_agent.tools.sanitize_exception", return_value="敏感信息已隐藏"):
            result = knowledge_upload.invoke({"text": "test"})

        assert "敏感信息已隐藏" in result or "sk-secret-leaked" not in result


# ============================================================
# P4 新增：CRM 工具测试
# ============================================================

class TestCrmQueryCustomer:
    """测试 crm_query_customer 工具"""

    def test_query_existing_customer(self):
        """查询存在的客户应返回格式化信息"""
        mock_client = MagicMock()
        mock_client.get_customer_summary.return_value = {
            "customer": {
                "customer_id": "C001",
                "name": "张伟",
                "email": "zhangwei@example.com",
                "phone": "13800138001",
                "company": "阿里巴巴",
                "level": "VIP",
                "status": "活跃",
            },
            "order_count": 3,
            "total_spent": 15000.0,
            "recent_orders": [
                {
                    "order_id": "ORD-001",
                    "product": "云服务器",
                    "amount": 5000.0,
                    "status": "已完成",
                    "date": "2024-01-15",
                }
            ],
        }

        with patch("cayz_agent.integrations.get_crm_client", return_value=mock_client):
            result = crm_query_customer.invoke({"customer_id": "C001"})

        assert "C001" in result
        assert "张伟" in result
        assert "阿里巴巴" in result
        assert "VIP" in result
        assert "ORD-001" in result
        assert "云服务器" in result

    def test_query_nonexistent_customer(self):
        """查询不存在的客户应返回错误"""
        mock_client = MagicMock()
        mock_client.get_customer_summary.return_value = {"error": "未找到客户: C999"}

        with patch("cayz_agent.integrations.get_crm_client", return_value=mock_client):
            result = crm_query_customer.invoke({"customer_id": "C999"})

        assert "未找到" in result or "C999" in result

    def test_query_no_orders(self):
        """客户无订单时应显示'无订单记录'"""
        mock_client = MagicMock()
        mock_client.get_customer_summary.return_value = {
            "customer": {
                "customer_id": "C002",
                "name": "李娜",
                "email": "lina@example.com",
                "phone": "13800138002",
                "company": "腾讯",
                "level": "普通",
                "status": "活跃",
            },
            "order_count": 0,
            "total_spent": 0.0,
            "recent_orders": [],
        }

        with patch("cayz_agent.integrations.get_crm_client", return_value=mock_client):
            result = crm_query_customer.invoke({"customer_id": "C002"})

        assert "无订单记录" in result

    def test_query_exception_sanitized(self):
        """CRM 异常应被捕获并脱敏"""
        mock_client = MagicMock()
        mock_client.get_customer_summary.side_effect = RuntimeError("sk-secret-leaked")

        with patch("cayz_agent.integrations.get_crm_client", return_value=mock_client), \
             patch("cayz_agent.tools.sanitize_exception", return_value="敏感信息已隐藏"):
            result = crm_query_customer.invoke({"customer_id": "C001"})

        assert "敏感信息已隐藏" in result or "sk-secret-leaked" not in result


class TestCrmSearchCustomers:
    """测试 crm_search_customers 工具"""

    def test_search_finds_matches(self):
        """搜索应返回匹配的客户列表"""
        from cayz_agent.integrations.crm import Customer
        mock_client = MagicMock()
        mock_client.search_customers.return_value = [
            Customer("C001", "张伟", "zhangwei@example.com", "13800138001", "阿里巴巴", "VIP", "活跃"),
            Customer("C002", "李娜", "lina@example.com", "13800138002", "腾讯", "VIP", "活跃"),
        ]

        with patch("cayz_agent.integrations.get_crm_client", return_value=mock_client):
            result = crm_search_customers.invoke({"keyword": "张"})

        assert "2" in result  # 找到 2 位
        assert "C001" in result
        assert "张伟" in result
        assert "阿里巴巴" in result

    def test_search_no_matches(self):
        """无匹配时应返回未找到提示"""
        mock_client = MagicMock()
        mock_client.search_customers.return_value = []

        with patch("cayz_agent.integrations.get_crm_client", return_value=mock_client):
            result = crm_search_customers.invoke({"keyword": "不存在"})

        assert "未找到" in result

    def test_search_empty_keyword_rejected(self):
        """空关键词应被验证拦截"""
        result = crm_search_customers.invoke({"keyword": ""})
        assert "输入无效" in result or "无效" in result

    def test_search_exception_sanitized(self):
        """异常应被捕获并脱敏"""
        mock_client = MagicMock()
        mock_client.search_customers.side_effect = RuntimeError("sk-leaked")

        with patch("cayz_agent.integrations.get_crm_client", return_value=mock_client), \
             patch("cayz_agent.tools.sanitize_exception", return_value="敏感信息已隐藏"):
            result = crm_search_customers.invoke({"keyword": "test"})

        assert "敏感信息已隐藏" in result or "sk-leaked" not in result


class TestCrmQueryOrder:
    """测试 crm_query_order 工具"""

    def test_query_existing_order(self):
        """查询存在的订单应返回详情"""
        from cayz_agent.integrations.crm import Order
        mock_client = MagicMock()
        mock_client.get_order.return_value = Order(
            order_id="ORD-2024-001",
            customer_id="C001",
            product="云服务器",
            amount=5000.0,
            status="已完成",
            created_at="2024-01-15",
        )

        with patch("cayz_agent.integrations.get_crm_client", return_value=mock_client):
            result = crm_query_order.invoke({"order_id": "ORD-2024-001"})

        assert "ORD-2024-001" in result
        assert "C001" in result
        assert "云服务器" in result
        assert "5000" in result
        assert "已完成" in result

    def test_query_nonexistent_order(self):
        """查询不存在的订单应返回未找到"""
        mock_client = MagicMock()
        mock_client.get_order.return_value = None

        with patch("cayz_agent.integrations.get_crm_client", return_value=mock_client):
            result = crm_query_order.invoke({"order_id": "ORD-999"})

        assert "未找到" in result or "ORD-999" in result

    def test_query_exception_sanitized(self):
        """异常应被捕获并脱敏"""
        mock_client = MagicMock()
        mock_client.get_order.side_effect = RuntimeError("sk-leaked")

        with patch("cayz_agent.integrations.get_crm_client", return_value=mock_client), \
             patch("cayz_agent.tools.sanitize_exception", return_value="敏感信息已隐藏"):
            result = crm_query_order.invoke({"order_id": "ORD-001"})

        assert "敏感信息已隐藏" in result or "sk-leaked" not in result


# ============================================================
# P4 新增：通知与邮件工具测试
# ============================================================

class TestSendWecomNotification:
    """测试 send_wecom_notification 工具"""

    def test_send_text_success(self):
        """文本消息发送成功应返回成功提示"""
        mock_notifier = MagicMock()
        mock_notifier.send_text.return_value = {"errcode": 0, "errmsg": "ok"}

        with patch("cayz_agent.integrations.get_notifier", return_value=mock_notifier):
            result = send_wecom_notification.invoke({"message": "测试消息", "msg_type": "text"})

        assert "成功" in result
        mock_notifier.send_text.assert_called_once_with("测试消息")

    def test_send_markdown_success(self):
        """markdown 消息应调用 send_markdown"""
        mock_notifier = MagicMock()
        mock_notifier.send_markdown.return_value = {"errcode": 0, "errmsg": "ok"}

        with patch("cayz_agent.integrations.get_notifier", return_value=mock_notifier):
            result = send_wecom_notification.invoke({"message": "# 标题", "msg_type": "markdown"})

        assert "成功" in result
        mock_notifier.send_markdown.assert_called_once_with("# 标题")

    def test_send_markdown_case_insensitive(self):
        """msg_type 大小写不敏感"""
        mock_notifier = MagicMock()
        mock_notifier.send_markdown.return_value = {"errcode": 0}

        with patch("cayz_agent.integrations.get_notifier", return_value=mock_notifier):
            send_wecom_notification.invoke({"message": "msg", "msg_type": "MARKDOWN"})

        mock_notifier.send_markdown.assert_called_once()

    def test_webhook_not_configured(self):
        """Webhook 未配置（errcode=-1）应返回提示"""
        mock_notifier = MagicMock()
        mock_notifier.send_text.return_value = {"errcode": -1}

        with patch("cayz_agent.integrations.get_notifier", return_value=mock_notifier):
            result = send_wecom_notification.invoke({"message": "test"})

        assert "未配置" in result or "WECOM_WEBHOOK_URL" in result

    def test_send_failure_returns_error(self):
        """发送失败应返回错误信息"""
        mock_notifier = MagicMock()
        mock_notifier.send_text.return_value = {"errcode": 40001, "errmsg": "invalid token"}

        with patch("cayz_agent.integrations.get_notifier", return_value=mock_notifier):
            result = send_wecom_notification.invoke({"message": "test"})

        assert "失败" in result
        assert "invalid token" in result

    def test_exception_sanitized(self):
        """异常应被捕获并脱敏"""
        mock_notifier = MagicMock()
        mock_notifier.send_text.side_effect = RuntimeError("sk-leaked")

        with patch("cayz_agent.integrations.get_notifier", return_value=mock_notifier), \
             patch("cayz_agent.tools.sanitize_exception", return_value="敏感信息已隐藏"):
            result = send_wecom_notification.invoke({"message": "test"})

        assert "敏感信息已隐藏" in result or "sk-leaked" not in result


class TestSendEmail:
    """测试 send_email 工具"""

    def test_send_success(self):
        """邮件发送成功应返回成功提示"""
        mock_sender = MagicMock()
        mock_sender.send.return_value = {
            "success": True,
            "to": ["user@example.com"],
            "subject": "测试主题",
        }

        with patch("cayz_agent.integrations.get_email_sender", return_value=mock_sender):
            result = send_email.invoke({
                "to": "user@example.com",
                "subject": "测试主题",
                "body": "测试内容",
            })

        assert "成功" in result
        assert "user@example.com" in result
        assert "测试主题" in result

    def test_send_multiple_recipients(self):
        """多个收件人（逗号分隔）应被正确解析"""
        mock_sender = MagicMock()
        mock_sender.send.return_value = {
            "success": True,
            "to": ["a@example.com", "b@example.com"],
            "subject": "主题",
        }

        with patch("cayz_agent.integrations.get_email_sender", return_value=mock_sender):
            send_email.invoke({
                "to": "a@example.com, b@example.com",
                "subject": "主题",
                "body": "内容",
            })

        mock_sender.send.assert_called_once_with(
            to_addrs=["a@example.com", "b@example.com"],
            subject="主题",
            body="内容",
            html=False,
        )

    def test_send_html_flag(self):
        """html=True 应传递给 sender"""
        mock_sender = MagicMock()
        mock_sender.send.return_value = {"success": True, "to": ["x@y.com"], "subject": "s"}

        with patch("cayz_agent.integrations.get_email_sender", return_value=mock_sender):
            send_email.invoke({
                "to": "x@y.com",
                "subject": "s",
                "body": "<b>html</b>",
                "html": True,
            })

        mock_sender.send.assert_called_once_with(
            to_addrs=["x@y.com"],
            subject="s",
            body="<b>html</b>",
            html=True,
        )

    def test_empty_recipients_rejected(self):
        """空收件人应被拒绝"""
        mock_sender = MagicMock()

        with patch("cayz_agent.integrations.get_email_sender", return_value=mock_sender):
            result = send_email.invoke({
                "to": "  ,  , ",
                "subject": "s",
                "body": "b",
            })

        assert "收件人" in result and "空" in result
        mock_sender.send.assert_not_called()

    def test_send_failure_returns_error(self):
        """发送失败应返回错误"""
        mock_sender = MagicMock()
        mock_sender.send.return_value = {"success": False, "error": "SMTP refused"}

        with patch("cayz_agent.integrations.get_email_sender", return_value=mock_sender):
            result = send_email.invoke({
                "to": "x@y.com",
                "subject": "s",
                "body": "b",
            })

        assert "失败" in result
        assert "SMTP refused" in result

    def test_exception_sanitized(self):
        """异常应被捕获并脱敏"""
        mock_sender = MagicMock()
        mock_sender.send.side_effect = RuntimeError("sk-leaked")

        with patch("cayz_agent.integrations.get_email_sender", return_value=mock_sender), \
             patch("cayz_agent.tools.sanitize_exception", return_value="敏感信息已隐藏"):
            result = send_email.invoke({
                "to": "x@y.com",
                "subject": "s",
                "body": "b",
            })

        assert "敏感信息已隐藏" in result or "sk-leaked" not in result

