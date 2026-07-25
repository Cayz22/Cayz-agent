"""
CRM 集成模块单元测试
"""
import pytest

from cayz_agent.integrations.crm import CRMClient, Customer, Order


class TestCRMClient:
    """测试 CRM 客户端"""

    def test_get_customer_by_id(self):
        """根据ID查询客户"""
        client = CRMClient(use_mock=True)
        customer = client.get_customer("C001")
        assert customer is not None
        assert customer.customer_id == "C001"
        assert customer.name == "张伟"
        assert customer.company == "阿里巴巴"

    def test_get_customer_not_found(self):
        """查询不存在的客户"""
        client = CRMClient(use_mock=True)
        customer = client.get_customer("C999")
        assert customer is None

    def test_search_customers_by_name(self):
        """按姓名搜索客户"""
        client = CRMClient(use_mock=True)
        results = client.search_customers("张")
        assert len(results) == 1
        assert results[0].name == "张伟"

    def test_search_customers_by_company(self):
        """按公司名搜索客户"""
        client = CRMClient(use_mock=True)
        results = client.search_customers("腾讯")
        assert len(results) == 1
        assert results[0].company == "腾讯科技"

    def test_search_customers_by_email(self):
        """按邮箱搜索客户"""
        client = CRMClient(use_mock=True)
        results = client.search_customers("lina@example.com")
        assert len(results) == 1
        assert results[0].email == "lina@example.com"

    def test_search_customers_empty_keyword(self):
        """空关键词返回空列表"""
        client = CRMClient(use_mock=True)
        results = client.search_customers("")
        assert len(results) == 0

    def test_search_customers_no_match(self):
        """无匹配结果"""
        client = CRMClient(use_mock=True)
        results = client.search_customers("不存在的客户")
        assert len(results) == 0

    def test_search_customers_case_insensitive(self):
        """搜索不区分大小写"""
        client = CRMClient(use_mock=True)
        results = client.search_customers("LINA")
        assert len(results) == 1

    def test_get_order_by_id(self):
        """根据订单号查询订单"""
        client = CRMClient(use_mock=True)
        order = client.get_order("ORD-2024-001")
        assert order is not None
        assert order.order_id == "ORD-2024-001"
        assert order.customer_id == "C001"
        assert order.product == "企业版AI助手年付"
        assert order.amount == 120000.00

    def test_get_order_not_found(self):
        """查询不存在的订单"""
        client = CRMClient(use_mock=True)
        order = client.get_order("ORD-9999")
        assert order is None

    def test_get_customer_orders(self):
        """查询客户的所有订单"""
        client = CRMClient(use_mock=True)
        orders = client.get_customer_orders("C001")
        assert len(orders) == 2
        assert all(o.customer_id == "C001" for o in orders)

    def test_get_customer_orders_nonexistent_customer(self):
        """查询不存在客户的订单返回空列表"""
        client = CRMClient(use_mock=True)
        orders = client.get_customer_orders("C999")
        assert len(orders) == 0

    def test_get_orders_by_status(self):
        """按状态筛选订单"""
        client = CRMClient(use_mock=True)
        completed = client.get_orders_by_status("已完成")
        assert len(completed) > 0
        assert all(o.status == "已完成" for o in completed)

    def test_get_customer_summary(self):
        """获取客户汇总信息"""
        client = CRMClient(use_mock=True)
        summary = client.get_customer_summary("C001")

        assert "error" not in summary
        assert summary["customer"]["name"] == "张伟"
        assert summary["order_count"] == 2
        assert summary["total_spent"] == 128000.00  # 120000 + 8000
        assert len(summary["recent_orders"]) == 2

    def test_get_customer_summary_not_found(self):
        """查询不存在客户的汇总返回错误"""
        client = CRMClient(use_mock=True)
        summary = client.get_customer_summary("C999")
        assert "error" in summary

    def test_get_customer_summary_total_spent_only_completed(self):
        """总消费只计算已完成的订单"""
        client = CRMClient(use_mock=True)
        # C002 有一个已完成(120000)和一个处理中(50000)
        summary = client.get_customer_summary("C002")
        assert summary["total_spent"] == 120000.00

    def test_non_mock_mode_raises_not_implemented(self):
        """P2-10：非 mock 模式未实现时应抛出 NotImplementedError 而非静默返回空"""
        with pytest.raises(NotImplementedError, match="CRM 真实 API 集成尚未实现"):
            CRMClient(use_mock=False)
