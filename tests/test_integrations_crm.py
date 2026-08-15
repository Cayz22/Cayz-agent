"""
CRM 集成模块单元测试

纯文件模式：CRMClient 数据完全源自 crm_customers.json / crm_orders.json。
测试通过 monkeypatch 将持久化文件重定向到临时目录并写入种子数据，避免污染真实数据。
"""

import pytest

from cayz_agent.integrations.crm import CRMClient, Customer, Order

# 种子数据（等价于纯文件模式下的初始 JSON 文件内容）
_SEED_CUSTOMERS = [
    {
        "customer_id": "C001",
        "name": "张伟",
        "email": "zhangwei@example.com",
        "phone": "13800138001",
        "company": "阿里巴巴",
        "level": "VIP",
        "status": "活跃",
        "archived_status": "",
    },
    {
        "customer_id": "C002",
        "name": "李娜",
        "email": "lina@example.com",
        "phone": "13800138002",
        "company": "腾讯科技",
        "level": "VIP",
        "status": "活跃",
        "archived_status": "",
    },
    {
        "customer_id": "C003",
        "name": "王强",
        "email": "wangqiang@example.com",
        "phone": "13800138003",
        "company": "字节跳动",
        "level": "普通",
        "status": "活跃",
        "archived_status": "",
    },
    {
        "customer_id": "C004",
        "name": "赵敏",
        "email": "zhaomin@example.com",
        "phone": "13800138004",
        "company": "美团",
        "level": "普通",
        "status": "待跟进",
        "archived_status": "",
    },
    {
        "customer_id": "C005",
        "name": "刘洋",
        "email": "liuyang@example.com",
        "phone": "13800138005",
        "company": "京东集团",
        "level": "VIP",
        "status": "流失",
        "archived_status": "",
    },
    {
        "customer_id": "C006",
        "name": "陈静",
        "email": "chenjing@example.com",
        "phone": "13800138006",
        "company": "百度",
        "level": "普通",
        "status": "活跃",
        "archived_status": "",
    },
    {
        "customer_id": "C007",
        "name": "杨光",
        "email": "yangguang@example.com",
        "phone": "13800138007",
        "company": "网易",
        "level": "试用",
        "status": "待跟进",
        "archived_status": "",
    },
    {
        "customer_id": "C008",
        "name": "黄磊",
        "email": "huanglei@example.com",
        "phone": "13800138008",
        "company": "小米科技",
        "level": "普通",
        "status": "活跃",
        "archived_status": "",
    },
]

_SEED_ORDERS = [
    {
        "order_id": "ORD-2026-001",
        "customer_id": "C001",
        "product": "企业版AI助手年付",
        "amount": 120000.00,
        "status": "已完成",
        "created_at": "2026-01-15",
        "archived_status": "",
    },
    {
        "order_id": "ORD-2026-002",
        "customer_id": "C001",
        "product": "API调用包(100万次)",
        "amount": 8000.00,
        "status": "已完成",
        "created_at": "2026-03-20",
        "archived_status": "",
    },
    {
        "order_id": "ORD-2026-003",
        "customer_id": "C002",
        "product": "企业版AI助手年付",
        "amount": 120000.00,
        "status": "已完成",
        "created_at": "2026-02-10",
        "archived_status": "",
    },
    {
        "order_id": "ORD-2026-004",
        "customer_id": "C002",
        "product": "定制模型训练",
        "amount": 50000.00,
        "status": "处理中",
        "created_at": "2026-06-01",
        "archived_status": "",
    },
    {
        "order_id": "ORD-2026-005",
        "customer_id": "C003",
        "product": "专业版月付",
        "amount": 999.00,
        "status": "已完成",
        "created_at": "2026-04-05",
        "archived_status": "",
    },
    {
        "order_id": "ORD-2026-007",
        "customer_id": "C004",
        "product": "专业版月付",
        "amount": 999.00,
        "status": "处理中",
        "created_at": "2026-06-15",
        "archived_status": "",
    },
    {
        "order_id": "ORD-2026-008",
        "customer_id": "C005",
        "product": "企业版AI助手年付",
        "amount": 120000.00,
        "status": "已退款",
        "created_at": "2026-01-20",
        "archived_status": "",
    },
    {
        "order_id": "ORD-2026-009",
        "customer_id": "C006",
        "product": "专业版月付",
        "amount": 999.00,
        "status": "已完成",
        "created_at": "2026-03-08",
        "archived_status": "",
    },
    {
        "order_id": "ORD-2026-010",
        "customer_id": "C008",
        "product": "API调用包(50万次)",
        "amount": 5000.00,
        "status": "已完成",
        "created_at": "2026-05-20",
        "archived_status": "",
    },
    {
        "order_id": "ORD-2026-011",
        "customer_id": "C008",
        "product": "专业版月付",
        "amount": 999.00,
        "status": "已取消",
        "created_at": "2026-06-02",
        "archived_status": "",
    },
]


def _make_client(tmp_path, monkeypatch, seed_customers=True, seed_orders=True):
    """隔离持久化文件到临时目录，写入种子数据后返回 CRMClient 实例"""
    import json

    customers_file = tmp_path / "customers.json"
    orders_file = tmp_path / "orders.json"
    monkeypatch.setattr("cayz_agent.integrations.crm._PERSIST_FILE", str(customers_file))
    monkeypatch.setattr("cayz_agent.integrations.crm._ORDER_PERSIST_FILE", str(orders_file))

    if seed_customers:
        customers_file.write_text(json.dumps(_SEED_CUSTOMERS, ensure_ascii=False), encoding="utf-8")
    if seed_orders:
        orders_file.write_text(json.dumps(_SEED_ORDERS, ensure_ascii=False), encoding="utf-8")

    return CRMClient()


class TestCRMClient:
    """测试 CRM 客户端"""

    def test_get_customer_by_id(self, tmp_path, monkeypatch):
        """根据ID查询客户"""
        client = _make_client(tmp_path, monkeypatch)
        customer = client.get_customer("C001")
        assert customer is not None
        assert customer.customer_id == "C001"
        assert customer.name == "张伟"
        assert customer.company == "阿里巴巴"

    def test_get_customer_not_found(self, tmp_path, monkeypatch):
        """查询不存在的客户"""
        client = _make_client(tmp_path, monkeypatch)
        customer = client.get_customer("C999")
        assert customer is None

    def test_search_customers_by_name(self, tmp_path, monkeypatch):
        """按姓名搜索客户"""
        client = _make_client(tmp_path, monkeypatch)
        results = client.search_customers("张")
        assert len(results) == 1
        assert results[0].name == "张伟"

    def test_search_customers_by_company(self, tmp_path, monkeypatch):
        """按公司名搜索客户"""
        client = _make_client(tmp_path, monkeypatch)
        results = client.search_customers("腾讯")
        assert len(results) == 1
        assert results[0].company == "腾讯科技"

    def test_search_customers_by_email(self, tmp_path, monkeypatch):
        """按邮箱搜索客户"""
        client = _make_client(tmp_path, monkeypatch)
        results = client.search_customers("lina@example.com")
        assert len(results) == 1
        assert results[0].email == "lina@example.com"

    def test_search_customers_empty_keyword(self, tmp_path, monkeypatch):
        """空关键词返回空列表"""
        client = _make_client(tmp_path, monkeypatch)
        results = client.search_customers("")
        assert len(results) == 0

    def test_search_customers_no_match(self, tmp_path, monkeypatch):
        """无匹配结果"""
        client = _make_client(tmp_path, monkeypatch)
        results = client.search_customers("不存在的客户")
        assert len(results) == 0

    def test_search_customers_case_insensitive(self, tmp_path, monkeypatch):
        """搜索不区分大小写"""
        client = _make_client(tmp_path, monkeypatch)
        results = client.search_customers("LINA")
        assert len(results) == 1

    def test_get_order_by_id(self, tmp_path, monkeypatch):
        """根据订单号查询订单"""
        client = _make_client(tmp_path, monkeypatch)
        order = client.get_order("ORD-2026-001")
        assert order is not None
        assert order.order_id == "ORD-2026-001"
        assert order.customer_id == "C001"
        assert order.product == "企业版AI助手年付"
        assert order.amount == 120000.00

    def test_get_order_not_found(self, tmp_path, monkeypatch):
        """查询不存在的订单"""
        client = _make_client(tmp_path, monkeypatch)
        order = client.get_order("ORD-9999")
        assert order is None

    def test_get_customer_orders(self, tmp_path, monkeypatch):
        """查询客户的所有订单"""
        client = _make_client(tmp_path, monkeypatch)
        orders = client.get_customer_orders("C001")
        assert len(orders) == 2
        assert all(o.customer_id == "C001" for o in orders)

    def test_get_customer_orders_nonexistent_customer(self, tmp_path, monkeypatch):
        """查询不存在客户的订单返回空列表"""
        client = _make_client(tmp_path, monkeypatch)
        orders = client.get_customer_orders("C999")
        assert len(orders) == 0

    def test_get_orders_by_status(self, tmp_path, monkeypatch):
        """按状态筛选订单"""
        client = _make_client(tmp_path, monkeypatch)
        completed = client.get_orders_by_status("已完成")
        assert len(completed) > 0
        assert all(o.status == "已完成" for o in completed)

    def test_get_customer_summary(self, tmp_path, monkeypatch):
        """获取客户汇总信息"""
        client = _make_client(tmp_path, monkeypatch)
        summary = client.get_customer_summary("C001")

        assert "error" not in summary
        assert summary["customer"]["name"] == "张伟"
        assert summary["order_count"] == 2
        assert summary["total_spent"] == 128000.00  # 120000 + 8000
        assert len(summary["recent_orders"]) == 2

    def test_get_customer_summary_not_found(self, tmp_path, monkeypatch):
        """查询不存在客户的汇总返回错误"""
        client = _make_client(tmp_path, monkeypatch)
        summary = client.get_customer_summary("C999")
        assert "error" in summary

    def test_get_customer_summary_total_spent_only_completed(self, tmp_path, monkeypatch):
        """总消费只计算已完成的订单"""
        client = _make_client(tmp_path, monkeypatch)
        # C002 有一个已完成(120000)和一个处理中(50000)
        summary = client.get_customer_summary("C002")
        assert summary["total_spent"] == 120000.00

    def test_empty_file_starts_empty(self, tmp_path, monkeypatch):
        """纯文件模式：无种子数据时以空数据启动"""
        client = _make_client(tmp_path, monkeypatch, seed_customers=False, seed_orders=False)
        assert client.list_customers() == []
        assert client.list_orders() == []

    # ---- 归档（软删除）----

    def test_archive_customer(self, tmp_path, monkeypatch):
        """归档客户：状态置为「已归档」"""
        client = _make_client(tmp_path, monkeypatch)
        archived = client.archive_customer("C001")
        assert archived is not None
        assert archived.status == "已归档"
        # 归档后业务查询视为删除
        assert client.get_customer("C001") is None

    def test_archive_customer_not_found(self, tmp_path, monkeypatch):
        """归档不存在的客户返回 None"""
        client = _make_client(tmp_path, monkeypatch)
        assert client.archive_customer("C999") is None

    def test_archive_customer_idempotent(self, tmp_path, monkeypatch):
        """重复归档同一客户幂等，不报错"""
        client = _make_client(tmp_path, monkeypatch)
        client.archive_customer("C001")
        again = client.archive_customer("C001")
        assert again.status == "已归档"

    def test_archive_order(self, tmp_path, monkeypatch):
        """归档订单：状态置为「已归档」"""
        client = _make_client(tmp_path, monkeypatch)
        archived = client.archive_order("ORD-2026-001")
        assert archived is not None
        assert archived.status == "已归档"
        # 归档后业务查询视为删除
        assert client.get_order("ORD-2026-001") is None

    def test_archive_order_not_found(self, tmp_path, monkeypatch):
        """归档不存在的订单返回 None"""
        client = _make_client(tmp_path, monkeypatch)
        assert client.archive_order("ORD-9999") is None

    def test_archived_customer_hidden_from_search(self, tmp_path, monkeypatch):
        """归档客户不在业务搜索中返回"""
        client = _make_client(tmp_path, monkeypatch)
        client.archive_customer("C001")
        assert client.search_customers("张伟") == []
        assert client.get_customer_summary("C001")["error"]

    def test_archived_order_hidden_from_customer_orders(self, tmp_path, monkeypatch):
        """归档订单不出现在客户订单列表"""
        client = _make_client(tmp_path, monkeypatch)
        # 先归档 C001 的一个订单
        client.archive_order("ORD-2026-001")
        orders = client.get_customer_orders("C001")
        assert all(o.order_id != "ORD-2026-001" for o in orders)

    def test_archived_listed_for_management(self, tmp_path, monkeypatch):
        """管理列表仍包含已归档客户（供恢复）"""
        client = _make_client(tmp_path, monkeypatch)
        client.archive_customer("C001")
        ids = [c.customer_id for c in client.list_customers()]
        assert "C001" in ids

    # ---- 恢复（取消归档）----

    def test_restore_customer_back_to_original(self, tmp_path, monkeypatch):
        """恢复客户：还原到归档前状态（C001 原为活跃）"""
        client = _make_client(tmp_path, monkeypatch)
        client.archive_customer("C001")
        # 归档后业务查询视为删除
        assert client.get_customer("C001") is None
        restored = client.restore_customer("C001")
        assert restored is not None
        assert restored.status == "活跃"
        assert restored.archived_status == ""

    def test_restore_customer_not_archived(self, tmp_path, monkeypatch):
        """未归档的客户恢复时状态不变"""
        client = _make_client(tmp_path, monkeypatch)
        restored = client.restore_customer("C001")
        assert restored.status == "活跃"

    def test_restore_customer_not_found(self, tmp_path, monkeypatch):
        """恢复不存在的客户返回 None"""
        client = _make_client(tmp_path, monkeypatch)
        assert client.restore_customer("C999") is None

    def test_restore_customer_fallback_default(self, tmp_path, monkeypatch):
        """历史已归档数据未记录原状态时，回退到默认活跃状态"""
        client = _make_client(tmp_path, monkeypatch)
        # 模拟历史数据：归档但未记录 archived_status
        c = client.get_customer("C001")
        c.archived_status = ""
        c.status = "已归档"
        restored = client.restore_customer("C001")
        assert restored.status == "活跃"

    def test_restore_order_back_to_original(self, tmp_path, monkeypatch):
        """恢复订单：还原到归档前状态（ORD-2026-001 原为已完成）"""
        client = _make_client(tmp_path, monkeypatch)
        client.archive_order("ORD-2026-001")
        # 归档后业务查询视为删除
        assert client.get_order("ORD-2026-001") is None
        restored = client.restore_order("ORD-2026-001")
        assert restored is not None
        assert restored.status == "已完成"
        assert restored.archived_status == ""

    def test_restore_order_not_found(self, tmp_path, monkeypatch):
        """恢复不存在的订单返回 None"""
        client = _make_client(tmp_path, monkeypatch)
        assert client.restore_order("ORD-9999") is None

    # ---- 彻底删除（物理删除，仅已归档数据）----

    def test_delete_customer_archived(self, tmp_path, monkeypatch):
        """彻底删除已归档客户：从持久化数据移除，不可恢复"""
        client = _make_client(tmp_path, monkeypatch)
        client.archive_customer("C001")
        deleted = client.delete_customer("C001")
        assert deleted is not None
        assert deleted.customer_id == "C001"
        # 彻底删除后客户完全消失（含管理列表）
        assert client.get_customer("C001") is None
        assert "C001" not in [c.customer_id for c in client.list_customers()]

    def test_delete_customer_rejects_active(self, tmp_path, monkeypatch):
        """未归档（活跃）客户禁止彻底删除，防止误删"""
        client = _make_client(tmp_path, monkeypatch)
        assert client.delete_customer("C001") is None
        # 活跃数据仍保留
        assert client.get_customer("C001") is not None

    def test_delete_customer_not_found(self, tmp_path, monkeypatch):
        """彻底删除不存在的客户返回 None"""
        client = _make_client(tmp_path, monkeypatch)
        assert client.delete_customer("C999") is None

    def test_delete_order_archived(self, tmp_path, monkeypatch):
        """彻底删除已归档订单：从持久化数据移除，不可恢复"""
        client = _make_client(tmp_path, monkeypatch)
        client.archive_order("ORD-2026-001")
        deleted = client.delete_order("ORD-2026-001")
        assert deleted is not None
        assert deleted.order_id == "ORD-2026-001"
        # 彻底删除后订单完全消失（含管理列表）
        assert client.get_order("ORD-2026-001") is None
        assert "ORD-2026-001" not in [o.order_id for o in client.list_orders()]

    def test_delete_order_rejects_active(self, tmp_path, monkeypatch):
        """未归档订单禁止彻底删除，防止误删"""
        client = _make_client(tmp_path, monkeypatch)
        assert client.delete_order("ORD-2026-001") is None
        # 有效订单仍保留
        assert client.get_order("ORD-2026-001") is not None

    def test_delete_order_not_found(self, tmp_path, monkeypatch):
        """彻底删除不存在的订单返回 None"""
        client = _make_client(tmp_path, monkeypatch)
        assert client.delete_order("ORD-9999") is None

    # ---- 列表 ----

    def test_list_customers_all(self, tmp_path, monkeypatch):
        """列出全部客户（含已归档）"""
        client = _make_client(tmp_path, monkeypatch)
        customers = client.list_customers()
        assert len(customers) == 8
        assert all(isinstance(c, Customer) for c in customers)

    def test_list_customers_keyword(self, tmp_path, monkeypatch):
        """带关键词列出客户时按姓名/公司过滤"""
        client = _make_client(tmp_path, monkeypatch)
        customers = client.list_customers("腾讯")
        assert len(customers) == 1
        assert customers[0].company == "腾讯科技"

    def test_list_orders_all(self, tmp_path, monkeypatch):
        """列出全部订单"""
        client = _make_client(tmp_path, monkeypatch)
        orders = client.list_orders()
        assert len(orders) > 0
        assert all(isinstance(o, Order) for o in orders)

    def test_list_orders_by_status(self, tmp_path, monkeypatch):
        """按状态过滤订单"""
        client = _make_client(tmp_path, monkeypatch)
        orders = client.list_orders("已完成")
        assert len(orders) > 0
        assert all(o.status == "已完成" for o in orders)

    # ---- 编辑 ----

    def test_update_customer(self, tmp_path, monkeypatch):
        """编辑客户字段并持久化"""
        client = _make_client(tmp_path, monkeypatch)
        updated = client.update_customer("C001", company="新公司", level="VIP", status="流失")
        assert updated is not None
        c = client.get_customer("C001")
        assert c.company == "新公司"
        assert c.level == "VIP"
        assert c.status == "流失"

    def test_update_customer_not_found(self, tmp_path, monkeypatch):
        """编辑不存在的客户返回 None"""
        client = _make_client(tmp_path, monkeypatch)
        assert client.update_customer("C999", name="x") is None

    def test_update_order(self, tmp_path, monkeypatch):
        """编辑订单字段并持久化"""
        client = _make_client(tmp_path, monkeypatch)
        updated = client.update_order("ORD-2026-001", product="新产品", amount=999.5, status="已取消")
        assert updated is not None
        o = client.get_order("ORD-2026-001")
        assert o.product == "新产品"
        assert o.amount == 999.5
        assert o.status == "已取消"

    def test_update_order_not_found(self, tmp_path, monkeypatch):
        """编辑不存在的订单返回 None"""
        client = _make_client(tmp_path, monkeypatch)
        assert client.update_order("ORD-9999", product="x") is None

    def test_update_order_customer_not_found(self, tmp_path, monkeypatch):
        """编辑订单时指定不存在的目标客户返回 None 且不修改"""
        client = _make_client(tmp_path, monkeypatch)
        assert client.update_order("ORD-2026-001", customer_id="C999") is None
        assert client.get_order("ORD-2026-001").customer_id == "C001"
