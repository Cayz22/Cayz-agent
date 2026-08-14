"""
CRM 系统集成

模拟企业 CRM 系统，提供客户信息查询和订单跟踪能力。
生产环境中可替换为对接真实 CRM API（如 Salesforce、HubSpot、纷享销客等）。

数据结构：
- 客户：ID、姓名、邮箱、电话、公司、等级、状态
- 订单：订单号、客户ID、产品、金额、状态、下单日期
"""

import json
import logging
import os
from dataclasses import dataclass

from ..config import get_settings

logger = logging.getLogger(__name__)

# 运行时新增客户的持久化文件（JSON），解决服务重启后新增客户丢失的问题
_PERSIST_FILE = "crm_customers.json"
# 运行时新增订单的持久化文件（JSON），与客户持久化同理
_ORDER_PERSIST_FILE = "crm_orders.json"


@dataclass
class Customer:
    """客户信息"""

    customer_id: str
    name: str
    email: str
    phone: str
    company: str
    level: str  # VIP / 普通 / 试用
    status: str  # 活跃 / 流失 / 待跟进


@dataclass
class Order:
    """订单信息"""

    order_id: str
    customer_id: str
    product: str
    amount: float
    status: str  # 已完成 / 处理中 / 已取消 / 已退款
    created_at: str


# ============================================================
# 模拟数据（生产环境替换为真实 API 调用）
# ============================================================

_MOCK_CUSTOMERS = [
    Customer("C001", "张伟", "zhangwei@example.com", "13800138001", "阿里巴巴", "VIP", "活跃"),
    Customer("C002", "李娜", "lina@example.com", "13800138002", "腾讯科技", "VIP", "活跃"),
    Customer("C003", "王强", "wangqiang@example.com", "13800138003", "字节跳动", "普通", "活跃"),
    Customer("C004", "赵敏", "zhaomin@example.com", "13800138004", "美团", "普通", "待跟进"),
    Customer("C005", "刘洋", "liuyang@example.com", "13800138005", "京东集团", "VIP", "流失"),
    Customer("C006", "陈静", "chenjing@example.com", "13800138006", "百度", "普通", "活跃"),
    Customer("C007", "杨光", "yangguang@example.com", "13800138007", "网易", "试用", "待跟进"),
    Customer("C008", "黄磊", "huanglei@example.com", "13800138008", "小米科技", "普通", "活跃"),
]

_MOCK_ORDERS = [
    Order("ORD-2026-001", "C001", "企业版AI助手年付", 120000.00, "已完成", "2026-01-15"),
    Order("ORD-2026-002", "C001", "API调用包(100万次)", 8000.00, "已完成", "2026-03-20"),
    Order("ORD-2026-003", "C002", "企业版AI助手年付", 120000.00, "已完成", "2026-02-10"),
    Order("ORD-2026-004", "C002", "定制模型训练", 50000.00, "处理中", "2026-06-01"),
    Order("ORD-2026-005", "C003", "专业版月付", 999.00, "已完成", "2026-04-05"),
    Order("ORD-2026-007", "C004", "专业版月付", 999.00, "处理中", "2026-06-15"),
    Order("ORD-2026-008", "C005", "企业版AI助手年付", 120000.00, "已退款", "2026-01-20"),
    Order("ORD-2026-009", "C006", "专业版月付", 999.00, "已完成", "2026-03-08"),
    Order("ORD-2026-010", "C008", "API调用包(50万次)", 5000.00, "已完成", "2026-05-20"),
    Order("ORD-2026-011", "C008", "专业版月付", 999.00, "已取消", "2026-06-02"),
]


class CRMClient:
    """
    CRM 客户端

    封装客户查询和订单跟踪操作。
    生产环境中将 _MOCK_* 数据替换为真实 API 请求即可。

    P2-10 修复：use_mock=False 时显式失败，避免静默返回空结果导致
    业务 Agent 向用户回复"未找到客户"（看似正常业务结果而非系统故障）。
    """

    def __init__(self, use_mock: bool = True):
        self.use_mock = use_mock
        if use_mock:
            self._customers = {c.customer_id: c for c in _MOCK_CUSTOMERS}
            self._orders = {o.order_id: o for o in _MOCK_ORDERS}
            # 加载历史新增客户，避免服务重启后丢失
            self._customers.update(self._load_persisted())
            # 加载历史新增订单，避免服务重启后丢失
            self._orders.update(self._load_persisted_orders())
        else:
            # P2-10：真实 API 模式未实现时显式失败，避免静默返回空结果
            raise NotImplementedError(
                "CRM 真实 API 集成尚未实现。请设置 crm_use_mock=True 使用模拟数据，"
                "或继承 CRMClient 并实现 _fetch_from_api 方法对接真实 CRM 系统。"
            )

    def _load_persisted(self) -> dict:
        """从持久化文件加载新增客户"""
        if not os.path.exists(_PERSIST_FILE):
            return {}
        try:
            with open(_PERSIST_FILE, encoding="utf-8") as f:
                data = json.load(f)
            return {
                item["customer_id"]: Customer(
                    customer_id=item["customer_id"],
                    name=item["name"],
                    email=item["email"],
                    phone=item["phone"],
                    company=item["company"],
                    level=item["level"],
                    status=item["status"],
                )
                for item in data
            }
        except Exception as e:
            logger.warning("加载 CRM 持久化客户失败: %s", e)
            return {}

    def _save_persisted(self) -> None:
        """将非内置新增客户写入持久化文件"""
        try:
            data = [
                {
                    "customer_id": c.customer_id,
                    "name": c.name,
                    "email": c.email,
                    "phone": c.phone,
                    "company": c.company,
                    "level": c.level,
                    "status": c.status,
                }
                for c in self._customers.values()
                if c.customer_id not in {m.customer_id for m in _MOCK_CUSTOMERS}
            ]
            with open(_PERSIST_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("保存 CRM 持久化客户失败: %s", e)

    def _load_persisted_orders(self) -> dict:
        """从持久化文件加载新增订单"""
        if not os.path.exists(_ORDER_PERSIST_FILE):
            return {}
        try:
            with open(_ORDER_PERSIST_FILE, encoding="utf-8") as f:
                data = json.load(f)
            return {
                item["order_id"]: Order(
                    order_id=item["order_id"],
                    customer_id=item["customer_id"],
                    product=item["product"],
                    amount=item["amount"],
                    status=item["status"],
                    created_at=item["created_at"],
                )
                for item in data
            }
        except Exception as e:
            logger.warning("加载 CRM 持久化订单失败: %s", e)
            return {}

    def _save_persisted_orders(self) -> None:
        """将非内置新增订单写入持久化文件"""
        try:
            data = [
                {
                    "order_id": o.order_id,
                    "customer_id": o.customer_id,
                    "product": o.product,
                    "amount": o.amount,
                    "status": o.status,
                    "created_at": o.created_at,
                }
                for o in self._orders.values()
                if o.order_id not in {m.order_id for m in _MOCK_ORDERS}
            ]
            with open(_ORDER_PERSIST_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("保存 CRM 持久化订单失败: %s", e)

    def get_customer(self, customer_id: str) -> Customer | None:
        """根据客户ID查询客户信息"""
        logger.info("CRM 查询客户: %s", customer_id)
        customer = self._customers.get(customer_id)
        if customer is None:
            logger.info("CRM 未找到客户: %s", customer_id)
        return customer

    def search_customers(self, keyword: str) -> list[Customer]:
        """
        按关键词搜索客户（支持姓名、公司、邮箱模糊匹配）
        """
        keyword = keyword.lower().strip()
        if not keyword:
            return []

        logger.info("CRM 搜索客户: keyword=%s", keyword)
        results = [
            c
            for c in self._customers.values()
            if keyword in c.name.lower() or keyword in c.company.lower() or keyword in c.email.lower()
        ]
        logger.info("CRM 搜索完成: 找到 %d 条结果", len(results))
        return results

    def get_order(self, order_id: str) -> Order | None:
        """根据订单号查询订单详情"""
        logger.info("CRM 查询订单: %s", order_id)
        order = self._orders.get(order_id)
        if order is None:
            logger.info("CRM 未找到订单: %s", order_id)
        return order

    def get_customer_orders(self, customer_id: str) -> list[Order]:
        """查询某客户的所有订单"""
        logger.info("CRM 查询客户订单: %s", customer_id)
        if customer_id not in self._customers:
            return []
        orders = [o for o in self._orders.values() if o.customer_id == customer_id]
        logger.info("CRM 查询完成: 客户 %s 有 %d 个订单", customer_id, len(orders))
        return orders

    def get_orders_by_status(self, status: str) -> list[Order]:
        """按状态筛选订单"""
        status = status.strip()
        logger.info("CRM 按状态查询订单: %s", status)
        orders = [o for o in self._orders.values() if o.status == status]
        logger.info("CRM 查询完成: 状态 %s 有 %d 个订单", status, len(orders))
        return orders

    def add_customer(
        self,
        name: str,
        email: str,
        phone: str,
        company: str,
        level: str = "普通",
        status: str = "活跃",
    ) -> Customer:
        """新增客户，自动分配客户ID"""
        # 自动生成客户ID（C009 起）
        existing_ids = [int(c.customer_id[1:]) for c in self._customers.values()]
        next_id = max(existing_ids) + 1 if existing_ids else 1
        customer_id = f"C{next_id:03d}"

        customer = Customer(
            customer_id=customer_id,
            name=name.strip(),
            email=email.strip(),
            phone=phone.strip(),
            company=company.strip(),
            level=level.strip(),
            status=status.strip(),
        )
        self._customers[customer_id] = customer
        self._save_persisted()
        logger.info("CRM 新增客户: %s (%s, %s)", customer_id, name, company)
        return customer

    def add_order(
        self,
        customer_id: str,
        product: str,
        amount: float,
        status: str = "处理中",
        created_at: str | None = None,
    ) -> Order:
        """新增订单，自动分配订单号。客户必须存在，否则返回 None。"""
        customer = self._customers.get(customer_id)
        if customer is None:
            logger.warning("CRM 新增订单失败: 客户不存在 %s", customer_id)
            return None

        # 自动生成订单号（ORD-YYYY-NNN）
        from datetime import datetime

        year = datetime.now().strftime("%Y")
        prefix = f"ORD-{year}-"
        existing_nums = [int(o.order_id[len(prefix) :]) for o in self._orders.values() if o.order_id.startswith(prefix)]
        next_num = (max(existing_nums) + 1) if existing_nums else 1
        order_id = f"{prefix}{next_num:03d}"

        if created_at is None:
            created_at = datetime.now().strftime("%Y-%m-%d")

        order = Order(
            order_id=order_id,
            customer_id=customer_id,
            product=product.strip(),
            amount=float(amount),
            status=status.strip(),
            created_at=created_at,
        )
        self._orders[order_id] = order
        self._save_persisted_orders()
        logger.info("CRM 新增订单: %s (%s, ¥%.2f)", order_id, product, amount)
        return order

    def get_customer_summary(self, customer_id: str) -> dict:
        """获取客户汇总信息（含订单统计）"""
        customer = self.get_customer(customer_id)
        if customer is None:
            return {"error": f"未找到客户: {customer_id}"}

        orders = self.get_customer_orders(customer_id)
        total_amount = sum(o.amount for o in orders if o.status == "已完成")

        return {
            "customer": {
                "customer_id": customer.customer_id,
                "name": customer.name,
                "email": customer.email,
                "phone": customer.phone,
                "company": customer.company,
                "level": customer.level,
                "status": customer.status,
            },
            "order_count": len(orders),
            "total_spent": total_amount,
            "recent_orders": [
                {
                    "order_id": o.order_id,
                    "product": o.product,
                    "amount": o.amount,
                    "status": o.status,
                    "date": o.created_at,
                }
                for o in sorted(orders, key=lambda x: x.created_at, reverse=True)[:5]
            ],
        }


# 全局单例
_crm_client: CRMClient | None = None


def get_crm_client() -> CRMClient:
    """获取 CRM 客户端单例"""
    global _crm_client
    if _crm_client is None:
        settings = get_settings()
        _crm_client = CRMClient(use_mock=settings.crm_use_mock)
    return _crm_client
