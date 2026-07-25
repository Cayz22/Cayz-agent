"""
CRM 系统集成

模拟企业 CRM 系统，提供客户信息查询和订单跟踪能力。
生产环境中可替换为对接真实 CRM API（如 Salesforce、HubSpot、纷享销客等）。

数据结构：
- 客户：ID、姓名、邮箱、电话、公司、等级、状态
- 订单：订单号、客户ID、产品、金额、状态、下单日期
"""

import logging
from dataclasses import dataclass

from ..config import get_settings

logger = logging.getLogger(__name__)


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
    Order("ORD-2024-001", "C001", "企业版AI助手年付", 120000.00, "已完成", "2024-01-15"),
    Order("ORD-2024-002", "C001", "API调用包(100万次)", 8000.00, "已完成", "2024-03-20"),
    Order("ORD-2024-003", "C002", "企业版AI助手年付", 120000.00, "已完成", "2024-02-10"),
    Order("ORD-2024-004", "C002", "定制模型训练", 50000.00, "处理中", "2024-06-01"),
    Order("ORD-2024-005", "C003", "专业版月付", 999.00, "已完成", "2024-04-05"),
    Order("ORD-2024-006", "C003", "API调用包(10万次)", 1000.00, "已完成", "2024-05-12"),
    Order("ORD-2024-007", "C004", "专业版月付", 999.00, "处理中", "2024-06-15"),
    Order("ORD-2024-008", "C005", "企业版AI助手年付", 120000.00, "已退款", "2024-01-20"),
    Order("ORD-2024-009", "C006", "专业版月付", 999.00, "已完成", "2024-03-08"),
    Order("ORD-2024-010", "C008", "API调用包(50万次)", 5000.00, "已完成", "2024-05-20"),
    Order("ORD-2024-011", "C008", "专业版月付", 999.00, "已取消", "2024-06-02"),
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
        else:
            # P2-10：真实 API 模式未实现时显式失败，避免静默返回空结果
            raise NotImplementedError(
                "CRM 真实 API 集成尚未实现。请设置 crm_use_mock=True 使用模拟数据，"
                "或继承 CRMClient 并实现 _fetch_from_api 方法对接真实 CRM 系统。"
            )

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
