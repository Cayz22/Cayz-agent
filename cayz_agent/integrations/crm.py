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

logger = logging.getLogger(__name__)

# 运行时新增客户的持久化文件（JSON），解决服务重启后新增客户丢失的问题
_PERSIST_FILE = "crm_customers.json"
# 运行时新增订单的持久化文件（JSON），与客户持久化同理
_ORDER_PERSIST_FILE = "crm_orders.json"
# 软删除/归档状态：归档（软删除）时把客户/订单状态置为此值，而非物理移除
_ARCHIVED_STATUS = "已归档"
# 恢复时的默认状态：仅当归档前未记录原始状态（历史已归档数据）时回退使用
_DEFAULT_CUSTOMER_STATUS = "活跃"
_DEFAULT_ORDER_STATUS = "处理中"


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
    archived_status: str = ""  # 归档前的原始状态，用于恢复（恢复后清空）


@dataclass
class Order:
    """订单信息"""

    order_id: str
    customer_id: str
    product: str
    amount: float
    status: str  # 已完成 / 处理中 / 已取消 / 已退款
    created_at: str
    archived_status: str = ""  # 归档前的原始状态，用于恢复（恢复后清空）


class CRMClient:
    """
    CRM 客户端

    封装客户查询和订单跟踪操作。
    纯文件模式：客户/订单数据完全由 crm_customers.json、crm_orders.json 决定，
    代码中不维护任何内置数据。文件缺失时以空数据启动。
    """

    def __init__(self):
        # 纯文件模式：直接加载持久化文件，文件为唯一数据源
        self._customers = self._load_persisted()
        self._orders = self._load_persisted_orders()

    def _load_persisted(self) -> dict:
        """从持久化文件加载全部客户（纯文件模式唯一数据源）"""
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
                    archived_status=item.get("archived_status", ""),
                )
                for item in data
            }
        except Exception as e:
            logger.warning("加载 CRM 持久化客户失败: %s", e)
            return {}

    def _save_persisted(self) -> None:
        """全量保存所有客户到 crm_customers.json，文件为唯一数据源"""
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
                    "archived_status": c.archived_status,
                }
                for c in self._customers.values()
            ]
            with open(_PERSIST_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("保存 CRM 持久化客户失败: %s", e)

    def _load_persisted_orders(self) -> dict:
        """从持久化文件加载全部订单（纯文件模式唯一数据源）"""
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
                    archived_status=item.get("archived_status", ""),
                )
                for item in data
            }
        except Exception as e:
            logger.warning("加载 CRM 持久化订单失败: %s", e)
            return {}

    def _save_persisted_orders(self) -> None:
        """全量保存所有订单到 crm_orders.json，文件为唯一数据源"""
        try:
            data = [
                {
                    "order_id": o.order_id,
                    "customer_id": o.customer_id,
                    "product": o.product,
                    "amount": o.amount,
                    "status": o.status,
                    "created_at": o.created_at,
                    "archived_status": o.archived_status,
                }
                for o in self._orders.values()
            ]
            with open(_ORDER_PERSIST_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("保存 CRM 持久化订单失败: %s", e)

    def get_customer(self, customer_id: str) -> Customer | None:
        """根据客户ID查询客户信息。已归档客户视为删除，返回 None。"""
        logger.info("CRM 查询客户: %s", customer_id)
        customer = self._customers.get(customer_id)
        if customer is None:
            logger.info("CRM 未找到客户: %s", customer_id)
            return None
        if customer.status == _ARCHIVED_STATUS:
            logger.info("CRM 客户已归档，视为不存在: %s", customer_id)
            return None
        return customer

    def search_customers(self, keyword: str) -> list[Customer]:
        """
        按关键词搜索客户（支持姓名、公司、邮箱模糊匹配）。已归档客户不参与搜索。
        """
        keyword = keyword.lower().strip()
        if not keyword:
            return []

        logger.info("CRM 搜索客户: keyword=%s", keyword)
        results = [
            c
            for c in self._customers.values()
            if c.status != _ARCHIVED_STATUS
            and (keyword in c.name.lower() or keyword in c.company.lower() or keyword in c.email.lower())
        ]
        logger.info("CRM 搜索完成: 找到 %d 条结果", len(results))
        return results

    def get_order(self, order_id: str) -> Order | None:
        """根据订单号查询订单详情。已归档订单视为删除，返回 None。"""
        logger.info("CRM 查询订单: %s", order_id)
        order = self._orders.get(order_id)
        if order is None:
            logger.info("CRM 未找到订单: %s", order_id)
            return None
        if order.status == _ARCHIVED_STATUS:
            logger.info("CRM 订单已归档，视为不存在: %s", order_id)
            return None
        return order

    def get_customer_orders(self, customer_id: str) -> list[Order]:
        """查询某客户的所有订单（不含已归档订单）"""
        logger.info("CRM 查询客户订单: %s", customer_id)
        if customer_id not in self._customers:
            return []
        orders = [o for o in self._orders.values() if o.customer_id == customer_id and o.status != _ARCHIVED_STATUS]
        logger.info("CRM 查询完成: 客户 %s 有 %d 个订单", customer_id, len(orders))
        return orders

    def get_orders_by_status(self, status: str) -> list[Order]:
        """按状态筛选订单。传「已归档」可查询已归档订单，传其他状态则排除已归档。"""
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

    def list_customers(self, keyword: str = "") -> list[Customer]:
        """列出全部客户（含已归档）。keyword 非空时按姓名/公司/邮箱模糊过滤。"""
        if keyword and keyword.strip():
            return self.search_customers(keyword)
        logger.info("CRM 列出客户: %d 条", len(self._customers))
        return list(self._customers.values())

    def list_orders(self, status: str = "") -> list[Order]:
        """列出全部订单（含已归档）。status 非空时按状态过滤。"""
        if status and status.strip():
            return self.get_orders_by_status(status)
        logger.info("CRM 列出订单: %d 条", len(self._orders))
        return list(self._orders.values())

    def update_customer(
        self,
        customer_id: str,
        name: str | None = None,
        email: str | None = None,
        phone: str | None = None,
        company: str | None = None,
        level: str | None = None,
        status: str | None = None,
    ) -> Customer | None:
        """编辑客户字段（仅更新传入的非 None 字段）。客户不存在返回 None。"""
        customer = self._customers.get(customer_id)
        if customer is None:
            logger.warning("CRM 编辑客户失败: 客户不存在 %s", customer_id)
            return None
        if name is not None:
            customer.name = str(name).strip()
        if email is not None:
            customer.email = str(email).strip()
        if phone is not None:
            customer.phone = str(phone).strip()
        if company is not None:
            customer.company = str(company).strip()
        if level is not None:
            customer.level = str(level).strip()
        if status is not None:
            customer.status = str(status).strip()
        self._save_persisted()
        logger.info("CRM 编辑客户: %s (%s)", customer_id, customer.name)
        return customer

    def update_order(
        self,
        order_id: str,
        customer_id: str | None = None,
        product: str | None = None,
        amount: float | None = None,
        status: str | None = None,
        created_at: str | None = None,
    ) -> Order | None:
        """编辑订单字段（仅更新传入的非 None 字段）。订单不存在返回 None，目标客户不存在时返回 None。"""
        order = self._orders.get(order_id)
        if order is None:
            logger.warning("CRM 编辑订单失败: 订单不存在 %s", order_id)
            return None
        if customer_id is not None:
            if customer_id not in self._customers:
                logger.warning("CRM 编辑订单失败: 目标客户不存在 %s", customer_id)
                return None
            order.customer_id = customer_id
        if product is not None:
            order.product = str(product).strip()
        if amount is not None:
            order.amount = float(amount)
        if status is not None:
            order.status = str(status).strip()
        if created_at is not None:
            order.created_at = str(created_at).strip()
        self._save_persisted_orders()
        logger.info("CRM 编辑订单: %s", order_id)
        return order

    def archive_customer(self, customer_id: str) -> Customer | None:
        """软删除/归档客户：将状态置为「已归档」并记录归档前状态。返回更新后的客户，客户不存在返回 None。"""
        customer = self._customers.get(customer_id)
        if customer is None:
            logger.warning("CRM 归档客户失败: 客户不存在 %s", customer_id)
            return None
        if customer.status == _ARCHIVED_STATUS:
            logger.info("CRM 客户已是归档状态，跳过: %s", customer_id)
            return customer
        customer.archived_status = customer.status  # 记录归档前状态，供恢复使用
        customer.status = _ARCHIVED_STATUS
        self._save_persisted()
        logger.info("CRM 归档客户: %s (%s)", customer_id, customer.name)
        return customer

    def restore_customer(self, customer_id: str) -> Customer | None:
        """恢复已归档客户：还原到归档前的状态。返回更新后的客户，客户不存在返回 None。"""
        customer = self._customers.get(customer_id)
        if customer is None:
            logger.warning("CRM 恢复客户失败: 客户不存在 %s", customer_id)
            return None
        if customer.status != _ARCHIVED_STATUS:
            logger.info("CRM 客户未归档，无需恢复: %s", customer_id)
            return customer
        # 还原到归档前状态；历史数据未记录时回退到默认活跃状态
        customer.status = customer.archived_status or _DEFAULT_CUSTOMER_STATUS
        customer.archived_status = ""
        self._save_persisted()
        logger.info("CRM 恢复客户: %s (%s) -> %s", customer_id, customer.name, customer.status)
        return customer

    def archive_order(self, order_id: str) -> Order | None:
        """软删除/归档订单：将状态置为「已归档」并记录归档前状态。返回更新后的订单，订单不存在返回 None。"""
        order = self._orders.get(order_id)
        if order is None:
            logger.warning("CRM 归档订单失败: 订单不存在 %s", order_id)
            return None
        if order.status == _ARCHIVED_STATUS:
            logger.info("CRM 订单已是归档状态，跳过: %s", order_id)
            return order
        order.archived_status = order.status  # 记录归档前状态，供恢复使用
        order.status = _ARCHIVED_STATUS
        self._save_persisted_orders()
        logger.info("CRM 归档订单: %s", order_id)
        return order

    def restore_order(self, order_id: str) -> Order | None:
        """恢复已归档订单：还原到归档前的状态。返回更新后的订单，订单不存在返回 None。"""
        order = self._orders.get(order_id)
        if order is None:
            logger.warning("CRM 恢复订单失败: 订单不存在 %s", order_id)
            return None
        if order.status != _ARCHIVED_STATUS:
            logger.info("CRM 订单未归档，无需恢复: %s", order_id)
            return order
        # 还原到归档前状态；历史数据未记录时回退到默认处理中状态
        order.status = order.archived_status or _DEFAULT_ORDER_STATUS
        order.archived_status = ""
        self._save_persisted_orders()
        logger.info("CRM 恢复订单: %s -> %s", order_id, order.status)
        return order

    def delete_customer(self, customer_id: str) -> Customer | None:
        """彻底删除（物理删除）客户。

        仅允许彻底删除已归档（软删除）的客户，防止误删活跃数据。
        删除后从持久化文件移除，不可恢复。返回被删除的客户，客户不存在或未归档时返回 None。
        """
        customer = self._customers.get(customer_id)
        if customer is None:
            logger.warning("CRM 彻底删除客户失败: 客户不存在 %s", customer_id)
            return None
        if customer.status != _ARCHIVED_STATUS:
            logger.warning("CRM 彻底删除客户失败: 仅已归档客户可彻底删除 %s（当前状态 %s）", customer_id, customer.status)
            return None
        del self._customers[customer_id]
        self._save_persisted()
        logger.info("CRM 彻底删除客户: %s (%s)", customer_id, customer.name)
        return customer

    def delete_order(self, order_id: str) -> Order | None:
        """彻底删除（物理删除）订单。

        仅允许彻底删除已归档（软删除）的订单，防止误删有效订单。
        删除后从持久化文件移除，不可恢复。返回被删除的订单，订单不存在或未归档时返回 None。
        """
        order = self._orders.get(order_id)
        if order is None:
            logger.warning("CRM 彻底删除订单失败: 订单不存在 %s", order_id)
            return None
        if order.status != _ARCHIVED_STATUS:
            logger.warning("CRM 彻底删除订单失败: 仅已归档订单可彻底删除 %s（当前状态 %s）", order_id, order.status)
            return None
        del self._orders[order_id]
        self._save_persisted_orders()
        logger.info("CRM 彻底删除订单: %s", order_id)
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
        _crm_client = CRMClient()
    return _crm_client
