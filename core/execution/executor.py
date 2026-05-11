"""实盘执行器

支持实盘交易执行：
- 订单路由：根据交易所规则路由订单
- 执行确认：跟踪订单状态
- 仓位同步：与券商系统同步持仓
- 错误处理：重试、降级、告警

注意：实盘执行需要接入券商API，本模块提供接口定义
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
import time

from ..strategy import Order

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """订单状态"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL_FILLED = "partial_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    ERROR = "error"


class OrderRoute(Enum):
    """订单路由"""
    SHANGHAI = "sh"
    SHENZHEN = "sz"


@dataclass
class FillInfo:
    """成交信息"""
    symbol: str
    direction: str
    price: float
    quantity: int
    timestamp: datetime
    commission: float = 0.0


@dataclass
class OrderRecord:
    """订单记录"""
    order_id: str
    symbol: str
    direction: str
    quantity: int
    filled_quantity: int = 0
    price: float = 0.0
    avg_fill_price: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    fills: List[FillInfo] = field(default_factory=list)
    error_message: str = ""
    retry_count: int = 0


class IBrokerAdapter(ABC):
    """券商适配器接口"""

    @abstractmethod
    def submit_order(self, order: Order) -> str:
        """提交订单，返回订单ID"""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        pass

    @abstractmethod
    def get_order_status(self, order_id: str) -> OrderStatus:
        """查询订单状态"""
        pass

    @abstractmethod
    def get_positions(self) -> Dict[str, int]:
        """获取当前持仓"""
        pass

    @abstractmethod
    def get_account_info(self) -> Dict:
        """获取账户信息"""
        pass


class OrderRouter:
    """订单路由器

    根据股票代码自动路由到对应交易所
    """

    SH_EXCHANGE_CODES = {'5', '6', '9'}
    SZ_EXCHANGE_CODES = {'0', '1', '2', '3'}

    @classmethod
    def get_route(cls, symbol: str) -> OrderRoute:
        """根据股票代码判断交易所"""
        if not symbol:
            return OrderRoute.SHENZHEN
        code = symbol[0]
        if code in cls.SH_EXCHANGE_CODES:
            return OrderRoute.SHANGHAI
        elif code in cls.SZ_EXCHANGE_CODES:
            return OrderRoute.SHENZHEN
        return OrderRoute.SHENZHEN


class LiveExecutor:
    """实盘执行器

    管理实盘订单执行：
    - 订单提交、取消、状态跟踪
    - 自动路由到对应交易所
    - 持仓同步
    - 错误重试
    """

    MAX_RETRY = 3
    RETRY_DELAY = 1.0

    def __init__(
        self,
        broker: IBrokerAdapter,
        sync_interval: int = 5,
    ):
        self.broker = broker
        self.sync_interval = sync_interval
        self._orders: Dict[str, OrderRecord] = {}
        self._last_sync: Optional[datetime] = None
        self._positions: Dict[str, int] = {}

    def submit(self, order: Order, price: float = None) -> str:
        """提交订单

        Args:
            order: 订单对象
            price: 执行价格（None则使用order.price）

        Returns:
            订单ID
        """
        route = OrderRouter.get_route(order.symbol)
        order.price = price or order.price

        logger.info(f"提交订单: {order.symbol} {order.direction} {order.quantity} @ {order.price}")

        try:
            order_id = self.broker.submit_order(order)
            self._orders[order_id] = OrderRecord(
                order_id=order_id,
                symbol=order.symbol,
                direction=order.direction,
                quantity=order.quantity,
                price=order.price,
                status=OrderStatus.SUBMITTED,
            )
            logger.info(f"订单已提交: {order_id}")
            return order_id
        except Exception as e:
            logger.error(f"订单提交失败: {e}")
            order_id = f"ERR_{int(time.time() * 1000)}"
            self._orders[order_id] = OrderRecord(
                order_id=order_id,
                symbol=order.symbol,
                direction=order.direction,
                quantity=order.quantity,
                price=order.price,
                status=OrderStatus.REJECTED,
                error_message=str(e),
            )
            return order_id

    def cancel(self, order_id: str) -> bool:
        """取消订单"""
        if order_id not in self._orders:
            logger.warning(f"订单不存在: {order_id}")
            return False

        try:
            success = self.broker.cancel_order(order_id)
            if success:
                self._orders[order_id].status = OrderStatus.CANCELLED
                logger.info(f"订单已取消: {order_id}")
            return success
        except Exception as e:
            logger.error(f"取消订单失败: {e}")
            return False

    def retry_failed_orders(self):
        """重试失败的订单"""
        for order_id, record in self._orders.items():
            if record.status == OrderStatus.REJECTED and record.retry_count < self.MAX_RETRY:
                record.retry_count += 1
                logger.info(f"重试订单 {order_id} (第{record.retry_count}次)")
                try:
                    order = Order(
                        symbol=record.symbol,
                        direction=record.direction,
                        quantity=record.quantity,
                        price=record.price,
                    )
                    new_id = self.broker.submit_order(order)
                    record.status = OrderStatus.SUBMITTED
                    logger.info(f"重试成功: {order_id} -> {new_id}")
                except Exception as e:
                    logger.error(f"重试失败: {e}")

    def sync_positions(self) -> Dict[str, int]:
        """同步持仓"""
        try:
            self._positions = self.broker.get_positions()
            self._last_sync = datetime.now()
            return self._positions
        except Exception as e:
            logger.error(f"持仓同步失败: {e}")
            return self._positions

    def get_pending_orders(self) -> List[OrderRecord]:
        """获取待成交订单"""
        return [
            r for r in self._orders.values()
            if r.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL_FILLED)
        ]

    def update_order_status(self, order_id: str):
        """更新订单状态"""
        if order_id not in self._orders:
            return

        try:
            status = self.broker.get_order_status(order_id)
            self._orders[order_id].status = status
            self._orders[order_id].updated_at = datetime.now()
        except Exception as e:
            logger.error(f"更新订单状态失败: {e}")

    def get_order(self, order_id: str) -> Optional[OrderRecord]:
        """获取订单记录"""
        return self._orders.get(order_id)

    @property
    def positions(self) -> Dict[str, int]:
        """当前持仓"""
        return self._positions

    @property
    def cash(self) -> float:
        """可用资金"""
        try:
            info = self.broker.get_account_info()
            return info.get('cash', 0.0)
        except Exception as e:
            logger.error(f"获取账户信息失败: {e}")
            return 0.0
