"""Mock Broker — 用于单元测试的固定响应 Broker。

所有接口返回预设值，不依赖外部行情，不执行真实计算。
方便在 CI / 单元测试中快速验证上层逻辑。
"""
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from .base import IBroker, BrokerOrder, OrderStatus, Quote, Position

logger = logging.getLogger(__name__)


class MockBroker(IBroker):
    """测试专用 Broker，返回固定响应。

    Parameters
    ----------
    cash : float
        固定返回的可用资金。
    positions : List[Position]
        固定返回的持仓列表。
    quote_price : float
        固定返回的行情价格。
    order_status : OrderStatus
        固定返回的订单状态（默认 FILLED）。
    """

    def __init__(self,
                 cash: float = 1_000_000,
                 positions: List[Position] = None,
                 quote_price: float = 10.0,
                 order_status: OrderStatus = OrderStatus.FILLED):
        self._cash = cash
        self._positions = positions or []
        self._quote_price = quote_price
        self._order_status = order_status
        self._orders: Dict[str, BrokerOrder] = {}
        self._connected = False

    # ---------- 连接 ----------
    def connect(self):
        self._connected = True
        logger.info("[MockBroker] 已连接(测试模式).")

    def disconnect(self):
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    # ---------- 行情 ----------
    def get_quote(self, symbol: str) -> Quote:
        return Quote(
            symbol=symbol,
            last_price=self._quote_price,
            bid=self._quote_price * 0.99,
            ask=self._quote_price * 1.01,
            volume=10000,
            timestamp=datetime.now(),
        )

    # ---------- 下单 ----------
    def buy(self, symbol: str, quantity: int, price: Optional[float] = None,
            order_type: str = "MARKET") -> BrokerOrder:
        if not self._connected:
            raise RuntimeError("Broker 未连接")
        order = self._make_order(symbol, "BUY", quantity, price, order_type)
        logger.info(f"[MockBroker] 模拟买入 {symbol} {quantity}")
        return order

    def sell(self, symbol: str, quantity: int, price: Optional[float] = None,
             order_type: str = "MARKET") -> BrokerOrder:
        if not self._connected:
            raise RuntimeError("Broker 未连接")
        order = self._make_order(symbol, "SELL", quantity, price, order_type)
        logger.info(f"[MockBroker] 模拟卖出 {symbol} {quantity}")
        return order

    # ---------- 查询 ----------
    def get_positions(self) -> List[Position]:
        return list(self._positions)

    def get_cash(self) -> float:
        return self._cash

    def get_orders(self, status: Optional[OrderStatus] = None) -> List[BrokerOrder]:
        orders = list(self._orders.values())
        if status is not None:
            orders = [o for o in orders if o.status == status]
        return orders

    # ---------- 撤单 ----------
    def cancel_order(self, order_id: str) -> bool:
        order = self._orders.get(order_id)
        if order is None:
            return False
        if order.status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED):
            return False
        order.status = OrderStatus.CANCELLED
        order.updated_at = datetime.now()
        return True

    # ---------- 内部 ----------
    def _make_order(self, symbol, direction, quantity, price, order_type) -> BrokerOrder:
        order = BrokerOrder(
            order_id=str(uuid.uuid4())[:16],
            symbol=symbol,
            direction=direction,
            quantity=quantity,
            price=price or self._quote_price,
            order_type=order_type,
            status=self._order_status,
        )
        if self._order_status == OrderStatus.FILLED:
            order.filled_quantity = quantity
        self._orders[order.order_id] = order
        return order

    # ---------- 测试辅助 ----------
    def set_cash(self, cash: float):
        self._cash = cash

    def set_positions(self, positions: List[Position]):
        self._positions = positions

    def set_quote_price(self, price: float):
        self._quote_price = price

    def set_order_status(self, status: OrderStatus):
        self._order_status = status
