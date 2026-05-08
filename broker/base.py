"""交易接口抽象基类。

定义所有 Broker 必须实现的接口，包括连接、下单、查询持仓/资金/行情、撤单等。
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any


class OrderStatus(Enum):
    """订单状态。"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL_FILLED = "partial_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class BrokerOrder:
    """Broker 层面的订单对象。"""
    order_id: str
    symbol: str
    direction: str
    quantity: int
    filled_quantity: int = 0
    price: Optional[float] = None
    order_type: str = "MARKET"
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    reason: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def remaining_quantity(self) -> int:
        return self.quantity - self.filled_quantity

    def to_dict(self) -> dict:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "direction": self.direction,
            "quantity": self.quantity,
            "filled_quantity": self.filled_quantity,
            "price": self.price,
            "order_type": self.order_type,
            "status": self.status.value,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "reason": self.reason,
        }


@dataclass
class Quote:
    """行情快照。"""
    symbol: str
    last_price: float
    bid: float = 0.0
    ask: float = 0.0
    volume: int = 0
    timestamp: Optional[datetime] = None


@dataclass
class Position:
    """持仓快照。"""
    symbol: str
    quantity: int
    avg_cost: float
    market_value: float = 0.0
    unrealized_pnl: float = 0.0


class IBroker(ABC):
    """交易接口抽象基类。"""

    @abstractmethod
    def connect(self):
        """建立与交易通道的连接。"""
        ...

    @abstractmethod
    def buy(self, symbol: str, quantity: int, price: Optional[float] = None,
            order_type: str = "MARKET") -> BrokerOrder:
        """买入下单。"""
        ...

    @abstractmethod
    def sell(self, symbol: str, quantity: int, price: Optional[float] = None,
             order_type: str = "MARKET") -> BrokerOrder:
        """卖出下单。"""
        ...

    @abstractmethod
    def get_positions(self) -> List[Position]:
        """获取当前持仓列表。"""
        ...

    @abstractmethod
    def get_cash(self) -> float:
        """获取可用资金。"""
        ...

    @abstractmethod
    def get_quote(self, symbol: str) -> Quote:
        """获取某只股票的最新行情。"""
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """撤销指定订单。"""
        ...

    @abstractmethod
    def get_orders(self, status: Optional[OrderStatus] = None) -> List[BrokerOrder]:
        """获取订单列表，可按状态过滤。"""
        ...

    def disconnect(self):
        """断开连接（可选实现）。"""
        pass

    def is_connected(self) -> bool:
        """是否已连接（可选实现）。"""
        return True
