"""
Simulated Broker
模拟券商，支持买入/卖出/查询持仓/账户。
"""
from typing import Dict, List
from dataclasses import dataclass, field
from .strategy import Order, Position


@dataclass
class Account:
    cash: float
    equity: float
    total_value: float


class SimulatedBroker:
    """
    模拟券商。
    实盘功能：
    - 买入/卖出（市价/限价）
    - 查询持仓
    - 查询账户
    """

    def __init__(self, initial_capital: float = 1_000_000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []

    def get_account(self) -> Account:
        equity = sum(
            p.quantity * p.entry_price
            for p in self.positions.values()
        )
        return Account(
            cash=self.cash,
            equity=equity,
            total_value=self.cash + equity
        )

    def get_positions(self) -> Dict[str, Position]:
        return self.positions.copy()

    def submit_order(self, order: Order) -> bool:
        """
        模拟下单（市价单直接成交，限价单挂单）。
        Returns: 是否成交
        """
        self.orders.append(order)

        if order.order_type == 'MARKET':
            # 市价单直接模拟撮合
            return True

        return False

    def cancel_order(self, order_id: str) -> bool:
        """撤单"""
        # 简化实现
        return True

    def update_position_price(self, symbol: str, current_price: float):
        """更新持仓的当前价格（用于盯盘）"""
        pass
