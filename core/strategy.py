"""
策略核心模块
定义选股器+择时器+仓位分配器的组装接口。
"""
from typing import List, Dict, Optional
from dataclasses import dataclass, field
import pandas as pd


@dataclass
class Signal:
    """交易信号"""
    symbol: str
    signal_type: 'SignalType'  # BUY/SELL/HOLD
    strength: float            # 信号强度 0-1
    price: float              # 参考价格
    reason: str = ""
    factors: Dict = field(default_factory=dict)


class SignalType:
    BUY = 1
    SELL = -1
    HOLD = 0


@dataclass
class Position:
    """持仓"""
    symbol: str
    quantity: int
    entry_price: float
    entry_date: str
    stop_loss: float = 0.0
    take_profit: float = 0.0

    @property
    def market_value(self):
        return self.quantity * self.entry_price


@dataclass
class Order:
    """订单"""
    symbol: str
    direction: str           # BUY / SELL
    quantity: int
    price: float = 0.0
    order_type: str = "MARKET"  # MARKET / LIMIT
    reason: str = ""


class QuantStrategy:
    """
    完整量化策略。
    组合：选股器 + 择时器 + 仓位分配器。
    """

    def __init__(self,
                 name: str,
                 selector,       # IStockSelector
                 timing,         # ITimingGenerator
                 portfolio,      # IPortfolioBuilder
                 top_n: int = 50):
        self.name = name
        self.selector = selector
        self.timing = timing
        self.portfolio = portfolio
        self.top_n = top_n

    def generate_orders(self,
                      factor_data: pd.DataFrame,
                      current_positions: Dict[str, Position],
                      cash: float,
                      date,
                      top_n: int = None) -> List[Order]:
        """
        生成订单。
        Step 1: 选股
        Step 2: 合并持仓候选池
        Step 3: 择时
        Step 4: 仓位分配
        Step 5: 生成订单
        """
        top_n = top_n or self.top_n
        orders = []

        # Step 1: 选股
        selected = self.selector.select(factor_data, date, top_n)

        # Step 2: 合并候选池
        current_symbols = list(current_positions.keys())
        candidate_pool = list(set(selected) | set(current_symbols))
        pool_data = factor_data[factor_data['symbol'].isin(candidate_pool)]

        # Step 3: 择时
        signals = self.timing.generate(pool_data, current_symbols, cash)

        # Step 4: 仓位分配
        allocation = self.portfolio.allocate(signals, cash, current_positions)

        # Step 5: 生成买入订单
        for symbol, shares in allocation.items():
            if symbol not in current_positions:
                orders.append(Order(
                    symbol=symbol,
                    direction='BUY',
                    quantity=shares,
                    price=signals[next(i for i, s in enumerate(signals) if s.symbol == symbol)].price
                                   if any(s.symbol == symbol for s in signals) else 0,
                    reason='新买入'
                ))

        # Step 6: 生成卖出订单（择时信号）
        for sig in signals:
            if sig.signal_type == SignalType.SELL:
                if sig.symbol in current_positions:
                    orders.append(Order(
                        symbol=sig.symbol,
                        direction='SELL',
                        quantity=current_positions[sig.symbol].quantity,
                        price=sig.price,
                        reason=sig.reason
                    ))

        return orders

    def get_description(self) -> str:
        selector_desc = self.selector.get_description() if hasattr(self.selector, 'get_description') else str(self.selector)
        return f"""策略: {self.name}
选股: {selector_desc}
择时: {self.timing.__class__.__name__}
仓位: {self.portfolio.__class__.__name__}"""
