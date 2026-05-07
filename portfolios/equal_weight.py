"""
仓位分配器基类
定义接口。
"""
from abc import ABC, abstractmethod
from typing import Dict, List


class IPortfolioBuilder(ABC):
    """仓位分配器接口"""

    @abstractmethod
    def allocate(self,
                 signals,  # List[Signal]
                 total_cash: float,
                 current_positions: Dict) -> Dict[str, int]:
        """
        分配仓位。
        Returns: {symbol: shares}
        """
        pass


class EqualWeightBuilder(IPortfolioBuilder):
    """
    等权重分配。
    每个标的分配等量资金。
    """

    def __init__(self, reserve_cash_ratio: float = 0.1):
        self.reserve_cash_ratio = reserve_cash_ratio

    def allocate(self, signals, total_cash: float,
                 current_positions: Dict) -> Dict[str, int]:
        from core.strategy import SignalType
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]

        if not buy_signals:
            return {}

        n = len(buy_signals)
        per_stock_cash = total_cash * (1 - self.reserve_cash_ratio) / n

        allocation = {}
        for sig in buy_signals:
            shares = int(per_stock_cash / sig.price / 100) * 100
            if shares >= 100:
                allocation[sig.symbol] = shares

        return allocation
