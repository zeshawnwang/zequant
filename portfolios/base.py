"""仓位分配器抽象基类。"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List


class IPortfolioBuilder(ABC):
    """仓位分配器接口。

    实现类须提供:
      - allocate(signals, total_cash, current_positions) -> Dict[str, int]
        返回 {symbol: shares} 的买入计划(必须是 100 股的整数倍)。
    """

    @abstractmethod
    def allocate(
        self,
        signals: List,  # List[core.strategy.Signal]
        total_cash: float,
        current_positions: Dict,
    ) -> Dict[str, int]:
        """根据 BUY 信号分配现金,返回 {symbol: shares}。"""