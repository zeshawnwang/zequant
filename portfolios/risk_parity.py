"""
风险平价仓位分配器
波动率越低，仓位越高；波动率越高，仓位越低。
"""
from typing import Dict, List
import numpy as np
from .base import IPortfolioBuilder


class RiskParityBuilder(IPortfolioBuilder):
    """
    风险平价仓位分配。
    原理：每只股票对组合风险的贡献相等。
    风险贡献 = 仓位权重 × 波动率
    => 权重 ∝ 1/波动率
    """

    def __init__(self,
                 volatility_factor: str = 'volatility_20',
                 max_weight: float = 0.15,
                 reserve_cash_ratio: float = 0.1):
        self.volatility_factor = volatility_factor
        self.max_weight = max_weight
        self.reserve_cash_ratio = reserve_cash_ratio

    def allocate(self, signals, total_cash: float,
                 current_positions: Dict) -> Dict[str, int]:
        from core.strategy import SignalType
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]

        if not buy_signals:
            return {}

        # 提取波动率
        volatilities = []
        valid_signals = []
        for sig in buy_signals:
            vol = sig.factors.get(self.volatility_factor, 0.2) if hasattr(sig, 'factors') else 0.2
            vol = max(vol, 0.01)  # 避免除零
            if vol > 0 and vol < 2:  # 过滤异常值
                volatilities.append(vol)
                valid_signals.append(sig)

        if not volatilities:
            # 无有效波动率，等权分配
            volatilities = [0.2] * len(valid_signals)

        # 风险平价权重
        inv_vol = np.array([1/v for v in volatilities])
        weights = inv_vol / inv_vol.sum()

        # 限制最大权重
        weights = np.minimum(weights, self.max_weight)
        weights = weights / weights.sum()  # 重新归一化

        usable_cash = total_cash * (1 - self.reserve_cash_ratio)
        allocation = {}
        for i, sig in enumerate(valid_signals):
            stock_cash = usable_cash * weights[i]
            shares = int(stock_cash / sig.price / 100) * 100
            if shares >= 100:
                allocation[sig.symbol] = shares

        return allocation
