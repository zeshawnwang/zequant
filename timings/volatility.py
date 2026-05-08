"""波动率择时器。

仅对【已持仓】标的产生信号:
  - vol > high_threshold: SELL(高波动减仓)
  - vol < low_threshold:  HOLD(低波动持有)

注意:本择时器**不会产生 BUY 信号**,需与会出 BUY 的择时器(如 TrendTiming)
配合 CompositeTiming 使用,或仅作为风控择时叠加在 TrendTiming 之后。
"""
from __future__ import annotations
from typing import List
import pandas as pd

from core.strategy import Signal, SignalType
from .base import ITimingGenerator


class VolatilityTiming(ITimingGenerator):
    """
    波动率择时。
    - vol > high_threshold: 高波动，减仓/清仓
    - vol < low_threshold: 低波动，持仓/加仓
    """

    def __init__(self,
                 volatility_factor: str = 'volatility_20',
                 high_threshold: float = 0.30,
                 low_threshold: float = 0.15,
                 reduce_ratio: float = 0.5):
        self.volatility_factor = volatility_factor
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.reduce_ratio = reduce_ratio  # 高波动时保留仓位比例

    def generate(self, factor_data: pd.DataFrame,
                positions: List[str], cash: float) -> List[Signal]:
        signals = []
        latest = factor_data.groupby('symbol').tail(1)

        for symbol in positions:
            if symbol not in latest['symbol'].values:
                continue

            row = latest[latest['symbol'] == symbol]
            if self.volatility_factor not in row.columns:
                continue

            vol = row[self.volatility_factor].iloc[-1]
            price = row['close'].iloc[-1] if 'close' in row.columns else 0

            if vol > self.high_threshold:
                signals.append(Signal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    strength=min(vol / 0.5, 1.0),
                    price=price,
                    reason=f"波动率高({vol:.3f}>{self.high_threshold})"
                ))
            elif vol < self.low_threshold:
                signals.append(Signal(
                    symbol=symbol,
                    signal_type=SignalType.HOLD,
                    strength=1 - vol,
                    price=price,
                    reason=f"波动率低({vol:.3f}<{self.low_threshold})"
                ))

        return signals
