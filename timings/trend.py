"""
趋势择时器
使用均线交叉和因子方向判断多空。
"""
from typing import List, Dict
import numpy as np
import pandas as pd
from enum import Enum


class SignalType(Enum):
    BUY = 1
    SELL = -1
    HOLD = 0


class Signal:
    def __init__(self, symbol: str, signal_type: SignalType,
                 strength: float, price: float, reason: str = ""):
        self.symbol = symbol
        self.signal_type = signal_type
        self.strength = strength
        self.price = price
        self.reason = reason

    def __repr__(self):
        return f"Signal({self.symbol}, {self.signal_type.name}, 强度={self.strength:.2f}, {self.reason})"


class ITimingGenerator:
    """择时器基类"""

    def generate(self, factor_data: pd.DataFrame,
                positions: List[str], cash: float) -> List[Signal]:
        raise NotImplementedError


class TrendTiming(ITimingGenerator):
    """
    趋势择时。
    规则：
    - 均线多头排列（短期 > 中期 > 长期）=> 看多
    - MACD > 0 => 看多
    - 跌破均线 => 看空
    """

    def __init__(self,
                 sma_short: int = 5,
                 sma_medium: int = 20,
                 buy_threshold: float = 0.6,
                 sell_threshold: float = 0.4):
        self.sma_short = sma_short
        self.sma_medium = sma_medium
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def generate(self, factor_data: pd.DataFrame,
                positions: List[str], cash: float) -> List[Signal]:
        signals = []
        for symbol in positions:
            df = factor_data[factor_data['symbol'] == symbol].tail(30)
            if len(df) < max(self.sma_short, self.sma_medium):
                continue

            score = self._calc_trend_score(df)
            latest_price = df['close'].iloc[-1]

            if score < self.sell_threshold:
                signals.append(Signal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    strength=1 - score,
                    price=latest_price,
                    reason=f"趋势转弱({score:.2f})"
                ))
            elif score > self.buy_threshold:
                signals.append(Signal(
                    symbol=symbol,
                    signal_type=SignalType.HOLD,
                    strength=score,
                    price=latest_price,
                    reason=f"趋势保持({score:.2f})"
                ))
        return signals

    def _calc_trend_score(self, df: pd.DataFrame) -> float:
        """计算趋势得分 0-1"""
        scores = []
        price = df['close']

        # 均线趋势
        if f'sma_{self.sma_short}' in df.columns and f'sma_{self.sma_medium}' in df.columns:
            sma_s = df[f'sma_{self.sma_short}'].iloc[-1]
            sma_m = df[f'sma_{self.sma_medium}'].iloc[-1]
            prev_sma_s = df[f'sma_{self.sma_short}'].iloc[-2]
            prev_sma_m = df[f'sma_{self.sma_medium}'].iloc[-2]

            # 金叉
            if sma_s > sma_m and prev_sma_s <= prev_sma_m:
                scores.append(1.0)
            # 死叉
            elif sma_s < sma_m and prev_sma_s >= prev_sma_m:
                scores.append(0.0)
            # 多头排列
            elif sma_s > sma_m:
                scores.append(0.8)
            else:
                scores.append(0.2)

        # MACD趋势
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            macd = df['macd'].iloc[-1]
            macd_s = df['macd_signal'].iloc[-1]
            scores.append(1.0 if macd > macd_s else 0.0)

        # 动量趋势
        if 'momentum_5' in df.columns and 'momentum_20' in df.columns:
            mom5 = df['momentum_5'].iloc[-1]
            mom20 = df['momentum_20'].iloc[-1]
            if mom5 > mom20 and mom5 > 0:
                scores.append(1.0)
            elif mom5 < mom20 or mom5 < 0:
                scores.append(0.0)
            else:
                scores.append(0.5)

        return np.mean(scores) if scores else 0.5
