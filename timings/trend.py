"""
趋势择时器
使用均线/MACD/动量打分对候选股票产生 BUY/SELL/HOLD 信号。

设计:
- 对候选池(factor_data 中出现的所有 symbol)逐只计算趋势分 score ∈ [0, 1]
- score > buy_threshold:
    * 若已持仓 → HOLD
    * 若未持仓 → BUY(供组合构建器分配仓位)
- score < sell_threshold:
    * 若已持仓 → SELL
    * 若未持仓 → 无信号
"""
from typing import List
import numpy as np
import pandas as pd

# 统一使用 core.strategy 中的 Signal/SignalType,避免与 portfolios 端比较失败
from core.strategy import Signal, SignalType


class ITimingGenerator:
    """择时器基类"""

    def generate(self, factor_data: pd.DataFrame,
                 positions: List[str], cash: float) -> List[Signal]:
        raise NotImplementedError


class TrendTiming(ITimingGenerator):
    """
    趋势择时。
    打分维度:
      - MACD: macd > macd_signal 看多
      - 动量: momentum_5 > 0 且 > momentum_20 看多
      - RSI: 30~70 偏中性,>70 偏空(反弹乏力),<30 偏空(下跌惯性)
    各维度 0/1 取均值得到 score。
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
        if factor_data is None or factor_data.empty:
            return []

        signals: List[Signal] = []
        held = set(positions or [])

        # 取每个 symbol 的最新一行用于打分
        latest = (
            factor_data.sort_values('date')
            .groupby('symbol')
            .tail(1)
        )

        for _, row in latest.iterrows():
            symbol = row['symbol']
            score = self._calc_trend_score(row)
            price = float(row['close']) if 'close' in row and pd.notna(row['close']) else 0.0
            if price <= 0:
                continue

            factors_dict = {
                k: row[k] for k in
                ('momentum_5', 'momentum_20', 'rsi_14', 'macd', 'macd_signal',
                 'volatility_20', 'volume_ratio', 'boll_position')
                if k in row.index and pd.notna(row[k])
            }

            if score >= self.buy_threshold:
                if symbol in held:
                    signals.append(Signal(
                        symbol=symbol, signal_type=SignalType.HOLD,
                        strength=score, price=price,
                        reason=f"趋势保持({score:.2f})",
                        factors=factors_dict,
                    ))
                else:
                    signals.append(Signal(
                        symbol=symbol, signal_type=SignalType.BUY,
                        strength=score, price=price,
                        reason=f"趋势看多({score:.2f})",
                        factors=factors_dict,
                    ))
            elif score <= self.sell_threshold:
                if symbol in held:
                    signals.append(Signal(
                        symbol=symbol, signal_type=SignalType.SELL,
                        strength=1 - score, price=price,
                        reason=f"趋势转弱({score:.2f})",
                        factors=factors_dict,
                    ))
                # 未持仓且趋势差,直接忽略
            # 中间区间不发信号

        return signals

    def _calc_trend_score(self, row) -> float:
        """对单行(单只股票最新一日)的因子值打分。"""
        scores = []

        # MACD
        macd = row.get('macd') if hasattr(row, 'get') else None
        macd_sig = row.get('macd_signal') if hasattr(row, 'get') else None
        if pd.notna(macd) and pd.notna(macd_sig):
            scores.append(1.0 if macd > macd_sig else 0.0)

        # 动量
        m5 = row.get('momentum_5') if hasattr(row, 'get') else None
        m20 = row.get('momentum_20') if hasattr(row, 'get') else None
        if pd.notna(m5) and pd.notna(m20):
            if m5 > 0 and m5 > m20:
                scores.append(1.0)
            elif m5 < 0:
                scores.append(0.0)
            else:
                scores.append(0.5)

        # RSI
        rsi = row.get('rsi_14') if hasattr(row, 'get') else None
        if pd.notna(rsi):
            if 50 <= rsi <= 70:
                scores.append(1.0)
            elif 30 <= rsi < 50:
                scores.append(0.5)
            else:
                scores.append(0.0)

        return float(np.mean(scores)) if scores else 0.5