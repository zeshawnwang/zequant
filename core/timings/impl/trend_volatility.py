"""趋势+波动率二重复合择时器。

将 TrendTiming 和 VolatilityTiming 结合为一个独立的择时器:
  - 趋势判断: MACD + 动量 + RSI 打分
  - 波动率风控: 高波动时强制减仓

信号逻辑:
  1. 波动率 > high_threshold → SELL (不看趋势)
  2. 波动率 < high_threshold 且趋势看多 → BUY
  3. 波动率 < high_threshold 且趋势看空 → SELL
  4. 中间区间 → HOLD

用法
----
    from timings.trend_volatility import TrendVolatilityTiming

    timing = TrendVolatilityTiming(
        sma_short=5,
        sma_medium=20,
        buy_threshold=0.6,
        sell_threshold=0.4,
        volatility_factor="volatility_20",
        high_threshold=0.05,    # 5% 年化波动率，高于此值减仓
        low_threshold=0.03,    # 3% 年化波动率，低于此值正常操作
    )
"""
from __future__ import annotations
from typing import List
import numpy as np
import pandas as pd

from core.strategy import Signal, SignalType
from core.timings.base.timing import ITimingGenerator


class TrendVolatilityTiming(ITimingGenerator):
    """
    趋势 + 波动率二重复合择时。

    核心逻辑:
      - 高波动环境: 强制减仓,不看趋势
      - 正常波动环境: 按趋势信号操作

    适用场景:
      - 牛市: 趋势信号 + 正常波动 → 持仓
      - 熊市: 高波动 → 自动减仓避险
      - 震荡市: 趋势信号切换 → 减少无效交易
    """

    def __init__(
        self,
        sma_short: int = 5,
        sma_medium: int = 20,
        buy_threshold: float = 0.6,
        sell_threshold: float = 0.4,
        volatility_factor: str = "volatility_20",
        high_threshold: float = 0.05,
        low_threshold: float = 0.03,
    ):
        self.sma_short = sma_short
        self.sma_medium = sma_medium
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.volatility_factor = volatility_factor
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold

    def generate(
        self,
        factor_data: pd.DataFrame,
        positions: List[str],
        cash: float,
        date=None,
    ) -> List[Signal]:
        if factor_data is None or factor_data.empty:
            return []

        signals: List[Signal] = []
        held = set(positions or [])

        df = factor_data
        if date is not None and "date" in df.columns:
            df = df[df["date"] < date]
        if df.empty:
            return []

        latest = df.sort_values("date").groupby("symbol").tail(1)

        for _, row in latest.iterrows():
            symbol = row["symbol"]
            price = (
                float(row["close"])
                if "close" in row and pd.notna(row["close"])
                else 0.0
            )
            if price <= 0:
                continue

            trend_score = self._calc_trend_score(row)
            vol = row.get(self.volatility_factor)
            vol = float(vol) if pd.notna(vol) else None

            factors_dict = {
                k: row[k]
                for k in (
                    "momentum_5",
                    "momentum_20",
                    "rsi_14",
                    "macd",
                    "macd_signal",
                    "volatility_20",
                    "volume_ratio",
                    "boll_position",
                )
                if k in row.index and pd.notna(row.get(k))
            }

            if vol is not None and vol > self.high_threshold:
                if symbol in held:
                    signals.append(
                        Signal(
                            symbol=symbol,
                            signal_type=SignalType.SELL,
                            strength=min(vol / 0.5, 1.0),
                            price=price,
                            reason=f"高波动减仓({vol:.3f}>{self.high_threshold})",
                            factors=factors_dict,
                        )
                    )
                continue

            if trend_score >= self.buy_threshold:
                if symbol in held:
                    signals.append(
                        Signal(
                            symbol=symbol,
                            signal_type=SignalType.HOLD,
                            strength=trend_score,
                            price=price,
                            reason=f"趋势看多({trend_score:.2f})",
                            factors=factors_dict,
                        )
                    )
                else:
                    signals.append(
                        Signal(
                            symbol=symbol,
                            signal_type=SignalType.BUY,
                            strength=trend_score,
                            price=price,
                            reason=f"趋势看多({trend_score:.2f})",
                            factors=factors_dict,
                        )
                    )
            elif trend_score <= self.sell_threshold:
                if symbol in held:
                    signals.append(
                        Signal(
                            symbol=symbol,
                            signal_type=SignalType.SELL,
                            strength=1 - trend_score,
                            price=price,
                            reason=f"趋势转弱({trend_score:.2f})",
                            factors=factors_dict,
                        )
                    )

        return signals

    def _calc_trend_score(self, row) -> float:
        """对单行(单只股票最新一日)的因子值打分。"""
        scores = []

        macd = row.get("macd") if hasattr(row, "get") else None
        macd_sig = row.get("macd_signal") if hasattr(row, "get") else None
        if pd.notna(macd) and pd.notna(macd_sig):
            scores.append(1.0 if macd > macd_sig else 0.0)

        m5 = row.get("momentum_5") if hasattr(row, "get") else None
        m20 = row.get("momentum_20") if hasattr(row, "get") else None
        if pd.notna(m5) and pd.notna(m20):
            if m5 > 0 and m5 > m20:
                scores.append(1.0)
            elif m5 < 0:
                scores.append(0.0)
            else:
                scores.append(0.5)

        rsi = row.get("rsi_14") if hasattr(row, "get") else None
        if pd.notna(rsi):
            if 50 <= rsi <= 70:
                scores.append(1.0)
            elif 30 <= rsi < 50:
                scores.append(0.5)
            else:
                scores.append(0.0)

        return float(np.mean(scores)) if scores else 0.5
