"""仓位管理器模块。

提供多种仓位确定方式。

仓位确定方式
------------
    FixedPositionSizer      : 固定仓位
    TrendPositionSizer     : 趋势仓位
    VolatilityPositionSizer : 波动率仓位
    RiskParityPositionSizer: 风险平价仓位

用法
----
    from core.signals.position import (
        FixedPositionSizer, TrendPositionSizer, VolatilityPositionSizer,
    )

    # 固定仓位
    sizer = FixedPositionSizer(position=0.8)

    # 趋势仓位
    sizer = TrendPositionSizer(
        bullish_threshold=0.6,
        bearish_threshold=0.4,
        max_position=1.0,
        min_position=0.0,
    )

    # 波动率仓位
    sizer = VolatilityPositionSizer(
        target_volatility=0.15,
        max_position=1.0,
    )

    position = sizer.get_position(date, market_data)
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
import pandas as pd


class IPositionSizer(ABC):
    """仓位确定器抽象基类。"""

    @abstractmethod
    def get_position(
        self,
        date: Any,
        market_data: Optional[pd.DataFrame] = None,
        current_position: float = 1.0,
    ) -> float:
        """
        返回仓位系数 (0~1)。

        Args:
            date: 当前日期
            market_data: 市场数据
            current_position: 当前仓位

        Returns:
            目标仓位系数 (0~1)
        """
        pass


class FixedPositionSizer(IPositionSizer):
    """固定仓位确定器。"""

    def __init__(self, position: float = 1.0):
        if not 0 <= position <= 1:
            raise ValueError(f"position must be in [0, 1], got {position}")
        self.position = position

    def get_position(
        self,
        date: Any,
        market_data: Optional[pd.DataFrame] = None,
        current_position: float = 1.0,
    ) -> float:
        return self.position


class TrendPositionSizer(IPositionSizer):
    """
    趋势仓位确定器。

    根据趋势信号强度确定仓位：
        - 强趋势（得分 > buy_threshold）→ 满仓
        - 弱趋势（得分 > sell_threshold）→ 半仓
        - 无趋势（得分 <= sell_threshold）→ 空仓

    也支持平滑过渡。
    """

    def __init__(
        self,
        bullish_threshold: float = 0.6,
        bearish_threshold: float = 0.4,
        max_position: float = 1.0,
        min_position: float = 0.0,
        smooth: bool = False,
    ):
        self.bullish_threshold = bullish_threshold
        self.bearish_threshold = bearish_threshold
        self.max_position = max_position
        self.min_position = min_position
        self.smooth = smooth
        self._last_position = max_position

    def get_position(
        self,
        date: Any,
        market_data: Optional[pd.DataFrame] = None,
        current_position: float = 1.0,
    ) -> float:
        if market_data is None or market_data.empty:
            return self._last_position

        df = market_data[market_data["date"] < date] if "date" in market_data.columns else market_data
        if df.empty:
            return self._last_position

        latest = df.sort_values("date").iloc[-1]

        trend_score = self._calc_trend_score(latest)

        if self.smooth:
            target = self._calc_continuous_position(trend_score)
            position = current_position * 0.7 + target * 0.3
        else:
            position = self._calc_discrete_position(trend_score)

        self._last_position = position
        return position

    def _calc_trend_score(self, row) -> float:
        scores = []

        macd = row.get("macd")
        macd_signal = row.get("macd_signal")
        if pd.notna(macd) and pd.notna(macd_signal):
            scores.append(1.0 if macd > macd_signal else 0.0)

        m5 = row.get("momentum_5")
        m20 = row.get("momentum_20")
        if pd.notna(m5) and pd.notna(m20):
            if m5 > 0 and m5 > m20:
                scores.append(1.0)
            elif m5 < 0:
                scores.append(0.0)
            else:
                scores.append(0.5)

        rsi = row.get("rsi_14")
        if pd.notna(rsi):
            if 50 <= rsi <= 70:
                scores.append(1.0)
            elif 30 <= rsi < 50:
                scores.append(0.5)
            else:
                scores.append(0.0)

        return sum(scores) / len(scores) if scores else 0.5

    def _calc_discrete_position(self, trend_score: float) -> float:
        if trend_score >= self.bullish_threshold:
            return self.max_position
        elif trend_score <= self.bearish_threshold:
            return self.min_position
        else:
            return (self.max_position + self.min_position) / 2

    def _calc_continuous_position(self, trend_score: float) -> float:
        if trend_score >= self.bullish_threshold:
            return self.max_position
        elif trend_score <= self.bearish_threshold:
            return self.min_position
        else:
            t = (trend_score - self.bearish_threshold) / (
                self.bullish_threshold - self.bearish_threshold
            )
            return self.min_position + t * (self.max_position - self.min_position)


class VolatilityPositionSizer(IPositionSizer):
    """
    波动率仓位确定器。

    根据市场波动率调整仓位：
        - 低波动 → 满仓
        - 高波动 → 降仓

    公式: position = min(max_position, target_volatility / current_volatility)
    """

    def __init__(
        self,
        volatility_factor: str = "volatility_20",
        target_volatility: float = 0.15,
        max_position: float = 1.0,
        min_position: float = 0.2,
        lookback: int = 20,
    ):
        self.volatility_factor = volatility_factor
        self.target_volatility = target_volatility
        self.max_position = max_position
        self.min_position = min_position
        self.lookback = lookback

    def get_position(
        self,
        date: Any,
        market_data: Optional[pd.DataFrame] = None,
        current_position: float = 1.0,
    ) -> float:
        if market_data is None or market_data.empty:
            return self.max_position

        df = market_data[market_data["date"] < date] if "date" in market_data.columns else market_data
        if df.empty:
            return self.max_position

        vol = df[self.volatility_factor].dropna()
        if vol.empty:
            return self.max_position

        recent_vol = vol.tail(self.lookback).mean()

        if recent_vol <= 0:
            return self.max_position

        position = self.target_volatility / recent_vol

        position = max(self.min_position, min(self.max_position, position))

        return position


class RiskParityPositionSizer(IPositionSizer):
    """
    风险平价仓位确定器。

    根据目标风险预算分配仓位，使得各资产对组合风险的贡献相等。
    """

    def __init__(
        self,
        volatility_factor: str = "volatility_20",
        target_risk: float = 0.15,
        max_position: float = 1.0,
    ):
        self.volatility_factor = volatility_factor
        self.target_risk = target_risk
        self.max_position = max_position

    def get_position(
        self,
        date: Any,
        market_data: Optional[pd.DataFrame] = None,
        current_position: float = 1.0,
    ) -> float:
        if market_data is None or market_data.empty:
            return self.max_position

        df = market_data[market_data["date"] < date] if "date" in market_data.columns else market_data
        if df.empty:
            return self.max_position

        vol = df[self.volatility_factor].dropna().tail(20).mean()

        if vol <= 0:
            return self.max_position

        position = self.target_risk / vol

        return min(self.max_position, max(0, position))


class CompositePositionSizer(IPositionSizer):
    """
    复合仓位确定器。

    组合多个仓位确定器的输出。
    """

    def __init__(
        self,
        sizers: list,
        weights: Optional[list] = None,
        mode: str = "average",
    ):
        """
        Args:
            sizers: 仓位确定器列表
            weights: 各确定器的权重
            mode: "average", "min", "max", "vote"
        """
        self.sizers = sizers
        self.weights = weights or [1.0] * len(sizers)
        self.mode = mode

    def get_position(
        self,
        date: Any,
        market_data: Optional[pd.DataFrame] = None,
        current_position: float = 1.0,
    ) -> float:
        positions = [
            sizer.get_position(date, market_data, current_position)
            for sizer in self.sizers
        ]

        if self.mode == "average":
            return sum(p * w for p, w in zip(positions, self.weights)) / sum(
                self.weights
            )
        elif self.mode == "min":
            return min(positions)
        elif self.mode == "max":
            return max(positions)
        elif self.mode == "vote":
            bullish = sum(1 for p in positions if p >= 0.6)
            bearish = sum(1 for p in positions if p <= 0.4)
            if bullish > bearish:
                return sum(p for p in positions if p >= 0.6) / max(1, bullish)
            elif bearish > bullish:
                return sum(p for p in positions if p <= 0.4) / max(1, bearish)
            else:
                return sum(positions) / len(positions)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
