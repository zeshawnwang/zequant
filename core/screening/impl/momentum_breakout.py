"""
动量突破选股器集合

根据《职业投资者如何分析股票选股》方法论,实现三类选股器:
  1. TrendBreakoutSelector: 趋势突破选股器
  2. OversoldReboundSelector: 超跌反弹选股器
  3. ChipConcentrationSelector: 筹码集中选股器

调用:
    from core.screening.impl.momentum_breakout import (
        TrendBreakoutSelector,
        OversoldReboundSelector,
        ChipConcentrationSelector,
    )
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from ..base.selector import IStockSelector


def _latest_per_symbol(df: pd.DataFrame, date=None) -> pd.DataFrame:
    """取每个 symbol 在 date 之前(严格小于)的最新一条数据。"""
    if df is None or df.empty:
        return pd.DataFrame()
    if date is not None and "date" in df.columns:
        df = df[df["date"] < date]
    if df.empty:
        return pd.DataFrame()
    return df.sort_values("date").groupby("symbol").tail(1).set_index("symbol")


# ===================================================================
#  1. TrendBreakoutSelector —— 趋势突破选股器
# ===================================================================

class TrendBreakoutSelector(IStockSelector):
    """趋势突破选股器。

    核心逻辑(多条件共振):
      - 均线多头排列过滤: MA5 > MA20 > MA60
      - 股价在60日均线上方
      - MA60 和 MA120 向上发散(斜率为正)
      - MACD 在零轴上方
      - 放量突破信号(量比 > 阈值)
    """

    def __init__(
        self,
        top_n: int = 50,
        min_volume_ratio: float = 1.5,
    ):
        """
        Args:
            top_n: 默认选股数量
            min_volume_ratio: 最低放量倍数(当日量/前20日均量)
        """
        self.top_n = top_n
        self.min_volume_ratio = min_volume_ratio

    @property
    def factor_names(self) -> List[str]:
        return [
            "ma5", "ma20", "ma60", "ma120",
            "ma_alignment_score", "ma60_trend", "ma120_trend",
            "macd_above_zero", "volume_breakout_ratio",
            "close",
        ]

    def select(self, factor_data: pd.DataFrame, date, top_n: int) -> List[str]:
        latest = _latest_per_symbol(factor_data, date)
        if latest.empty:
            return []

        mask = pd.Series(True, index=latest.index)

        if {"ma5", "ma20", "ma60"}.issubset(latest.columns):
            mask &= (latest["ma5"] > latest["ma20"]) & (latest["ma20"] > latest["ma60"])

        if "ma60" in latest.columns:
            mask &= latest["close"] > latest["ma60"]

        if "ma60_trend" in latest.columns:
            mask &= latest["ma60_trend"] > 0
        if "ma120_trend" in latest.columns:
            mask &= latest["ma120_trend"] > 0

        if "macd_above_zero" in latest.columns:
            mask &= latest["macd_above_zero"] > 0

        if "volume_breakout_ratio" in latest.columns:
            mask &= latest["volume_breakout_ratio"] >= self.min_volume_ratio

        candidates = latest[mask].index.tolist()

        if "volume_breakout_ratio" in latest.columns:
            candidates_df = latest.loc[candidates]
            candidates_df = candidates_df.sort_values("volume_breakout_ratio", ascending=False)
            candidates = candidates_df.head(top_n or self.top_n).index.tolist()
        else:
            candidates = candidates[: top_n or self.top_n]

        return candidates

    def get_description(self) -> str:
        return "趋势突破选股器: 均线多头+放量突破+MACD零轴上方"


# ===================================================================
#  2. OversoldReboundSelector —— 超跌反弹选股器
# ===================================================================

class OversoldReboundSelector(IStockSelector):
    """超跌反弹选股器。

    核心逻辑(三共振):
      - 日线MACD在0轴下方形成圆弧底 + 金叉
      - 股价站上21日线但离99日线空间大
      - 5日、21日均线由陡峭向下转为走平并勾头向上
      - KDJ低位金叉或二次金叉(用MACD金叉替代)
    """

    def __init__(
        self,
        top_n: int = 50,
        ma_short: int = 5,
        ma_medium: int = 21,
        ma_long: int = 99,
    ):
        """
        Args:
            top_n: 默认选股数量
            ma_short: 短期均线窗口
            ma_medium: 中期均线窗口
            ma_long: 长期均线窗口
        """
        self.top_n = top_n
        self.ma_short = ma_short
        self.ma_medium = ma_medium
        self.ma_long = ma_long

    @property
    def factor_names(self) -> List[str]:
        return [
            "macd", "macd_signal", "macd_above_zero",
            "macd_golden_cross", "macd_arc_bottom",
            "ma5", "ma20", "close",
            "ma_angle_20",
        ]

    def select(self, factor_data: pd.DataFrame, date, top_n: int) -> List[str]:
        latest = _latest_per_symbol(factor_data, date)
        if latest.empty:
            return []

        score = pd.Series(0.0, index=latest.index)

        if "macd_arc_bottom" in latest.columns:
            arc_score = pd.to_numeric(latest["macd_arc_bottom"], errors="coerce").fillna(0)
            score += arc_score * 2

        if "macd_golden_cross" in latest.columns:
            golden = pd.to_numeric(latest["macd_golden_cross"], errors="coerce").fillna(0)
            score += golden * 2

        if "macd_above_zero" in latest.columns:
            below_zero = (latest["macd_above_zero"] == 0).astype(float)
            score += below_zero

        close = pd.to_numeric(latest["close"], errors="coerce")
        ma_medium_col = f"ma{self.ma_medium}"
        ma_long_col = f"ma{self.ma_long}"

        if ma_medium_col in latest.columns:
            ma_med = pd.to_numeric(latest[ma_medium_col], errors="coerce")
            above_medium = (close > ma_med).astype(float)
            score += above_medium * 1.5

        if ma_long_col in latest.columns:
            ma_lng = pd.to_numeric(latest[ma_long_col], errors="coerce")
            space = (ma_lng - close) / close.replace(0, np.nan)
            score += space.clip(0, 0.5) * 2

        if "ma_angle_20" in latest.columns:
            angle = pd.to_numeric(latest["ma_angle_20"], errors="coerce")
            turning = (angle > -2).astype(float)
            score += turning

        if "macd" in latest.columns and "macd_signal" in latest.columns:
            macd = pd.to_numeric(latest["macd"], errors="coerce")
            sig = pd.to_numeric(latest["macd_signal"], errors="coerce")
            macd_above_sig = (macd > sig).astype(float)
            score += macd_above_sig

        score = score.fillna(0)

        n = top_n or self.top_n
        ranked = score.sort_values(ascending=False).head(n)
        return ranked[ranked > 0].index.tolist()

    def get_description(self) -> str:
        return "超跌反弹选股器: MACD圆弧底+金叉+站上中期均线+短期均线走平向上"


# ===================================================================
#  3. ChipConcentrationSelector —— 筹码集中选股器
# ===================================================================

class ChipConcentrationSelector(IStockSelector):
    """筹码集中选股器。

    核心逻辑:
      - 量能萎缩到极限 (volume_contraction < 0.5)
      - 筹码集中度低 (chip_concentration < 阈值)
      - 均线粘合 (ma_convergence < 阈值)
      - 蓄势充分后放量突破
    """

    def __init__(
        self,
        top_n: int = 50,
        max_volume_contraction: float = 0.5,
        max_chip_concentration: float = 0.05,
        max_ma_convergence: float = 0.05,
        min_breakout_volume: float = 1.5,
    ):
        """
        Args:
            top_n: 默认选股数量
            max_volume_contraction: 最大量缩比(5日均量/20日均量)
            max_chip_concentration: 最大筹码集中度
            max_ma_convergence: 最大均线粘合度
            min_breakout_volume: 突破时最低量比
        """
        self.top_n = top_n
        self.max_volume_contraction = max_volume_contraction
        self.max_chip_concentration = max_chip_concentration
        self.max_ma_convergence = max_ma_convergence
        self.min_breakout_volume = min_breakout_volume

    @property
    def factor_names(self) -> List[str]:
        return [
            "volume_contraction",
            "chip_concentration",
            "ma_convergence",
            "volume_breakout_ratio",
            "box_breakout",
            "breakout_strength",
        ]

    def select(self, factor_data: pd.DataFrame, date, top_n: int) -> List[str]:
        latest = _latest_per_symbol(factor_data, date)
        if latest.empty:
            return []

        score = pd.Series(0.0, index=latest.index)
        mask = pd.Series(True, index=latest.index)

        if "volume_contraction" in latest.columns:
            vol_c = pd.to_numeric(latest["volume_contraction"], errors="coerce")
            mask &= vol_c < self.max_volume_contraction
            score += (1 - vol_c.clip(0, 1)) * 2

        if "chip_concentration" in latest.columns:
            chip_c = pd.to_numeric(latest["chip_concentration"], errors="coerce")
            mask &= chip_c < self.max_chip_concentration
            score += (1 - chip_c.clip(0, 0.2) / 0.2) * 2

        if "ma_convergence" in latest.columns:
            ma_conv = pd.to_numeric(latest["ma_convergence"], errors="coerce")
            mask &= ma_conv < self.max_ma_convergence
            score += (1 - ma_conv.clip(0, 0.2) / 0.2) * 2

        if "volume_breakout_ratio" in latest.columns:
            vol_r = pd.to_numeric(latest["volume_breakout_ratio"], errors="coerce")
            score += vol_r.clip(0, 5) * 0.5

        if "breakout_strength" in latest.columns:
            bs = pd.to_numeric(latest["breakout_strength"], errors="coerce")
            score += bs.clip(0, 5) * 1.0

        if "box_breakout" in latest.columns:
            bb = pd.to_numeric(latest["box_breakout"], errors="coerce").fillna(0)
            score += bb * 3

        score = score.fillna(0)
        candidates = latest[mask].index.intersection(score.index)

        n = top_n or self.top_n
        ranked = score.loc[candidates].sort_values(ascending=False).head(n)
        return ranked[ranked > 0].index.tolist()

    def get_description(self) -> str:
        return "筹码集中选股器: 量缩+筹码集中+均线粘合+放量突破"
