"""
市场状态识别择时器

根据《职业投资者如何分析股票选股》方法论,实现牛熊识别择时器:
  - MarketRegimeTiming: 牛熊识别择时器,根据市场状态动态调整策略
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import IntEnum
import numpy as np
import pandas as pd

from ..base.timing import ITimingGenerator


class SignalType(IntEnum):
    """信号类型。"""
    SELL = -1
    HOLD = 0
    BUY = 1


@dataclass
class Signal:
    """择时信号。"""
    symbol: str
    signal_type: SignalType
    strength: float
    price: float
    reason: str = ""
    factors: Optional[Dict[str, Any]] = None


class MarketRegimeTiming(ITimingGenerator):
    """牛熊识别择时器。

    核心逻辑:
      - 判断当前市场是牛市还是熊市
      - 牛市: 使用高β策略,进攻为主,推荐高β股票
      - 熊市/调整市: 使用低β策略,防御为主,推荐低β股票或空仓
    """

    def __init__(
        self,
        inner_timing=None,
        bull_beta_threshold: float = 1.0,
        bear_beta_threshold: float = 0.8,
        bull_buy_threshold: float = 0.6,
        bull_sell_threshold: float = 0.4,
        bear_buy_threshold: float = 0.8,
        bear_sell_threshold: float = 0.5,
    ):
        self.inner_timing = inner_timing
        self.bull_beta_threshold = bull_beta_threshold
        self.bear_beta_threshold = bear_beta_threshold
        self.bull_buy_threshold = bull_buy_threshold
        self.bull_sell_threshold = bull_sell_threshold
        self.bear_buy_threshold = bear_buy_threshold
        self.bear_sell_threshold = bear_sell_threshold

    def generate(
        self,
        factor_data: pd.DataFrame,
        positions: List[str],
        cash: float,
        date=None,
    ) -> List[Signal]:
        if factor_data is None or factor_data.empty:
            return []

        regime = self._detect_regime(factor_data, date)

        if regime == "bull":
            buy_threshold = self.bull_buy_threshold
            sell_threshold = self.bull_sell_threshold
        elif regime == "bear":
            buy_threshold = self.bear_buy_threshold
            sell_threshold = self.bear_sell_threshold
        else:
            buy_threshold = (self.bull_buy_threshold + self.bear_buy_threshold) / 2
            sell_threshold = (self.bull_sell_threshold + self.bear_sell_threshold) / 2

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
            score = self._calc_score(row)
            price = float(row["close"]) if "close" in row and pd.notna(row.get("close")) else 0.0
            if price <= 0:
                continue

            if regime == "bear":
                beta = row.get("beta_20") if hasattr(row, "get") else None
                if pd.notna(beta) and beta > self.bear_beta_threshold:
                    if symbol not in held:
                        continue

            if regime == "bull":
                beta = row.get("beta_20") if hasattr(row, "get") else None
                if pd.notna(beta) and beta < self.bull_beta_threshold:
                    score *= 0.7

            factors_dict = {
                k: row[k] for k in (
                    "momentum_5", "momentum_20", "rsi_14",
                    "macd", "macd_signal", "volatility_20",
                    "volume_ratio", "boll_position", "beta_20",
                )
                if k in row.index and pd.notna(row.get(k))
            }
            factors_dict["market_regime"] = regime

            if score >= buy_threshold:
                if symbol in held:
                    signals.append(Signal(
                        symbol=symbol, signal_type=SignalType.HOLD,
                        strength=score, price=price,
                        reason=f"{regime}市趋势保持({score:.2f})",
                        factors=factors_dict,
                    ))
                else:
                    signals.append(Signal(
                        symbol=symbol, signal_type=SignalType.BUY,
                        strength=score, price=price,
                        reason=f"{regime}市趋势看多({score:.2f})",
                        factors=factors_dict,
                    ))
            elif score <= sell_threshold:
                if symbol in held:
                    signals.append(Signal(
                        symbol=symbol, signal_type=SignalType.SELL,
                        strength=1 - score, price=price,
                        reason=f"{regime}市趋势转弱({score:.2f})",
                        factors=factors_dict,
                    ))

        return signals

    def _detect_regime(self, factor_data: pd.DataFrame, date=None) -> str:
        """判断当前市场是牛市、熊市还是震荡市。"""
        df = factor_data
        if date is not None and "date" in df.columns:
            df = df[df["date"] < date]
        if df.empty:
            return "neutral"

        latest = df.sort_values("date").groupby("symbol").tail(1)
        if latest.empty:
            return "neutral"

        regime_score = 0.0
        vote_count = 0

        if {"ma5", "ma20", "ma60"}.issubset(latest.columns):
            ma5 = pd.to_numeric(latest["ma5"], errors="coerce")
            ma20 = pd.to_numeric(latest["ma20"], errors="coerce")
            ma60 = pd.to_numeric(latest["ma60"], errors="coerce")
            bull_ratio = ((ma5 > ma20) & (ma20 > ma60)).mean()
            regime_score += (bull_ratio - 0.5) * 2
            vote_count += 1

        if "macd_above_zero" in latest.columns:
            macd_up = pd.to_numeric(latest["macd_above_zero"], errors="coerce").fillna(0)
            ratio = macd_up.mean()
            regime_score += (ratio - 0.5) * 2
            vote_count += 1

        if {"close", "ma60"}.issubset(latest.columns):
            close = pd.to_numeric(latest["close"], errors="coerce")
            ma60 = pd.to_numeric(latest["ma60"], errors="coerce")
            above_ratio = (close > ma60).mean()
            regime_score += (above_ratio - 0.5) * 2
            vote_count += 1

        if "ma60_trend" in latest.columns:
            ma60_trend = pd.to_numeric(latest["ma60_trend"], errors="coerce").fillna(0)
            up_ratio = (ma60_trend > 0).mean()
            regime_score += (up_ratio - 0.5) * 2
            vote_count += 1

        if vote_count == 0:
            return "neutral"

        avg_score = regime_score / vote_count

        if avg_score > 0.3:
            return "bull"
        elif avg_score < -0.3:
            return "bear"
        else:
            return "neutral"

    def _calc_score(self, row) -> float:
        """对单只股票打分 [0, 1]。"""
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
