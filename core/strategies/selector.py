"""
市场状态感知策略选择器 — 根据当前市场状态自动切换最佳策略。

用法:
    from core.strategies.selector import MarketStateSelector

    selector = MarketStateSelector()
    state = selector.detect_market_state(market_data)
    rec = selector.recommend(state)
    allocation = selector.allocate(market_data)
"""
from __future__ import annotations
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd


class MarketStateSelector:
    STATES = ["bull", "bear", "oscillate", "recovery"]

    def __init__(self):
        self.state_strategies = {
            "bull": [
                {"strategy": "mf_d10_rp", "weight": 0.6, "note": "牛市无择时满仓"},
                {"strategy": "mf_vol_d10_rp", "weight": 0.2, "note": "波动保护"},
                {"strategy": "chip_rp", "weight": 0.2, "note": "低波动防御"},
            ],
            "bear": [
                {"strategy": "chip_vol_rp", "weight": 0.5, "note": "熊市主打防守"},
                {"strategy": "chip_covrp", "weight": 0.3, "note": "低风险底仓"},
                {"strategy": "mf_vol_d10_rp", "weight": 0.2, "note": "少量做多"},
            ],
            "oscillate": [
                {"strategy": "mf50_chip50_combo", "weight": 0.4, "note": "均衡配置"},
                {"strategy": "chip_covrp", "weight": 0.3, "note": "低回撤底仓"},
                {"strategy": "ga_d10", "weight": 0.3, "note": "GA进攻"},
            ],
            "recovery": [
                {"strategy": "mf60_chip40_combo", "weight": 0.5, "note": "温和进攻"},
                {"strategy": "osr_d10", "weight": 0.3, "note": "超跌反弹"},
                {"strategy": "mf_vol_d10_rp", "weight": 0.2, "note": "波动保护"},
            ],
        }

    def detect_market_state(
        self, market_data: Optional[pd.DataFrame] = None,
        *,
        above_ma200: Optional[float] = None,
        ma5_slope: Optional[float] = None,
        ma20_slope: Optional[float] = None,
        ma60_slope: Optional[float] = None,
        ma5: Optional[float] = None,
        ma20: Optional[float] = None,
        ma60: Optional[float] = None,
    ) -> str:
        if market_data is not None:
            return self._detect_from_df(market_data)
        return self._detect_from_params(
            above_ma200=above_ma200,
            ma5_slope=ma5_slope,
            ma20_slope=ma20_slope,
            ma60_slope=ma60_slope,
            ma5=ma5,
            ma20=ma20,
            ma60=ma60,
        )

    def _detect_from_df(self, df: pd.DataFrame) -> str:
        df = df.copy()
        if "close" in df.columns:
            df["ma5"] = df["close"].rolling(5).mean()
            df["ma20"] = df["close"].rolling(20).mean()
            df["ma60"] = df["close"].rolling(60).mean()
            df["ma200"] = df["close"].rolling(200).mean()

        latest = df.iloc[-1] if len(df) > 0 else None
        if latest is None:
            return "oscillate"

        close = latest.get("close", np.nan)
        ma5_val = latest.get("ma5", np.nan)
        ma20_val = latest.get("ma20", np.nan)
        ma60_val = latest.get("ma60", np.nan)
        ma200_val = latest.get("ma200", np.nan)

        above_ma200 = (close - ma200_val) / ma200_val if pd.notna(close) and pd.notna(ma200_val) and ma200_val != 0 else None
        ma5_slope_val = latest.get("ma5_slope", None)
        ma20_slope_val = latest.get("ma20_slope", None)

        if ma5_slope_val is None and pd.notna(ma5_val):
            lookback = min(5, len(df) - 1)
            if lookback >= 2:
                ma5_prev = df["ma5"].iloc[-lookback]
                ma5_slope_val = (ma5_val - ma5_prev) / ma5_prev if pd.notna(ma5_prev) and ma5_prev != 0 else 0.0
            else:
                ma5_slope_val = 0.0

        return self._detect_from_params(
            above_ma200=above_ma200,
            ma5_slope=ma5_slope_val,
            ma20_slope=ma20_slope_val,
            ma5=ma5_val,
            ma20=ma20_val,
            ma60=ma60_val,
        )

    def _detect_from_params(
        self,
        above_ma200: Optional[float] = None,
        ma5_slope: Optional[float] = None,
        ma20_slope: Optional[float] = None,
        ma60_slope: Optional[float] = None,
        ma5: Optional[float] = None,
        ma20: Optional[float] = None,
        ma60: Optional[float] = None,
    ) -> str:
        above_ma200 = above_ma200 if above_ma200 is not None else 0.0
        ma5_slope = ma5_slope if ma5_slope is not None else 0.0
        ma20_slope = ma20_slope if ma20_slope is not None else 0.0
        ma60_slope = ma60_slope if ma60_slope is not None else 0.0

        if above_ma200 > 0 and ma20_slope > 0:
            return "bull"

        if above_ma200 < 0 and ma20_slope < 0 and ma60_slope < 0:
            return "bear"

        if above_ma200 < 0 and ma5_slope > 0.005:
            return "recovery"

        if ma5 is not None and ma20 is not None and ma60 is not None:
            if pd.notna(ma5) and pd.notna(ma20) and pd.notna(ma60):
                spread = abs(ma5 - ma20) / ma20 + abs(ma20 - ma60) / ma60
                if spread < 0.03:
                    return "oscillate"

        if above_ma200 < 0 and ma5_slope > 0:
            return "recovery"

        return "oscillate"

    def recommend(self, state: str) -> List[Dict[str, Any]]:
        return self.state_strategies.get(state, self.state_strategies["oscillate"])

    def allocate(self, market_data: Optional[pd.DataFrame] = None, **kwargs) -> Dict[str, float]:
        state = self.detect_market_state(market_data, **kwargs)
        recs = self.recommend(state)
        return {r["strategy"]: r["weight"] for r in recs}
