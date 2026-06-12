"""趋势评分共享工具。

提供统一的趋势评分函数，被 trend.py / market_regime.py / position.py 共用，
避免 MACD / 动量 / RSI 打分逻辑在三处重复。
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def calc_trend_score(row) -> float:
    """统一趋势评分：MACD / 动量 / RSI 三因子打分。

    Parameters
    ----------
    row : pd.Series or dict-like
        包含 ``macd``, ``macd_signal``, ``momentum_5``,
        ``momentum_20``, ``rsi_14`` 等字段。

    Returns
    -------
    float
        [0, 1] 区间分数，缺失所有因子时返回 0.5。
    """
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
