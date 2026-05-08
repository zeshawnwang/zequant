"""MultiFactorSelector / FactorRankSelector 冒烟测试。

只用内存 DataFrame 构造场景,不接触 DuckDB。
"""
from __future__ import annotations
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd
import pytest

from screening.factor_rank import FactorRankSelector
from screening.multi_factor import MultiFactorSelector


def _build_factor_panel() -> pd.DataFrame:
    """5 只股票 * 1 个交易日,factor_a / factor_b 两列因子。"""
    date = pd.Timestamp("2024-03-01")
    rows = [
        {"symbol": "A", "date": date, "factor_a": 1.0, "factor_b": -1.0},
        {"symbol": "B", "date": date, "factor_a": 2.0, "factor_b": -2.0},
        {"symbol": "C", "date": date, "factor_a": 3.0, "factor_b":  0.5},
        {"symbol": "D", "date": date, "factor_a": 4.0, "factor_b":  1.5},
        {"symbol": "E", "date": date, "factor_a": 5.0, "factor_b":  2.0},
    ]
    return pd.DataFrame(rows)


def test_factor_rank_descending_top3():
    df = _build_factor_panel()
    sel = FactorRankSelector("factor_a", ascending=False, top_n=3)
    # date 设为数据次日,确保数据能被选中(只用 date 之前的数据)
    picks = sel.select(df, date=pd.Timestamp("2024-03-02"), top_n=3)
    assert picks == ["E", "D", "C"]


def test_factor_rank_ascending_top2():
    df = _build_factor_panel()
    sel = FactorRankSelector("factor_a", ascending=True, top_n=2)
    picks = sel.select(df, date=pd.Timestamp("2024-03-02"), top_n=2)
    assert picks == ["A", "B"]


def test_multi_factor_positive_weight_picks_high_combined():
    df = _build_factor_panel()
    # factor_a + factor_b 的组合分数:A=-2 / B=-2 / C=3.5 / D=5.5 / E=7
    # zscore 后排序效果一致:E > D > C > B > A
    sel = MultiFactorSelector({"factor_a": 1.0, "factor_b": 1.0}, top_n=3,
                              winsorize=0.0)
    picks = sel.select(df, date=pd.Timestamp("2024-03-02"), top_n=3)
    assert picks[0] == "E"
    assert "D" in picks[:3]
    assert "A" not in picks[:3]


def test_multi_factor_negative_weight_flips_direction():
    df = _build_factor_panel()
    # 给 factor_a 负权重,应选出 factor_a 最低的几只
    sel = MultiFactorSelector({"factor_a": -1.0}, top_n=2, winsorize=0.0)
    picks = sel.select(df, date=pd.Timestamp("2024-03-02"), top_n=2)
    assert picks == ["A", "B"]


def test_multi_factor_from_summary():
    """from_summary 过滤 |IR| >= 阈值后能正常构造选股器。"""
    summary = pd.DataFrame([
        {"factor_name": "factor_a", "ir": 0.30, "ic_mean": 0.02},
        {"factor_name": "factor_b", "ir": -0.25, "ic_mean": -0.01},
        {"factor_name": "factor_c", "ir": 0.05, "ic_mean": 0.001},   # 会被过滤
    ])
    sel = MultiFactorSelector.from_summary(summary, top_n=3, min_abs_ir=0.2)
    assert set(sel.weights.keys()) == {"factor_a", "factor_b"}


def test_multi_factor_rejects_empty_weights():
    with pytest.raises(ValueError):
        MultiFactorSelector({}, top_n=10)


def test_multi_factor_rejects_empty_summary():
    """summary 中没有任何 |ir|>=阈值的因子时应抛 RuntimeError。"""
    summary = pd.DataFrame([
        {"factor_name": "factor_a", "ir": 0.01, "ic_mean": 0.0},
    ])
    with pytest.raises(RuntimeError):
        MultiFactorSelector.from_summary(summary, min_abs_ir=0.2)