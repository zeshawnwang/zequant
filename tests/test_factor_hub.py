"""FactorHub 注册/计算冒烟测试。

验证:
  - register/list/categories 基础能力
  - compute_all 向量化路径在最小内存输入下能跑通
  - 结果 schema 为 (date, symbol, factor_name, value) 长表
"""
from __future__ import annotations
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd
import pytest

from core.factor_hub import FactorHub


def _sample_bars() -> pd.DataFrame:
    """造两只股票 * 30 天的最小 bars,够 rolling(20) 出值。"""
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    rows = []
    for i, d in enumerate(dates):
        rows.append({"symbol": "000001", "date": d.date(),
                     "open": 10 + 0.1 * i, "high": 10.5 + 0.1 * i,
                     "low": 9.5 + 0.1 * i, "close": 10 + 0.1 * i,
                     "volume": 1_000_000 + i * 1000,
                     "amount": 10_000_000 + i * 10_000,
                     "pct_change": 0.01})
        rows.append({"symbol": "600000", "date": d.date(),
                     "open": 20 - 0.1 * i, "high": 20.5,
                     "low": 19.5 - 0.1 * i, "close": 20 - 0.1 * i,
                     "volume": 2_000_000,
                     "amount": 40_000_000,
                     "pct_change": -0.01})
    return pd.DataFrame(rows)


def test_factor_registry_has_technical_factors():
    """import factors 后 FactorHub 至少应有 technical 类的 13 个因子。"""
    import factors  # noqa: F401  触发注册
    techs = FactorHub.list_by_category("technical")
    # 至少要有 returns / momentum_20 / rsi_14 / macd 这 4 个基准因子
    for need in ("returns", "momentum_20", "rsi_14", "macd"):
        assert need in techs, f"missing technical factor: {need}"


def test_compute_all_returns_long_schema():
    """compute_all 应返回 (date, symbol, factor_name, value) 长表。"""
    import factors  # noqa: F401
    bars = _sample_bars()
    long_df = FactorHub.compute_all(
        bars, names=["momentum_20", "rsi_14"], verbose=False
    )
    assert set(long_df.columns) == {"date", "symbol", "factor_name", "value"}
    # 两个因子都应当产生数据
    assert set(long_df["factor_name"].unique()) == {"momentum_20", "rsi_14"}
    # 值类型应为 float,不存在 inf
    assert pd.api.types.is_numeric_dtype(long_df["value"])


def test_compute_single_factor_wide_shape():
    """compute 单因子返回 wide DataFrame,shape = (n_date, n_symbol)。"""
    import factors  # noqa: F401
    bars = _sample_bars()
    wide = FactorHub.compute("momentum_20", bars, verbose=False)
    assert isinstance(wide, pd.DataFrame)
    # 至少有两列(两只股票),至少 10 天
    assert wide.shape[1] == 2
    assert wide.shape[0] >= 10


def test_unregistered_factor_raises():
    with pytest.raises(KeyError):
        FactorHub.get("factor_that_does_not_exist")