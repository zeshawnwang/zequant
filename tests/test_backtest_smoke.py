"""BacktestEngine 端到端最小冒烟测试。

只用内存因子表 + 不挂 Universe 的简化场景,验证:
  - run() 能把 BUY 信号落到 trades / final_positions
  - report 含完整字段(equity_curve / total_return / final_value)
  - selection_log 至少有 1 条选股记录
"""
from __future__ import annotations
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd

from core.backtest import BacktestEngine
from core.strategy import QuantStrategy
from screening.factor_rank import FactorRankSelector
from timings.trend import TrendTiming
from portfolios.equal_weight import EqualWeightBuilder


def _make_factor_panel() -> pd.DataFrame:
    """造 3 只股票 * 25 个交易日的最小因子面板,带 close + 趋势打分所需因子。"""
    dates = pd.date_range("2024-01-02", periods=25, freq="B")
    rows = []
    for i, d in enumerate(dates):
        # A: 强趋势上行(macd > signal、动量正、RSI 60)
        rows.append({"symbol": "A", "date": d, "open": 10.0 + 0.2 * i, "close": 10.0 + 0.2 * i,
                     "momentum_5": 0.05, "momentum_20": 0.10,
                     "macd": 0.30, "macd_signal": 0.10, "rsi_14": 60.0,
                     "volatility_20": 0.10, "volume_ratio": 1.0,
                     "boll_position": 0.6})
        # B: 趋势平庸(打分中性)
        rows.append({"symbol": "B", "date": d, "open": 20.0 + 0.05 * i, "close": 20.0 + 0.05 * i,
                     "momentum_5": 0.01, "momentum_20": 0.01,
                     "macd": 0.05, "macd_signal": 0.05, "rsi_14": 50.0,
                     "volatility_20": 0.15, "volume_ratio": 1.0,
                     "boll_position": 0.5})
        # C: 趋势走弱(macd<signal、RSI 偏低)
        rows.append({"symbol": "C", "date": d, "open": 30.0 - 0.2 * i, "close": 30.0 - 0.2 * i,
                     "momentum_5": -0.05, "momentum_20": -0.08,
                     "macd": -0.20, "macd_signal": 0.10, "rsi_14": 25.0,
                     "volatility_20": 0.30, "volume_ratio": 1.0,
                     "boll_position": 0.2})
    return pd.DataFrame(rows)


def test_backtest_engine_runs_end_to_end():
    factor_data = _make_factor_panel()

    selector = FactorRankSelector("momentum_20", ascending=False, top_n=2)
    timing = TrendTiming(buy_threshold=0.5, sell_threshold=0.4)
    portfolio = EqualWeightBuilder(reserve_cash_ratio=0.0)
    strategy = QuantStrategy(
        name="SmokeStrategy",
        selector=selector,
        timing=timing,
        portfolio=portfolio,
        top_n=2,
    )

    engine = BacktestEngine(
        initial_capital=100_000,
        fee_config={"stamp_tax": 0.001, "transfer_fee": 0.00002,
                    "commission": 0.0003, "min_commission": 5,
                    "slippage": 0.0005},
        risk_config={"max_position_pct": 0.6, "max_total_position": 0.95,
                     "stop_loss": 0.10, "take_profit": 0.25},
        universe=None,
    )

    report = engine.run(
        strategy=strategy,
        factor_data=factor_data,
        start_date="2024-01-02",
        end_date="2024-02-15",
    )

    # 1) 资金/收益指标字段完整
    assert report.initial_capital == 100_000
    assert report.final_value > 0
    # 2) 强趋势股 A 应当被选中
    assert any("A" in rec.get("selected", []) for rec in report.selection_log)
    # 3) 至少产生过一笔 BUY 交易
    assert any(t.direction == "BUY" for t in report.trades)
    # 4) equity_curve 不为空
    assert not report.equity_curve.empty
    # 5) pretty_print 不抛异常
    text = report.pretty_print(top_positions=5, top_selections=2)
    assert "回测报告" in text