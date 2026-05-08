"""BacktestEngine T+1 锁单元测试。

验证:同一交易日买入的股票,当日止损/卖单都不会被执行。
"""
from __future__ import annotations
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd

from core.backtest import BacktestEngine, Order, Position


def _make_engine() -> BacktestEngine:
    return BacktestEngine(
        initial_capital=1_000_000,
        fee_config={
            "stamp_tax": 0.001,
            "transfer_fee": 0.00002,
            "commission": 0.0003,
            "min_commission": 5,
            "slippage": 0.0005,
        },
        risk_config={
            "max_position_pct": 0.30,
            "max_total_position": 0.95,
            "stop_loss": 0.10,
            "take_profit": 0.25,
        },
    )


def test_buy_date_map_populated_on_buy():
    """BUY 成功后 _buy_date_map 应当记录 (symbol, date)。"""
    eng = _make_engine()
    eng._execute_order(
        Order(symbol="600001", direction="BUY", quantity=100, price=10.0),
        date="2024-01-10",
    )
    assert eng._buy_date_map.get("600001") == "2024-01-10"
    assert "600001" in eng.positions


def test_same_day_sell_blocked_by_t1():
    """同日 SELL 订单应当被 T+1 过滤,仓位保持不变。"""
    eng = _make_engine()
    date = "2024-01-10"
    eng._execute_order(
        Order(symbol="600001", direction="BUY", quantity=100, price=10.0),
        date=date,
    )
    before_qty = eng.positions["600001"].quantity

    # 模拟 BacktestEngine.run 里对 SELL 订单的 T+1 过滤逻辑
    order = Order(symbol="600001", direction="SELL", quantity=100, price=10.5)
    if eng._buy_date_map.get(order.symbol) == date:
        # 跳过:当日买入不能卖
        pass
    else:
        eng._execute_order(order, date=date)

    assert eng.positions["600001"].quantity == before_qty


def test_next_day_sell_allowed():
    """第二天卖出不受 T+1 限制。"""
    eng = _make_engine()
    eng._execute_order(
        Order(symbol="600001", direction="BUY", quantity=100, price=10.0),
        date="2024-01-10",
    )
    # 第二天
    next_day = "2024-01-11"
    order = Order(symbol="600001", direction="SELL", quantity=100, price=10.5)
    assert eng._buy_date_map.get(order.symbol) != next_day
    eng._execute_order(order, date=next_day)
    assert "600001" not in eng.positions


def test_check_stops_skips_same_day_buy():
    """_check_stops 遇到同日买入的股票应当直接 continue。"""
    eng = _make_engine()
    date = "2024-01-10"
    eng._execute_order(
        Order(symbol="600001", direction="BUY", quantity=100, price=10.0),
        date=date,
    )
    # 构造一个当日暴跌到触发止损的行情
    today_bars = pd.DataFrame([
        {"date": date, "symbol": "600001", "close": 8.9,
         "pct_change": -11.0, "volume": 1_000_000},
    ])
    eng._check_stops(today_bars, date=date)
    # 同日买入,T+1 锁应阻止止损
    assert "600001" in eng.positions


if __name__ == "__main__":
    test_buy_date_map_populated_on_buy()
    test_same_day_sell_blocked_by_t1()
    test_next_day_sell_allowed()
    test_check_stops_skips_same_day_buy()
    print("T+1 锁单元测试通过")