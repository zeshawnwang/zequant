#!/usr/bin/env python3
"""基于 SignalStrategy 的回测示例。

展示:
  1. 从数据库加载多标的量价数据(附已预计算的因子)
  2. 使用信号策略选择股票
  3. 运行回测得到报告
  4. 输出绩效统计

用法:
    python3 scripts/example_backtest_signal.py
    python3 scripts/example_backtest_signal.py --strategy sma_cross
"""
from __future__ import annotations
import sys
import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

import argparse
from datetime import datetime, timedelta

import pandas as pd

from core.config import load_config
from core.datasourcesourcebase import Database
from core.execution.impl.backtest import BacktestEngine
from core.monitor.impl.performance import PerformanceMonitor, PerformanceReport
from core.strategies.impl.example_signal_strategy import STRATEGIES
from core.strategies.impl.config_signal_strategy import get_config, list_configs


def load_bars(db: Database, start_date: str, end_date: str) -> pd.DataFrame:
    """从数据库加载指定区间多标的日线数据, DataFrame 包含 date/symbol/open/high/low/close/volume 列。"""
    bars = db.get_daily_bars(start_date=start_date, end_date=end_date)
    if bars.empty:
        raise ValueError(f"未找到 {start_date}~{end_date} 区间数据,请先运行 fetch_data")
    return bars


def main() -> None:
    parser = argparse.ArgumentParser(description="回测示例(信号策略)")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--start", default=None,
                        help="开始日期,默认三个月前")
    parser.add_argument("--end", default=None,
                        help="结束日期,默认昨天")
    parser.add_argument("--capital", type=float, default=100_000,
                        help="初始资金,默认 100k")
    parser.add_argument("--strategy", default="example_signal",
                        choices=list(STRATEGIES.keys()),
                        help="信号策略名")
    args = parser.parse_args()

    cfg = load_config(args.config)
    db_path = cfg["database"]["path"]
    db = Database(db_path)

    end = args.end or (datetime.now() - timedelta(1)).strftime("%Y-%m-%d")
    start = args.start or (datetime.now() - timedelta(90)).strftime("%Y-%m-%d")

    print(f"加载数据: {start} ~ {end}")

    try:
        bars = load_bars(db, start, end)
    except ValueError as e:
        print(f"错误: {e}")
        db.close()
        return

    print(f"共 {bars['symbol'].nunique()} 个股, {len(bars)} 行")

    print(f"创建策略: {args.strategy}")
    strategy_cls = STRATEGIES[args.strategy]
    strategy_config = get_config(args.strategy)

    strategy = strategy_cls(selector_config=strategy_config)

    engine = BacktestEngine(
        initial_capital=args.capital,
        fee_config={"commission": 0.0003, "min_fee": 5},
    )

    print(f"运行回测, 初始资金 {args.capital:,.0f} ...")
    report = engine.run(
        strategy=strategy,
        factor_data=bars,
        start_date=start,
        end_date=end,
    )

    report.pretty_print()

    monitor = PerformanceMonitor(report)
    perf_report: PerformanceReport = monitor.analyze()

    print(f"\n{'='*60}")
    print(f"夏普比率:     {perf_report.sharpe_ratio:.2f}")
    print(f"最大回撤:     {perf_report.max_drawdown:.2%}")
    print(f"年化收益率:   {perf_report.annual_return:.2%}")
    print(f"总交易次数:   {perf_report.total_trades}")
    print(f"胜率:         {perf_report.win_rate:.2%}")
    print(f"{'='*60}")

    db.close()


if __name__ == "__main__":
    main()
