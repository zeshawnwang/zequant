#!/usr/bin/env python3
"""评估和对比选股器。

用法:
    python scripts/evaluate_selectors.py --start 2024-01-01 --end 2024-12-31 --top-n 50
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.database import Database
from core.selector_evaluator import SelectorEvaluator
from core.screening import (
    FactorRankSelector,
    MultiFactorSelector,
    TrendBreakoutSelector,
    OversoldReboundSelector,
    ChipConcentrationSelector,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def main():
    parser = argparse.ArgumentParser(description="评估选股器")
    parser.add_argument("--start", type=str, required=True, help="起始日期 YYYY-MM-DD")
    parser.add_argument("--end", type=str, required=True, help="结束日期 YYYY-MM-DD")
    parser.add_argument("--top-n", type=int, default=50, help="每次选股数量")
    parser.add_argument("--rebalance-freq", type=str, default="W", choices=["D", "W", "M", "Q"], help="调仓频率")
    args = parser.parse_args()

    db = Database()
    evaluator = SelectorEvaluator(db)

    # 准备要评估的选股器列表
    selectors = [
        FactorRankSelector(factor_name="momentum_20", ascending=False),
        FactorRankSelector(factor_name="volatility_20", ascending=True),
        TrendBreakoutSelector(),
        OversoldReboundSelector(),
        ChipConcentrationSelector(),
    ]

    # 批量对比
    print(f"\n开始评估 {len(selectors)} 个选股器...")
    comparison_df = evaluator.compare_selectors(
        selectors=selectors,
        start_date=args.start,
        end_date=args.end,
        rebalance_freq=args.rebalance_freq,
        top_n=args.top_n,
    )

    if comparison_df.empty:
        print("没有可用的评估结果")
        return

    # 打印对比表格
    print("\n" + "=" * 100)
    print("选股器对比汇总")
    print("=" * 100)

    # 精简显示关键指标
    key_cols = [
        "selector_name",
        "selector_description",
        "total_return",
        "excess_return",
        "annual_return",
        "max_drawdown",
        "sharpe_ratio",
        "win_rate",
        "turnover_rate",
    ]
    print(comparison_df[key_cols].to_string(index=False))

    # 单独打印每个选股器的详细报告
    print("\n" + "=" * 100)
    print("各选股器详细报告")
    print("=" * 100)

    for selector in selectors:
        report = evaluator.evaluate(
            selector=selector,
            start_date=args.start,
            end_date=args.end,
            rebalance_freq=args.rebalance_freq,
            top_n=args.top_n,
        )
        report.pretty_print()


if __name__ == "__main__":
    main()
