#!/usr/bin/env python3
"""因子计算脚本(走 FactorHub 单一注册中心)。

支持按 category 或显式 --names 指定要算哪些因子。默认计算 technical 类别(13 个传统因子)。

示例:
    python3 scripts/compute_factors.py                       # 全市场 + technical
    python3 scripts/compute_factors.py --start 2024-01-01
    python3 scripts/compute_factors.py --category alpha101   # 算 alpha101 全部
    python3 scripts/compute_factors.py --names returns rsi_14 macd
    python3 scripts/compute_factors.py --symbols 000001 600519
"""
from __future__ import annotations
import sys
import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

import argparse

from core.config import load_config, get_db_path
from core.database import Database
from core.factor import FactorRunner
from core.factor_hub import FactorHub


def main() -> None:
    parser = argparse.ArgumentParser(description="因子计算(FactorHub)")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--symbols", nargs="*", help="指定股票列表(默认全市场)")
    parser.add_argument("--start", default=None,
                        help="起始日期,默认读 config.backtest.start_date")
    parser.add_argument("--category", default="technical",
                        help="因子分类: technical / alpha101(默认 technical)")
    parser.add_argument("--names", nargs="*",
                        help="显式指定因子名(优先于 --category)")
    parser.add_argument("--quiet", action="store_true", help="不打印每因子进度")
    args = parser.parse_args()

    cfg = load_config(args.config)
    start = args.start or cfg["backtest"]["start_date"]

    db = Database(get_db_path(cfg))
    runner = FactorRunner(db)

    names = args.names
    target_desc = ", ".join(names) if names else f"category={args.category}"
    print(f"开始计算因子 [{target_desc}],起始 {start} ...")

    long_df = runner.compute_all(
        symbols=args.symbols,
        start_date=start,
        names=names,
        category=args.category,
        verbose=not args.quiet,
    )
    if long_df.empty:
        print("无新数据")
    else:
        n_factor = long_df["factor_name"].nunique()
        n_sym = long_df["symbol"].nunique()
        print(f"完成: {n_factor} 个因子 × {n_sym} 只股票, 总 {len(long_df)} 条")
        print(f"因子总览: 已注册 {len(FactorHub.list_all())} 个,"
              f"分类: {FactorHub.categories()}")

    db.close()


if __name__ == "__main__":
    main()