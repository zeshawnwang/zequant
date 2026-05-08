#!/usr/bin/env python3
"""数据采集脚本 —— 从 AKShare 增量抓取日线。

用法:
    python3 scripts/fetch_data.py --all          # 刷新全市场名册
    python3 scripts/fetch_data.py 000001         # 抓单只
    python3 scripts/fetch_data.py --batch 100    # 抓前 N 只
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
from core.data_fetcher import IncrementalFetcher
from core.data_checker import DataQualityChecker


def main() -> None:
    parser = argparse.ArgumentParser(description="获取股票日线数据")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("symbol", nargs="?", default=None, help="股票代码")
    parser.add_argument("--batch", type=int, default=None, help="批量获取前 N 只")
    parser.add_argument("--start", default=None, help="起始日期 YYYYMMDD")
    parser.add_argument("--all", action="store_true", help="刷新全市场股票名册")
    args = parser.parse_args()

    cfg = load_config(args.config)
    db = Database(get_db_path(cfg))
    fetcher = IncrementalFetcher(db)

    if args.all:
        print("获取全市场股票列表...")
        symbols_df = fetcher.fetch_all_symbols()
        print(f"获取到 {len(symbols_df)} 只股票")
        db.close()
        return

    if args.symbol:
        print(f"获取 {args.symbol} ...")
        df = fetcher.fetch_daily_bars(args.symbol, args.start)
        if not df.empty:
            issues = DataQualityChecker.check(df, args.symbol)
            if issues:
                print(f"数据问题: {issues}")
            else:
                print(f"获取成功: {len(df)} 条, {df['date'].min()} ~ {df['date'].max()}")
    elif args.batch:
        symbols_df = db.get_symbols()
        symbols = symbols_df["symbol"].tolist()[: args.batch]
        print(f"批量获取 {len(symbols)} 只股票...")
        results = fetcher.fetch_batch(symbols)
        success = sum(1 for v in results.values() if v > 0)
        print(f"完成: {success}/{len(symbols)} 只成功获取数据")
    else:
        print("请指定股票代码或使用 --batch / --all")

    db.close()


if __name__ == "__main__":
    main()