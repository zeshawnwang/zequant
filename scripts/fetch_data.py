#!/usr/bin/env python3
"""
数据采集脚本
从AKShare增量获取日线数据。
用法：
  python scripts/fetch_data.py              # 获取全市场
  python scripts/fetch_data.py 000001       # 获取单只
  python scripts/fetch_data.py --batch 100  # 获取前100只
"""
import sys
sys.path.insert(0, '.')

import argparse
from core.database import Database
from core.data_fetcher import IncrementalFetcher
from core.data_checker import DataQualityChecker

def main():
    parser = argparse.ArgumentParser(description='获取股票日线数据')
    parser.add_argument('symbol', nargs='?', default=None, help='股票代码')
    parser.add_argument('--batch', type=int, default=None, help='批量获取前N只')
    parser.add_argument('--start', default=None, help='起始日期 YYYYMMDD')
    parser.add_argument('--all', action='store_true', help='获取全市场股票列表')
    args = parser.parse_args()

    db = Database()
    fetcher = IncrementalFetcher(db)

    if args.all:
        print("获取全市场股票列表...")
        symbols_df = fetcher.fetch_all_symbols()
        print(f"获取到 {len(symbols_df)} 只股票")
        return

    if args.symbol:
        # 单只
        print(f"获取 {args.symbol} ...")
        df = fetcher.fetch_daily_bars(args.symbol, args.start)
        if not df.empty:
            issues = DataQualityChecker.check(df, args.symbol)
            if issues:
                print(f"数据问题: {issues}")
            else:
                print(f"获取成功: {len(df)} 条, {df['date'].min()} ~ {df['date'].max()}")
    elif args.batch:
        # 批量
        symbols_df = db.get_symbols()
        symbols = symbols_df['symbol'].tolist()[:args.batch]
        print(f"批量获取 {len(symbols)} 只股票...")
        results = fetcher.fetch_batch(symbols)
        success = sum(1 for v in results.values() if v > 0)
        print(f"完成: {success}/{len(symbols)} 只成功获取数据")
    else:
        print("请指定股票代码或使用 --batch / --all")

    db.close()

if __name__ == "__main__":
    main()
