#!/usr/bin/env python3
"""
因子计算脚本
批量计算所有因子的增量更新。
"""
import sys
import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

import argparse
from core.database import Database
from core.factor import FactorRunner

def main():
    parser = argparse.ArgumentParser(description='计算技术因子')
    parser.add_argument('--symbols', nargs='*', help='指定股票列表')
    parser.add_argument('--start', default='2019-01-01', help='起始日期')
    args = parser.parse_args()

    db = Database()
    runner = FactorRunner(db)

    print(f"开始计算因子 (从 {args.start})...")
    factors = runner.compute_all(symbols=args.symbols, start_date=args.start)

    if not factors.empty:
        print(f"因子计算完成: {len(factors)} 条")
        print(f"因子列: {[c for c in factors.columns if c not in ['date', 'symbol']]}")
    else:
        print("无新数据")

    db.close()

if __name__ == "__main__":
    main()
