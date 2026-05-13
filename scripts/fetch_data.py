#!/usr/bin/env python3
"""增量获取 A 股行情数据并写入数据库。

支持自动补缺失交易日、逐日补量价与复权因子全字段:
    - 日行情: open/high/low/close/volume/amount
    - 复权因子: adjust_factor / factor

定期(每日)执行可保证本地数据库与市场同步。
"""
from __future__ import annotations
import sys
import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

import argparse
import logging

from core.config import load_config
from core.database import Database
from core.data.fetcher import IncrementalFetcher
from core.data.checker import DataQualityChecker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
_log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="增量获取 A 股行情数据")
    parser.add_argument("--config", default="config/config.yaml",
                        help="配置文件路径")
    parser.add_argument("--symbols", nargs="*", default=None,
                        help="指定股票代码(默认自动发现所有 A 股)")
    parser.add_argument("--start", default=None,
                        help="起始日期 (YYYYMMDD/YYYY-MM-DD), 默认 config 中设置")
    parser.add_argument("--end", default=None,
                        help="结束日期, 默认今天")
    parser.add_argument("--no-check", action="store_true",
                        help="跳过数据质量检查")
    args = parser.parse_args()

    cfg = load_config(args.config)
    db_path = cfg["database"]["path"]
    db = Database(db_path)

    start = args.start or cfg.get("data", {}).get("start_date", None)
    end = args.end

    _log.info("开始获取数据, 数据库=%s", db_path)

    fetcher = IncrementalFetcher(db)
    fetcher.fetch_all(symbols=args.symbols, start_date=start, end_date=end)

    if not args.no_check:
        checker = DataQualityChecker(db)
        issues = checker.check_all(start=start, end=end)
        if issues:
            _log.warning("发现 %d 个数据质量问题:", len(issues))
            for issue in issues:
                _log.warning("  %s", issue)
        else:
            _log.info("数据质量检查通过")

    db.close()
    _log.info("完成")


if __name__ == "__main__":
    main()
