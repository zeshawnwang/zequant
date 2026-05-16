"""每日生产调度器。

0830  pull:   数据更新
0900  factor: 因子计算
0930  signal: 生成调仓信号
1530  save:   收盘快照
1600  report: 日报告

用法
----
    python3 -m live.runner                    # 全流程
    python3 -m live.runner --mode signals     # 只生成信号
    python3 -m live.runner --mode report      # 只生成报告
"""
from __future__ import annotations
import argparse
import logging
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from live.storage.positions import PositionStorage
from live.monitor.dashboard import Dashboard

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("live.runner")


class Runner:
    def __init__(self, mode: str = "full"):
        self.mode = mode
        self.storage = PositionStorage()

    def run(self):
        if self.mode in ("full", "update"):
            logger.info("阶段1: 数据更新")
            try:
                from core.data.daily_updater import DailyUpdater
                updater = DailyUpdater()
                updater.run(fetch_bars=False, compute_factors=True, compute_technical=True)
            except ImportError:
                logger.warning("DailyUpdater 未就绪，跳过数据更新")

        if self.mode in ("full", "signals"):
            logger.info("阶段2: 信号生成")
            from live.signals.generator import SignalGenerator
            from live.signals.combiner import SignalCombiner
            gen = SignalGenerator()
            orders = gen.generate()
            if orders:
                combined = SignalCombiner.combine(orders)
                logger.info("今日调仓: %d 笔", len(combined))

        if self.mode in ("full", "save"):
            logger.info("阶段3: 持仓快照")
            self.storage.save_snapshot()

        if self.mode in ("full", "report"):
            logger.info("阶段4: 日报告")
            Dashboard.generate_report()

        logger.info("完成")


def main():
    parser = argparse.ArgumentParser(description="ZEquant 生产调度器")
    parser.add_argument("--mode", default="full", choices=["full", "update", "signals", "save", "report"])
    args = parser.parse_args()
    Runner(mode=args.mode).run()


if __name__ == "__main__":
    main()
