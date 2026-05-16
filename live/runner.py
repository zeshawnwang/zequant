"""
每日生产调度器 — 邮件通知模式（无券商API）。

每日流水线：
  08:30  数据拉取 + 因子计算
  09:00  信号生成 + 调仓清单
  09:05  邮件发送 / 文件保存

用法：
    python3 -m live.runner                     # 全流程
    python3 -m live.runner --mode signals       # 只生成信号+发邮件
    python3 -m live.runner --mode email         # 只发邮件
"""
from __future__ import annotations
import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("live.runner")


class Runner:
    def __init__(self, mode: str = "full"):
        self.mode = mode
        self.config = self._load_config()

    def _load_config(self) -> dict:
        """加载实盘配置。"""
        import yaml
        cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        with open(cfg_path) as f:
            return yaml.safe_load(f)

    def run(self):
        if self.mode in ("full", "update"):
            self._step_update()

        if self.mode in ("full", "signals"):
            orders = self._step_signals()
        else:
            orders = []

        if self.mode in ("full", "email"):
            self._step_email(orders)

        logger.info("调度完成")

    def _step_update(self):
        """数据更新。"""
        logger.info("阶段1: 数据更新")
        try:
            from core.datasource.daily_updater import DailyUpdater
            updater = DailyUpdater()
            updater.run(fetch_bars=False, compute_factors=True, compute_technical=True)
        except Exception as e:
            logger.warning("数据更新失败(可跳过): %s", e)

    def _step_signals(self) -> List[Dict]:
        """生成调仓信号 + 读当前持仓。"""
        logger.info("阶段2: 信号生成")

        try:
            from live.notification import Mailer
            from live.storage.positions import PositionStorage
        except ImportError:
            logger.warning("信号生成模块未就绪，使用占位数据")
            return self._placeholder_signals()

        # 生成策略信号
        signals = self._generate_signals()

        # 读当前持仓
        storage = PositionStorage()
        positions = storage.load_today_positions()

        # 转为调仓清单
        mailer = Mailer(self.config.get("mail", {}))
        cash = self.config.get("account", {}).get("initial_cash", 100000)
        orders = mailer.format_orders(signals, positions, cash)

        logger.info("今日调仓: %d 笔买入, %d 笔卖出",
                     sum(1 for o in orders if o["direction"] == "买入"),
                     sum(1 for o in orders if o["direction"] == "卖出"))

        # 保存持仓快照
        try:
            storage.save_snapshot(
                strategy=self.config.get("strategies", {}).get("primary", "default"),
                total_value=0.0,
                cash=cash,
                positions={s: 100 for s in [sig["symbol"] for sig in signals[:5]]},
                orders=orders,
            )
        except Exception as e:
            logger.warning("持仓保存失败: %s", e)

        return orders

    def _generate_signals(self) -> List[Dict]:
        """生成策略信号。未来接入真正的 Pipeline.signals()。

        Returns:
            [{symbol, weight, price, reason}, ...]
        """
        prim = self.config.get("strategies", {}).get("primary", "mf_vol_d10_rp")
        logger.info("策略: %s (MVP占位，待接入Pipeline)", prim)
        return []

    def _placeholder_signals(self) -> List[Dict]:
        """占位信号（依赖未安装时使用）。"""
        return [
            {"symbol": "000001", "weight": 0.05, "price": "—", "reason": "MVP占位"},
            {"symbol": "000858", "weight": 0.05, "price": "—", "reason": "MVP占位"},
        ]

    def _step_email(self, orders: List[Dict]):
        """发送/保存调仓清单。"""
        logger.info("阶段3: 邮件通知")

        today = datetime.now().strftime("%Y-%m-%d")
        subject = f"[ZEquant] {today} 调仓信号 ({len(orders)}笔)"

        config = self.config.get("mail", {})
        from live.notification import Mailer
        mailer = Mailer(config)

        # 持仓信息
        positions = {o["symbol"]: o.get("shares", 0) for o in orders if o["direction"] == "买入"}

        # 额外报告（当前策略配置+资金状态）
        report = (
            f"策略: {self.config['strategies']['primary']} + {self.config['strategies']['secondary']}\n"
            f"资金: {self.config['account']['initial_cash']:,.0f} 元\n"
            f"最大持仓: {self.config['account']['max_positions']} 只\n"
            f"单票上限: {self.config['account']['max_single_weight']:.0%}\n"
            f"模式: {self.config['mode']}"
        )

        mailer.send_daily_signal(subject, orders, positions, report)


def main():
    parser = argparse.ArgumentParser(description="ZEquant 每日调度器(邮件模式)")
    parser.add_argument("--mode", default="full",
                        choices=["full", "update", "signals", "email"])
    args = parser.parse_args()
    Runner(mode=args.mode).run()


if __name__ == "__main__":
    main()
