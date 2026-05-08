"""Live Engine — 实时/模拟交易引擎。

将策略产生的信号通过 Broker 接口转化为真实或模拟成交，
支持定时轮询、风控检查、日志记录。
"""
import logging
import time
from datetime import datetime
from typing import Callable, Dict, List, Optional

import pandas as pd

from ..broker.base import IBroker, BrokerOrder, OrderStatus
from .strategy import QuantStrategy, Order, Position

logger = logging.getLogger(__name__)


class LiveEngine:
    """实时交易引擎。

    Parameters
    ----------
    broker : IBroker
        交易接口实例（PaperBroker / MockBroker / 实盘 Broker）。
    strategy : QuantStrategy
        策略实例。
    data_provider : Callable
        数据供给函数，签名: data_provider() -> pd.DataFrame。
        返回的 DataFrame 必须包含 strategy 所需的列（如 symbol, close, factor 等）。
    poll_interval : int
        轮询间隔（秒），默认 60。
    risk_config : dict
        风控配置，目前支持 max_position_pct, max_total_position。
    """

    def __init__(self,
                 broker: IBroker,
                 strategy: QuantStrategy,
                 data_provider: Callable[[], pd.DataFrame],
                 poll_interval: int = 60,
                 risk_config: dict = None):
        self.broker = broker
        self.strategy = strategy
        self.data_provider = data_provider
        self.poll_interval = poll_interval
        self.risk_config = risk_config or {}
        self._running = False
        self._trade_log: List[dict] = []

    # ---------- 主循环 ----------
    def run(self, once: bool = False):
        """启动引擎。once=True 时只执行一次（用于测试或手动触发）。"""
        self.broker.connect()
        self._running = True
        logger.info("[LiveEngine] 启动")

        try:
            while self._running:
                self._tick()
                if once:
                    break
                time.sleep(self.poll_interval)
        except KeyboardInterrupt:
            logger.info("[LiveEngine] 收到中断信号，停止")
        finally:
            self.stop()

    def stop(self):
        self._running = False
        try:
            self.broker.disconnect()
        except Exception as e:
            logger.warning(f"[LiveEngine] 断开 Broker 时出错: {e}")
        logger.info("[LiveEngine] 已停止")

    # ---------- 单次执行 ----------
    def _tick(self):
        now = datetime.now()
        logger.info(f"[LiveEngine] ===== 轮询 {now.isoformat()} =====")

        # 1. 获取数据
        try:
            factor_data = self.data_provider()
        except Exception as e:
            logger.error(f"[LiveEngine] 数据获取失败: {e}")
            return

        if factor_data is None or factor_data.empty:
            logger.warning("[LiveEngine] 数据为空，跳过本次轮询")
            return

        # 2. 获取当前状态
        positions = self.broker.get_positions()
        cash = self.broker.get_cash()
        current_positions: Dict[str, Position] = {
            p.symbol: Position(
                symbol=p.symbol,
                quantity=p.quantity,
                entry_price=p.avg_cost,
                entry_date="",
            )
            for p in positions
        }

        # 3. 生成订单
        try:
            orders = self.strategy.generate_orders(
                factor_data, current_positions, cash, now
            )
        except Exception as e:
            logger.error(f"[LiveEngine] 策略生成订单失败: {e}")
            return

        if not orders:
            logger.info("[LiveEngine] 无交易信号")
            return

        # 4. 执行订单（带风控）
        for order in orders:
            self._execute(order, cash, current_positions)

    # ---------- 订单执行 ----------
    def _execute(self, order: Order, cash: float, current_positions: Dict[str, Position]):
        if order.quantity <= 0 or order.price <= 0:
            logger.warning(f"[LiveEngine] 非法订单，跳过: {order}")
            return

        # 风控检查
        allowed, reason = self._risk_check(order, cash, current_positions)
        if not allowed:
            logger.warning(f"[LiveEngine] 风控拦截 {order.symbol} {order.direction}: {reason}")
            self._log_trade(order, None, reason)
            return

        try:
            if order.direction == "BUY":
                broker_order = self.broker.buy(
                    symbol=order.symbol,
                    quantity=order.quantity,
                    price=order.price,
                    order_type=order.order_type,
                )
            elif order.direction == "SELL":
                broker_order = self.broker.sell(
                    symbol=order.symbol,
                    quantity=order.quantity,
                    price=order.price,
                    order_type=order.order_type,
                )
            else:
                logger.warning(f"[LiveEngine] 未知方向 {order.direction}")
                return
        except Exception as e:
            logger.error(f"[LiveEngine] 下单异常 {order.symbol}: {e}")
            self._log_trade(order, None, f"异常: {e}")
            return

        if broker_order.status == OrderStatus.REJECTED:
            logger.warning(f"[LiveEngine] 订单被拒绝: {broker_order.reason}")
            self._log_trade(order, broker_order, broker_order.reason)
            return

        logger.info(
            f"[LiveEngine] 订单提交成功 {broker_order.order_id} "
            f"{order.symbol} {order.direction} {order.quantity} @ {order.price}"
        )
        self._log_trade(order, broker_order, "OK")

    # ---------- 风控 ----------
    def _risk_check(self, order: Order, cash: float,
                    current_positions: Dict[str, Position]) -> tuple:
        max_pos_pct = self.risk_config.get("max_position_pct", 0.15)
        max_total_pct = self.risk_config.get("max_total_position", 0.85)

        # 计算当前总市值（近似）
        pos_value = sum(p.quantity * p.entry_price for p in current_positions.values())
        total_value = cash + pos_value

        if order.direction == "BUY":
            order_value = order.price * order.quantity
            if total_value > 0 and order_value / total_value > max_pos_pct:
                return False, f"单股仓位超限({order_value/total_value:.1%}>{max_pos_pct:.1%})"
            new_total = (pos_value + order_value) / total_value if total_value > 0 else 0
            if new_total > max_total_pct:
                return False, f"总仓位超限({new_total:.1%}>{max_total_pct:.1%})"
        return True, "OK"

    # ---------- 日志 ----------
    def _log_trade(self, order: Order, broker_order: Optional[BrokerOrder], result: str):
        rec = {
            "timestamp": datetime.now().isoformat(),
            "symbol": order.symbol,
            "direction": order.direction,
            "quantity": order.quantity,
            "price": order.price,
            "order_type": order.order_type,
            "reason": order.reason,
            "result": result,
            "order_id": broker_order.order_id if broker_order else None,
        }
        self._trade_log.append(rec)

    def get_trade_log(self) -> pd.DataFrame:
        """获取交易日志 DataFrame。"""
        return pd.DataFrame(self._trade_log)

    def export_trade_log(self, path: str):
        """导出交易日志到 CSV。"""
        df = self.get_trade_log()
        df.to_csv(path, index=False, encoding="utf-8-sig")
        logger.info(f"[LiveEngine] 交易日志已导出: {path}")
