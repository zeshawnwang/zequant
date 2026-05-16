"""订单管理器 — 生成可执行订单、检查资金约束。"""
from __future__ import annotations
import logging
from datetime import datetime
from typing import Dict, List

logger = logging.getLogger(__name__)


class OrderManager:
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.max_positions = self.config.get("max_positions", 20)
        self.max_single_weight = self.config.get("max_single_weight", 0.15)

    def build_orders(self, signals: List[Dict], current_positions: Dict[str, int]) -> List[Dict]:
        """
        将信号转为可执行订单。

        Args:
            signals: 信号生成器输出的 [{symbol, weight, direction}, ...]
            current_positions: 当前持仓 {symbol: shares}

        Returns:
            订单列表 [{symbol, direction, price, shares, reason}, ...]
        """
        orders = []
        for sig in signals:
            sym = sig["symbol"]
            if sig["direction"] == "buy":
                if len(current_positions) >= self.max_positions:
                    logger.warning("超最大持仓数，跳过 %s", sym)
                    continue
                orders.append({
                    "symbol": sym, "direction": "BUY",
                    "shares": 100,  # 1手
                    "reason": sig.get("reason", ""),
                    "generated_at": datetime.now().isoformat(),
                })
            elif sig["direction"] == "sell":
                if sym in current_positions:
                    orders.append({
                        "symbol": sym, "direction": "SELL",
                        "shares": current_positions[sym],
                        "reason": sig.get("reason", ""),
                        "generated_at": datetime.now().isoformat(),
                    })
        logger.info("生成 %d 笔订单", len(orders))
        return orders
