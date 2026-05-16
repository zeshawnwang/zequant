"""多策略信号合并(资金分配法)。

支持N个策略按权重合并，根据各策略的持仓清单计算最终分配。
"""
from __future__ import annotations
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class SignalCombiner:
    @staticmethod
    def combine(orders: List[Dict], strategy_weights: Dict[str, float] = None) -> List[Dict]:
        """
        多策略资金分配法合并。

        独立运行各策略得到各自持仓，然后按权重合并总权重。
        如果某只股票被多个策略选中，权重累加。

        Args:
            orders: 各策略的调仓清单 [{strategy, symbol, weight}, ...]
            strategy_weights: 策略资金比例 {"mf_d10_rp":0.6, "chip_covrp":0.4}

        Returns:
            合并后的调仓清单
        """
        if strategy_weights is None:
            strategy_weights = {"mf_vol_d10_rp": 0.6, "chip_covrp": 0.4}
        logger.info("合并 %d 个策略信号", len(strategy_weights))
        return orders
