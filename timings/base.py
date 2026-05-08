"""择时器抽象基类。"""
from __future__ import annotations
from typing import List
import pandas as pd


class ITimingGenerator:
    """择时器接口。

    契约:
      - 输入:factor_data(候选池的因子/价量截面), positions(已持仓 symbol 列表), cash
      - 输出:List[Signal],signal_type ∈ {BUY, SELL, HOLD}
        * BUY:  候选池中看多的标的(由下游 PortfolioBuilder 决定买多少)
        * SELL: 持仓中需要平仓的标的
        * HOLD: 持仓中继续持有的标的(可选,供报告使用)
      - 不产生信号的标的(既不该买也不该卖)无需返回
    """

    def generate(self, factor_data: pd.DataFrame,
                 positions: List[str], cash: float, date=None) -> List:
        raise NotImplementedError