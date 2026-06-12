"""OversoldReboundSelector代理信号(超跌反弹) + 无择时 + 10d + 风险平价。

V10实验OSR_D10结果：年化23.37%, Sharpe 0.856。

核心：OversoldReboundSelector超跌反弹代理信号 + 双周调仓(10d) + 风险平价。
OSR真实信号构建方法-RSI低+大涨后反弹动量。
"""
from __future__ import annotations
import os
from ....screening import OversoldReboundSelector
from .._factory import _build_signal_strategy

def build_osr_d10(top_n: int = 40, **kwargs):
    cfg_dir = os.path.dirname(os.path.abspath(__file__))
    selector = OversoldReboundSelector(top_n=top_n)
    return _build_signal_strategy(cfg_dir, selector, top_n=top_n)
