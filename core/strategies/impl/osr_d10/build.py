"""OversoldReboundSelector代理信号(超跌反弹) + 无择时 + 10d + 风险平价。

V10实验OSR_D10结果：年化23.37%, Sharpe 0.856。

核心：OversoldReboundSelector超跌反弹代理信号 + 双周调仓(10d) + 风险平价。
OSR真实信号构建方法-RSI低+大涨后反弹动量。
"""
from __future__ import annotations
import json, os
from ...base.strategy import SignalStrategy
from ....screening import OversoldReboundSelector
from ....signals import LayeredComposer, MaxSingleWeightConstraint
from ....risk import RiskManager

def build_osr_d10(top_n: int = 40, **kwargs) -> SignalStrategy:
    cfg_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(cfg_dir, "config.json")) as f:
        cfg = json.load(f)
    selector = OversoldReboundSelector(top_n=top_n)
    composer = LayeredComposer(top_n=top_n, constraints=[
        MaxSingleWeightConstraint(max_weight=cfg["composer"]["constraints"][0]["max_single_weight"])])
    risk = RiskManager(config=cfg.get("risk", {}))
    return SignalStrategy(name=cfg["strategy"]["name"], selector=selector,
        position_sizer=None, composer=composer, risk_manager=risk, top_n=top_n)
