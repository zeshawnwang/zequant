"""OversoldRebound代理信号 + VolTiming择时 + 等权 + 10d。

V10实验OSR_Vol_EQ_D10结果：年化24.35%, Sharpe 0.970, 回撤-28.44%。

核心：OversoldReboundSelector超跌反弹代理信号 + VolatilityTiming波动率择时
+ 等权分配 + 双周调仓(10d)。
"""
from __future__ import annotations
import json, os
from ...base.strategy import SignalStrategy
from ....screening import OversoldReboundSelector
from ....timings import VolatilityTiming
from ....signals import LayeredComposer, MaxSingleWeightConstraint
from ....risk import RiskManager

def build_osr_vol_eq_d10(top_n: int = 40, **kwargs) -> SignalStrategy:
    cfg_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(cfg_dir, "config.json")) as f:
        cfg = json.load(f)
    selector = OversoldReboundSelector(top_n=top_n)
    timing = VolatilityTiming(**cfg["timing"]["params"])
    composer = LayeredComposer(top_n=top_n, constraints=[
        MaxSingleWeightConstraint(max_weight=cfg["composer"]["constraints"][0]["max_single_weight"])])
    risk = RiskManager(config=cfg.get("risk", {}))
    return SignalStrategy(name=cfg["strategy"]["name"], selector=selector,
        position_sizer=timing, composer=composer, risk_manager=risk, top_n=top_n)
