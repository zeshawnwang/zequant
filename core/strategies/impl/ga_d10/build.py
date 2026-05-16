"""GA优化+10d+RP — 全量GA+新技术因子优化结果。

X5实验GA(rf=10)结果：年化33.59%, Sharpe 1.435, 回撤-16.33%。

核心：MultiFactorSelector(GA优化66因子权重) + 双周调仓(10d) + 风险平价。
全量GA最高Sharpe(1.435) + 回撤最低(-16.33%)，唯一同时满足回撤<20%+Sharpe>1.4的策略。
"""
from __future__ import annotations
import json, os
from ...base.strategy import SignalStrategy
from ....screening import MultiFactorSelector
from ....signals import LayeredComposer, MaxSingleWeightConstraint
from ....risk import RiskManager

def build_ga_d10(top_n: int = 40, **kwargs) -> SignalStrategy:
    cfg_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(cfg_dir, "config.json")) as f:
        cfg = json.load(f)
    selector = MultiFactorSelector(weights=cfg["selector"]["weights"],
        winsorize=cfg["selector"].get("winsorize", 0.01), top_n=top_n, normalize_weights=True)
    composer = LayeredComposer(top_n=top_n, constraints=[
        MaxSingleWeightConstraint(max_weight=cfg["composer"]["constraints"][0]["max_single_weight"])])
    risk = RiskManager(config=cfg.get("risk", {}))
    return SignalStrategy(name=cfg["strategy"]["name"], selector=selector,
        position_sizer=None, composer=composer, risk_manager=risk, top_n=top_n)
