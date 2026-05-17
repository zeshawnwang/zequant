"""MF+VolTiming+双周+RP — V6系列最优平衡策略。

V6实验B03_MF+VolTiming+D10结果：年化26.13%, Sharpe 1.334, 回撤-22.59%。
最接近目标(回撤<20%)的组合，也是Sharpe最高的稳定策略。

核心：MultiFactorSelector(V1权重) + VolTiming风控 + 双周调仓(10d) + 风险平价
"""
from __future__ import annotations
import json, os
from ...base.strategy import SignalStrategy
from ....screening import MultiFactorSelector
from ....timings import VolatilityTiming
from ....signals import LayeredComposer, MaxSingleWeightConstraint
from ....risk import RiskManager

def build_mf_vol_d10_rp(top_n: int = 20, **kwargs) -> SignalStrategy:
    cfg_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(cfg_dir, "config.json")) as f:
        cfg = json.load(f)
    selector = MultiFactorSelector(weights=cfg["selector"]["weights"],
        winsorize=cfg["selector"].get("winsorize", 0.01), top_n=top_n, normalize_weights=True)
    timing = VolatilityTiming(volatility_factor=cfg["timing"]["params"]["volatility_factor"],
        high_threshold=cfg["timing"]["params"]["high_threshold"],
        low_threshold=cfg["timing"]["params"]["low_threshold"])
    composer = LayeredComposer(top_n=top_n, constraints=[
        MaxSingleWeightConstraint(max_weight=cfg["composer"]["constraints"][0]["max_single_weight"])])
    risk = RiskManager(config=cfg.get("risk", {}))
    return SignalStrategy(name=cfg["strategy"]["name"], selector=selector,
        position_sizer=timing, composer=composer, risk_manager=risk, top_n=top_n)
