"""MF+TrendTiming+5d+RP — 温和择时+回撤控制。

V6实验B01_MF+TrendTiming+D5结果：年化20.63%, Sharpe 0.555, 回撤-26.22%。

核心：MultiFactorSelector(V1权重) + TrendTiming温和择时 + 5天调仓 + 风险平价
"""
from __future__ import annotations
import json, os
from ...base.strategy import SignalStrategy
from ....screening import MultiFactorSelector
from ....timings import TrendTiming
from ....signals import LayeredComposer, MaxSingleWeightConstraint
from ....risk import RiskManager

def build_mf_trend_d5_rp(top_n: int = 40, **kwargs) -> SignalStrategy:
    cfg_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(cfg_dir, "config.json")) as f:
        cfg = json.load(f)
    selector = MultiFactorSelector(weights=cfg["selector"]["weights"],
        winsorize=cfg["selector"].get("winsorize", 0.01), top_n=top_n, normalize_weights=True)
    timing = TrendTiming(**cfg["timing"]["params"])
    composer = LayeredComposer(top_n=top_n, constraints=[
        MaxSingleWeightConstraint(max_weight=cfg["composer"]["constraints"][0]["max_single_weight"])])
    risk = RiskManager(config=cfg.get("risk", {}))
    return SignalStrategy(name=cfg["strategy"]["name"], selector=selector,
        position_sizer=timing, composer=composer, risk_manager=risk, top_n=top_n)
