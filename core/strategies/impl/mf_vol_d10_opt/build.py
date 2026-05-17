"""MF+VolTiming+10d+RP — 调优最佳版。

来自tune_mfvol_v2系列调优：vol_lookback=60, ht=0.25, mhd=5。
全区间Sharpe≈1.28, 熊市年化≈30%, 熊市回撤≈-12%。

核心：MultiFactorSelector(V1权重) + VolTiming(ht=0.25) + 10d调仓 + 风险平价
优化点：ht从0.30降至0.25提升熊市Sharpe，vol_lookback使用60
"""
from __future__ import annotations
import json, os
from ...base.strategy import SignalStrategy
from ....screening import MultiFactorSelector
from ....timings import VolatilityTiming
from ....signals import LayeredComposer, MaxSingleWeightConstraint
from ....risk import RiskManager

def build_mf_vol_d10_opt(top_n: int = 20, **kwargs) -> SignalStrategy:
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
