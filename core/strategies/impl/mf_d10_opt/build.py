"""MF(V1权重) + 30只持仓 + 5天最低持有 + 10天调仓 — 调优最佳版。

来自tune_mf_v2系列调优：tn=30, mhd=5, rf=10。
全区间年化≈39%, Sharpe≈1.30, 熊市年化≈54%, 熊市回撤≈-15%。

核心：MultiFactorSelector(V1权重) + 无择时 + 10d调仓 + 风险平价
优化点：top_n从40降至30集中仓位，min_hold_days从10降至5提高灵活性
"""
from __future__ import annotations
import json, os
from ...base.strategy import SignalStrategy
from ....screening import MultiFactorSelector
from ....signals import LayeredComposer, MaxSingleWeightConstraint
from ....risk import RiskManager

def build_mf_d10_opt(top_n: int = 30, **kwargs) -> SignalStrategy:
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
