"""C01 Layered — 择时层筛选+MF选图+D5+RP。

C01组合：先用TrendTiming信号(>0.6)过滤出趋势向上的股票池，
再用MF(多因子评分)从池中排名选前40只，最终用风险平价分配。
D5调仓频率，回撤仅-7.55%。

源自X2实验。
"""
from __future__ import annotations
import json, os
from ...base.strategy import SignalStrategy
from ....screening import MultiFactorSelector
from ....timings import TrendTiming
from ....signals import LayeredComposer, MaxSingleWeightConstraint
from ....risk import RiskManager


def build_c01_layered_d5(top_n: int = 40, **kwargs) -> SignalStrategy:
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
