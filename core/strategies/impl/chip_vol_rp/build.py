"""Chip+VolTiming+3d+RP — 筹码集中+波动率风控。

V8实验Chip_Vol_D3结果：年化10.37%, Sharpe 1.101, 回撤-10.51%。

核心：ChipConcentrationSelector + VolatilityTiming风控 + 周频3d + 风险平价。
回撤控制冠军——全程最大回撤仅-10.51%。
"""
from __future__ import annotations
import json, os
from ...base.strategy import SignalStrategy
from ....screening.impl.momentum_breakout import ChipConcentrationSelector
from ....timings import VolatilityTiming
from ....signals import LayeredComposer, MaxSingleWeightConstraint
from ....risk import RiskManager

def build_chip_vol_rp(top_n: int = 40, **kwargs) -> SignalStrategy:
    cfg_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(cfg_dir, "config.json")) as f:
        cfg = json.load(f)
    selector = ChipConcentrationSelector(top_n=top_n,
        max_volume_contraction=cfg["selector"].get("max_volume_contraction", 0.5),
        max_chip_concentration=cfg["selector"].get("max_chip_concentration", 0.05),
        max_ma_convergence=cfg["selector"].get("max_ma_convergence", 0.05))
    timing = VolatilityTiming(**cfg["timing"]["params"])
    composer = LayeredComposer(top_n=top_n, constraints=[
        MaxSingleWeightConstraint(max_weight=cfg["composer"]["constraints"][0]["max_single_weight"])])
    risk = RiskManager(config=cfg.get("risk", {}))
    return SignalStrategy(name=cfg["strategy"]["name"], selector=selector,
        position_sizer=timing, composer=composer, risk_manager=risk, top_n=top_n)
