"""Chip_CovRP(40%) + Chip_Equal(40%) + Chip_Vol(20%) 资金分配组合。

资金分配法：40%资金分配给chip_covrp，40%给chip_equal，20%给chip_vol。
此处定义单个ChipConcentrationSelector子策略，组合在回测框架层实现。

全区间年化≈4.9%, 熊市基本持平。
"""
from __future__ import annotations
import json, os
from ...base.strategy import SignalStrategy
from ....screening.impl.momentum_breakout import ChipConcentrationSelector
from ....signals import LayeredComposer, MaxSingleWeightConstraint
from ....risk import RiskManager

def build_chip_combo(top_n: int = 40, **kwargs) -> SignalStrategy:
    cfg_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(cfg_dir, "config.json")) as f:
        cfg = json.load(f)
    selector = ChipConcentrationSelector(top_n=top_n,
        max_volume_contraction=cfg["selector"].get("max_volume_contraction", 0.5),
        max_chip_concentration=cfg["selector"].get("max_chip_concentration", 0.05),
        max_ma_convergence=cfg["selector"].get("max_ma_convergence", 0.05))
    composer = LayeredComposer(top_n=top_n, constraints=[
        MaxSingleWeightConstraint(max_weight=cfg["composer"]["constraints"][0]["max_single_weight"])])
    risk = RiskManager(config=cfg.get("risk", {}))
    return SignalStrategy(name=cfg["strategy"]["name"], selector=selector,
        position_sizer=None, composer=composer, risk_manager=risk, top_n=top_n)
