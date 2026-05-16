"""ChipConcentrationSelector + 等权分配 + 3d。

V8实验Chip_Equal_D3结果：年化17.43%, Sharpe 1.401, 回撤-11.19%。

核心：ChipConcentrationSelector + 无择时 + 3d调仓 + 等权分配。
Sharpe 1.401为所有Chip系列策略中最高。
"""
from __future__ import annotations
import json, os
from ...base.strategy import SignalStrategy
from ....screening.impl.momentum_breakout import ChipConcentrationSelector
from ....signals import LayeredComposer, MaxSingleWeightConstraint
from ....risk import RiskManager

def build_chip_equal_d3(top_n: int = 40, **kwargs) -> SignalStrategy:
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
