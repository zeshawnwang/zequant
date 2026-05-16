"""MF_D10(0.5)×Chip_CovRP_D3(0.5)多策略组合。

V10实验MF50_ChipCovRP_Combo结果：年化22.78%, Sharpe 1.343, 回撤-17.01%。

核心：50%资金跑MF_D10(tn=50,mhd=10) + 50%资金跑Chip_CovRP(tn=40)。
利用MF的高收益弹性和Chip_CovRP的极致回撤防护，Sharpe 1.343为组合系列最高。
"""
from __future__ import annotations
import json, os
from ...base.strategy import SignalStrategy
from ....screening import MultiFactorSelector
from ....signals import LayeredComposer, MaxSingleWeightConstraint
from ....risk import RiskManager

def build_mf50_chipcovrp50_combo(top_n: int = 40, **kwargs) -> SignalStrategy:
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
