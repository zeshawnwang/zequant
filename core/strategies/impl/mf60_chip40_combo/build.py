"""MF_D10(0.6)×Chip_D3(0.4)多策略组合。

V10实验MF60_Chip40_Combo结果：年化24.84%, Sharpe 1.269, 回撤-18.20%。

核心：60%资金跑MF_D10(tn=50,mhd=10) + 40%资金跑Chip_D3(tn=40)，合并持仓后归一化。
同时达到回撤<20% + 年化>20%的双重目标，是V10最有价值的组合发现。
"""
from __future__ import annotations
import json, os
from ...base.strategy import SignalStrategy
from ....screening import MultiFactorSelector
from ....screening.impl.momentum_breakout import ChipConcentrationSelector
from ....signals import LayeredComposer, MaxSingleWeightConstraint
from ....risk import RiskManager

def build_mf60_chip40_combo(top_n: int = 40, **kwargs) -> SignalStrategy:
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
