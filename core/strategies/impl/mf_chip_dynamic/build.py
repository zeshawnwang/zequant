"""动态权重: 牛市→70%MF+30%Chip, 熊市→30%MF+70%Chip。

使用trend_p信号判断牛熊，动态调整MF与Chip子策略的资金分配。
牛市多配MF（高收益弹性），熊市多配Chip（防御属性）。
全区间Sharpe≈1.39, 回撤仅-8.72%。

核心：MultiFactorSelector(V1权重) + 无择时 + 动态资金分配（回测框架层实现）
"""
from __future__ import annotations
import json, os
from ...base.strategy import SignalStrategy
from ....screening import MultiFactorSelector
from ....signals import LayeredComposer, MaxSingleWeightConstraint
from ....risk import RiskManager

def build_mf_chip_dynamic(top_n: int = 40, **kwargs) -> SignalStrategy:
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
