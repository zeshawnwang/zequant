"""GA_D10(0.6)×Chip_CovRP(0.4)多策略组合。

X7实验ga_covrp_combo结果：年化42.67%, Sharpe 0.986, 回撤-57.72%。

核心：60%资金跑GA_D10(RP,tp=40,rf=10) + 40%资金跑Chip_CovRP(CovRP,tp=40,rf=10)，
合并持仓后归一化。GA提供高收益弹性，Chip_CovRP提供防御缓冲。
"""
from __future__ import annotations
import json, os
from ...base.strategy import SignalStrategy
from ....screening import MultiFactorSelector
from ....signals import LayeredComposer, MaxSingleWeightConstraint
from ....risk import RiskManager

def build_ga_covrp_combo(top_n: int = 40, **kwargs) -> SignalStrategy:
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
