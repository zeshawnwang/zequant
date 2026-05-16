"""V4 MF+RP基准策略 — V3/V4迭代的基准对照。

核心：MultiFactorSelector(V1权重) + 风险平价 + 无择时 + 周频调仓。
作为所有后续实验的基准线，不做任何择时干预。

适用市场环境：全市场，作为纯选股能力的基准评估
回撤控制手段：仅靠风险平价分散
约束：回撤大（>80%），不能独立实盘
"""
from __future__ import annotations
import json, os

from ...base.strategy import SignalStrategy
from ....screening import MultiFactorSelector
from ....signals import LayeredComposer, MaxSingleWeightConstraint
from ....risk import RiskManager


def build_v4_mf_rp(top_n: int = 40, **kwargs) -> SignalStrategy:
    cfg_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(cfg_dir, "config.json")) as f:
        cfg = json.load(f)
    user_cfg = kwargs.get("strategy_config") or {}

    selector = MultiFactorSelector(
        weights=cfg["selector"]["weights"],
        winsorize=cfg["selector"].get("winsorize", 0.01),
        top_n=top_n,
        normalize_weights=True,
    )

    composer = LayeredComposer(
        top_n=top_n,
        constraints=[
            MaxSingleWeightConstraint(max_weight=cfg.get("composer", {}).get("constraints", [{}])[0].get("max_single_weight", 0.10)),
        ],
    )

    risk_manager = RiskManager(config=cfg.get("risk", {}))

    return SignalStrategy(
        name=cfg["strategy"]["name"],
        selector=selector,
        position_sizer=None,
        composer=composer,
        risk_manager=risk_manager,
        top_n=top_n,
    )
