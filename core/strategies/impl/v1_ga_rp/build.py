"""V1 GA风险平价策略 — V1迭代的最终产出。

核心：GA优化出的50因子权重 + 风险平价(波动率倒数)仓位分配 + 周频调仓。
是V1~V4迭代中年化收益最高的纯多因子策略（+45.46% / Sharpe 1.187）。

适用市场环境：强牛市/结构牛/反弹市场表现突出
回撤控制手段：风险平价分配（低波动个股高权重）分散风险
约束：最大回撤约50%，熊市中无明显防御能力
"""
from __future__ import annotations
import json, os

from ...base.strategy import SignalStrategy
from ....screening import MultiFactorSelector
from ....signals import LayeredComposer, MaxSingleWeightConstraint
from ....risk import RiskManager


def build_v1_ga_rp(top_n: int = 40, **kwargs) -> SignalStrategy:
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
