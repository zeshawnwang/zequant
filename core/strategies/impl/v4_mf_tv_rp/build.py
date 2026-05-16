"""V4 MF+TV择时风控策略 — V4迭代的最优风控方案。

核心：MultiFactorSelector(V1权重) + TrendVolatilityTiming择时 + 风险平价 + 周频调仓。
将V1强选股和V4 TV择时的风控能力结合。

适用市场环境：高波动市场/熊市（回撤控制最好-19%）
回撤控制手段：TV择时的波动率风控（vol>5%强制减仓）
约束：平均仓位仅24%，牛市中收益有限
"""
from __future__ import annotations
import json, os

from ...base.strategy import SignalStrategy
from ....screening import MultiFactorSelector
from ....timings import TrendVolatilityTiming
from ....signals import LayeredComposer, MaxSingleWeightConstraint
from ....risk import RiskManager


def build_v4_mf_tv_rp(top_n: int = 40, **kwargs) -> SignalStrategy:
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

    timing = TrendVolatilityTiming(**cfg["timing"]["params"])

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
        position_sizer=timing,
        composer=composer,
        risk_manager=risk_manager,
        top_n=top_n,
    )
