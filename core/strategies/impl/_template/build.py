"""
{StrategyName} — 策略设计说明

设计理念：
  [为什么选这个搭配]

适用市场：
  [这个策略在什么市场表现好]

回撤控制手段：
  [具体靠什么控制回撤]

历史来源：
  源自 experiments/XX 号实验，验证通过后迁移至此。
"""
from __future__ import annotations
from typing import Dict, List, Optional
import json, os

from ...base.strategy import SignalStrategy
from ...base.factory import StrategyFactory
from ....screening import MultiFactorSelector
from ....timings import TrendTiming, TrendVolatilityTiming, MarketRegimeTiming
from ....positioners import RPPortfolioWeights
from ....signals import (
    LayeredComposer,
    DirectComposer,
    MaxSingleWeightConstraint,
    ReserveCashConstraint,
)
from ....risk import RiskManager


def build_{strategy_name}(top_n: int = 40, **kwargs) -> SignalStrategy:
    """构建 {StrategyName} 策略。

    Args:
        top_n: 选股数量
        **kwargs: 支持 strategy_config 覆盖默认配置

    Returns:
        SignalStrategy 实例
    """
    cfg_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(cfg_dir, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    
    user_cfg = kwargs.get("strategy_config") or {}

    # ── 选股器 ──
    selector = MultiFactorSelector(
        weights=cfg["selector"]["weights"],
        winsorize=cfg["selector"].get("winsorize", 0.01),
        top_n=top_n,
    )

    # ── 择时器 ──
    timing_type = cfg["timing"]["type"]
    if timing_type == "TrendTiming":
        timing = TrendTiming(**cfg["timing"].get("params", {}))
    elif timing_type == "TrendVolatilityTiming":
        timing = TrendVolatilityTiming(**cfg["timing"].get("params", {}))
    elif timing_type == "MarketRegimeTiming":
        timing = MarketRegimeTiming(**cfg["timing"].get("params", {}))
    else:
        timing = None

    # ── 信号组合器 ──
    composer_type = cfg["composer"]["type"]
    top_n = cfg["composer"].get("top_n", top_n)
    constraints = []
    for c in cfg["composer"].get("constraints", []):
        if "max_single_weight" in c:
            constraints.append(MaxSingleWeightConstraint(max_weight=c["max_single_weight"]))
        if "reserve_cash" in c:
            constraints.append(ReserveCashConstraint(reserve_ratio=c["reserve_cash"]))
    
    if composer_type == "LayeredComposer":
        composer = LayeredComposer(top_n=top_n, constraints=constraints)
    elif composer_type == "DirectComposer":
        composer = DirectComposer(constraints=constraints)
    else:
        composer = LayeredComposer(top_n=top_n, constraints=constraints)

    # ── 风控 ──
    risk_manager = RiskManager(config=cfg.get("risk", {}))

    return SignalStrategy(
        name=cfg["strategy"]["name"],
        selector=selector,
        position_sizer=timing,
        composer=composer,
        risk_manager=risk_manager,
        top_n=top_n,
    )
