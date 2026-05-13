"""Alpha101 多因子策略族

支持以下策略:
    - alpha101_manual: 配置文件中的手工权重
    - alpha101_from_registry: 从 factor_registry 表拉权重

新架构版本。
"""
from __future__ import annotations
from typing import Dict, Optional
import pandas as pd

from ..base.strategy import SignalStrategy
from .hub import register_strategy
from ...screening import MultiFactorSelector
from ...signals import (
    LayeredComposer,
    MaxSingleWeightConstraint,
    ReserveCashConstraint,
    TrendPositionSizer,
)
from ...risk import RiskManager
from ...database import Database


TIMING_FACTORS = [
    "macd", "macd_signal", "momentum_5", "momentum_20",
    "rsi_14", "volatility_20", "volume_ratio", "boll_position",
]


def _get_strategy_config(kwargs: dict, strategy_name: str) -> dict:
    return kwargs.get("strategy_config") or {}


def _build_multi_factor_strategy(
    weights: Dict[str, float],
    top_n: int = 50,
    strategy_name: str = "MultiFactor",
    **kwargs
) -> SignalStrategy:
    cfg = _get_strategy_config(kwargs, strategy_name)
    top_n = top_n if top_n else cfg.get("top_n", 50)

    selector = MultiFactorSelector(
        weights,
        top_n=top_n * 3,
    )

    position_sizer = TrendPositionSizer(
        bullish_threshold=0.55,
        bearish_threshold=0.40,
    )

    composer = LayeredComposer(
        top_n=top_n,
        constraints=[
            MaxSingleWeightConstraint(max_weight=0.1),
            ReserveCashConstraint(reserve_ratio=0.1),
        ],
    )

    risk_manager = RiskManager(
        stop_loss=StopLoss(method="fixed", threshold=0.10),
        max_total_exposure=0.9,
        max_single_position=0.1,
    )

    return SignalStrategy(
        name=strategy_name,
        selector=selector,
        position_sizer=position_sizer,
        composer=composer,
        risk_manager=risk_manager,
        top_n=top_n,
    )


@register_strategy(
    "alpha101_manual",
    category="multi_factor",
    timing_factors=TIMING_FACTORS,
    description="Alpha101 多因子合成：手工权重",
)
def build_alpha101_manual(top_n: int = None, weights: Dict[str, float] = None, **kwargs) -> SignalStrategy:
    cfg = _get_strategy_config(kwargs, "alpha101_manual")
    top_n = top_n if top_n is not None else cfg.get("top_n", 50)

    if weights is None:
        weights = cfg.get("weights")
    if weights is None:
        weights = {
            "momentum_20": 0.6,
            "volatility_20": -0.4,
            "a3": -0.5,
            "a101": 0.5,
        }

    return _build_multi_factor_strategy(
        weights=weights,
        top_n=top_n,
        strategy_name="Alpha101Manual",
        **kwargs,
    )


@register_strategy(
    "alpha101_from_registry",
    category="multi_factor",
    timing_factors=TIMING_FACTORS,
    description="Alpha101 多因子：从 factor_registry 自动拉取",
)
def build_alpha101_from_registry(db=None, top_n: int = None, min_abs_ir: float = None, **kwargs) -> SignalStrategy:
    cfg = _get_strategy_config(kwargs, "alpha101_from_registry")
    top_n = top_n if top_n is not None else cfg.get("top_n", 50)
    if min_abs_ir is None:
        min_abs_ir = cfg.get("min_abs_ir", 0.2)

    if db is None:
        db = Database()

    selector = MultiFactorSelector.from_registry(
        db,
        top_n=top_n * 3,
        min_abs_ir=min_abs_ir,
    )

    weights = selector.weights if hasattr(selector, "weights") else {}

    return _build_multi_factor_strategy(
        weights=weights,
        top_n=top_n,
        strategy_name="Alpha101FromRegistry",
        **kwargs,
    )
