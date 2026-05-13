"""动量与低波动策略(横截面)。

两个策略通过 @register_strategy 注册到 StrategyHub:
    - momentum_top50 : 买入 momentum_20 排名前 N 的股票,趋势择时,等权分仓
    - low_vol_top50  : 买入 volatility_20 最低的 N 只股票,趋势择时,风险平价分仓

新架构版本请参考 strategies/__init__.py
"""
from __future__ import annotations

from ..base.strategy import SignalStrategy
from .hub import register_strategy
from ...screening import FactorRankSelector
from ...signals import (
    LayeredComposer,
    DirectComposer,
    MaxSingleWeightConstraint,
    ReserveCashConstraint,
    TrendPositionSizer,
    VolatilityPositionSizer,
)
from ...risk import RiskManager


def _get_strategy_config(kwargs: dict, strategy_name: str) -> dict:
    return kwargs.get("strategy_config") or {}


@register_strategy(
    "momentum_top50",
    category="cross_section",
    timing_factors=["momentum_5", "momentum_20", "macd", "rsi_14"],
    description="按 momentum_20 选动量股,趋势择时,等权重持有",
)
def build_momentum_strategy(top_n: int = None, **kwargs) -> SignalStrategy:
    cfg = _get_strategy_config(kwargs, "momentum_top50")
    top_n = top_n if top_n is not None else cfg.get("top_n", 50)

    selector = FactorRankSelector(
        factor_name=cfg.get("selector", {}).get("factor_name", "momentum_20"),
        ascending=False,
        top_n=top_n * 3,
    )

    position_sizer = TrendPositionSizer(
        bullish_threshold=0.6,
        bearish_threshold=0.4,
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
        name="MomentumTop50",
        selector=selector,
        position_sizer=position_sizer,
        composer=composer,
        risk_manager=risk_manager,
        top_n=top_n,
    )


@register_strategy(
    "low_vol_top50",
    category="cross_section",
    timing_factors=["momentum_5", "momentum_20", "macd", "rsi_14", "volatility_20"],
    description="选 volatility_20 最低的股票,趋势择时,风险平价",
)
def build_low_vol_strategy(top_n: int = None, **kwargs) -> SignalStrategy:
    cfg = _get_strategy_config(kwargs, "low_vol_top50")
    top_n = top_n if top_n is not None else cfg.get("top_n", 50)

    selector = FactorRankSelector(
        factor_name=cfg.get("selector", {}).get("factor_name", "volatility_20"),
        ascending=True,
        top_n=top_n * 3,
    )

    position_sizer = VolatilityPositionSizer(
        volatility_factor="volatility_20",
        target_volatility=0.2,
        max_position=0.8,
    )

    composer = DirectComposer(
        constraints=[
            MaxSingleWeightConstraint(max_weight=0.15),
            ReserveCashConstraint(reserve_ratio=0.1),
        ],
    )

    risk_manager = RiskManager(config={
        'stop_loss': 0.10,
        'take_profit': 0.12,
        'max_position_pct': 0.15,
        'max_total_position': 0.9,
    })

    return SignalStrategy(
        name="LowVolatility",
        selector=selector,
        position_sizer=position_sizer,
        composer=composer,
        risk_manager=risk_manager,
        top_n=top_n,
    )
