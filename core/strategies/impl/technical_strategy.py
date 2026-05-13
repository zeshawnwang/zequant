"""技术分析策略集合

根据《职业投资者如何分析股票选股》方法论,实现以下策略:
  - trend_breakout: 趋势突破策略
  - oversold_rebound: 超跌反弹策略

新架构版本。
"""
from __future__ import annotations

from ..base.strategy import SignalStrategy
from .hub import register_strategy
from ...screening import TrendBreakoutSelector, OversoldReboundSelector
from ...signals import (
    LayeredComposer,
    MaxSingleWeightConstraint,
    ReserveCashConstraint,
    TrendPositionSizer,
)
from ...risk import RiskManager


def _get_strategy_config(kwargs: dict) -> dict:
    return kwargs.get("strategy_config") or {}


@register_strategy(
    "trend_breakout",
    category="technical_analysis",
    timing_factors=["momentum_5", "momentum_20", "macd", "rsi_14", "volume_ratio", "boll_position"],
    description="趋势突破策略: 均线多头+放量突破+MACD零轴上方",
)
def build_trend_breakout_strategy(top_n: int = None, **kwargs) -> SignalStrategy:
    cfg = _get_strategy_config(kwargs)
    top_n = top_n if top_n is not None else cfg.get("top_n", 30)

    selector = TrendBreakoutSelector(top_n=top_n * 3)

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

    risk_manager = RiskManager(config={
        'stop_loss': 0.10,
        'take_profit': 0.12,
        'max_position_pct': 0.1,
        'max_total_position': 0.9,
    })

    return SignalStrategy(
        name="TrendBreakout",
        selector=selector,
        position_sizer=position_sizer,
        composer=composer,
        risk_manager=risk_manager,
        top_n=top_n,
    )


@register_strategy(
    "oversold_rebound",
    category="technical_analysis",
    timing_factors=["momentum_5", "momentum_20", "macd", "rsi_14", "volume_ratio"],
    description="超跌反弹策略: RSI超卖+价格低位+放量反弹",
)
def build_oversold_rebound_strategy(top_n: int = None, **kwargs) -> SignalStrategy:
    cfg = _get_strategy_config(kwargs)
    top_n = top_n if top_n is not None else cfg.get("top_n", 30)

    selector = OversoldReboundSelector(top_n=top_n * 3)

    position_sizer = TrendPositionSizer(
        bullish_threshold=0.5,
        bearish_threshold=0.3,
    )

    composer = LayeredComposer(
        top_n=top_n,
        constraints=[
            MaxSingleWeightConstraint(max_weight=0.08),
            ReserveCashConstraint(reserve_ratio=0.15),
        ],
    )

    risk_manager = RiskManager(
        stop_loss=StopLoss(method="fixed", threshold=0.08),
        max_total_exposure=0.85,
        max_single_position=0.08,
    )

    return SignalStrategy(
        name="OversoldRebound",
        selector=selector,
        position_sizer=position_sizer,
        composer=composer,
        risk_manager=risk_manager,
        top_n=top_n,
    )
