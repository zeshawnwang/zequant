"""新架构示例策略 - 积木式拼装

展示如何用 SignalStrategy 架构，选择不同的模块来组合策略。

可用积木：
┌─────────┐  ┌───────────────┐  ┌──────────────────┐  ┌─────────────────┐
│ 选股层  │  │ 仓位管理层    │  │ 信号组合层      │  │ 风控层         │
├─────────┤  ├───────────────┤  ├──────────────────┤  ├─────────────────┤
│FactorRank│ │TrendPosition  │ │LayeredComposer  │ │RiskManager     │
│-momentum│ │-volatility    │ │DirectComposer   │ │-单票约束      │
│-low_vol │ │               │ │VoteComposer     │ │-总仓位约束    │
└─────────┘  └───────────────┘  └──────────────────┘  └─────────────────┘

示例策略配置：
1. 动量策略 (momentum_v2)
2. 低波动策略 (low_vol_v2)
3. 趋势+波动复合择时 (trend_vol_v2)
"""
from __future__ import annotations

from ..base.strategy import SignalStrategy
from ...signals import (
    LayeredComposer,
    DirectComposer,
    MaxSingleWeightConstraint,
    MaxTotalPositionConstraint,
    ReserveCashConstraint,
    TrendPositionSizer,
    VolatilityPositionSizer,
)
from ...risk import RiskManager
from ...screening import FactorRankSelector
from ...timings import TrendVolatilityTiming


def build_momentum_v2(top_n: int = 30, **kwargs) -> SignalStrategy:
    """
    动量策略 v2 (新架构)
    """
    cfg = kwargs.get("strategy_config") or {}

    selector = FactorRankSelector(
        factor_name=cfg.get("selector", {}).get("factor_name", "momentum_20"),
        ascending=cfg.get("selector", {}).get("ascending", False),
        top_n=top_n * 3,
    )

    position_sizer = TrendPositionSizer(
        bullish_threshold=cfg.get("position_sizer", {}).get("bullish_threshold", 0.6),
        bearish_threshold=cfg.get("position_sizer", {}).get("bearish_threshold", 0.4),
    )

    composer = LayeredComposer(
        top_n=top_n,
        constraints=[
            MaxSingleWeightConstraint(max_weight=0.05),
            ReserveCashConstraint(reserve_ratio=0.05),
        ],
    )

    risk_manager = RiskManager(config={
        'stop_loss': 0.10,
        'take_profit': 0.30,
        'max_position_pct': 0.05,
        'max_total_position': 0.95,
    })

    return SignalStrategy(
        name="MomentumV2",
        selector=selector,
        position_sizer=position_sizer,
        composer=composer,
        risk_manager=risk_manager,
        top_n=top_n,
    )


def build_low_vol_v2(top_n: int = 30, **kwargs) -> SignalStrategy:
    """
    低波动策略 v2 (新架构)
    """
    cfg = kwargs.get("strategy_config") or {}

    selector = FactorRankSelector(
        factor_name=cfg.get("selector", {}).get("factor_name", "volatility_20"),
        ascending=cfg.get("selector", {}).get("ascending", True),
        top_n=top_n * 3,
    )

    position_sizer = VolatilityPositionSizer(
        volatility_factor=cfg.get("position_sizer", {}).get("volatility_factor", "volatility_20"),
        target_volatility=cfg.get("position_sizer", {}).get("target_volatility", 0.20),
        max_position=cfg.get("position_sizer", {}).get("max_position", 1.0),
    )

    composer = DirectComposer(
        constraints=[
            MaxSingleWeightConstraint(max_weight=0.10),
            ReserveCashConstraint(reserve_ratio=0.10),
        ],
    )

    risk_manager = RiskManager(config={
        'stop_loss': 0.20,
        'take_profit': 0.40,
        'max_position_pct': 0.10,
        'max_total_position': 0.90,
    })

    return SignalStrategy(
        name="LowVolV2",
        selector=selector,
        position_sizer=position_sizer,
        composer=composer,
        risk_manager=risk_manager,
        top_n=top_n,
    )


def build_trend_vol_v2(top_n: int = 30, **kwargs) -> SignalStrategy:
    """
    趋势+波动复合择时策略 v2 (新架构)
    """
    cfg = kwargs.get("strategy_config") or {}

    selector = FactorRankSelector(
        factor_name=cfg.get("selector", {}).get("factor_name", "momentum_20"),
        ascending=False,
        top_n=top_n * 3,
    )

    class TrendVolatilityPositionSizer:
        def __init__(self, sma_short=5, sma_medium=20, buy_threshold=0.6,
                     sell_threshold=0.4, volatility_factor="volatility_20",
                     high_threshold=0.05, low_threshold=0.03):
            self.timing = TrendVolatilityTiming(
                sma_short=sma_short,
                sma_medium=sma_medium,
                buy_threshold=buy_threshold,
                sell_threshold=sell_threshold,
                volatility_factor=volatility_factor,
                high_threshold=high_threshold,
                low_threshold=low_threshold,
            )

        def get_position(self, date, market_data=None, current_position=1.0) -> float:
            if market_data is None or len(market_data) == 0:
                return 0.0
            signals = self.timing.generate(market_data, {}, 0, date)
            if not signals:
                return 0.0
            for sig in signals:
                if hasattr(sig, 'signal_type'):
                    if sig.signal_type == 1:
                        return 0.8
            return 0.0

    position_sizer = TrendVolatilityPositionSizer(
        sma_short=cfg.get("position_sizer", {}).get("sma_short", 5),
        sma_medium=cfg.get("position_sizer", {}).get("sma_medium", 20),
        buy_threshold=cfg.get("position_sizer", {}).get("buy_threshold", 0.6),
        sell_threshold=cfg.get("position_sizer", {}).get("sell_threshold", 0.4),
        volatility_factor="volatility_20",
        high_threshold=0.05,
        low_threshold=0.03,
    )

    composer = LayeredComposer(
        top_n=top_n,
        constraints=[
            MaxSingleWeightConstraint(max_weight=0.05),
            ReserveCashConstraint(reserve_ratio=0.20),
        ],
    )

    risk_manager = RiskManager(config={
        'stop_loss': 0.08,
        'take_profit': 0.20,
        'max_position_pct': 0.05,
        'max_total_position': 0.80,
    })

    return SignalStrategy(
        name="TrendVolV2",
        selector=selector,
        position_sizer=position_sizer,
        composer=composer,
        risk_manager=risk_manager,
        top_n=top_n,
    )


STRATEGIES = {
    "momentum_v2": build_momentum_v2,
    "low_vol_v2": build_low_vol_v2,
    "trend_vol_v2": build_trend_vol_v2,
}


if __name__ == "__main__":
    print("新架构策略构建器已加载")
    print("\n可用策略:")
    for name in STRATEGIES.keys():
        print(f"  - {name}")
