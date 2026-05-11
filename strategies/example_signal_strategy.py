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

from core.strategy.base import SignalStrategy
from core.signals import (
    LayeredComposer,
    DirectComposer,
    MaxSingleWeightConstraint,
    MaxTotalPositionConstraint,
    MinPositionConstraint,
    ReserveCashConstraint,
)
from core.signals.position import TrendPositionSizer, VolatilityPositionSizer
from core.risk import RiskManager, StopLoss, TakeProfit
from screening.factor_rank import FactorRankSelector
from timings.trend_volatility import TrendVolatilityTiming


# =============================================================================
# 示例策略 1: 动量策略 v2
# 积木组合:
#   - 选股: FactorRankSelector (momentum_20, 降序)
#   - 仓位: TrendPositionSizer (趋势择时)
#   - 组合: LayeredComposer (先择时再分配)
#   - 风控: RiskManager (单票5%, 总仓位95%, 固定止损10%)
# =============================================================================
def build_momentum_v2(top_n: int = 30, **kwargs) -> SignalStrategy:
    """
    动量策略 v2 (新架构)
    """
    cfg = kwargs.get("strategy_config") or {}

    # 积木 1: 选股器
    selector = FactorRankSelector(
        factor_name=cfg.get("selector", {}).get("factor_name", "momentum_20"),
        ascending=cfg.get("selector", {}).get("ascending", False),
        top_n=top_n * 3,
    )

    # 积木 2: 仓位管理器
    position_sizer = TrendPositionSizer(
        sma_short=cfg.get("position_sizer", {}).get("sma_short", 5),
        sma_medium=cfg.get("position_sizer", {}).get("sma_medium", 20),
        buy_threshold=cfg.get("position_sizer", {}).get("buy_threshold", 0.6),
        sell_threshold=cfg.get("position_sizer", {}).get("sell_threshold", 0.4),
    )

    # 积木 3: 信号组合器
    composer = LayeredComposer(
        top_n=top_n,
        constraints=[
            MaxSingleWeightConstraint(max_weight=0.05),
            ReserveCashConstraint(reserve_ratio=0.05),
        ],
    )

    # 积木 4: 风控管理器
    risk_manager = RiskManager(
        constraints=[
            MaxTotalPositionConstraint(max_total=0.95),
            MinPositionConstraint(min_single=0.005),
        ],
        stop_loss=StopLoss(method="fixed", threshold=0.10),
        take_profit=TakeProfit(method="fixed", threshold=0.30),
        max_total_exposure=0.95,
        max_single_position=0.05,
    )

    # 组装策略
    return SignalStrategy(
        name="MomentumV2",
        selector=selector,
        position_sizer=position_sizer,
        composer=composer,
        risk_manager=risk_manager,
        top_n=top_n,
    )


# =============================================================================
# 示例策略 2: 低波动策略 v2
# 积木组合:
#   - 选股: FactorRankSelector (volatility_20, 升序)
#   - 仓位: VolatilityPositionSizer (波动率择时)
#   - 组合: DirectComposer (直接相乘)
#   - 风控: RiskManager (单票10%, 总仓位90%, 移动止损20%)
# =============================================================================
def build_low_vol_v2(top_n: int = 30, **kwargs) -> SignalStrategy:
    """
    低波动策略 v2 (新架构)
    """
    cfg = kwargs.get("strategy_config") or {}

    # 积木 1: 选股器
    selector = FactorRankSelector(
        factor_name=cfg.get("selector", {}).get("factor_name", "volatility_20"),
        ascending=cfg.get("selector", {}).get("ascending", True),
        top_n=top_n * 3,
    )

    # 积木 2: 仓位管理器
    position_sizer = VolatilityPositionSizer(
        volatility_factor=cfg.get("position_sizer", {}).get("volatility_factor", "volatility_20"),
        target_volatility=cfg.get("position_sizer", {}).get("target_volatility", 0.20),
        max_position=cfg.get("position_sizer", {}).get("max_position", 1.0),
    )

    # 积木 3: 信号组合器
    composer = DirectComposer(
        constraints=[
            MaxSingleWeightConstraint(max_weight=0.10),
            ReserveCashConstraint(reserve_ratio=0.10),
        ],
    )

    # 积木 4: 风控管理器
    risk_manager = RiskManager(
        constraints=[
            MaxTotalPositionConstraint(max_total=0.90),
        ],
        stop_loss=StopLoss(method="trailing", threshold=0.20, lookback=20),
        max_total_exposure=0.90,
        max_single_position=0.10,
    )

    # 组装策略
    return SignalStrategy(
        name="LowVolV2",
        selector=selector,
        position_sizer=position_sizer,
        composer=composer,
        risk_manager=risk_manager,
        top_n=top_n,
    )


# =============================================================================
# 示例策略 3: 趋势+波动复合择时策略
# 积木组合:
#   - 选股: FactorRankSelector (momentum_20, 降序)
#   - 仓位: 复合择时 (TrendVolatilityTiming 适配到 PositionSizer)
#   - 组合: LayeredComposer (先择时再分配)
#   - 风控: 严格风控 (单票5%, 总仓位80%, 双重止损)
# =============================================================================
def build_trend_vol_v2(top_n: int = 30, **kwargs) -> SignalStrategy:
    """
    趋势+波动复合择时策略 v2 (新架构)
    """
    cfg = kwargs.get("strategy_config") or {}

    # 积木 1: 选股器
    selector = FactorRankSelector(
        factor_name=cfg.get("selector", {}).get("factor_name", "momentum_20"),
        ascending=False,
        top_n=top_n * 3,
    )

    # 积木 2: 复合仓位管理器 (用之前创建的 TrendVolatilityTiming)
    # 注意：需要适配旧择时器到新 PositionSizer 接口
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
            """返回仓位系数 0~1"""
            if market_data is None or len(market_data) == 0:
                return 0.0
            # 从旧择时器获取信号，映射到仓位系数
            signals = self.timing.generate(market_data, {}, 0, date)
            if not signals:
                return 0.0
            # 简单映射：有任意买入信号则仓位系数 0.8，否则 0
            for sig in signals:
                if hasattr(sig, 'signal_type'):
                    if sig.signal_type == 1:  # BUY
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

    # 积木 3: 信号组合器
    composer = LayeredComposer(
        top_n=top_n,
        constraints=[
            MaxSingleWeightConstraint(max_weight=0.05),
            ReserveCashConstraint(reserve_ratio=0.20),
        ],
    )

    # 积木 4: 风控管理器
    risk_manager = RiskManager(
        constraints=[
            MaxTotalPositionConstraint(max_total=0.80),
            MinPositionConstraint(min_single=0.01),
        ],
        stop_loss=StopLoss(method="fixed", threshold=0.08),
        take_profit=TakeProfit(method="fixed", threshold=0.20),
        max_total_exposure=0.80,
        max_single_position=0.05,
    )

    # 组装策略
    return SignalStrategy(
        name="TrendVolV2",
        selector=selector,
        position_sizer=position_sizer,
        composer=composer,
        risk_manager=risk_manager,
        top_n=top_n,
    )


# =============================================================================
# 策略列表
# =============================================================================
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
    
    print("\n示例用法:")
    print("""
    from strategies.example_signal_strategy import STRATEGIES
    
    # 选择策略
    strategy_builder = STRATEGIES["momentum_v2"]
    
    # 构建策略（可传入自定义配置）
    strategy = strategy_builder(
        top_n=30,
        strategy_config={
            "selector": {"factor_name": "momentum_20"},
            "position_sizer": {"sma_short": 5, "sma_medium": 20},
        }
    )
    
    print(f"策略已构建: {strategy.name}")
    """)
