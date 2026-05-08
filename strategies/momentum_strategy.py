"""动量与低波动策略(横截面)。

两个策略通过 @register_strategy 注册到 StrategyHub:
    - momentum_top50 : 买入 momentum_20 排名前 N 的股票,趋势择时,等权分仓
    - low_vol_top50  : 买入 volatility_20 最低的 N 只股票,趋势择时,风险平价分仓

调用方:
    from core.strategy_hub import StrategyHub
    import strategies                           # 触发注册
    strat = StrategyHub.create("momentum_top50", top_n=50)
"""
from __future__ import annotations

from core.strategy import QuantStrategy
from core.strategy_hub import register_strategy
from screening.factor_rank import FactorRankSelector
from timings.trend import TrendTiming
from portfolios.equal_weight import EqualWeightBuilder
from portfolios.risk_parity import RiskParityBuilder


# TrendTiming 依赖的技术因子,写在 StrategyHub 元数据里,
# 回测脚本据此知道要额外加载这些列(无需用户手配)
_TIMING_FACTORS = ["momentum_5", "momentum_20", "macd", "rsi_14"]


@register_strategy(
    "momentum_top50",
    category="cross_section",
    timing_factors=_TIMING_FACTORS,
    description="按 momentum_20 选动量股,趋势择时,等权重持有",
)
def build_momentum_strategy(top_n: int = 50, **_) -> QuantStrategy:
    """高动量策略:动量越高越优先,等权重买入。"""
    selector = FactorRankSelector(
        factor_name="momentum_20",
        ascending=False,
        top_n=top_n * 3,
    )
    return QuantStrategy(
        name="MomentumStrategy",
        selector=selector,
        timing=TrendTiming(sma_short=5, sma_medium=20,
                           buy_threshold=0.6, sell_threshold=0.4),
        portfolio=EqualWeightBuilder(reserve_cash_ratio=0.1),
        top_n=top_n,
    )


@register_strategy(
    "low_vol_top50",
    category="cross_section",
    timing_factors=_TIMING_FACTORS + ["volatility_20"],
    description="按 volatility_20 选低波动股,趋势择时,风险平价分配",
)
def build_low_vol_strategy(top_n: int = 50, **_) -> QuantStrategy:
    """低波动策略:波动率越低越优先,风险平价分配仓位。"""
    selector = FactorRankSelector(
        factor_name="volatility_20",
        ascending=True,
        top_n=top_n * 3,
    )
    return QuantStrategy(
        name="LowVolStrategy",
        selector=selector,
        timing=TrendTiming(sma_short=5, sma_medium=20,
                           buy_threshold=0.6, sell_threshold=0.4),
        portfolio=RiskParityBuilder(volatility_factor="volatility_20", max_weight=0.15),
        top_n=top_n,
    )