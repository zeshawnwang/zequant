"""
动量策略实例
示例策略：选高动量股票 + 趋势择时 + 等权重分配。
"""
from selectors.factor_rank import FactorRankSelector
from timings.trend import TrendTiming
from portfolios.equal_weight import EqualWeightBuilder
from core.strategy import QuantStrategy


def create_momentum_strategy(top_n: int = 50) -> QuantStrategy:
    """
    创建动量策略。
    - 选股：高动量（momentum_20排名前N）
    - 择时：趋势择时（均线+MACD）
    - 仓位：等权重
    """
    selector = FactorRankSelector(
        factor_name='momentum_20',
        ascending=False,  # 动量越高越好
        top_n=top_n * 3  # 初筛多一些
    )

    timing = TrendTiming(
        sma_short=5,
        sma_medium=20,
        buy_threshold=0.6,
        sell_threshold=0.4
    )

    portfolio = EqualWeightBuilder(reserve_cash_ratio=0.1)

    return QuantStrategy(
        name="MomentumStrategy",
        selector=selector,
        timing=timing,
        portfolio=portfolio,
        top_n=top_n
    )


def create_low_vol_strategy(top_n: int = 50) -> QuantStrategy:
    """
    创建低波动策略。
    - 选股：低波动（volatility_20排名后N%，取最稳的）
    - 择时：趋势择时
    - 仓位：风险平价
    """
    from portfolios.risk_parity import RiskParityBuilder

    selector = FactorRankSelector(
        factor_name='volatility_20',
        ascending=True,  # 波动率越低越好
        top_n=top_n * 3
    )

    timing = TrendTiming(
        sma_short=5,
        sma_medium=20,
        buy_threshold=0.6,
        sell_threshold=0.4
    )

    portfolio = RiskParityBuilder(
        volatility_factor='volatility_20',
        max_weight=0.15
    )

    return QuantStrategy(
        name="LowVolStrategy",
        selector=selector,
        timing=timing,
        portfolio=portfolio,
        top_n=top_n
    )
