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


def _get_strategy_config(kwargs: dict, strategy_name: str) -> dict:
    """从 kwargs 中提取策略配置,优先使用传入的 strategy_config。

    若调用方传入了 strategy_config,则直接返回;
    否则返回空 dict,工厂函数将使用默认值。
    """
    return kwargs.get("strategy_config") or {}


@register_strategy(
    "momentum_top50",
    category="cross_section",
    timing_factors=_TIMING_FACTORS,
    description="按 momentum_20 选动量股,趋势择时,等权重持有",
)
def build_momentum_strategy(top_n: int = None, **kwargs) -> QuantStrategy:
    """高动量策略:动量越高越优先,等权重买入。

    参数优先级: 传入的 top_n > 配置文件中的 top_n > 默认值 50
    """
    # 读取策略专属配置
    cfg = _get_strategy_config(kwargs, "momentum_top50")

    # 选股参数
    top_n = top_n if top_n is not None else cfg.get("top_n", 50)
    selector_cfg = cfg.get("selector", {})
    factor_name = selector_cfg.get("factor_name", "momentum_20")
    ascending = selector_cfg.get("ascending", False)
    pool_multiplier = selector_cfg.get("pool_multiplier", 3)

    # 择时参数
    timing_cfg = cfg.get("timing", {})
    sma_short = timing_cfg.get("sma_short", 5)
    sma_medium = timing_cfg.get("sma_medium", 20)
    buy_threshold = timing_cfg.get("buy_threshold", 0.6)
    sell_threshold = timing_cfg.get("sell_threshold", 0.4)

    # 仓位分配参数
    portfolio_cfg = cfg.get("portfolio", {})
    reserve_cash_ratio = portfolio_cfg.get("reserve_cash_ratio", 0.1)

    selector = FactorRankSelector(
        factor_name=factor_name,
        ascending=ascending,
        top_n=top_n * pool_multiplier,
    )
    return QuantStrategy(
        name="MomentumStrategy",
        selector=selector,
        timing=TrendTiming(
            sma_short=sma_short,
            sma_medium=sma_medium,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
        ),
        portfolio=EqualWeightBuilder(reserve_cash_ratio=reserve_cash_ratio),
        top_n=top_n,
    )


@register_strategy(
    "low_vol_top50",
    category="cross_section",
    timing_factors=_TIMING_FACTORS + ["volatility_20"],
    description="按 volatility_20 选低波动股,趋势择时,风险平价分配",
)
def build_low_vol_strategy(top_n: int = None, **kwargs) -> QuantStrategy:
    """低波动策略:波动率越低越优先,风险平价分配仓位。

    参数优先级: 传入的 top_n > 配置文件中的 top_n > 默认值 50
    """
    # 读取策略专属配置
    cfg = _get_strategy_config(kwargs, "low_vol_top50")

    # 选股参数
    top_n = top_n if top_n is not None else cfg.get("top_n", 50)
    selector_cfg = cfg.get("selector", {})
    factor_name = selector_cfg.get("factor_name", "volatility_20")
    ascending = selector_cfg.get("ascending", True)
    pool_multiplier = selector_cfg.get("pool_multiplier", 3)

    # 择时参数
    timing_cfg = cfg.get("timing", {})
    sma_short = timing_cfg.get("sma_short", 5)
    sma_medium = timing_cfg.get("sma_medium", 20)
    buy_threshold = timing_cfg.get("buy_threshold", 0.6)
    sell_threshold = timing_cfg.get("sell_threshold", 0.4)

    # 仓位分配参数
    portfolio_cfg = cfg.get("portfolio", {})
    volatility_factor = portfolio_cfg.get("volatility_factor", "volatility_20")
    max_weight = portfolio_cfg.get("max_weight", 0.15)

    selector = FactorRankSelector(
        factor_name=factor_name,
        ascending=ascending,
        top_n=top_n * pool_multiplier,
    )
    return QuantStrategy(
        name="LowVolStrategy",
        selector=selector,
        timing=TrendTiming(
            sma_short=sma_short,
            sma_medium=sma_medium,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
        ),
        portfolio=RiskParityBuilder(
            volatility_factor=volatility_factor,
            max_weight=max_weight,
        ),
        top_n=top_n,
    )
