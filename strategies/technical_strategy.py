"""技术分析策略集合

根据《职业投资者如何分析股票选股》方法论,实现以下策略:
  - trend_breakout: 趋势突破策略
  - oversold_rebound: 超跌反弹策略
  - chip_concentration: 筹码集中策略
  - beta_adaptive: β自适应策略

调用:
    import strategies.technical_strategy  # 触发 @register_strategy 副作用
"""
from __future__ import annotations

from core.strategy import QuantStrategy
from core.strategy_hub import register_strategy
from screening.momentum_breakout import (
    TrendBreakoutSelector,
    OversoldReboundSelector,
    ChipConcentrationSelector,
)
from timings.trend import TrendTiming
from timings.market_regime import MarketRegimeTiming
from portfolios.equal_weight import EqualWeightBuilder


_TIMING_FACTORS_TECH = [
    "momentum_5", "momentum_20", "macd", "rsi_14",
    "volume_ratio", "boll_position",
]


def _get_strategy_config(kwargs: dict) -> dict:
    return kwargs.get("strategy_config") or {}


@register_strategy(
    "trend_breakout",
    category="technical_analysis",
    timing_factors=_TIMING_FACTORS_TECH,
    description="趋势突破策略: 均线多头+放量突破+MACD零轴上方",
)
def build_trend_breakout_strategy(top_n: int = None, **kwargs) -> QuantStrategy:
    """趋势突破策略。

    选股: 均线多头排列 + 股价在60日均线上方 + 放量突破
    择时: 趋势择时(MACD + 动量 + RSI)
    仓位: 等权重分配
    """
    cfg = _get_strategy_config(kwargs)

    top_n = top_n if top_n is not None else cfg.get("top_n", 50)
    selector_cfg = cfg.get("selector", {})
    min_volume_ratio = selector_cfg.get("min_volume_ratio", 1.5)

    timing_cfg = cfg.get("timing", {})
    buy_threshold = timing_cfg.get("buy_threshold", 0.6)
    sell_threshold = timing_cfg.get("sell_threshold", 0.4)

    portfolio_cfg = cfg.get("portfolio", {})
    reserve_cash_ratio = portfolio_cfg.get("reserve_cash_ratio", 0.1)

    selector = TrendBreakoutSelector(
        top_n=top_n * 3,
        min_volume_ratio=min_volume_ratio,
    )
    return QuantStrategy(
        name="TrendBreakout",
        selector=selector,
        timing=TrendTiming(
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
        ),
        portfolio=EqualWeightBuilder(reserve_cash_ratio=reserve_cash_ratio),
        top_n=top_n,
    )


@register_strategy(
    "oversold_rebound",
    category="technical_analysis",
    timing_factors=_TIMING_FACTORS_TECH + ["ma5", "ma20", "macd_golden_cross"],
    description="超跌反弹策略: MACD圆弧底+金叉+站上中期均线",
)
def build_oversold_rebound_strategy(top_n: int = None, **kwargs) -> QuantStrategy:
    """超跌反弹策略。

    选股: MACD在零轴下方形成圆弧底+金叉 + 股价站上21日线 + 短期均线走平向上
    择时: 趋势择时
    仓位: 等权重分配
    """
    cfg = _get_strategy_config(kwargs)

    top_n = top_n if top_n is not None else cfg.get("top_n", 50)
    selector_cfg = cfg.get("selector", {})
    ma_short = selector_cfg.get("ma_short", 5)
    ma_medium = selector_cfg.get("ma_medium", 21)
    ma_long = selector_cfg.get("ma_long", 99)

    timing_cfg = cfg.get("timing", {})
    buy_threshold = timing_cfg.get("buy_threshold", 0.6)
    sell_threshold = timing_cfg.get("sell_threshold", 0.4)

    portfolio_cfg = cfg.get("portfolio", {})
    reserve_cash_ratio = portfolio_cfg.get("reserve_cash_ratio", 0.1)

    selector = OversoldReboundSelector(
        top_n=top_n * 3,
        ma_short=ma_short,
        ma_medium=ma_medium,
        ma_long=ma_long,
    )
    return QuantStrategy(
        name="OversoldRebound",
        selector=selector,
        timing=TrendTiming(
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
        ),
        portfolio=EqualWeightBuilder(reserve_cash_ratio=reserve_cash_ratio),
        top_n=top_n,
    )


@register_strategy(
    "chip_concentration",
    category="technical_analysis",
    timing_factors=_TIMING_FACTORS_TECH,
    description="筹码集中策略: 量缩+筹码集中+均线粘合+放量突破",
)
def build_chip_concentration_strategy(top_n: int = None, **kwargs) -> QuantStrategy:
    """筹码集中策略。

    选股: 量能萎缩到极限 + 筹码集中度低 + 均线粘合 + 蓄势充分后放量突破
    择时: 趋势择时
    仓位: 等权重分配
    """
    cfg = _get_strategy_config(kwargs)

    top_n = top_n if top_n is not None else cfg.get("top_n", 50)
    selector_cfg = cfg.get("selector", {})
    max_volume_contraction = selector_cfg.get("max_volume_contraction", 0.5)
    max_chip_concentration = selector_cfg.get("max_chip_concentration", 0.05)
    max_ma_convergence = selector_cfg.get("max_ma_convergence", 0.05)
    min_breakout_volume = selector_cfg.get("min_breakout_volume", 1.5)

    timing_cfg = cfg.get("timing", {})
    buy_threshold = timing_cfg.get("buy_threshold", 0.6)
    sell_threshold = timing_cfg.get("sell_threshold", 0.4)

    portfolio_cfg = cfg.get("portfolio", {})
    reserve_cash_ratio = portfolio_cfg.get("reserve_cash_ratio", 0.1)

    selector = ChipConcentrationSelector(
        top_n=top_n * 3,
        max_volume_contraction=max_volume_contraction,
        max_chip_concentration=max_chip_concentration,
        max_ma_convergence=max_ma_convergence,
        min_breakout_volume=min_breakout_volume,
    )
    return QuantStrategy(
        name="ChipConcentration",
        selector=selector,
        timing=TrendTiming(
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
        ),
        portfolio=EqualWeightBuilder(reserve_cash_ratio=reserve_cash_ratio),
        top_n=top_n,
    )


@register_strategy(
    "beta_adaptive",
    category="technical_analysis",
    timing_factors=_TIMING_FACTORS_TECH + ["beta_20", "beta_60"],
    description="β自适应策略: 牛熊识别后动态切换高低β",
)
def build_beta_adaptive_strategy(top_n: int = None, **kwargs) -> QuantStrategy:
    """β自适应策略。

    选股: 多因子综合选股(动量+低波)
    择时: 市场状态识别(牛熊判断),牛市用高β进攻,熊市用低β防御
    仓位: 等权重分配
    """
    cfg = _get_strategy_config(kwargs)

    top_n = top_n if top_n is not None else cfg.get("top_n", 50)

    timing_cfg = cfg.get("timing", {})
    bull_beta_threshold = timing_cfg.get("bull_beta_threshold", 1.0)
    bear_beta_threshold = timing_cfg.get("bear_beta_threshold", 0.8)
    bull_buy_threshold = timing_cfg.get("bull_buy_threshold", 0.6)
    bull_sell_threshold = timing_cfg.get("bull_sell_threshold", 0.4)
    bear_buy_threshold = timing_cfg.get("bear_buy_threshold", 0.8)
    bear_sell_threshold = timing_cfg.get("bear_sell_threshold", 0.5)

    portfolio_cfg = cfg.get("portfolio", {})
    reserve_cash_ratio = portfolio_cfg.get("reserve_cash_ratio", 0.1)

    from screening.factor_rank import FactorRankSelector

    selector = FactorRankSelector(
        factor_name="momentum_20",
        ascending=False,
        top_n=top_n * 3,
    )
    return QuantStrategy(
        name="BetaAdaptive",
        selector=selector,
        timing=MarketRegimeTiming(
            bull_beta_threshold=bull_beta_threshold,
            bear_beta_threshold=bear_beta_threshold,
            bull_buy_threshold=bull_buy_threshold,
            bull_sell_threshold=bull_sell_threshold,
            bear_buy_threshold=bear_buy_threshold,
            bear_sell_threshold=bear_sell_threshold,
        ),
        portfolio=EqualWeightBuilder(reserve_cash_ratio=reserve_cash_ratio),
        top_n=top_n,
    )
