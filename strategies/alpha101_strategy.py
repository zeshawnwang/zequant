"""Alpha101 多因子策略族(3 种注册到 StrategyHub)。

| 注册名                       | requires_evaluation | 因子来源                      |
|------------------------------|---------------------|-------------------------------|
| alpha101_manual              | False               | 配置文件中的手工权重          |
| alpha101_from_registry       | False               | 从 factor_registry 表拉权重    |
| alpha101_walk_forward        | True                | 由 FactorEvaluator 在线评估    |

择时器与组合器统一:
- TrendTiming(短均线 5 / 长均线 20,买阈 0.55,卖阈 0.4)
- EqualWeightBuilder(保留 10% 现金)

新策略只需在本文件加一个 @register_strategy 工厂函数即可。
"""
from __future__ import annotations
from typing import Dict, Optional
import pandas as pd

from core.strategy import QuantStrategy
from core.strategy_hub import register_strategy
from screening.multi_factor import MultiFactorSelector
from timings.trend import TrendTiming
from portfolios.equal_weight import EqualWeightBuilder


# 公共择时器与组合器(三个 alpha101 策略共用,确保对照可比)
TIMING_FACTORS = [
    "macd", "macd_signal", "momentum_5", "momentum_20",
    "rsi_14", "volatility_20", "volume_ratio", "boll_position",
]


def _get_strategy_config(kwargs: dict, strategy_name: str) -> dict:
    """从 kwargs 中提取策略配置,优先使用传入的 strategy_config。

    若调用方传入了 strategy_config,则直接返回;
    否则返回空 dict,工厂函数将使用默认值。
    """
    return kwargs.get("strategy_config") or {}


def _make_timing(cfg: dict) -> TrendTiming:
    """根据配置创建 TrendTiming 择时器。

    若配置为空,则使用默认参数。
    """
    timing_cfg = cfg.get("timing", {})
    return TrendTiming(
        sma_short=timing_cfg.get("sma_short", 5),
        sma_medium=timing_cfg.get("sma_medium", 20),
        buy_threshold=timing_cfg.get("buy_threshold", 0.55),
        sell_threshold=timing_cfg.get("sell_threshold", 0.4),
    )


def _make_portfolio(cfg: dict) -> EqualWeightBuilder:
    """根据配置创建 EqualWeightBuilder 仓位分配器。

    若配置为空,则使用默认参数。
    """
    portfolio_cfg = cfg.get("portfolio", {})
    return EqualWeightBuilder(
        reserve_cash_ratio=portfolio_cfg.get("reserve_cash_ratio", 0.1)
    )


# ===== 1) 手工权重 ======================================================

@register_strategy(
    "alpha101_manual",
    category="multi_factor",
    timing_factors=TIMING_FACTORS,
    description="Alpha101 多因子合成:手工权重(动量+波动+a3 反转+a101 强势)",
)
def build_alpha101_manual(
    top_n: int = None,
    weights: Optional[Dict[str, float]] = None,
    **kwargs,
) -> QuantStrategy:
    """权重静态写死的多因子策略(快速基线)。

    参数优先级: 传入的参数 > 配置文件中的参数 > 代码默认值
    """
    # 读取策略专属配置
    cfg = _get_strategy_config(kwargs, "alpha101_manual")

    # 选股参数
    top_n = top_n if top_n is not None else cfg.get("top_n", 50)

    # 因子权重:传入的 weights 优先级最高,其次配置文件,最后代码默认值
    if weights is None:
        weights = cfg.get("weights")
    if weights is None:
        weights = {
            "momentum_20": 0.6,        # 顺势
            "volatility_20": -0.4,     # 反向(低波偏好)
            "a3": -0.5,                # 短期反转
            "a101": 0.5,               # 当日强势
        }

    selector = MultiFactorSelector(weights, top_n=top_n * 3)
    return QuantStrategy(
        name="Alpha101Manual",
        selector=selector,
        timing=_make_timing(cfg),
        portfolio=_make_portfolio(cfg),
        top_n=top_n,
    )


# ===== 2) 从 factor_registry 表拉权重 ===================================

@register_strategy(
    "alpha101_from_registry",
    category="multi_factor",
    timing_factors=TIMING_FACTORS,
    description="Alpha101 多因子:从 factor_registry 自动拉 |IR|≥阈值的因子加权",
)
def build_alpha101_from_registry(
    db,
    top_n: int = None,
    min_abs_ir: float = None,
    **kwargs,
) -> QuantStrategy:
    """需先跑 evaluate_factors.py 把评估结果落到 factor_registry。

    参数优先级: 传入的参数 > 配置文件中的参数 > 代码默认值
    """
    # 读取策略专属配置
    cfg = _get_strategy_config(kwargs, "alpha101_from_registry")

    # 选股参数
    top_n = top_n if top_n is not None else cfg.get("top_n", 50)
    if min_abs_ir is None:
        min_abs_ir = cfg.get("min_abs_ir", 0.2)

    selector = MultiFactorSelector.from_registry(
        db, top_n=top_n * 3, min_abs_ir=min_abs_ir,
    )
    return QuantStrategy(
        name="Alpha101FromRegistry",
        selector=selector,
        timing=_make_timing(cfg),
        portfolio=_make_portfolio(cfg),
        top_n=top_n,
    )


# ===== 3) Walk-forward 评估驱动 ========================================

@register_strategy(
    "alpha101_walk_forward",
    category="multi_factor",
    requires_evaluation=True,
    timing_factors=TIMING_FACTORS,
    eval_factor_filter="alpha",   # 评估候选限定为 a* 因子
    description="Alpha101 walk-forward:在评估期算 IR,样本外用 |IR| 前 N 个因子合成",
)
def build_alpha101_walk_forward(
    db,
    eval_summary: pd.DataFrame,
    top_n: int = None,
    top_factors: int = None,
    **kwargs,
) -> QuantStrategy:
    """评估期与回测期严格分离,避免前视偏差。

    Args:
        eval_summary: FactorEvaluator.evaluate_all() 返回的 summary
        top_factors:  取 |IR| 前 N 个因子做合成

    参数优先级: 传入的参数 > 配置文件中的参数 > 代码默认值
    """
    if eval_summary is None or eval_summary.empty:
        raise ValueError("alpha101_walk_forward 要求传入非空 eval_summary")

    # 读取策略专属配置
    cfg = _get_strategy_config(kwargs, "alpha101_walk_forward")

    # 选股参数
    top_n = top_n if top_n is not None else cfg.get("top_n", 30)
    if top_factors is None:
        top_factors = cfg.get("top_factors", 8)

    df = eval_summary.copy()
    df["abs_ir"] = df["ir"].abs()
    top = df.sort_values("abs_ir", ascending=False).head(top_factors)
    selector = MultiFactorSelector.from_summary(top, top_n=top_n * 3, min_abs_ir=0.0)
    return QuantStrategy(
        name=f"Alpha101WalkForward_top{top_factors}",
        selector=selector,
        timing=_make_timing(cfg),
        portfolio=_make_portfolio(cfg),
        top_n=top_n,
    )
