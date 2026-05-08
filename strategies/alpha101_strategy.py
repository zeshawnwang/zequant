"""Alpha101 多因子策略族(3 种注册到 StrategyHub)。

| 注册名                       | requires_evaluation | 因子来源                      |
|------------------------------|---------------------|-------------------------------|
| alpha101_manual              | False               | 代码里写死的手工权重          |
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


def _make_timing() -> TrendTiming:
    return TrendTiming(
        sma_short=5, sma_medium=20,
        buy_threshold=0.55, sell_threshold=0.4,
    )


def _make_portfolio() -> EqualWeightBuilder:
    return EqualWeightBuilder(reserve_cash_ratio=0.1)


# ===== 1) 手工权重 ======================================================

@register_strategy(
    "alpha101_manual",
    category="multi_factor",
    timing_factors=TIMING_FACTORS,
    description="Alpha101 多因子合成:手工权重(动量+波动+a3 反转+a101 强势)",
)
def build_alpha101_manual(
    top_n: int = 50,
    weights: Optional[Dict[str, float]] = None,
    **_,
) -> QuantStrategy:
    """权重静态写死的多因子策略(快速基线)。"""
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
        timing=_make_timing(),
        portfolio=_make_portfolio(),
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
    top_n: int = 50,
    min_abs_ir: float = 0.2,
    **_,
) -> QuantStrategy:
    """需先跑 evaluate_factors.py 把评估结果落到 factor_registry。"""
    selector = MultiFactorSelector.from_registry(
        db, top_n=top_n * 3, min_abs_ir=min_abs_ir,
    )
    return QuantStrategy(
        name="Alpha101FromRegistry",
        selector=selector,
        timing=_make_timing(),
        portfolio=_make_portfolio(),
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
    top_n: int = 30,
    top_factors: int = 8,
    **_,
) -> QuantStrategy:
    """评估期与回测期严格分离,避免前视偏差。

    Args:
        eval_summary: FactorEvaluator.evaluate_all() 返回的 summary
        top_factors:  取 |IR| 前 N 个因子做合成
    """
    if eval_summary is None or eval_summary.empty:
        raise ValueError("alpha101_walk_forward 要求传入非空 eval_summary")
    df = eval_summary.copy()
    df["abs_ir"] = df["ir"].abs()
    top = df.sort_values("abs_ir", ascending=False).head(top_factors)
    selector = MultiFactorSelector.from_summary(top, top_n=top_n * 3, min_abs_ir=0.0)
    return QuantStrategy(
        name=f"Alpha101WalkForward_top{top_factors}",
        selector=selector,
        timing=_make_timing(),
        portfolio=_make_portfolio(),
        top_n=top_n,
    )