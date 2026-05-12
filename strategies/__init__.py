"""新架构策略库（全 SignalStrategy 实现）。

所有策略都使用新的信号流架构实现。

导入路径：
- 选股器: from core.screening import FactorRankSelector, MultiFactorSelector, etc.
- 择时器: from core.timings.impl.trend import TrendTiming (适配器)
- 仓位管理: from core.signals.position import TrendPositionSizer, VolatilityPositionSizer
- 组合器: from core.signals import LayeredComposer, DirectComposer
- 风控: from core.risk import RiskManager
"""
from __future__ import annotations
from typing import Dict, List, Optional, Any
import pandas as pd

try:
    from core.strategy.hub import register_strategy, StrategyHub
except ImportError:
    from core.strategy_hub_legacy import register_strategy, StrategyHub

from core.strategy import SignalStrategy
from core.screening import (
    FactorRankSelector,
    MultiFactorSelector,
)
from core.signals import (
    LayeredComposer,
    DirectComposer,
    IComposer,
    MaxSingleWeightConstraint,
    MaxTotalPositionConstraint,
    ReserveCashConstraint,
)
from core.signals.position import TrendPositionSizer, VolatilityPositionSizer
from core.risk import RiskManager, StopLoss, TakeProfit


def _get_strategy_config(kwargs: dict, strategy_name: str) -> dict:
    """从 kwargs 中提取策略配置。"""
    return kwargs.get("strategy_config") or {}


def _build_momentum_base(top_n: int, factor_name: str, ascending: bool = False, **kwargs) -> SignalStrategy:
    """构建动量策略基础（内部共用）。"""
    cfg = _get_strategy_config(kwargs, "momentum")
    
    selector = FactorRankSelector(
        factor_name=factor_name,
        ascending=ascending,
        top_n=top_n * 3,
    )
    
    position_sizer = TrendPositionSizer(
        bullish_threshold=cfg.get("position_sizer", {}).get("bullish_threshold", 0.6),
        bearish_threshold=cfg.get("position_sizer", {}).get("bearish_threshold", 0.4),
        max_position=cfg.get("position_sizer", {}).get("max_position", 1.0),
        min_position=cfg.get("position_sizer", {}).get("min_position", 0.0),
    )
    
    composer = LayeredComposer(
        top_n=top_n,
        constraints=[
            MaxSingleWeightConstraint(max_weight=0.1),
            ReserveCashConstraint(reserve_ratio=0.1),
        ],
    )
    
    risk_manager = RiskManager(
        constraints=[
            MaxTotalPositionConstraint(max_position=0.9),
        ],
        stop_loss=StopLoss(method="fixed", threshold=0.10),
        max_total_exposure=0.9,
        max_single_position=0.1,
    )
    
    return SignalStrategy(
        name="MomentumStrategy",
        selector=selector,
        position_sizer=position_sizer,
        composer=composer,
        risk_manager=risk_manager,
        top_n=top_n,
    )


@register_strategy(
    "momentum_top50",
    category="cross_section",
    timing_factors=["momentum_5", "momentum_20", "macd", "rsi_14"],
    description="动量策略：按 momentum_20 选前 50，趋势择时，等权重",
)
def build_momentum_strategy_v2(top_n: int = None, **kwargs) -> SignalStrategy:
    """新架构动量策略：按 momentum_20 选前 N，趋势择时。"""
    cfg = _get_strategy_config(kwargs, "momentum_top50")
    top_n = top_n if top_n is not None else cfg.get("top_n", 50)
    
    return _build_momentum_base(
        top_n=top_n,
        factor_name=cfg.get("selector", {}).get("factor_name", "momentum_20"),
        ascending=False,
        strategy_name="MomentumTop50_v2",
        **kwargs,
    )


@register_strategy(
    "low_vol_top50",
    category="cross_section",
    timing_factors=["momentum_5", "momentum_20", "macd", "rsi_14", "volatility_20"],
    description="低波动率策略：选 volatility_20 最低的 50 只，趋势择时，风险平价",
)
def build_low_vol_strategy_v2(top_n: int = None, **kwargs) -> SignalStrategy:
    """新架构低波动策略：选波动率最低的 N 只，波动率择时。"""
    cfg = _get_strategy_config(kwargs, "low_vol_top50")
    top_n = top_n if top_n is not None else cfg.get("top_n", 50)
    
    selector = FactorRankSelector(
        factor_name=cfg.get("selector", {}).get("factor_name", "volatility_20"),
        ascending=True,
        top_n=top_n * 3,
    )
    
    position_sizer = VolatilityPositionSizer(
        volatility_factor=cfg.get("position_sizer", {}).get("volatility_factor", "volatility_20"),
        target_volatility=cfg.get("position_sizer", {}).get("target_volatility", 0.2),
        max_position=cfg.get("position_sizer", {}).get("max_position", 0.8),
    )
    
    composer = DirectComposer(
        constraints=[
            MaxSingleWeightConstraint(max_weight=0.15),
            ReserveCashConstraint(reserve_ratio=0.1),
        ],
    )
    
    risk_manager = RiskManager(
        constraints=[
            MaxTotalPositionConstraint(max_position=0.9),
        ],
        stop_loss=StopLoss(method="fixed", threshold=0.10),
        max_total_exposure=0.9,
        max_single_position=0.15,
    )
    
    return SignalStrategy(
        name="LowVolatility_v2",
        selector=selector,
        position_sizer=position_sizer,
        composer=composer,
        risk_manager=risk_manager,
        top_n=top_n,
    )


def _build_multi_factor_base(
    weights: Dict[str, float],
    top_n: int = 30,
    strategy_name: str = "MultiFactor",
    **kwargs
) -> SignalStrategy:
    """多因子策略基础构建。"""
    cfg = _get_strategy_config(kwargs, strategy_name)
    
    selector = MultiFactorSelector(
        weights,
        top_n=top_n * 3,
    )
    
    position_sizer = TrendPositionSizer(
        bullish_threshold=cfg.get("position_sizer", {}).get("bullish_threshold", 0.6),
        bearish_threshold=cfg.get("position_sizer", {}).get("bearish_threshold", 0.4),
        max_position=cfg.get("position_sizer", {}).get("max_position", 1.0),
        min_position=cfg.get("position_sizer", {}).get("min_position", 0.0),
    )
    
    composer = LayeredComposer(
        top_n=top_n,
        constraints=[
            MaxSingleWeightConstraint(max_weight=0.1),
            ReserveCashConstraint(reserve_ratio=0.1),
        ],
    )
    
    risk_manager = RiskManager(
        constraints=[
            MaxTotalPositionConstraint(max_position=0.9),
        ],
        stop_loss=StopLoss(method="fixed", threshold=0.10),
        max_total_exposure=0.9,
        max_single_position=0.1,
    )
    
    return SignalStrategy(
        name=strategy_name,
        selector=selector,
        position_sizer=position_sizer,
        composer=composer,
        risk_manager=risk_manager,
        top_n=top_n,
    )


@register_strategy(
    "alpha101_manual",
    category="multi_factor",
    timing_factors=["macd", "macd_signal", "momentum_5", "momentum_20", "rsi_14", "volatility_20"],
    description="Alpha101 多因子合成：手工权重（动量+低波动率+反转）",
)
def build_alpha101_manual_v2(
    top_n: int = None,
    weights: Optional[Dict[str, float]] = None,
    **kwargs
) -> SignalStrategy:
    """新架构 Alpha101 手工权重策略。"""
    cfg = _get_strategy_config(kwargs, "alpha101_manual")
    top_n = top_n if top_n is not None else cfg.get("top_n", 50)
    
    if weights is None:
        weights = cfg.get("weights")
    if weights is None:
        weights = {
            "momentum_20": 0.6,
            "volatility_20": -0.4,
            "a3": -0.5,
            "a101": 0.5,
        }
    
    return _build_multi_factor_base(
        weights=weights,
        top_n=top_n,
        strategy_name="Alpha101Manual_v2",
        **kwargs,
    )


@register_strategy(
    "alpha101_from_registry",
    category="multi_factor",
    timing_factors=["macd", "macd_signal", "momentum_5", "momentum_20", "rsi_14", "volatility_20"],
    description="Alpha101 多因子：从 factor_registry 自动拉取 |IR| ≥ 0.2 的因子",
)
def build_alpha101_from_registry_v2(
    db=None,
    top_n: int = None,
    min_abs_ir: float = None,
    **kwargs
) -> SignalStrategy:
    """新架构 Alpha101 从注册表构建策略。"""
    cfg = _get_strategy_config(kwargs, "alpha101_from_registry")
    top_n = top_n if top_n is not None else cfg.get("top_n", 50)
    if min_abs_ir is None:
        min_abs_ir = cfg.get("min_abs_ir", 0.2)
    
    if db is None:
        from core.database import Database
        db = Database()
    
    selector = MultiFactorSelector.from_registry(
        db,
        top_n=top_n * 3,
        min_abs_ir=min_abs_ir,
    )
    
    # 提取权重用于策略构建
    weights = selector.weights if hasattr(selector, "weights") else {}
    
    return _build_multi_factor_base(
        weights=weights,
        top_n=top_n,
        strategy_name="Alpha101FromRegistry_v2",
        **kwargs,
    )


@register_strategy(
    "alpha101_walk_forward",
    category="multi_factor",
    requires_evaluation=True,
    timing_factors=["macd", "macd_signal", "momentum_5", "momentum_20", "rsi_14", "volatility_20"],
    eval_factor_filter="alpha",
    description="Alpha101 walk-forward：评估期算 IR，样本外用 |IR| 前 N 个因子",
)
def build_alpha101_walk_forward_v2(
    db=None,
    eval_summary: pd.DataFrame = None,
    top_n: int = None,
    top_factors: int = None,
    **kwargs
) -> SignalStrategy:
    """新架构 Alpha101 Walk-Forward 策略。"""
    if eval_summary is None or eval_summary.empty:
        raise ValueError("alpha101_walk_forward_v2 要求传入非空 eval_summary")
    
    cfg = _get_strategy_config(kwargs, "alpha101_walk_forward")
    top_n = top_n if top_n is not None else cfg.get("top_n", 30)
    if top_factors is None:
        top_factors = cfg.get("top_factors", 8)
    
    df = eval_summary.copy()
    df["abs_ir"] = df["ir"].abs()
    top = df.sort_values("abs_ir", ascending=False).head(top_factors)
    selector = MultiFactorSelector.from_summary(top, top_n=top_n * 3, min_abs_ir=0.0)
    
    weights = top.set_index("factor_name")["ir"].to_dict()
    
    return _build_multi_factor_base(
        weights=weights,
        top_n=top_n,
        strategy_name=f"Alpha101WalkForward_v2_top{top_factors}",
        **kwargs,
    )


_strategy_hub = StrategyHub()
register_strategy = _strategy_hub.register
create = _strategy_hub.create
get_meta = _strategy_hub.get_meta
list_all = _strategy_hub.list_all
list_by_category = _strategy_hub.list_by_category
categories = _strategy_hub.categories
describe = _strategy_hub.describe
