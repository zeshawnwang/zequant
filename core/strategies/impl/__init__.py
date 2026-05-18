"""策略实现。

提供预定义的策略构建函数。
"""
from __future__ import annotations
from typing import Dict, List, Optional, Any
import pandas as pd

from ..base.strategy import SignalStrategy
from .hub import register_strategy, StrategyHub
from ...screening import FactorRankSelector, MultiFactorSelector
from ...signals import TrendPositionSizer, VolatilityPositionSizer
from ...signals import LayeredComposer, DirectComposer, MaxSingleWeightConstraint, ReserveCashConstraint


def _get_strategy_config(kwargs: dict, strategy_name: str) -> dict:
    """从 kwargs 中提取策略配置。"""
    return kwargs.get("strategy_config") or {}


_strategy_hub = StrategyHub()
register_strategy = _strategy_hub.register
create = _strategy_hub.create
get_meta = _strategy_hub.get_meta
list_all = _strategy_hub.list_all
list_by_category = _strategy_hub.list_by_category
categories = _strategy_hub.categories
describe = _strategy_hub.describe


# V1~V4 落地策略
from .v1_ga_rp import build_v1_ga_rp
from .v4_mf_rp import build_v4_mf_rp
from .v4_mf_tv_rp import build_v4_mf_tv_rp

# V5~V6 新发掘策略
from .mf_vol_d10_rp import build_mf_vol_d10_rp
from .mf_trend_d5_rp import build_mf_trend_d5_rp
from .mf_d10_rp import build_mf_d10_rp

# V8 Chip 筹码集中策略
from .chip_rp import build_chip_rp
from .chip_vol_rp import build_chip_vol_rp

# V10 新发掘策略
from .mf60_chip40_combo import build_mf60_chip40_combo
from .mf50_chip50_combo import build_mf50_chip50_combo
from .chip_covrp import build_chip_covrp
from .chip_equal_d3 import build_chip_equal_d3
from .osr_d10 import build_osr_d10
from .osr_vol_eq_d10 import build_osr_vol_eq_d10

# X5 GA优化策略
from .ga_d10 import build_ga_d10
from .ga_d5 import build_ga_d5

# V10 组合策略
from .mf50_chipcovrp50_combo import build_mf50_chipcovrp50_combo

# X7 GA+CovRP组合策略
from .ga_covrp_combo import build_ga_covrp_combo

# X8 C01 Layered策略
from .c01_layered_d5 import build_c01_layered_d5

# 2026-05-17 修正Pipeline验证策略
from .mf_d10_opt_0517 import build_mf_d10_opt_0517

# 2026-05-18 紧急事件处理策略
from .mf_d10_emergency_v1 import build_mf_d10_emergency_v1

# ── 策略注册 ──
register_strategy("mf_d10_opt_0517", category="multi_factor",
    description="MF+双周+RP 2026-05-17修正版，真实费率+Universe过滤后三区间通过")(
    build_mf_d10_opt_0517)

register_strategy("mf_d10_emergency_v1", category="multi_factor",
    description="MF_D10+个股止损10%+大盘熔断4%/3天恢复 紧急处理版本，综合分65.0")(
    build_mf_d10_emergency_v1)
