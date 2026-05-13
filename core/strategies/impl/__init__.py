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
