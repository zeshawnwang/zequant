"""选股器模块。

目录结构:
    core/screening/
    ├── base/
    │   ├── selector.py     (基类 IStockSelector)
    │   └── selector_hub.py (SelectorHub 注册中心)
    └── impl/
        ├── factor_rank.py (FactorRankSelector)
        ├── multi_factor.py (MultiFactorSelector)
        ├── fundamental.py (FundamentalSelector)
        └── momentum_breakout.py (TrendBreakoutSelector, OversoldReboundSelector, ChipConcentrationSelector)
"""
from .base.selector import IStockSelector
from .base.selector_hub import SelectorHub, register_selector, _selector_hub
from .impl.factor_rank import FactorRankSelector
from .impl.multi_factor import MultiFactorSelector
from .impl.fundamental import FundamentalSelector
from .impl.momentum_breakout import (
    TrendBreakoutSelector,
    OversoldReboundSelector,
    ChipConcentrationSelector,
)


def list_selectors() -> list:
    """列出所有已注册的选股器。"""
    return _selector_hub.list_all()


def create_selector(name: str, **kwargs):
    """按名创建选股器。"""
    return _selector_hub.create(name, **kwargs)


__all__ = [
    'IStockSelector',
    'SelectorHub',
    'register_selector',
    'list_selectors',
    'create_selector',
    'FactorRankSelector',
    'MultiFactorSelector',
    'FundamentalSelector',
    'TrendBreakoutSelector',
    'OversoldReboundSelector',
    'ChipConcentrationSelector',
]
