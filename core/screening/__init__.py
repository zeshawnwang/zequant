"""选股器模块。

目录结构:
    core/screening/
    ├── base/
    │   └── selector.py     (基类 IStockSelector)
    └── impl/
        ├── factor_rank.py (FactorRankSelector)
        ├── multi_factor.py (MultiFactorSelector)
        ├── fundamental.py (FundamentalSelector)
        └── momentum_breakout.py (TrendBreakoutSelector, OversoldReboundSelector, ChipConcentrationSelector)
"""
from .base.selector import IStockSelector
from .impl.factor_rank import FactorRankSelector
from .impl.multi_factor import MultiFactorSelector
from .impl.fundamental import FundamentalSelector
from .impl.momentum_breakout import (
    TrendBreakoutSelector,
    OversoldReboundSelector,
    ChipConcentrationSelector,
)

__all__ = [
    'IStockSelector',
    'FactorRankSelector',
    'MultiFactorSelector',
    'FundamentalSelector',
    'TrendBreakoutSelector',
    'OversoldReboundSelector',
    'ChipConcentrationSelector',
]
