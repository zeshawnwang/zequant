"""screening 模块 —— 选股器(stock screening / selection)集合。

本包原名 stock_selectors,为彻底消除与 Python 标准库 selectors 的命名歧义,
统一更名为 screening(语义更贴近"选股/筛选",且与 stdlib 零冲突)。

公开接口:
  - IStockSelector:       抽象基类
  - FactorRankSelector:   单因子排序选股
  - MultiFactorSelector:  多因子加权合成选股
  - FundamentalSelector:  基本面三因子选股(业绩增长+估值合理+盈利稳健)
"""
from .base import IStockSelector
from .factor_rank import FactorRankSelector
from .multi_factor import MultiFactorSelector
from .fundamental import FundamentalSelector

__all__ = ["IStockSelector", "FactorRankSelector", "MultiFactorSelector", "FundamentalSelector"]