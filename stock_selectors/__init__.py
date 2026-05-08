"""stock_selectors module

注意:本包原名 `selectors`,因与 Python 标准库模块 `selectors` 重名,
在某些 import 顺序下会引发循环 import,故重命名。
"""
from .factor_rank import FactorRankSelector, IStockSelector

__all__ = ['FactorRankSelector', 'IStockSelector']