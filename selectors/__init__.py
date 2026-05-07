"""selectors module"""
from .factor_rank import FactorRankSelector, IStockSelector
from .combo import CompositeSelector

__all__ = ['FactorRankSelector', 'IStockSelector', 'CompositeSelector']
