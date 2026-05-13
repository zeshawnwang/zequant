"""执行模块。

负责回测和实盘执行。
"""
from .impl.backtest import BacktestEngine

__all__ = ["BacktestEngine"]
