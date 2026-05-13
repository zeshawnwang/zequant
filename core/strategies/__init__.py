"""策略模块。

目录结构：
  - base/: 策略基类 (SignalStrategy, CompositeStrategy)
  - impl/: 具体策略实现
"""
from .base.strategy import SignalStrategy, CompositeStrategy, TargetPosition, StrategySignal
from .impl.hub import register_strategy, StrategyHub, create, get_meta, list_all, list_by_category, categories, describe

__all__ = [
    "SignalStrategy",
    "CompositeStrategy",
    "TargetPosition",
    "StrategySignal",
    "register_strategy",
    "StrategyHub",
    "create",
    "get_meta",
    "list_all",
    "list_by_category",
    "categories",
    "describe",
]
