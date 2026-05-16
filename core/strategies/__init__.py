"""策略模块。

目录结构：
  - base/: 策略基类 (SignalStrategy, CompositeStrategy, StrategyFactory)
  - impl/: 具体策略实现和注册中心
"""
from .base.strategy import SignalStrategy, CompositeStrategy, TargetPosition, StrategySignal
from .base.factory import StrategyFactory
from .impl.hub import register_strategy, StrategyHub, create, get_meta, list_all, list_by_category, categories, describe

__all__ = [
    "SignalStrategy",
    "CompositeStrategy",
    "TargetPosition",
    "StrategySignal",
    "StrategyFactory",
    "register_strategy",
    "StrategyHub",
    "create",
    "get_meta",
    "list_all",
    "list_by_category",
    "categories",
    "describe",
]
