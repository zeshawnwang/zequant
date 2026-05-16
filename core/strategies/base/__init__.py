"""策略基类和工厂。"""
from .strategy import SignalStrategy, CompositeStrategy, TargetPosition, StrategySignal, IStrategy, Order, Position
from .factory import StrategyFactory

__all__ = [
    "SignalStrategy",
    "CompositeStrategy",
    "TargetPosition",
    "StrategySignal",
    "IStrategy",
    "Order",
    "Position",
    "StrategyFactory",
]
