"""策略模块（新架构）。

该模块包含策略相关的定义：
  - IStrategy: 策略接口
  - SignalStrategy: 信号流驱动策略基类
  - CompositeStrategy: 组合策略
  - TargetPosition, StrategySignal: 数据类型
"""
from __future__ import annotations

# 新架构导出
from .base import (
    IStrategy,
    SignalStrategy,
    CompositeStrategy,
    TargetPosition,
    StrategySignal,
)

# 兼容旧架构（如果需要）
try:
    from ..strategy_legacy import QuantStrategy, Order, Position, Signal, SignalType
except ImportError:
    pass

__all__ = [
    "IStrategy",
    "SignalStrategy",
    "CompositeStrategy",
    "TargetPosition",
    "StrategySignal",
    "QuantStrategy",  # 兼容
    "Order",
    "Position",
    "Signal",
    "SignalType",
]
