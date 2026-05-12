"""core 核心模块（新架构）。

所有模块都已迁移到子目录，基类与实现分离。

设计原则:
  - 顶层只暴露轻量、无副作用的类型
  - 新架构优先使用 SignalStrategy
  - 旧架构兼容性保留但逐步淘汰
"""
from __future__ import annotations
import logging

# ----- 日志:为整个项目配置默认 handler -----
def _setup_default_logger():
    root = logging.getLogger()
    if any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        return
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    root.addHandler(h)
    root.setLevel(logging.INFO)

_setup_default_logger()

# ----- 新架构（推荐）类型 -----
from .strategy import SignalStrategy, CompositeStrategy, TargetPosition, StrategySignal, IStrategy
from .signals import IComposer, LayeredComposer, DirectComposer, WeightedComposer, VoteComposer
from .risk import RiskManager, IConstraint
from .execution import BacktestEngine, LiveExecutor

# 兼容旧架构（保留但推荐迁移到新架构）
try:
    from .strategy_legacy import QuantStrategy, Order, Position, Signal, SignalType
except ImportError:
    pass


# ----- 数据类型（新旧架构通用）-----
from .database import Database
from .data_checker import DataQualityChecker
from .data_validator import DataValidator, ValidationReport, validate_data
from .fee import FeeCalculator, TradeCost

__all__ = [
    # 数据库与工具
    "Database",
    "DataQualityChecker",
    "DataValidator",
    "ValidationReport",
    "validate_data",
    "FeeCalculator",
    "TradeCost",
    
    # 新架构（推荐）
    "IStrategy",
    "SignalStrategy",
    "CompositeStrategy",
    "TargetPosition",
    "StrategySignal",
    
    # 信号组合器
    "IComposer",
    "LayeredComposer",
    "DirectComposer",
    "WeightedComposer",
    "VoteComposer",
    
    # 风控
    "RiskManager",
    "IConstraint",
    
    # 执行引擎
    "BacktestEngine",
    "LiveExecutor",
    
    # 兼容旧架构（请迁移到新架构）
    "QuantStrategy",
    "Order",
    "Position",
    "Signal",
    "SignalType",
]
