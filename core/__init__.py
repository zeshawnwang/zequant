"""core 核心模块（新架构）。

所有模块都已迁移到子目录，基类与实现分离。

设计原则:
  - 顶层只暴露轻量、无副作用的类型
  - 新架构优先使用 SignalStrategy
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
from .strategies.base.strategy import SignalStrategy, CompositeStrategy, TargetPosition, StrategySignal, IStrategy
from .signals import LayeredComposer, DirectComposer
from .execution import BacktestEngine

# ----- 数据类型（新旧架构通用）-----
from .database import Database
from .datasource.checker import DataQualityChecker
from .datasource.validator import DataValidator, ValidationReport, validate_data
from .risk import FeeCalculator, TradeCost, RiskManager

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
    "LayeredComposer",
    "DirectComposer",

    # 风控
    "RiskManager",

    # 执行引擎
    "BacktestEngine",
]
