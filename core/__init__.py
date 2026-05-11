"""core 核心模块。

为避免 `import core` 触发因子库加载(101 个 alpha 公式),本 __init__
**不**导入 FactorRunner / 因子注册表。需要时显式 `from core.factor import FactorRunner`。

设计原则:
  - 顶层只暴露**轻量、无副作用**的类型(数据结构、抽象基类、工具类)
  - 重量级模块(因子计算、回测、评估)按需 from xxx import yyy
"""
from __future__ import annotations
import logging

# ----- 日志:为整个项目配置默认 handler;子模块用 logging.getLogger(__name__) 自动继承 -----
def _setup_default_logger():
    # 项目根 logger(名为 "core" / "factors" / "strategies" 等都会继承 root)
    root = logging.getLogger()
    if any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        return  # 已有 handler,避免重复
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    root.addHandler(h)
    root.setLevel(logging.INFO)

_setup_default_logger()

# ----- 轻量类型(纯数据,加载零开销)-----
from .database import Database
from .data_checker import DataQualityChecker
from .data_validator import DataValidator, ValidationReport, validate_data
from .fee import FeeCalculator, RiskManager, TradeCost
from .strategy import QuantStrategy, Order, Position, Signal, SignalType

__all__ = [
    "Database",
    "DataQualityChecker",
    "DataValidator",
    "ValidationReport",
    "validate_data",
    "FeeCalculator",
    "RiskManager",
    "TradeCost",
    "QuantStrategy",
    "Order",
    "Position",
    "Signal",
    "SignalType",
]

def _lazy_import_submodules():
    """按需导入子模块（避免启动时加载重量级模块）"""
    pass