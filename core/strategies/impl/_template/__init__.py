"""{StrategyName} — 策略组装入口。

从 build.py 导入构建函数，供 strategies.hub 注册使用。
"""
from .build import build_{strategy_name}

__all__ = ["build_{strategy_name}"]
