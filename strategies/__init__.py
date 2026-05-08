"""策略包入口。

导入本包(`import strategies`)即触发各策略模块中的 @register_strategy
装饰器执行,把策略注册到 [`StrategyHub`](../core/strategy_hub.py:1)。

新增策略只需在本目录加 `xxx_strategy.py`,并在下方追加一行 `from . import xxx_strategy`。
"""
# 触发注册副作用 ------------------------------------------------------------
from . import momentum_strategy   # noqa: F401  动量 / 低波
from . import alpha101_strategy    # noqa: F401  Alpha101 多因子族(3 种)

__all__ = [
    "momentum_strategy",
    "alpha101_strategy",
]