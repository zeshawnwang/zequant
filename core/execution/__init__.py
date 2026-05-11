"""执行层模块

职责：
- BacktestEngine：事件驱动回测引擎，适配SignalStrategy架构
- LiveExecutor：实盘执行器
- OrderRouter：订单路由和执行

架构设计：
- 执行器与策略解耦，支持多种策略类型
- 支持T+1、止损止盈、Universe过滤
- 支持滑点、费用计算
"""
from .backtest import BacktestEngine, BacktestReport, Trade, FinalPosition
from .executor import LiveExecutor, OrderRouter

__all__ = [
    'BacktestEngine',
    'BacktestReport',
    'Trade',
    'FinalPosition',
    'LiveExecutor',
    'OrderRouter',
]
