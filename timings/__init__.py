"""timings 模块 —— 择时器(产生 BUY/SELL/HOLD 信号)集合。

公开接口:
  - ITimingGenerator: 抽象基类
  - TrendTiming:       趋势择时(MACD + 动量 + RSI)
  - VolatilityTiming:  波动率择时(仅产生 SELL/HOLD,不建仓)
  - CompositeTiming:   复合择时(投票 / 加权)
  - MarketRegimeTiming: 牛熊识别择时(根据市场状态动态调整策略)
"""
from .base import ITimingGenerator
from .trend import TrendTiming
from .volatility import VolatilityTiming
from .combo import CompositeTiming
from .market_regime import MarketRegimeTiming

__all__ = ["ITimingGenerator", "TrendTiming", "VolatilityTiming", "CompositeTiming", "MarketRegimeTiming"]