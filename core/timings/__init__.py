"""择时器模块。

择时器的作用：判断市场环境，输出仓位系数 (0~1)，决定当前的风险暴露水平。

目录结构:
    core/timings/
    ├── base/
    │   ├── timing.py       (基类 ITimingGenerator)
    │   └── timing_hub.py   (TimingHub 注册中心)
    └── impl/
        ├── trend.py        (TrendTiming - 趋势择时)
        ├── volatility.py   (VolatilityTiming - 波动率择时)
        ├── trend_volatility.py (TrendVolatilityTiming - 趋势+波动率复合择时)
        ├── combo.py (CompositeTiming - 复合择时器)
        └── market_regime.py (MarketRegimeTiming - 牛熊识别择时器)
"""
from .base.timing import ITimingGenerator
from .base.timing_hub import TimingHub, register_timing, _timing_hub
from .impl.trend import TrendTiming
from .impl.volatility import VolatilityTiming
from .impl.trend_volatility import TrendVolatilityTiming
from .impl.combo import CompositeTiming
from .impl.market_regime import MarketRegimeTiming


def list_timings() -> list:
    """列出所有已注册的择时器。"""
    return _timing_hub.list_all()


def create_timing(name: str, **kwargs):
    """按名创建择时器。"""
    return _timing_hub.create(name, **kwargs)


__all__ = [
    'ITimingGenerator',
    'TimingHub',
    'register_timing',
    'list_timings',
    'create_timing',
    'TrendTiming',
    'VolatilityTiming',
    'TrendVolatilityTiming',
    'CompositeTiming',
    'MarketRegimeTiming',
]
