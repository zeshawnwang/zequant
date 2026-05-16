"""择时器注册中心。

提供择时器的按名注册和创建功能。

用法:
    hub = TimingHub()

    @hub.register("trend")
    class TrendTiming(ITimingGenerator):
        ...

    timing = hub.create("trend", ...)
"""
from __future__ import annotations
from typing import Dict, List, Optional, Type, Callable, Any
from dataclasses import dataclass, field
import logging

from .timing import ITimingGenerator

logger = logging.getLogger(__name__)


@dataclass
class TimingMeta:
    """择时器元数据。"""
    name: str
    factory: Callable
    description: str = ""
    category: str = ""


class TimingHub:
    """择时器注册中心。"""

    def __init__(self):
        self._registry: Dict[str, TimingMeta] = {}

    def register(
        self,
        name: str,
        category: str = "",
        description: str = "",
    ) -> Callable:
        """装饰器：注册择时器类。"""
        def decorator(cls: Type) -> Type:
            meta = TimingMeta(
                name=name,
                factory=cls,
                description=description or cls.__doc__ or "",
                category=category,
            )
            self._registry[name] = meta
            logger.info(f"注册择时器: {name} (category={category})")
            return cls
        return decorator

    def create(self, name: str, **kwargs) -> ITimingGenerator:
        """创建择时器实例。"""
        if name not in self._registry:
            raise KeyError(f"择时器 '{name}' 未注册。可用: {list(self._registry.keys())}")
        meta = self._registry[name]
        instance = meta.factory(**kwargs)
        if not isinstance(instance, ITimingGenerator):
            raise TypeError(f"择时器 '{name}' 必须实现 ITimingGenerator 接口")
        return instance

    def list_all(self) -> List[str]:
        return list(self._registry.keys())

    def list_by_category(self, category: str) -> List[str]:
        return [n for n, m in self._registry.items() if m.category == category]

    @property
    def categories(self) -> List[str]:
        return list(set(m.category for m in self._registry.values()))


_timing_hub = TimingHub()

# 注册内置择时器
from ..impl.trend import TrendTiming
from ..impl.volatility import VolatilityTiming
from ..impl.trend_volatility import TrendVolatilityTiming
from ..impl.combo import CompositeTiming
from ..impl.market_regime import MarketRegimeTiming

_timing_hub.register("trend", category="trend", description="趋势择时(MACD/动量/RSI)")(TrendTiming)
_timing_hub.register("volatility", category="volatility", description="波动率择时")(VolatilityTiming)
_timing_hub.register("trend_volatility", category="composite", description="趋势+波动率复合择时")(TrendVolatilityTiming)
_timing_hub.register("composite", category="composite", description="复合择时(投票/加权)")(CompositeTiming)
_timing_hub.register("market_regime", category="market", description="牛熊识别择时")(MarketRegimeTiming)

register_timing = _timing_hub.register
