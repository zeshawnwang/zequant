"""选股器注册中心。

提供选股器的按名注册和创建功能。

用法:
    hub = SelectorHub()

    @hub.register("factor_rank")
    class FactorRankSelector(IStockSelector):
        ...

    selector = hub.create("factor_rank", factor_name="momentum_20")
"""
from __future__ import annotations
from typing import Dict, List, Optional, Type, Callable, Any
from dataclasses import dataclass, field
import logging

from .selector import IStockSelector

logger = logging.getLogger(__name__)


@dataclass
class SelectorMeta:
    """选股器元数据。"""
    name: str
    factory: Callable
    description: str = ""
    category: str = ""


class SelectorHub:
    """选股器注册中心。"""

    def __init__(self):
        self._registry: Dict[str, SelectorMeta] = {}

    def register(
        self,
        name: str,
        category: str = "",
        description: str = "",
    ) -> Callable:
        """装饰器：注册选股器类。"""
        def decorator(cls: Type) -> Type:
            meta = SelectorMeta(
                name=name,
                factory=cls,
                description=description or cls.__doc__ or "",
                category=category,
            )
            self._registry[name] = meta
            logger.info(f"注册选股器: {name} (category={category})")
            return cls
        return decorator

    def create(self, name: str, **kwargs) -> IStockSelector:
        """创建选股器实例。"""
        if name not in self._registry:
            raise KeyError(f"选股器 '{name}' 未注册。可用: {list(self._registry.keys())}")
        meta = self._registry[name]
        instance = meta.factory(**kwargs)
        if not isinstance(instance, IStockSelector):
            raise TypeError(f"选股器 '{name}' 必须实现 IStockSelector 接口")
        return instance

    def list_all(self) -> List[str]:
        return list(self._registry.keys())

    def list_by_category(self, category: str) -> List[str]:
        return [n for n, m in self._registry.items() if m.category == category]

    @property
    def categories(self) -> List[str]:
        return list(set(m.category for m in self._registry.values()))


_selector_hub = SelectorHub()

# 注册内置选股器
from ..impl.factor_rank import FactorRankSelector
from ..impl.multi_factor import MultiFactorSelector
from ..impl.fundamental import FundamentalSelector
from ..impl.momentum_breakout import TrendBreakoutSelector, OversoldReboundSelector, ChipConcentrationSelector

_selector_hub.register("factor_rank", category="single_factor", description="单因子排名选股")(FactorRankSelector)
_selector_hub.register("multi_factor", category="multi_factor", description="多因子合成选股")(MultiFactorSelector)
_selector_hub.register("fundamental", category="fundamental", description="基本面选股")(FundamentalSelector)
_selector_hub.register("trend_breakout", category="technical", description="趋势突破选股")(TrendBreakoutSelector)
_selector_hub.register("oversold_rebound", category="technical", description="超跌反弹选股")(OversoldReboundSelector)
_selector_hub.register("chip_concentration", category="technical", description="筹码集中选股")(ChipConcentrationSelector)

register_selector = _selector_hub.register
