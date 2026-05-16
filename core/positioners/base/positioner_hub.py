"""仓位确定器注册中心。

提供仓位确定器的按名注册和创建功能。

用法:
    hub = PositionSizerHub()

    @hub.register("my_sizer")
    class MySizer(IPositionSizer):
        ...

    sizer = hub.create("my_sizer", ...)
"""
from __future__ import annotations
from typing import Dict, List, Optional, Type, Callable, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class PositionSizerMeta:
    name: str
    factory: Callable
    description: str = ""


class PositionSizerHub:
    """仓位确定器注册中心。"""

    def __init__(self):
        self._registry: Dict[str, PositionSizerMeta] = {}

    def register(self, name: str, description: str = "") -> Callable:
        def decorator(cls: Type) -> Type:
            meta = PositionSizerMeta(name=name, factory=cls, description=description or cls.__doc__ or "")
            self._registry[name] = meta
            logger.info(f"注册仓位确定器: {name}")
            return cls
        return decorator

    def create(self, name: str, **kwargs) -> Any:
        if name not in self._registry:
            raise KeyError(f"仓位确定器 '{name}' 未注册。可用: {list(self._registry.keys())}")
        meta = self._registry[name]
        instance = meta.factory(**kwargs)
        return instance

    def list_all(self) -> List[str]:
        return list(self._registry.keys())


_position_sizer_hub = PositionSizerHub()

from ...signals.impl.position import FixedPositionSizer, TrendPositionSizer, VolatilityPositionSizer, RiskParityPositionSizer, CompositePositionSizer

_position_sizer_hub.register("fixed", description="固定仓位")(FixedPositionSizer)
_position_sizer_hub.register("trend", description="趋势仓位")(TrendPositionSizer)
_position_sizer_hub.register("volatility", description="波动率仓位")(VolatilityPositionSizer)
_position_sizer_hub.register("risk_parity", description="风险平价仓位")(RiskParityPositionSizer)
_position_sizer_hub.register("composite", description="复合仓位")(CompositePositionSizer)

register_position_sizer = _position_sizer_hub.register
