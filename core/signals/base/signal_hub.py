"""信号组合器注册中心。

提供组合器的按名注册和创建功能。
"""
from __future__ import annotations
from typing import Dict, List, Optional, Type, Callable, Any
from dataclasses import dataclass, field
import logging

from .composer import IComposer

logger = logging.getLogger(__name__)


@dataclass
class ComposerMeta:
    name: str
    factory: Callable
    description: str = ""


class ComposerHub:
    """信号组合器注册中心。"""

    def __init__(self):
        self._registry: Dict[str, ComposerMeta] = {}

    def register(self, name: str, description: str = "") -> Callable:
        def decorator(cls: Type) -> Type:
            meta = ComposerMeta(name=name, factory=cls, description=description or cls.__doc__ or "")
            self._registry[name] = meta
            logger.info(f"注册组合器: {name}")
            return cls
        return decorator

    def create(self, name: str, **kwargs) -> IComposer:
        if name not in self._registry:
            raise KeyError(f"组合器 '{name}' 未注册。可用: {list(self._registry.keys())}")
        meta = self._registry[name]
        instance = meta.factory(**kwargs)
        if not isinstance(instance, IComposer):
            raise TypeError(f"组合器 '{name}' 必须实现 IComposer 接口")
        return instance

    def list_all(self) -> List[str]:
        return list(self._registry.keys())


_composer_hub = ComposerHub()

from .composer import LayeredComposer, DirectComposer, WeightedComposer, VoteComposer

_composer_hub.register("layered", description="分层组合(先择时再分配)")(LayeredComposer)
_composer_hub.register("direct", description="直接组合(选股×择时)")(DirectComposer)
_composer_hub.register("weighted", description="加权组合")(WeightedComposer)
_composer_hub.register("vote", description="投票组合")(VoteComposer)

register_composer = _composer_hub.register
