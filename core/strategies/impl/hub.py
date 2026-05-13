"""策略注册中心。

提供策略注册和创建功能。
"""
from __future__ import annotations
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class StrategyMeta:
    """策略元数据。"""
    name: str
    category: str
    factory: Callable
    description: str = ""
    timing_factors: List[str] = field(default_factory=list)
    requires_evaluation: bool = False
    eval_factor_filter: str = ""


class StrategyHub:
    """
    策略注册中心。

    使用方式：
        hub = StrategyHub()
        hub.register("策略名", category="分类", timing_factors=[...])(策略工厂函数)

        # 创建策略
        strategy = hub.create("策略名", top_n=50)

        # 列出所有策略
        all_strategies = hub.list_all()

        # 按分类列出
        cross_section = hub.list_by_category("cross_section")
    """

    def __init__(self):
        self._registry: Dict[str, StrategyMeta] = {}

    def register(
        self,
        name: str,
        category: str = "general",
        timing_factors: Optional[List[str]] = None,
        requires_evaluation: bool = False,
        eval_factor_filter: str = "",
        description: str = "",
    ) -> Callable:
        """
        装饰器：注册策略。

        使用示例：
            @hub.register("momentum", category="cross_section")
            def build_momentum(top_n=50):
                ...
        """
        def decorator(factory: Callable) -> Callable:
            meta = StrategyMeta(
                name=name,
                category=category,
                factory=factory,
                description=description,
                timing_factors=timing_factors or [],
                requires_evaluation=requires_evaluation,
                eval_factor_filter=eval_factor_filter,
            )
            self._registry[name] = meta
            logger.info(f"注册策略: {name} (category={category})")
            return factory

        return decorator

    def create(self, strategy_name: str, **kwargs) -> Any:
        """创建策略实例。"""
        if strategy_name not in self._registry:
            available = list(self._registry.keys())
            raise KeyError(f"策略 '{strategy_name}' 未注册。可用策略: {available}")

        meta = self._registry[strategy_name]
        return meta.factory(**kwargs)

    def get_meta(self, strategy_name: str) -> Optional[StrategyMeta]:
        """获取策略元数据。"""
        return self._registry.get(strategy_name)

    def list_all(self) -> List[str]:
        """列出所有策略名称。"""
        return list(self._registry.keys())

    def list_by_category(self, category: str) -> List[str]:
        """按分类列出策略。"""
        return [
            name for name, meta in self._registry.items()
            if meta.category == category
        ]

    @property
    def categories(self) -> List[str]:
        """获取所有分类。"""
        return list(set(meta.category for meta in self._registry.values()))

    def describe(self, strategy_name: str) -> str:
        """获取策略描述。"""
        if strategy_name not in self._registry:
            return f"策略 '{strategy_name}' 未找到"

        meta = self._registry[strategy_name]
        lines = [
            f"策略: {meta.name}",
            f"分类: {meta.category}",
            f"描述: {meta.description}",
        ]
        if meta.timing_factors:
            lines.append(f"择时因子: {', '.join(meta.timing_factors)}")
        return "\n".join(lines)


_default_hub = StrategyHub()

register_strategy = _default_hub.register
create = _default_hub.create
get_meta = _default_hub.get_meta
list_all = _default_hub.list_all
list_by_category = _default_hub.list_by_category
categories = _default_hub.categories
describe = _default_hub.describe
