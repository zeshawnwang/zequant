"""策略注册中心（新架构）。

设计目标：
  - 统一管理所有策略（新旧架构都支持）
  - 提供策略工厂功能
  - 记录策略元信息

策略分类：
  - cross_section: 截面策略
  - multi_factor: 多因子策略
  - timing: 择时策略
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any


@dataclass
class StrategyMeta:
    """策略元信息。"""
    name: str
    category: str
    factory: Callable[..., Any]
    description: str = ""
    requires_evaluation: bool = False
    timing_factors: List[str] = field(default_factory=list)
    eval_factor_filter: Optional[str] = None


class StrategyHub:
    """策略注册中心，支持多实例。"""

    def __init__(self):
        self._registry: Dict[str, StrategyMeta] = {}

    def register(
        self,
        name: str,
        category: str = "default",
        description: str = "",
        requires_evaluation: bool = False,
        timing_factors: Optional[List[str]] = None,
        eval_factor_filter: Optional[str] = None,
    ):
        """装饰器: 把工厂函数注册到 StrategyHub。

        Args:
            name: 策略唯一名称
            category: 分类标签 (cross_section/multi_factor/timing/...)
            description: 描述
            requires_evaluation: 是否依赖因子评估结果
            timing_factors: 择时器依赖的技术因子列表
            eval_factor_filter: 评估驱动策略的因子前缀过滤
        """
        def deco(func):
            self._registry[name] = StrategyMeta(
                name=name,
                category=category,
                factory=func,
                description=description or (func.__doc__ or "").strip(),
                requires_evaluation=requires_evaluation,
                timing_factors=list(timing_factors or []),
                eval_factor_filter=eval_factor_filter,
            )
            return func
        return deco

    def create(self, name: str, **kwargs):
        """按名称构造策略实例。kwargs 透传给工厂函数。"""
        if name not in self._registry:
            raise KeyError(
                f"策略未注册: {name}\n"
                f"可用策略: {self.list_all()}"
            )
        return self._registry[name].factory(**kwargs)

    def get_meta(self, name: str) -> StrategyMeta:
        """获取策略元信息。"""
        return self._registry[name]

    def list_all(self) -> List[str]:
        """列出所有策略。"""
        return sorted(self._registry.keys())

    def list_by_category(self, category: str) -> List[str]:
        """按分类列出策略。"""
        return sorted(
            name for name, meta in self._registry.items()
            if meta.category == category
        )

    def categories(self) -> List[str]:
        """列出所有策略分类。"""
        return sorted(set(meta.category for meta in self._registry.values()))

    def describe(self, name: str) -> str:
        """描述一个策略。"""
        meta = self.get_meta(name)
        tag = " [需评估]" if meta.requires_evaluation else ""
        return f"[{meta.category}]{tag} {meta.name}: {meta.description}"


# 默认全局实例
_hub = StrategyHub()

# 便捷装饰器（绑定到默认实例）
register_strategy = _hub.register

# 便捷函数（绑定到默认实例）
create = _hub.create
get_meta = _hub.get_meta
list_all = _hub.list_all
list_by_category = _hub.list_by_category
categories = _hub.categories
describe = _hub.describe
