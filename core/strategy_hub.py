"""StrategyHub —— 策略注册中心。

设计目标
--------
"用什么选股器 / 用哪些因子 / 择时器如何接"是策略开发者的事,
回测脚本只按名字拉策略、不触碰内部装配。

策略分两类
----------
1) 静态参数化策略(requires_evaluation=False,默认):
   - 权重/因子在代码里写死或由调用方传入
   - factory 签名:`def factory(db=None, top_n=50, **kwargs) -> QuantStrategy`
   - 调用:`StrategyHub.create("momentum_top50", db=db, top_n=30)`

2) 评估驱动策略(requires_evaluation=True):
   - 需要先跑因子评估(IC/IR)才能确定权重 —— 典型:Alpha101 walk-forward
   - factory 签名:`def factory(db, eval_summary, top_n=50, **kwargs) -> QuantStrategy`
     * `eval_summary`: FactorEvaluator.evaluate_all() 的返回 DataFrame
   - 调用:`StrategyHub.create(name, db=db, eval_summary=summary, top_n=30)`
   - 元信息上声明 timing_factors 帮助调用方一并拉取择时所需数据

用法示例
--------
    from core.strategy_hub import register_strategy, StrategyHub

    @register_strategy("momentum_top50", category="cross_section",
                       description="按 momentum_20 选高动量股")
    def build_momentum(top_n=50, **_):
        from core.strategy import QuantStrategy
        ...

    @register_strategy("alpha101_walk_forward",
                       category="multi_factor",
                       requires_evaluation=True,
                       timing_factors=["macd", "momentum_20", "rsi_14"],
                       description="Alpha101 walk-forward 多因子合成")
    def build_alpha101_wf(db, eval_summary, top_n=30, top_factors=8, **_):
        from screening.multi_factor import MultiFactorSelector
        top = eval_summary.nlargest(top_factors, "ir", keep="first")
        sel = MultiFactorSelector.from_summary(top, top_n=top_n)
        ...
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


@dataclass
class StrategyMeta:
    """策略元信息:在注册时被装饰器填充,调用端据此决定如何准备上下文。"""
    name: str
    category: str
    factory: Callable[..., object]
    description: str = ""
    requires_evaluation: bool = False
    # 回测时除选股因子外还必须加载的技术因子(例:择时器依赖)
    timing_factors: List[str] = field(default_factory=list)
    # 评估驱动策略:对评估候选因子做前缀过滤(如 "alpha" → 仅评估 a* 因子)
    # None 表示评估库内所有因子;调用端在准备 eval_summary 时按此字段筛选
    eval_factor_filter: Optional[str] = None


class StrategyHub:
    """策略注册中心。"""

    _registry: Dict[str, StrategyMeta] = {}

    # ---- 注册 -----------------------------------------------------------

    @classmethod
    def register(
        cls,
        name: str,
        category: str = "default",
        description: str = "",
        requires_evaluation: bool = False,
        timing_factors: Optional[List[str]] = None,
        eval_factor_filter: Optional[str] = None,
    ):
        """装饰器:把 factory 注册到 StrategyHub。

        Args:
            name:                 策略名(唯一)
            category:             分类标签(cross_section / multi_factor / ...)
            description:          一句话描述,也可用 factory 的 docstring
            requires_evaluation:  是否依赖因子评估结果
            timing_factors:       择时器依赖的技术因子清单(额外加载)
            eval_factor_filter:   评估候选因子前缀(如 "alpha"),仅 requires_evaluation 时生效
        """
        def deco(func):
            cls._registry[name] = StrategyMeta(
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

    # ---- 创建 -----------------------------------------------------------

    @classmethod
    def create(cls, name: str, **kwargs):
        """按名字构造策略实例。kwargs 透传给 factory。"""
        if name not in cls._registry:
            raise KeyError(
                f"strategy not registered: {name}\n"
                f"available: {cls.list_all()}"
            )
        return cls._registry[name].factory(**kwargs)

    # ---- 查询 -----------------------------------------------------------

    @classmethod
    def get_meta(cls, name: str) -> StrategyMeta:
        return cls._registry[name]

    @classmethod
    def list_all(cls) -> List[str]:
        return sorted(cls._registry.keys())

    @classmethod
    def list_by_category(cls, category: str) -> List[str]:
        return sorted(n for n, m in cls._registry.items() if m.category == category)

    @classmethod
    def categories(cls) -> List[str]:
        return sorted(set(m.category for m in cls._registry.values()))

    @classmethod
    def describe(cls, name: str) -> str:
        m = cls.get_meta(name)
        tag = " [需评估]" if m.requires_evaluation else ""
        return f"[{m.category}]{tag} {m.name}: {m.description}"


# 便捷别名
register_strategy = StrategyHub.register