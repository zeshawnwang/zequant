"""
FactorHub - 因子注册中心(可插拔)

核心抽象:
  @register_factor(name="a3", category="alpha101", requires=["open", "volume"])
  def alpha3(ctx: FactorContext) -> pd.DataFrame:
      return -corr(rank_(ctx.open), rank_(ctx.volume), 10)

调用方:
  FactorHub.compute_all(bars, names=["a3", "momentum_20"])  # 自动按需 pivot
  FactorHub.list_all()                                       # 列出全部已注册
  FactorHub.list_by_category("alpha101")                     # 分类列出

设计原则:
  1. 因子只声明输入字段(requires),Hub 自动准备 wide 格式
  2. 因子不直接接 db / 全局变量,只接 ctx(纯函数,易测试)
  3. 同名注册自动覆盖(便于热修)
  4. 计算流水线可观测(每个因子的耗时、输出形状)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set
import time
import numpy as np
import pandas as pd


@dataclass
class FactorContext:
    """每次计算时一次性 pivot 的 wide 表集合,所有因子共享。"""
    open: Optional[pd.DataFrame] = None
    high: Optional[pd.DataFrame] = None
    low: Optional[pd.DataFrame] = None
    close: Optional[pd.DataFrame] = None
    volume: Optional[pd.DataFrame] = None
    amount: Optional[pd.DataFrame] = None
    vwap: Optional[pd.DataFrame] = None
    returns: Optional[pd.DataFrame] = None
    # 行业中性化要用,先留空
    industry: Optional[pd.DataFrame] = None
    # 缓存:供因子之间复用中间结果
    cache: Dict[str, pd.DataFrame] = field(default_factory=dict)


@dataclass
class FactorMeta:
    name: str
    category: str
    requires: List[str]
    func: Callable[[FactorContext], pd.DataFrame]
    description: str = ""


class FactorHub:
    """全局因子仓库(单例式)。"""

    _registry: Dict[str, FactorMeta] = {}

    # ============== 注册 ==============

    @classmethod
    def register(
        cls,
        name: str,
        category: str = "misc",
        requires: Optional[List[str]] = None,
        description: str = "",
    ):
        """装饰器:把一个 (ctx) -> wide DataFrame 函数注册为因子。"""
        def deco(func):
            meta = FactorMeta(
                name=name,
                category=category,
                requires=list(requires or ["close"]),
                func=func,
                description=description or (func.__doc__ or "").strip(),
            )
            cls._registry[name] = meta
            return func
        return deco

    @classmethod
    def get(cls, name: str) -> FactorMeta:
        if name not in cls._registry:
            raise KeyError(f"factor not registered: {name}")
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

    # ============== 计算 ==============

    @classmethod
    def _build_context(cls, bars: pd.DataFrame, required_fields: Set[str]) -> FactorContext:
        """按需 pivot,只 pivot 真正用到的字段。"""
        bars = bars.copy()
        bars["date"] = pd.to_datetime(bars["date"])
        bars = bars.sort_values(["date", "symbol"])

        # 自动派生 vwap
        if "vwap" in required_fields:
            if "amount" in bars.columns and "volume" in bars.columns:
                bars["vwap"] = bars["amount"] / bars["volume"].replace(0, np.nan)
                bars["vwap"] = bars["vwap"].fillna(bars.get("close"))
            else:
                bars["vwap"] = bars.get("close", np.nan)

        # 自动派生 returns
        if "returns" in required_fields and "returns" not in bars.columns:
            tmp = bars.pivot_table(index="date", columns="symbol",
                                    values="close", aggfunc="last").sort_index()
            ret = tmp.pct_change().stack(dropna=True).reset_index()
            ret.columns = ["date", "symbol", "returns"]
            bars = bars.merge(ret, on=["date", "symbol"], how="left")

        ctx = FactorContext()

        def _pivot(col):
            if col not in bars.columns:
                return None
            return bars.pivot_table(
                index="date", columns="symbol", values=col, aggfunc="last"
            ).sort_index().astype(float)

        for f in required_fields:
            if hasattr(ctx, f):
                setattr(ctx, f, _pivot(f))
        return ctx

    @classmethod
    def compute(
        cls,
        name: str,
        bars: pd.DataFrame,
        ctx: Optional[FactorContext] = None,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """单因子计算,返回 wide DataFrame。"""
        meta = cls.get(name)
        if ctx is None:
            ctx = cls._build_context(bars, set(meta.requires))
        t0 = time.time()
        wide = meta.func(ctx)
        wide = wide.replace([np.inf, -np.inf], np.nan)
        if verbose:
            print(f"  {name:>6s}  shape={wide.shape}  ({time.time()-t0:.2f}s)")
        return wide

    @classmethod
    def compute_all(
        cls,
        bars: pd.DataFrame,
        names: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """批量计算,返回 long 表 (date, symbol, factor_name, value)。"""
        names = names or cls.list_all()
        # 一次性 pivot,所有因子共享
        all_required: Set[str] = set()
        for n in names:
            if n in cls._registry:
                all_required.update(cls._registry[n].requires)
        ctx = cls._build_context(bars, all_required)

        long_frames = []
        for n in names:
            if n not in cls._registry:
                if verbose:
                    print(f"  [WARN] not registered: {n}")
                continue
            try:
                wide = cls.compute(n, bars, ctx=ctx, verbose=verbose)
            except Exception as e:
                if verbose:
                    print(f"  [ERROR] {n} failed: {e}")
                continue
            melted = wide.stack(dropna=True).reset_index()
            melted.columns = ["date", "symbol", "value"]
            melted["factor_name"] = n
            long_frames.append(melted[["date", "symbol", "factor_name", "value"]])

        if not long_frames:
            return pd.DataFrame(columns=["date", "symbol", "factor_name", "value"])
        out = pd.concat(long_frames, ignore_index=True)
        out["date"] = pd.to_datetime(out["date"]).dt.date
        return out


# 便捷别名
register_factor = FactorHub.register