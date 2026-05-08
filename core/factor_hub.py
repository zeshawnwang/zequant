"""
FactorHub - 因子注册中心(可插拔)

核心抽象:
  @register_factor(name="a3", category="alpha101", requires=["open", "volume"])
  def alpha3(ctx: FactorContext) -> pd.DataFrame:
      return -corr(rank_(ctx.open), rank_(ctx.volume), 10)

调用方:
  factor_hub.compute_all(bars, names=["a3", "momentum_20"])  # 自动按需 pivot
  factor_hub.list_all()                                       # 列出全部已注册
  factor_hub.list_by_category("alpha101")                     # 分类列出

设计原则:
  1. 因子只声明输入字段(requires),Hub 自动准备 wide 格式
  2. 因子不直接接 db / 全局变量,只接 ctx(纯函数,易测试)
  3. 同名注册自动覆盖(便于热修)
  4. 计算流水线可观测(每个因子的耗时、输出形状)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple
import os
import time
import warnings
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
    """因子仓库，支持多实例。"""

    def __init__(self):
        self._registry: Dict[str, FactorMeta] = {}

    # ============== 注册 ==============

    def register(
        self,
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
            self._registry[name] = meta
            return func
        return deco

    def get(self, name: str) -> FactorMeta:
        if name not in self._registry:
            raise KeyError(f"factor not registered: {name}")
        return self._registry[name]

    def list_all(self) -> List[str]:
        return sorted(self._registry.keys())

    def list_by_category(self, category: str) -> List[str]:
        return sorted(n for n, m in self._registry.items() if m.category == category)

    def categories(self) -> List[str]:
        return sorted(set(m.category for m in self._registry.values()))

    # ============== 计算 ==============

    def _build_context(self, bars: pd.DataFrame, required_fields: Set[str]) -> FactorContext:
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

    def compute(
        self,
        name: str,
        bars: pd.DataFrame,
        ctx: Optional[FactorContext] = None,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """单因子计算,返回 wide DataFrame。"""
        meta = self.get(name)
        if ctx is None:
            ctx = self._build_context(bars, set(meta.requires))
        t0 = time.time()
        wide = meta.func(ctx)
        wide = wide.replace([np.inf, -np.inf], np.nan)
        if verbose:
            print(f"  {name:>6s}  shape={wide.shape}  ({time.time()-t0:.2f}s)")
        return wide

    def compute_all(
        self,
        bars: pd.DataFrame,
        names: Optional[List[str]] = None,
        verbose: bool = True,
        n_jobs: int = 1,
    ) -> pd.DataFrame:
        """批量计算,返回 long 表 (date, symbol, factor_name, value)。

        Args:
            bars: K 线数据
            names: 指定要计算的因子列表,None 则计算全部已注册因子
            verbose: 是否打印每个因子的耗时与形状
            n_jobs: 并行进程数,1 为串行,-1 为使用全部 CPU 核心
        """
        names = names or self.list_all()
        # 一次性 pivot,所有因子共享
        all_required: Set[str] = set()
        for n in names:
            if n in self._registry:
                all_required.update(self._registry[n].requires)
        ctx = self._build_context(bars, all_required)

        if n_jobs == 1:
            return self._compute_all_sequential(names, bars, ctx, verbose)

        return self._compute_all_parallel(names, bars, ctx, verbose, n_jobs)

    def _compute_all_sequential(
        self,
        names: List[str],
        bars: pd.DataFrame,
        ctx: FactorContext,
        verbose: bool,
    ) -> pd.DataFrame:
        """串行计算所有因子。"""
        long_frames = []
        for n in names:
            if n not in self._registry:
                if verbose:
                    print(f"  [WARN] not registered: {n}")
                continue
            try:
                wide = self.compute(n, bars, ctx=ctx, verbose=verbose)
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

    @staticmethod
    def _worker_compute(
        name: str,
        func: Callable[[FactorContext], pd.DataFrame],
        requires: List[str],
        bars: pd.DataFrame,
        verbose: bool,
    ) -> Tuple[str, Optional[pd.DataFrame], Optional[str]]:
        """工作进程执行的计算函数。

        由于进程间无法共享 ctx,每个工作进程需要独立构建所需的上下文。
        返回 (name, wide_df_or_None, error_msg_or_None)
        """
        try:
            # 在子进程中重新构建上下文(只构建该因子需要的字段)
            hub = FactorHub()
            ctx = hub._build_context(bars, set(requires))
            t0 = time.time()
            wide = func(ctx)
            wide = wide.replace([np.inf, -np.inf], np.nan)
            if verbose:
                print(f"  {name:>6s}  shape={wide.shape}  ({time.time()-t0:.2f}s)")
            return name, wide, None
        except Exception as e:
            return name, None, str(e)

    def _compute_all_parallel(
        self,
        names: List[str],
        bars: pd.DataFrame,
        ctx: FactorContext,
        verbose: bool,
        n_jobs: int,
    ) -> pd.DataFrame:
        """并行计算所有因子。"""
        import concurrent.futures

        if n_jobs == -1:
            n_jobs = os.cpu_count() or 1
        n_jobs = max(1, n_jobs)

        # 准备任务参数(只传递可序列化的数据)
        tasks = []
        valid_names = []
        for n in names:
            if n not in self._registry:
                if verbose:
                    print(f"  [WARN] not registered: {n}")
                continue
            meta = self._registry[n]
            tasks.append((n, meta.func, meta.requires, bars, verbose))
            valid_names.append(n)

        if not tasks:
            return pd.DataFrame(columns=["date", "symbol", "factor_name", "value"])

        results = []

        # 尝试使用 ProcessPoolExecutor
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
                futures = {
                    executor.submit(self._worker_compute, *task): task[0]
                    for task in tasks
                }
                for future in concurrent.futures.as_completed(futures):
                    name = futures[future]
                    try:
                        _, wide, err = future.result()
                        if err is not None:
                            if verbose:
                                print(f"  [ERROR] {name} failed: {err}")
                            continue
                        results.append((name, wide))
                    except Exception as e:
                        if verbose:
                            print(f"  [ERROR] {name} failed: {e}")
                        continue
        except (TypeError, AttributeError, ImportError, OSError) as exc:
            # 进程池启动失败(如函数不可 pickle、资源不足等),优雅回退到串行
            warnings.warn(
                f"ProcessPoolExecutor 启动失败({exc}),回退到串行计算。"
                f"若因子函数使用了闭包/lambda,请改用模块级函数。",
                RuntimeWarning,
                stacklevel=2,
            )
            return self._compute_all_sequential(valid_names, bars, ctx, verbose)

        # 组装结果
        long_frames = []
        for name, wide in results:
            if wide is None:
                continue
            melted = wide.stack(dropna=True).reset_index()
            melted.columns = ["date", "symbol", "value"]
            melted["factor_name"] = name
            long_frames.append(melted[["date", "symbol", "factor_name", "value"]])

        if not long_frames:
            return pd.DataFrame(columns=["date", "symbol", "factor_name", "value"])
        out = pd.concat(long_frames, ignore_index=True)
        out["date"] = pd.to_datetime(out["date"]).dt.date
        return out


# 默认全局实例，保持向后兼容
_factor_hub = FactorHub()

# 便捷别名（绑定到默认全局实例）
register_factor = _factor_hub.register

# 为了兼容旧的类级调用方式，暴露默认实例的方法作为模块级函数
compute_all = _factor_hub.compute_all
compute = _factor_hub.compute
get = _factor_hub.get
list_all = _factor_hub.list_all
list_by_category = _factor_hub.list_by_category
categories = _factor_hub.categories
