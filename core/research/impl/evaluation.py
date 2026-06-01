"""因子评估体系。

提供完整的因子评估、排名、监控功能。

评估指标
--------
    - IC (Information Coefficient): 信息系数
    - IR (Information Ratio): 信息比率
    - IC T-stat: IC 的统计显著性
    - 分层收益: Top/Bottom 分组收益
    - 单调性: 分组收益是否单调
    - 换手率: 因子换手频率

用法
----
    from core.research.impl.evaluation import FactorEvaluator

    # 评估因子
    evaluator = FactorEvaluator(db)
    summary = evaluator.evaluate(factors=["a1", "a16"], period=("2020-01-01", "2020-06-30"))
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class FactorEvaluationResult:
    """因子评估结果。"""
    factor_name: str
    ic_mean: float
    ic_std: float
    ir: float
    ic_t_stat: float
    turnover: float
    top_group_return: float
    bottom_group_return: float
    monotonic: bool
    n_days: int

    def to_dict(self) -> Dict:
        return {
            "factor_name": self.factor_name,
            "ic_mean": self.ic_mean,
            "ic_std": self.ic_std,
            "ir": self.ir,
            "ic_t_stat": self.ic_t_stat,
            "turnover": self.turnover,
            "top_group_return": self.top_group_return,
            "bottom_group_return": self.bottom_group_return,
            "monotonic": self.monotonic,
            "n_days": self.n_days,
        }


class IFactorEvaluator(ABC):
    """因子评估器抽象基类。"""

    @abstractmethod
    def evaluate(self, factor_name: str, period: Tuple[str, str]) -> FactorEvaluationResult:
        """评估单个因子。"""
        pass

    @abstractmethod
    def evaluate_all(self, factor_names: List[str], period: Tuple[str, str]) -> pd.DataFrame:
        """批量评估多个因子。"""
        pass

    @abstractmethod
    def rank_factors(self, period: Tuple[str, str], top_n: int = 20) -> pd.DataFrame:
        """因子排名。"""
        pass


class FactorEvaluator(IFactorEvaluator):
    """因子评估器。"""

    def __init__(self, db):
        self.db = db

    def evaluate(
        self,
        factor_name: str,
        period: Tuple[str, str],
        forward_days: int = 5,
    ) -> FactorEvaluationResult:
        """
        评估单个因子。

        Args:
            factor_name: 因子名
            period: (开始日期, 结束日期)
            forward_days: 前瞻收益窗口

        Returns:
            FactorEvaluationResult
        """
        start_date, end_date = period

        ic_series = self._calc_ic_series(factor_name, start_date, end_date, forward_days)
        if ic_series.empty:
            return FactorEvaluationResult(
                factor_name=factor_name,
                ic_mean=0, ic_std=0, ir=0, ic_t_stat=0,
                turnover=0, top_group_return=0, bottom_group_return=0,
                monotonic=False, n_days=0,
            )

        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        ir = ic_mean / ic_std if ic_std > 0 else 0

        from scipy import stats
        n = len(ic_series)
        ic_t_stat = (ic_mean / ic_std * np.sqrt(n)) if ic_std > 0 else 0

        turnover = self._calc_turnover(factor_name, start_date, end_date)

        top_ret, bottom_ret, monotonic = self._calc_group_returns(
            factor_name, start_date, end_date, forward_days
        )

        return FactorEvaluationResult(
            factor_name=factor_name,
            ic_mean=ic_mean,
            ic_std=ic_std,
            ir=ir,
            ic_t_stat=ic_t_stat,
            turnover=turnover,
            top_group_return=top_ret,
            bottom_group_return=bottom_ret,
            monotonic=monotonic,
            n_days=n,
        )

    def evaluate_all(
        self,
        factor_names: List[str],
        period: Tuple[str, str],
        forward_days: int = 5,
    ) -> pd.DataFrame:
        """批量评估多个因子。"""
        results = []
        for name in factor_names:
            result = self.evaluate(name, period, forward_days)
            results.append(result.to_dict())
        return pd.DataFrame(results).sort_values("ir", ascending=False)

    def rank_factors(
        self,
        period: Tuple[str, str],
        top_n: int = 20,
        min_ir: float = 0.0,
    ) -> pd.DataFrame:
        """因子排名。"""
        start_date, end_date = period
        all_factors = self.db.list_factor_columns()
        df = self.evaluate_all(all_factors, period)
        df = df[df["ir"].abs() >= min_ir]
        return df.head(top_n)

    def _calc_ic_series(
        self,
        factor_name: str,
        start_date: str,
        end_date: str,
        forward_days: int,
    ) -> pd.Series:
        """计算 IC 序列。"""
        factor_data = self.db.get_factors(
            start_date=start_date,
            end_date=end_date,
            factor_names=[factor_name],
            with_close=True,
        )
        if factor_data.empty:
            return pd.Series()

        end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=forward_days * 2 + 10)
        bars = self.db.get_daily_bars(
            start_date=start_date,
            end_date=end_ts.strftime("%Y-%m-%d"),
        )
        if bars.empty:
            return pd.Series()

        factor_data["date"] = pd.to_datetime(factor_data["date"])
        bars["date"] = pd.to_datetime(bars["date"])

        merged = factor_data[["date", "symbol", factor_name, "close"]].merge(
            bars[["date", "symbol", "close"]],
            on=["date", "symbol"],
            suffixes=("", "_future"),
            how="inner",
        )

        if merged.empty or f"close_future" not in merged.columns:
            return pd.Series()

        merged["fwd_ret"] = (
            merged["close_future"] / merged["close"] - 1
        ).fillna(0)

        daily_ic = merged.groupby("date").apply(
            lambda x: x[factor_name].corr(x["fwd_ret"])
        )
        return daily_ic.dropna()

    def _calc_turnover(
        self,
        factor_name: str,
        start_date: str,
        end_date: str,
        top_n: int = 50,
    ) -> float:
        """计算换手率。"""
        factor_data = self.db.get_factors(
            start_date=start_date,
            end_date=end_date,
            factor_names=[factor_name],
        )
        if factor_data.empty:
            return 0.0

        dates = sorted(factor_data["date"].unique())
        if len(dates) < 2:
            return 0.0

        turnovers = []
        for i in range(len(dates) - 1):
            d1, d2 = dates[i], dates[i + 1]
            df1 = factor_data[factor_data["date"] == d1].nlargest(top_n, factor_name)
            df2 = factor_data[factor_data["date"] == d2].nlargest(top_n, factor_name)

            symbols1 = set(df1["symbol"])
            symbols2 = set(df2["symbol"])

            if len(symbols1) > 0:
                turnover = len(symbols1 - symbols2) / len(symbols1)
                turnovers.append(turnover)

        return np.mean(turnovers) if turnovers else 0.0

    def _calc_group_returns(
        self,
        factor_name: str,
        start_date: str,
        end_date: str,
        forward_days: int,
        n_groups: int = 5,
    ) -> Tuple[float, float, bool]:
        """计算分组收益。"""
        factor_data = self.db.get_factors(
            start_date=start_date,
            end_date=end_date,
            factor_names=[factor_name],
            with_close=True,
        )
        if factor_data.empty:
            return 0.0, 0.0, False

        end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=forward_days * 2 + 10)
        bars = self.db.get_daily_bars(
            start_date=start_date,
            end_date=end_ts.strftime("%Y-%m-%d"),
        )
        if bars.empty:
            return 0.0, 0.0, False

        factor_data["date"] = pd.to_datetime(factor_data["date"])
        bars["date"] = pd.to_datetime(bars["date"])

        merged = factor_data[["date", "symbol", factor_name, "close"]].merge(
            bars[["date", "symbol", "close"]],
            on=["date", "symbol"],
            suffixes=("", "_future"),
            how="inner",
        )

        if merged.empty or "close_future" not in merged.columns:
            return 0.0, 0.0, False

        merged["fwd_ret"] = (
            merged["close_future"] / merged["close"] - 1
        ).fillna(0)

        merged["group"] = merged.groupby("date")[factor_name].transform(
            lambda x: pd.qcut(x, n_groups, labels=False, duplicates="drop")
        )

        group_returns = merged.groupby(["date", "group"])["fwd_ret"].mean().unstack()

        if group_returns.empty or group_returns.shape[1] < 2:
            return 0.0, 0.0, False

        top_ret = group_returns.iloc[:, -1].mean()
        bottom_ret = group_returns.iloc[:, 0].mean()

        group_means = group_returns.mean()
        monotonic = all(
            group_means.iloc[i] < group_means.iloc[i + 1]
            for i in range(len(group_means) - 1)
        )

        return top_ret, bottom_ret, monotonic


class FactorMonitor:
    """因子监控器。"""

    def __init__(self, db):
        self.db = db
        self.evaluator = FactorEvaluator(db)

    def check_decay(
        self,
        factor_name: str,
        recent_periods: int = 5,
        period_days: int = 60,
    ) -> Dict:
        """
        检查因子衰减。

        Args:
            factor_name: 因子名
            recent_periods: 最近几个周期
            period_days: 每个周期的天数

        Returns:
            衰减报告
        """
        end_date = pd.Timestamp.today()
        ir_history = []

        for i in range(recent_periods):
            start = end_date - pd.Timedelta(days=period_days)
            result = self.evaluator.evaluate(
                factor_name,
                (start.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")),
            )
            ir_history.append(result.ir)
            end_date = start - pd.Timedelta(days=1)

        ir_history = list(reversed(ir_history))

        trend = "stable"
        if len(ir_history) >= 3:
            if all(ir_history[i] > ir_history[i+1] for i in range(len(ir_history)-1)):
                trend = "decaying"
            elif all(ir_history[i] < ir_history[i+1] for i in range(len(ir_history)-1)):
                trend = "improving"

        return {
            "factor_name": factor_name,
            "ir_history": ir_history,
            "current_ir": ir_history[-1] if ir_history else 0,
            "avg_ir": np.mean(ir_history) if ir_history else 0,
            "trend": trend,
            "decay_rate": (
                (ir_history[-1] - ir_history[0]) / abs(ir_history[0])
                if ir_history and ir_history[0] != 0
                else 0
            ),
        }

    def check_universe(
        self,
        factor_names: List[str],
        period: Tuple[str, str],
    ) -> pd.DataFrame:
        """检查因子在不同股票池的表现。"""
        evaluator = FactorEvaluator(self.db)
        results = []

        for name in factor_names:
            result = evaluator.evaluate(name, period)
            results.append(result.to_dict())

        return pd.DataFrame(results).sort_values("ir", ascending=False)
