"""基本面三因子选股器 (FundamentalSelector)

核心思想（源自雪球「漫慢投资」的选股方法）:
    用三大确定性因子做初筛，过滤掉大部分不确定性股票，构建基础股票池。

第一步: 三大确定性因子初筛，构建基础股票池

第一步三因子:
    1. 业绩高增长: 至少 N 年净利润增速 > growth_threshold%
    2. 估值合理:   PE < max_pe 且 PB < max_pb
    3. 盈利稳健:   EPS 连续 M 年 > min_eps

使用示例:
    sel = FundamentalSelector(
        max_pe=20, max_pb=2,
        min_eps=0.3, eps_years=3,
        growth_threshold=10, growth_years=2,
        top_n=50,
    )
    picks = sel.select(factor_data, date='2024-06-01', top_n=50)
"""
from __future__ import annotations
from typing import List
import pandas as pd

from core.screening.base.selector import IStockSelector


class FundamentalSelector(IStockSelector):
    """基本面三因子初筛选股器。"""

    def __init__(
        self,
        max_pe: float = 20,
        max_pb: float = 2,
        min_eps: float = 0.3,
        eps_years: int = 3,
        growth_threshold: float = 10,
        growth_years: int = 2,
        top_n: int = 100,
    ):
        """
        Args:
            max_pe:           最大 PE（市盈率）阈值
            max_pb:           最大 PB（市净率）阈值
            min_eps:          最低 EPS（每股收益）阈值
            eps_years:        要求 EPS 连续多少年 > min_eps
            growth_threshold: 净利润增速阈值（%）
            growth_years:     要求多少年净利润增速 > growth_threshold
            top_n:            默认选股数量
        """
        self.max_pe = max_pe
        self.max_pb = max_pb
        self.min_eps = min_eps
        self.eps_years = eps_years
        self.growth_threshold = growth_threshold
        self.growth_years = growth_years
        self.top_n = top_n

    @property
    def factor_names(self) -> list:
        return ["pe", "pb", "eps", "net_profit_growth"]

    def select(self, factor_data: pd.DataFrame, date, top_n: int) -> List[str]:
        if factor_data is None or factor_data.empty:
            return []

        df = factor_data.copy()
        if "date" in df.columns:
            date_ts = pd.Timestamp(date)
            df["date"] = pd.to_datetime(df["date"])
            df = df[df["date"] < date_ts]

        if df.empty:
            return []

        latest = (
            df.sort_values("date")
            .groupby("symbol")
            .tail(1)
            .set_index("symbol")
        )

        required = ["pe", "pb", "eps", "net_profit_growth"]
        missing = [c for c in required if c not in latest.columns]
        if missing:
            raise RuntimeError(
                f"FundamentalSelector 需要以下因子列: {required}, "
                f"缺失: {missing}。请先运行 compute_factors.py 计算基本面因子。"
            )

        latest = latest.dropna(subset=required)

        candidates = self._filter_three_factors(latest)
        if candidates.empty:
            return []

        candidates = self._score_and_rank(candidates)

        n = top_n or self.top_n
        return candidates.head(n).index.tolist()

    def _filter_three_factors(self, latest: pd.DataFrame) -> pd.DataFrame:
        """第一步: 三因子初筛。"""
        mask = (
            (latest["pe"] < self.max_pe)
            & (latest["pb"] < self.max_pb)
            & (latest["eps"] > self.min_eps)
            & (latest["net_profit_growth"] > self.growth_threshold)
        )
        return latest[mask].copy()

    def _score_and_rank(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """对初筛结果打分排序。

        打分维度:
        - PE 越低越好（标准化后取负）
        - PB 越低越好（标准化后取负）
        - 净利润增速越高越好（标准化）

        综合得分 = -zscore(PE) - zscore(PB) + zscore(growth)
        """
        df = candidates.copy()
        scores = []

        if "pe" in df.columns and df["pe"].std() > 0:
            scores.append(-_zscore(df["pe"]))
        if "pb" in df.columns and df["pb"].std() > 0:
            scores.append(-_zscore(df["pb"]))
        if "net_profit_growth" in df.columns and df["net_profit_growth"].std() > 0:
            scores.append(_zscore(df["net_profit_growth"]))

        if scores:
            df["composite_score"] = pd.concat(scores, axis=1).sum(axis=1)
        else:
            df["composite_score"] = 0

        df = df.sort_values("composite_score", ascending=False)
        return df

    def get_description(self) -> str:
        return (
            f"基本面三因子[PE<{self.max_pe}, PB<{self.max_pb}, "
            f"EPS>{self.min_eps}, 增速>{self.growth_threshold}%]"
        )


def _zscore(s: pd.Series) -> pd.Series:
    """标准化。"""
    s = pd.to_numeric(s, errors="coerce").astype(float)
    mu, sd = s.mean(), s.std(ddof=0)
    if sd is None or sd == 0 or pd.isna(sd):
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sd
