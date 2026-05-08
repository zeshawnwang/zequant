"""因子排名选股器

按单个因子排序选股,支持升序/降序、分位数过滤。
"""
from __future__ import annotations
from typing import List
import pandas as pd

from .base import IStockSelector


class FactorRankSelector(IStockSelector):
    """
    因子排名选股。
    按因子值排序,支持:
    - ascending: 从小到大(低估值)或从大到小(高动量)
    - min/max_factor_value: 值域过滤
    - quantile_filter: 分位数过滤
    """

    def __init__(self,
                 factor_name: str,
                 ascending: bool = False,
                 top_n: int = 100,
                 min_factor_value: float = None,
                 max_factor_value: float = None,
                 quantile_filter: float = None):
        self.factor_name = factor_name
        self.ascending = ascending
        self.top_n = top_n
        self.min_factor_value = min_factor_value
        self.max_factor_value = max_factor_value
        self.quantile_filter = quantile_filter

    @property
    def factor_names(self) -> List[str]:
        """该选股器消费的因子列(供回测脚本提前加载用)。"""
        return [self.factor_name]

    def select(self, factor_data: pd.DataFrame, date, top_n: int) -> List[str]:
        if factor_data is None or factor_data.empty:
            return []

        # 取每只股票在 date 之前的最后一条
        if 'date' in factor_data.columns:
            df = factor_data[factor_data['date'] <= date]
        else:
            df = factor_data

        if df.empty or self.factor_name not in df.columns:
            return []

        latest = df.sort_values('date').groupby('symbol').tail(1)
        latest = latest.dropna(subset=[self.factor_name])

        # 以 symbol 为索引,便于取值
        series = latest.set_index('symbol')[self.factor_name]

        # 分位数过滤
        if self.quantile_filter is not None and not series.empty:
            q = series.quantile(self.quantile_filter)
            if self.ascending:
                series = series[series <= q]
            else:
                series = series[series >= q]

        # 值域过滤
        if self.min_factor_value is not None:
            series = series[series >= self.min_factor_value]
        if self.max_factor_value is not None:
            series = series[series <= self.max_factor_value]

        ranked = series.sort_values(ascending=self.ascending)
        return ranked.head(top_n).index.tolist()

    def get_description(self) -> str:
        direction = "低" if self.ascending else "高"
        return f"因子选股:{self.factor_name}({direction}优)"