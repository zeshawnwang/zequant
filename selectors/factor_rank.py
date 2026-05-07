"""
因子排名选股器
按单个因子排序选股，支持升序/降序、分位数过滤。
"""
from typing import List
import pandas as pd


class IStockSelector:
    """选股器基类"""

    def select(self, factor_data: pd.DataFrame, date, top_n: int) -> List[str]:
        raise NotImplementedError

    def get_description(self) -> str:
        return self.__class__.__name__


class FactorRankSelector(IStockSelector):
    """
    因子排名选股。
    按因子值排序，支持：
    - ascending: 从小到大（低估值）或从大到小（高动量）
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

    def select(self, factor_data: pd.DataFrame, date, top_n: int) -> List[str]:
        # 获取指定日期之前的数据
        if 'date' in factor_data.columns:
            df = factor_data[factor_data['date'] <= date]
            latest = df.groupby('symbol').tail(1)
        else:
            latest = factor_data.groupby('symbol').tail(1)

        if self.factor_name not in latest.columns:
            return []

        result = latest[self.factor_name].dropna()

        # 分位数过滤
        if self.quantile_filter is not None:
            q = result.quantile(self.quantile_filter)
            if self.ascending:
                result = result[result >= q]
            else:
                result = result[result <= q]

        # 值域过滤
        if self.min_factor_value is not None:
            result = result[result >= self.min_factor_value]
        if self.max_factor_value is not None:
            result = result[result <= self.max_factor_value]

        # 排序
        if self.ascending:
            ranked = result.sort_values(ascending=True)
        else:
            ranked = result.sort_values(ascending=False)

        return ranked.head(top_n).index.tolist() if hasattr(result, 'index') else ranked.head(top_n).index.to_list()

    def get_description(self) -> str:
        direction = "低" if self.ascending else "高"
        return f"因子选股:{self.factor_name}({direction}优)"
