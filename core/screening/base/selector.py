"""选股器抽象基类。"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional
import pandas as pd


class IStockSelector(ABC):
    """选股器接口。

    实现类须实现:
      - select(factor_data, date, top_n) -> List[str]: 返回 symbol 列表
    """

    @abstractmethod
    def select(
        self,
        factor_data: pd.DataFrame,
        date,
        top_n: int,
    ) -> List[str]:
        """
        从因子数据中选出 top_n 个股票。

        Args:
            factor_data: 因子截面数据
            date: 当前日期
            top_n: 选股数量

        Returns:
            股票代码列表
        """
        pass

    def get_description(self) -> str:
        return self.__class__.__name__
