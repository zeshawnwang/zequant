"""选股器抽象基类。"""
from __future__ import annotations
from typing import List
import pandas as pd


class IStockSelector:
    """选股器接口。

    实现类须提供:
      - select(factor_data, date, top_n) -> List[str]: 返回 symbol 列表
      - get_description() -> str: 用于日志/报告的简短描述
    """

    def select(self, factor_data: pd.DataFrame, date, top_n: int) -> List[str]:
        raise NotImplementedError

    def get_description(self) -> str:
        return self.__class__.__name__
