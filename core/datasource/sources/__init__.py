"""
数据源抽象层 — 统一接口 + 注册机制。

所有数据源实现三个方法：
  fetch_bars(symbol, start_date, end_date) -> pd.DataFrame
  fetch_symbols() -> pd.DataFrame
  name -> str

新增数据源只需在 sources/ 目录新建文件并注册，Fetcher 自动发现。
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type
import pandas as pd


class BaseDataSource(ABC):
    """数据源基类。"""

    @property
    @abstractmethod
    def name(self) -> str:
        """数据源名称标识。"""

    @abstractmethod
    def fetch_bars(self, symbol: str, start_date: str,
                   end_date: str, adjust: str = "qfq") -> pd.DataFrame:
        """获取单只股票日线。

        Returns:
            DataFrame 必须包含: date, symbol, open, high, low, close, volume, amount, pct_change
            空数据时返回空 DataFrame。
        """

    @abstractmethod
    def fetch_symbols(self) -> pd.DataFrame:
        """获取全市场股票列表。

        Returns:
            DataFrame 必须包含: symbol, name
            空数据时返回空 DataFrame。
        """


class DataSourceRegistry:
    """数据源注册中心。"""

    _sources: Dict[str, Type[BaseDataSource]] = {}

    @classmethod
    def register(cls, source_cls: Type[BaseDataSource]):
        """注册一个数据源。"""
        name = source_cls.__name__.replace("Source", "").lower()
        cls._sources[name] = source_cls

    @classmethod
    def list(cls) -> Dict[str, Type[BaseDataSource]]:
        """列出所有已注册数据源。"""
        return dict(cls._sources)

    @classmethod
    def get(cls, name: str) -> Optional[Type[BaseDataSource]]:
        """按名称获取数据源类。"""
        return cls._sources.get(name)


def register_source(cls):
    """装饰器：注册数据源。"""
    DataSourceRegistry.register(cls)
    return cls
