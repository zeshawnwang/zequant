"""数据模块。

负责数据获取、检查、验证等相关功能。
"""
from __future__ import annotations

from .checker import DataQualityChecker
from .fetcher import IncrementalFetcher
from .validator import DataValidator, ValidationReport, validate_data

# LOF/ETF 基金代码前缀（共享常量，供 akshare_source / data_updater 等使用）
FUND_PREFIXES = (
    "15", "16", "50", "51", "52", "56", "58",
    "159", "160", "161", "162", "163", "164", "165", "166", "167", "168",
    "500", "501", "502", "510", "511", "512", "513", "515", "517", "518", "588",
)


def is_fund_symbol(symbol: str) -> bool:
    """判断是否是 LOF/ETF 基金代码"""
    s = str(symbol).zfill(6)
    return s.startswith(FUND_PREFIXES)


__all__ = [
    "DataQualityChecker",
    "IncrementalFetcher",
    "DataValidator",
    "ValidationReport",
    "validate_data",
    "FUND_PREFIXES",
    "is_fund_symbol",
]
