"""数据模块。

负责数据获取、检查、验证等相关功能。
"""
from __future__ import annotations

from .checker import DataQualityChecker
from .fetcher import IncrementalFetcher
from .validator import DataValidator, ValidationReport, validate_data

__all__ = [
    "DataQualityChecker",
    "IncrementalFetcher",
    "DataValidator",
    "ValidationReport",
    "validate_data",
]
