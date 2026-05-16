"""
多源兜底获取器 — 自动尝试所有已注册数据源。

策略：
  1. 按优先级顺序（akshare → baostock → efinance）依次尝试
  2. 第一个返回非空数据的源即被采用
  3. 全部失败则返回空 DataFrame

用法：
    from core.datasource.fallback_fetcher import FallbackFetcher
    fetcher = FallbackFetcher()
    df = fetcher.fetch_bars("000001", "2026-05-01", "2026-05-15")
"""
from __future__ import annotations
import logging
from typing import List, Optional
import pandas as pd

from .sources import DataSourceRegistry, BaseDataSource

logger = logging.getLogger(__name__)

# 兜底优先级（配置为 str list，按序尝试）
DEFAULT_PRIORITY = ["akshare", "baostock", "efinance"]


class FallbackFetcher:
    """多源兜底获取器。"""

    def __init__(self, priority: List[str] = None):
        # 确保注册所有 sources
        import core.datasource.sources.akshare_source  # noqa: F401
        import core.datasource.sources.baostock_source  # noqa: F401
        import core.datasource.sources.efinance_source  # noqa: F401

        self.priority = priority or DEFAULT_PRIORITY
        self._instances: dict = {}

    def _get_source(self, name: str) -> Optional[BaseDataSource]:
        """获取或创建数据源实例。"""
        if name not in self._instances:
            cls = DataSourceRegistry.get(name)
            if cls is not None:
                self._instances[name] = cls()
        return self._instances.get(name)

    def _try_fetch_bars(self, symbol: str, start_date: str,
                        end_date: str) -> pd.DataFrame:
        """依次尝试所有数据源。"""
        errors = []
        for src_name in self.priority:
            src = self._get_source(src_name)
            if src is None:
                continue
            try:
                df = src.fetch_bars(symbol, start_date, end_date)
                if df is not None and not df.empty:
                    logger.debug("✅ %s (%s) %s %d条",
                                 src_name, symbol, start_date, len(df))
                    return df
            except Exception as e:
                errors.append(f"{src_name}: {e}")
                continue

        if errors:
            logger.debug("全部数据源失败 %s: %s", symbol, "; ".join(errors))
        return pd.DataFrame()

    def fetch_bars(self, symbol: str, start_date: str,
                   end_date: str = None) -> pd.DataFrame:
        """获取单只股票日线（兜底链）。"""
        from datetime import datetime
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        return self._try_fetch_bars(symbol, start_date, end_date)

    def fetch_symbols(self) -> pd.DataFrame:
        """获取全市场股票列表。"""
        for src_name in self.priority:
            src = self._get_source(src_name)
            if src is None:
                continue
            try:
                df = src.fetch_symbols()
                if df is not None and not df.empty:
                    logger.info("✅ %s: 获取股票列表 %d只", src_name, len(df))
                    return df
            except Exception as e:
                logger.debug("%s 获取股票列表失败: %s", src_name, e)
                continue
        return pd.DataFrame()

    def fetch_batch_bars(self, symbols: List[str], start_date: str,
                         end_date: str = None, report_every: int = 200
                         ) -> dict:
        """批量获取多只股票。

        Returns:
            {symbol: rows_fetched}
        """
        results = {}
        total = len(symbols)
        logger.info("批量获取 %d 只股票日线 [兜底链=%s]...",
                    total, "+".join(self.priority))
        for i, sym in enumerate(symbols):
            if (i + 1) % report_every == 0:
                logger.info("  [%d/%d] %s ...", i + 1, total, sym)
            df = self.fetch_bars(sym, start_date, end_date)
            results[sym] = len(df)
        fetched = sum(1 for v in results.values() if v > 0)
        logger.info("批量获取完成: %d/%d 只有数据", fetched, total)
        return results

    def list_available(self) -> List[str]:
        """列出当前可用的数据源（能导入的）。"""
        available = []
        for name in self.priority:
            src = self._get_source(name)
            if src is not None:
                available.append(name)
        return available
