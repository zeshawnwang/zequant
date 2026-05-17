"""数据增量抓取模块。

负责把每日行情、股票名册等增量写入本地 DuckDB。
使用 FallbackFetcher 兜底链（akshare → baostock → efinance），自动切换可用数据源。
"""
import logging
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Optional
from ..database import Database

logger = logging.getLogger(__name__)


class IncrementalFetcher:
    """增量获取数据,使用 FallbackFetcher 自动切换可用数据源。"""

    def __init__(self, db: Database):
        self.db = db

    def fetch_daily_bars(self, symbol: str, start_date: str = None) -> pd.DataFrame:
        """
        增量获取日线数据(仅获取本地该 symbol 最大日期之后的数据)。
        """
        from .fallback_fetcher import FallbackFetcher
        fetcher = FallbackFetcher()

        # 按 symbol 维度确定起始日期,避免用全局最大日期误伤新股票
        if start_date is None:
            local_max = self.db.get_symbol_max_date(symbol)
            if local_max:
                start_date = (local_max + timedelta(days=1)).strftime("%Y-%m-%d")
            else:
                start_date = "2019-01-01"
        else:
            start_date = start_date[:10]

        # 起点晚于今日,无增量可取,直接返回
        today = datetime.now().strftime("%Y-%m-%d")
        if start_date > today:
            return pd.DataFrame()

        df = fetcher.fetch_bars(symbol, start_date, end_date=today)
        if df is None or df.empty:
            if not fetcher.list_available():
                logger.error("所有数据源均不可用")
            return pd.DataFrame()

        self.db.upsert_daily_bars(df)
        return df

    def fetch_all_symbols(self, with_list_date: bool = True) -> pd.DataFrame:
        """获取全市场股票列表。

        Args:
            with_list_date: 若为 True,从 daily_bars 表推断各 symbol 的最早交易日
                作为 list_date(用于 Universe 过滤"上市不满 N 天")。
                这是廉价兜底:不额外调 API,纯本地 SQL 聚合。
        """
        from .fallback_fetcher import FallbackFetcher
        fetcher = FallbackFetcher()
        df = fetcher.fetch_symbols()

        if df is None or df.empty:
            logger.error("所有数据源获取股票列表均失败")
            return pd.DataFrame()

        df["symbol"] = df["symbol"].astype(str).str.zfill(6)
        if "name" in df.columns:
            df["name"] = df["name"].astype(str)
        else:
            df["name"] = ""

        df["market"] = df["symbol"].apply(
            lambda x: "SH" if x.startswith(("6", "9")) else
                      ("BJ" if x.startswith(("4", "8")) else "SZ")
        )
        df["delist_date"] = None
        df["sector"] = None

        # 从本地 daily_bars 推断 list_date(各 symbol 的最早一条 K 线日期)
        if with_list_date:
            try:
                first_dates = self.db.conn.execute(
                    "SELECT symbol, MIN(date) AS list_date "
                    "FROM daily_bars GROUP BY symbol"
                ).df()
                if not first_dates.empty:
                    df = df.merge(first_dates, on="symbol", how="left")
                else:
                    df["list_date"] = None
            except Exception as e:
                logger.warning("推断 list_date 失败: %s", e)
                df["list_date"] = None
        else:
            df["list_date"] = None

        cols = ["symbol", "name", "market", "list_date", "delist_date", "sector"]
        cols = [c for c in cols if c in df.columns]
        self.db.save_symbols(df[cols])
        return df

    def fetch_batch(self, symbols: list = None, max_n: int = 100) -> dict:
        """批量获取多只股票数据。

        Returns:
            {symbol: rows_fetched}
        """
        if symbols is None:
            sym_df = self.db.get_symbols()
            symbols = sym_df["symbol"].tolist()[:max_n]

        results = {}
        for i, sym in enumerate(symbols):
            logger.info("[%d/%d] 获取 %s ...", i + 1, len(symbols), sym)
            df = self.fetch_daily_bars(sym)
            results[sym] = len(df)

        return results

    def fetch_all(self, symbols: list = None, start_date: str = None, end_date: str = None) -> dict:
        """获取全市场增量日线数据。

        Args:
            symbols: 指定股票列表, None则全市场
            start_date: 起始日期
            end_date: 结束日期(默认今天)

        Returns:
            {symbol: rows_fetched}
        """
        if symbols is None:
            sym_df = self.db.get_symbols()
            symbols = sym_df["symbol"].tolist()
            # 如果本地无符号表则从数据源获取
            if symbols:
                logger.info("从本地符号表获取 %d 只股票", len(symbols))
            else:
                logger.info("本地符号表为空,从数据源获取")
                self.fetch_all_symbols()
                sym_df = self.db.get_symbols()
                symbols = sym_df["symbol"].tolist()

        total = len(symbols)
        results = {}
        logger.info("开始全市场数据获取: %d 只股票, 起始=%s", total, start_date or "auto")
        for i, sym in enumerate(symbols):
            if (i + 1) % 200 == 0:
                logger.info("[%d/%d] %s ...", i + 1, total, sym)
            df = self.fetch_daily_bars(sym, start_date=start_date)
            results[sym] = len(df)

        fetched = sum(1 for v in results.values() if v > 0)
        logger.info("数据获取完成: %d/%d 只股票有新数据", fetched, total)
        return results
