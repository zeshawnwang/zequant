"""
Data Fetcher
Incremental fetching from AKShare (primary) and Tushare (fallback).
"""
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Optional
from .database import Database


class IncrementalFetcher:
    """增量获取数据，自动从AKShare拉取，upsert到DuckDB。"""

    def __init__(self, db: Database):
        self.db = db

    def fetch_daily_bars(self, symbol: str, start_date: str = None) -> pd.DataFrame:
        """
        增量获取日线数据（仅获取本地最大日期之后的数据）。
        """
        try:
            import akshare as ak
        except ImportError:
            print("请安装 akshare: pip install akshare")
            return pd.DataFrame()

        # 确定起始日期
        if start_date is None:
            local_max = self.db.get_max_date("daily_bars", "date")
            if local_max:
                start_date = str(local_max + timedelta(days=1))
            else:
                start_date = "20190101"

        try:
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date.replace("-", ""),
                end_date=datetime.now().strftime("%Y%m%d"),
                adjust="qfq"
            )
        except Exception as e:
            print(f"获取 {symbol} 失败: {e}")
            return pd.DataFrame()

        if df is None or df.empty:
            return pd.DataFrame()

        # 标准化列名
        rename = {
            "日期": "date",
            "股票代码": "symbol",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "涨跌幅": "pct_change",
            "涨跌额": "price_change",
            "换手率": "turnover",
        }
        df = df.rename(columns=rename)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["symbol"] = df["symbol"].astype(str)

        # 保留需要的列
        cols = ["date", "symbol", "open", "high", "low", "close",
                "volume", "amount", "pct_change"]
        df = df[[c for c in cols if c in df.columns]]

        # 写入数据库
        self.db.upsert_daily_bars(df)
        return df

    def fetch_all_symbols(self) -> pd.DataFrame:
        """获取全市场股票列表"""
        try:
            import akshare as ak
        except ImportError:
            return pd.DataFrame()

        try:
            df = ak.stock_info_a_code_name()
        except Exception as e:
            print(f"获取股票列表失败: {e}")
            return pd.DataFrame()

        df = df.rename(columns={
            "代码": "symbol",
            "名称": "name"
        })
        df["symbol"] = df["symbol"].astype(str)
        df["market"] = df["symbol"].apply(
            lambda x: "SH" if x.startswith("6") else "SZ"
        )
        df["list_date"] = None
        df["delist_date"] = None
        df["sector"] = None

        self.db.save_symbols(df[["symbol", "name", "market", "list_date", "delist_date", "sector"]])
        return df

    def fetch_batch(self, symbols: list = None, max_n: int = 100) -> dict:
        """
        批量获取多只股票数据。
        Returns: {symbol: rows_fetched}
        """
        if symbols is None:
            df = self.db.get_symbols()
            symbols = df["symbol"].tolist()[:max_n]

        results = {}
        for i, sym in enumerate(symbols):
            print(f"[{i+1}/{len(symbols)}] 获取 {sym} ...")
            df = self.fetch_daily_bars(sym)
            results[sym] = len(df)

        return results
