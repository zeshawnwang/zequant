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
        增量获取日线数据(仅获取本地该 symbol 最大日期之后的数据)。
        """
        try:
            import akshare as ak
        except ImportError:
            print("请安装 akshare: pip install akshare")
            return pd.DataFrame()

        # 按 symbol 维度确定起始日期,避免用全局最大日期误伤新股票
        if start_date is None:
            local_max = self.db.get_symbol_max_date(symbol)
            if local_max:
                start_date = (local_max + timedelta(days=1)).strftime("%Y%m%d")
            else:
                start_date = "20190101"
        else:
            start_date = start_date.replace("-", "")

        # 起点晚于今日,无增量可取,直接返回
        today = datetime.now().strftime("%Y%m%d")
        if start_date > today:
            return pd.DataFrame()

        try:
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date,
                end_date=today,
                adjust="qfq"
            )
        except Exception as e:
            print(f"获取 {symbol} 失败: {e}")
            return pd.DataFrame()

        if df is None or df.empty:
            return pd.DataFrame()

        # 标准化列名(兼容中英两套)
        rename = {
            "日期": "date", "股票代码": "symbol",
            "开盘": "open", "收盘": "close",
            "最高": "high", "最低": "low",
            "成交量": "volume", "成交额": "amount",
            "涨跌幅": "pct_change", "涨跌额": "price_change",
            "换手率": "turnover",
        }
        df = df.rename(columns=rename)

        # AKShare 有时返回的 symbol 列为空,统一写死
        df["symbol"] = str(symbol).zfill(6)
        df["date"] = pd.to_datetime(df["date"]).dt.date

        cols = ["date", "symbol", "open", "high", "low", "close",
                "volume", "amount", "pct_change"]
        df = df[[c for c in cols if c in df.columns]]

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

        if df is None or df.empty:
            return pd.DataFrame()

        # 兼容新旧两套列名:旧版 "代码/名称",新版 "code/name"
        df = df.rename(columns={
            "代码": "symbol",
            "名称": "name",
            "code": "symbol",
        })

        if "symbol" not in df.columns:
            print(f"未识别的列名: {df.columns.tolist()}")
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
        df["list_date"] = None
        df["delist_date"] = None
        df["sector"] = None

        cols = ["symbol", "name", "market", "list_date", "delist_date", "sector"]
        self.db.save_symbols(df[cols])
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
