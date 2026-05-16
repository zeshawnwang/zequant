"""
AKShare 数据源（主数据源）。

特点：数据全、免费、无需 token，但逐只拉取有频率限制。
"""
from __future__ import annotations
import logging
import time
from typing import Optional
import pandas as pd

from . import BaseDataSource, register_source

logger = logging.getLogger(__name__)

# 请求间隔(秒),避免触发反爬
REQUEST_INTERVAL = 0.15


@register_source
class AkshareSource(BaseDataSource):

    @property
    def name(self) -> str:
        return "akshare"

    def fetch_bars(self, symbol: str, start_date: str,
                   end_date: str, adjust: str = "qfq") -> pd.DataFrame:
        try:
            import akshare as ak
        except ImportError:
            logger.warning("akshare 未安装")
            return pd.DataFrame()

        try:
            time.sleep(REQUEST_INTERVAL)
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
                adjust=adjust,
            )
        except Exception as e:
            logger.debug("akshare 获取 %s 失败: %s", symbol, e)
            return pd.DataFrame()

        if df is None or df.empty:
            return pd.DataFrame()

        rename = {
            "日期": "date", "股票代码": "symbol",
            "开盘": "open", "收盘": "close",
            "最高": "high", "最低": "low",
            "成交量": "volume", "成交额": "amount",
            "涨跌幅": "pct_change", "涨跌额": "price_change",
            "换手率": "turnover",
        }
        df = df.rename(columns=rename)
        df["symbol"] = str(symbol).zfill(6)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        cols = ["date", "symbol", "open", "high", "low", "close",
                "volume", "amount", "pct_change"]
        return df[[c for c in cols if c in df.columns]]

    def fetch_symbols(self) -> pd.DataFrame:
        try:
            import akshare as ak
        except ImportError:
            return pd.DataFrame()
        try:
            df = ak.stock_info_a_code_name()
            df = df.rename(columns={"代码": "symbol", "名称": "name",
                                     "code": "symbol"})
            df["symbol"] = df["symbol"].astype(str).str.zfill(6)
            return df[["symbol", "name"]]
        except Exception as e:
            logger.warning("akshare 获取股票列表失败: %s", e)
            return pd.DataFrame()
