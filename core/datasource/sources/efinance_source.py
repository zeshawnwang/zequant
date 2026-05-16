"""
EFinance 数据源（兜底方案2）。

特点：免费、无需 token、基于东方财富、单次可批量获取。
缺点：依赖东方财富接口，网络限制时不可用。
"""
from __future__ import annotations
import logging
import time
import pandas as pd

from . import BaseDataSource, register_source

logger = logging.getLogger(__name__)

REQUEST_INTERVAL = 0.2


@register_source
class EfinanceSource(BaseDataSource):

    @property
    def name(self) -> str:
        return "efinance"

    def fetch_bars(self, symbol: str, start_date: str,
                   end_date: str, adjust: str = "qfq") -> pd.DataFrame:
        try:
            import efinance as ef
        except ImportError:
            return pd.DataFrame()

        try:
            time.sleep(REQUEST_INTERVAL)
            market = 1 if symbol.startswith(("6", "9")) else 0
            df = ef.stock.get_quote_history(
                symbol, klt=101,  # 日线
                fqt=1 if adjust == "qfq" else 0,  # 前复权
                beg=start_date[:10].replace("-", ""),
                end=end_date[:10].replace("-", ""),
            )
        except Exception as e:
            logger.debug("efinance 获取 %s 失败: %s", symbol, e)
            return pd.DataFrame()

        if df is None or df.empty:
            return pd.DataFrame()

        rename = {
            "日期": "date", "股票名称": "name",
            "开盘": "open", "收盘": "close", "最高": "high", "最低": "low",
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
            import efinance as ef
        except ImportError:
            return pd.DataFrame()
        try:
            df = ef.stock.get_realtime_quotes()
            return df[["股票代码", "股票名称"]].rename(
                columns={"股票代码": "symbol", "股票名称": "name"}
            )
        except Exception as e:
            logger.warning("efinance 获取股票列表失败: %s", e)
            return pd.DataFrame()
