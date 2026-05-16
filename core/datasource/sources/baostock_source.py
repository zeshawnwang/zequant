"""
BaoStock 数据源（兜底方案1）。

特点：免费、无需 token、基于 TCP 长连接、批量拉取稳定。
缺点：需要 login/logout, 字段较少(无 turnover)。
"""
from __future__ import annotations
import logging
import time
from typing import Optional
import pandas as pd

from . import BaseDataSource, register_source

logger = logging.getLogger(__name__)

# 市场代码映射
_BAO_MARKET = {"SH": "sh.", "SZ": "sz.", "BJ": "bj."}


def _infer_market(symbol: str) -> str:
    if symbol.startswith(("6", "9")):
        return "sh."
    if symbol.startswith(("4", "8")):
        return "bj."
    return "sz."


@register_source
class BaostockSource(BaseDataSource):

    def __init__(self):
        self._logged_in = False
        self._lg = None

    def _ensure_login(self):
        if not self._logged_in:
            import baostock as bs
            self._lg = bs.login()
            self._logged_in = True
            logger.debug("baostock 登录: %s", self._lg.error_code)

    def _logout(self):
        if self._logged_in:
            import baostock as bs
            bs.logout()
            self._logged_in = False

    @property
    def name(self) -> str:
        return "baostock"

    def fetch_bars(self, symbol: str, start_date: str,
                   end_date: str, adjust: str = "qfq") -> pd.DataFrame:
        try:
            import baostock as bs
        except ImportError:
            return pd.DataFrame()

        self._ensure_login()
        try:
            code = _infer_market(symbol) + symbol
            adjust_flag = "2" if adjust == "qfq" else "1"  # 2=前复权 1=不复权
            rs = bs.query_history_k_data_plus(
                code,
                "date,open,high,low,close,volume,amount,pctChg",
                start_date=start_date[:10], end_date=end_date[:10],
                frequency="d", adjustflag=adjust_flag,
            )
            data = []
            while (rs.error_code == "0") and rs.next():
                data.append(rs.get_row_data())
            if not data:
                return pd.DataFrame()

            df = pd.DataFrame(data, columns=[
                "date", "open", "high", "low", "close",
                "volume", "amount", "pct_change",
            ])
            df["symbol"] = str(symbol).zfill(6)
            df["date"] = pd.to_datetime(df["date"]).dt.date
            for c in ["open", "high", "low", "close", "volume", "amount"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df["pct_change"] = pd.to_numeric(df["pct_change"], errors="coerce")
            return df[["date", "symbol", "open", "high", "low", "close",
                       "volume", "amount", "pct_change"]]

        except Exception as e:
            logger.debug("baostock 获取 %s 失败: %s", symbol, e)
            return pd.DataFrame()

    def fetch_symbols(self) -> pd.DataFrame:
        try:
            import baostock as bs
        except ImportError:
            return pd.DataFrame()

        self._ensure_login()
        try:
            rs = bs.query_all_stock()
            data = []
            while (rs.error_code == "0") and rs.next():
                row = rs.get_row_data()
                data.append(row)
            if not data:
                return pd.DataFrame()
            df = pd.DataFrame(data, columns=["code", "name", "ipoDate", "outDate", "type", "status"])
            df = df[df["status"] == "1"]  # 只取上市状态
            df["symbol"] = df["code"].str.replace(r'^[a-z]+\.', "", regex=True)
            return df[["symbol", "name"]]
        except Exception as e:
            logger.warning("baostock 获取股票列表失败: %s", e)
            return pd.DataFrame()
