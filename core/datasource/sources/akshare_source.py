"""
AKShare 数据源（主数据源）。

特点：数据全、免费、无需 token，但逐只拉取有频率限制。
支持 A 股股票和 LOF/ETF 基金。
"""
from __future__ import annotations
import logging
import time
from typing import Optional, List
import pandas as pd

from . import BaseDataSource, register_source
from .. import FUND_PREFIXES, is_fund_symbol

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

        symbol = str(symbol).zfill(6)

        # LOF/ETF 基金使用不同的 API
        if is_fund_symbol(symbol):
            return self._fetch_fund_bars(symbol, start_date, end_date)

        # A 股股票
        return self._fetch_stock_bars(symbol, start_date, end_date, adjust)

    def _fetch_stock_bars(self, symbol: str, start_date: str,
                          end_date: str, adjust: str = "qfq") -> pd.DataFrame:
        """获取 A 股股票日线数据"""
        try:
            import akshare as ak
        except ImportError:
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
            logger.debug("akshare 获取股票 %s 失败: %s", symbol, e)
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

    def _fetch_fund_bars(self, symbol: str, start_date: str,
                         end_date: str) -> pd.DataFrame:
        """获取 LOF/ETF 基金日线数据（净值数据）"""
        try:
            import akshare as ak
        except ImportError:
            return pd.DataFrame()

        start_date_fmt = start_date.replace("-", "")
        end_date_fmt = end_date.replace("-", "")

        try:
            time.sleep(REQUEST_INTERVAL)
            # 使用东方财富基金净值 API
            df = ak.fund_etf_fund_info_em(fund=symbol, start_date=start_date_fmt, end_date=end_date_fmt)
        except Exception as e:
            logger.debug("akshare 获取基金 %s 失败: %s", symbol, e)
            return pd.DataFrame()

        if df is None or df.empty:
            return pd.DataFrame()

        # 标准化字段名
        rename_map = {
            "净值日期": "date",
            "单位净值": "close",
            "累计净值": "accumulated_nav",
            "日增长率": "pct_change",
            "申购状态": "subscribe_status",
            "赎回状态": "redeem_status",
        }
        df = df.rename(columns=rename_map)

        # 确保必需列存在
        if "date" not in df.columns or "close" not in df.columns:
            return pd.DataFrame()

        df["symbol"] = str(symbol).zfill(6)
        df["date"] = pd.to_datetime(df["date"]).dt.date

        # 处理数值列
        for col in ["close", "pct_change"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # 处理涨跌幅（可能是字符串如 "-2.07%"，需要提取数值）
        if "pct_change" in df.columns and df["pct_change"].dtype == object:
            df["pct_change"] = df["pct_change"].str.rstrip("%").astype(float)

        # 基金净值数据有限，设置默认值
        df["open"] = df["close"]  # 开盘价用收盘价代替
        df["high"] = df["close"]
        df["low"] = df["close"]
        df["volume"] = 0
        df["amount"] = 0

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

    def fetch_fund_symbols(self) -> pd.DataFrame:
        """获取 LOF/ETF 基金列表"""
        try:
            import akshare as ak
        except ImportError:
            return pd.DataFrame()

        fund_list = []

        # 获取 ETF 基金列表
        try:
            time.sleep(REQUEST_INTERVAL)
            etf_df = ak.fund_etf_spot_em()
            if etf_df is not None and not etf_df.empty:
                etf_df = etf_df.rename(columns={
                    "基金代码": "symbol", "基金简称": "name"
                })
                etf_df["symbol"] = etf_df["symbol"].astype(str).str.zfill(6)
                etf_df["type"] = "ETF"
                fund_list.append(etf_df[["symbol", "name", "type"]])
        except Exception as e:
            logger.debug("获取 ETF 列表失败: %s", e)

        # 获取 LOF 基金列表
        try:
            time.sleep(REQUEST_INTERVAL)
            lof_df = ak.fund_lof_spot_em()
            if lof_df is not None and not lof_df.empty:
                lof_df = lof_df.rename(columns={
                    "基金代码": "symbol", "基金简称": "name"
                })
                lof_df["symbol"] = lof_df["symbol"].astype(str).str.zfill(6)
                lof_df["type"] = "LOF"
                fund_list.append(lof_df[["symbol", "name", "type"]])
        except Exception as e:
            logger.debug("获取 LOF 列表失败: %s", e)

        if not fund_list:
            return pd.DataFrame()

        combined = pd.concat(fund_list, ignore_index=True)
        return combined[["symbol", "name", "type"]]

    def fetch_batch_fund_bars(self, symbols: List[str], start_date: str,
                               end_date: str) -> dict:
        """批量获取基金日线数据"""
        results = {}
        for sym in symbols:
            df = self._fetch_fund_bars(sym, start_date, end_date)
            results[sym] = len(df)
        return results
