"""
Factor Computation Engine (Polars)
计算技术因子，支持增量更新。
"""
import polars as pl
import pandas as pd
from typing import Optional
from .database import Database


class FactorCalculator:
    """
    基于Polars的因子计算引擎。
    支持增量计算，只需提供原始K线数据。
    """

    @staticmethod
    def compute(df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有技术因子。
        输入: raw daily_bars (pandas DataFrame)
        输出: factor DataFrame
        """
        if df is None or df.empty:
            return pd.DataFrame()

        # 转换为 Polars
        pl_df = pl.from_pandas(df)

        # 先排序（按日期和股票）
        pl_df = pl_df.sort(["symbol", "date"])

        # 计算收益
        pl_df = pl_df.with_columns([
            pl.col("close").pct_change().over("symbol").alias("returns")
        ])

        # 动量因子
        pl_df = pl_df.with_columns([
            pl.col("close").pct_change(5).over("symbol").alias("momentum_5"),
            pl.col("close").pct_change(20).over("symbol").alias("momentum_20"),
        ])

        # RSI
        pl_df = FactorCalculator._rsi(pl_df, "close", 14)

        # MACD
        pl_df = FactorCalculator._macd(pl_df, "close", 12, 26, 9)

        # Bollinger Bands
        pl_df = FactorCalculator._bollinger(pl_df, "close", 20, 2)

        # 成交量因子
        pl_df = FactorCalculator._volume_factors(pl_df, "volume", 20)

        # 波动率
        pl_df = pl_df.with_columns([
            pl.col("returns").rolling_std(20).over("symbol").alias("volatility_20")
        ])

        # 转回 pandas
        result = pl_df.to_pandas()
        return result

    @staticmethod
    def _rsi(df: pl.DataFrame, price_col: str, window: int) -> pl.DataFrame:
        diff = df[price_col].diff()
        gain = diff.clip(lower_bound=0)
        loss = (-diff).clip(lower_bound=0)
        avg_gain = gain.rolling(window, min_periods=1).mean()
        avg_loss = loss.rolling(window, min_periods=1).mean()
        rs = avg_gain / avg_loss
        return df.with_columns([
            (100 - 100 / (1 + rs)).alias(f"rsi_{window}")
        ])

    @staticmethod
    def _macd(df: pl.DataFrame, price_col: str,
              fast: int, slow: int, signal: int) -> pl.DataFrame:
        ema_fast = df[price_col].ewm_span(fast, adjust=False).mean()
        ema_slow = df[price_col].ewm_span(slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm_span(signal, adjust=False).mean()
        macd_hist = macd - macd_signal
        return df.with_columns([
            macd.alias("macd"),
            macd_signal.alias("macd_signal"),
            macd_hist.alias("macd_hist"),
        ])

    @staticmethod
    def _bollinger(df: pl.DataFrame, price_col: str,
                   window: int, std_mult: int) -> pl.DataFrame:
        middle = df[price_col].rolling(window, min_periods=1).mean()
        std = df[price_col].rolling(window, min_periods=1).std()
        upper = middle + std_mult * std
        lower = middle - std_mult * std
        position = (df[price_col] - lower) / (upper - lower)
        return df.with_columns([
            upper.alias("boll_upper"),
            middle.alias("boll_middle"),
            lower.alias("boll_lower"),
            position.alias("boll_position"),
        ])

    @staticmethod
    def _volume_factors(df: pl.DataFrame, vol_col: str, window: int) -> pl.DataFrame:
        avg_vol = df[vol_col].rolling(window, min_periods=1).mean()
        volume_ratio = df[vol_col] / avg_vol
        return df.with_columns([
            volume_ratio.alias("volume_ratio"),
        ])


class FactorRunner:
    """因子批量计算 Runner。"""

    def __init__(self, db: Database):
        self.db = db

    def compute_all(self, symbols: list = None, start_date: str = None) -> pd.DataFrame:
        """
        计算并保存所有因子的增量更新。
        """
        # 获取原始K线数据
        if symbols:
            bars = self.db.get_daily_bars()
            bars = bars[bars["symbol"].isin(symbols)]
        else:
            bars = self.db.get_daily_bars()

        if start_date:
            bars = bars[bars["date"] >= start_date]

        if bars.empty:
            print("无新数据需要计算因子")
            return pd.DataFrame()

        # 计算因子
        factors = FactorCalculator.compute(bars)

        # 保存到数据库
        if not factors.empty:
            self.db.save_factors(factors)

        print(f"因子计算完成: {len(factors)} 条记录")
        return factors
