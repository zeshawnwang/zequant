"""
Factor Computation Engine (Polars)
计算技术因子,支持按 symbol 分组的增量计算。

修复说明(相对旧版):
- 旧版用 `df[col].rolling(...)` / `df[col].ewm_span(...)` 不按 symbol 分组,
  导致跨股票的值被串在一起计算 EMA / rolling,因子严重污染。
- 新版所有窗口函数都走 `pl.col(...).rolling_*(...).over("symbol")` 或
  `pl.col(...).ewm_mean(..., adjust=False).over("symbol")`,按股票独立计算。
"""
import polars as pl
import pandas as pd
from typing import List

from .database import Database


class FactorCalculator:
    """基于 Polars 的因子计算引擎。"""

    @staticmethod
    def compute(df: pd.DataFrame) -> pd.DataFrame:
        """
        输入: raw daily_bars (pandas.DataFrame),需含 symbol/date/close/volume
        输出: pandas.DataFrame,包含所有因子列
        """
        if df is None or df.empty:
            return pd.DataFrame()

        needed = {"symbol", "date", "close", "volume"}
        missing = needed - set(df.columns)
        if missing:
            raise ValueError(f"compute() missing columns: {missing}")

        # 数值列强转 float,避免 DECIMAL 在 polars 里出现意外行为
        pl_df = pl.from_pandas(df).with_columns([
            pl.col("close").cast(pl.Float64),
            pl.col("volume").cast(pl.Float64),
        ]).sort(["symbol", "date"])

        # ===== 基础收益 & 动量 =====
        pl_df = pl_df.with_columns([
            pl.col("close").pct_change().over("symbol").alias("returns"),
            pl.col("close").pct_change(5).over("symbol").alias("momentum_5"),
            pl.col("close").pct_change(20).over("symbol").alias("momentum_20"),
        ])

        # ===== RSI(14) =====
        pl_df = pl_df.with_columns([
            pl.col("close").diff().over("symbol").alias("_diff")
        ])
        pl_df = pl_df.with_columns([
            pl.when(pl.col("_diff") > 0).then(pl.col("_diff")).otherwise(0.0).alias("_gain"),
            pl.when(pl.col("_diff") < 0).then(-pl.col("_diff")).otherwise(0.0).alias("_loss"),
        ])
        pl_df = pl_df.with_columns([
            pl.col("_gain").rolling_mean(window_size=14, min_periods=1).over("symbol").alias("_avg_gain"),
            pl.col("_loss").rolling_mean(window_size=14, min_periods=1).over("symbol").alias("_avg_loss"),
        ])
        pl_df = pl_df.with_columns([
            (100 - 100 / (1 + pl.col("_avg_gain") / pl.when(pl.col("_avg_loss") == 0).then(1e-12).otherwise(pl.col("_avg_loss"))))
            .alias("rsi_14")
        ])

        # ===== MACD(12,26,9) =====
        pl_df = pl_df.with_columns([
            pl.col("close").ewm_mean(span=12, adjust=False).over("symbol").alias("_ema_fast"),
            pl.col("close").ewm_mean(span=26, adjust=False).over("symbol").alias("_ema_slow"),
        ])
        pl_df = pl_df.with_columns([
            (pl.col("_ema_fast") - pl.col("_ema_slow")).alias("macd")
        ])
        pl_df = pl_df.with_columns([
            pl.col("macd").ewm_mean(span=9, adjust=False).over("symbol").alias("macd_signal"),
        ])
        pl_df = pl_df.with_columns([
            (pl.col("macd") - pl.col("macd_signal")).alias("macd_hist")
        ])

        # ===== Bollinger(20, 2) =====
        pl_df = pl_df.with_columns([
            pl.col("close").rolling_mean(window_size=20, min_periods=1).over("symbol").alias("boll_middle"),
            pl.col("close").rolling_std(window_size=20, min_periods=2).over("symbol").alias("_boll_std"),
        ])
        pl_df = pl_df.with_columns([
            (pl.col("boll_middle") + 2 * pl.col("_boll_std")).alias("boll_upper"),
            (pl.col("boll_middle") - 2 * pl.col("_boll_std")).alias("boll_lower"),
        ])
        pl_df = pl_df.with_columns([
            pl.when((pl.col("boll_upper") - pl.col("boll_lower")) == 0)
            .then(None)
            .otherwise(
                (pl.col("close") - pl.col("boll_lower"))
                / (pl.col("boll_upper") - pl.col("boll_lower"))
            )
            .alias("boll_position")
        ])

        # ===== 成交量比 =====
        pl_df = pl_df.with_columns([
            pl.col("volume").rolling_mean(window_size=20, min_periods=1).over("symbol").alias("_avg_vol")
        ])
        pl_df = pl_df.with_columns([
            pl.when(pl.col("_avg_vol") == 0)
            .then(None)
            .otherwise(pl.col("volume") / pl.col("_avg_vol"))
            .alias("volume_ratio")
        ])

        # ===== 波动率(20 日收益标准差) =====
        pl_df = pl_df.with_columns([
            pl.col("returns").rolling_std(window_size=20, min_periods=5).over("symbol").alias("volatility_20")
        ])

        # 丢掉中间列
        drop_cols = [c for c in ("_diff", "_gain", "_loss", "_avg_gain", "_avg_loss",
                                 "_ema_fast", "_ema_slow", "_boll_std", "_avg_vol")
                     if c in pl_df.columns]
        if drop_cols:
            pl_df = pl_df.drop(drop_cols)

        # to_pandas() 在 Polars 中默认走 pyarrow,缺它时回退到字典构造
        try:
            result = pl_df.to_pandas()
        except (ModuleNotFoundError, ImportError):
            result = pd.DataFrame({c: pl_df[c].to_list() for c in pl_df.columns})

        # 清洗:把 ±inf 统一替换为 NaN(DuckDB DECIMAL 列不接受 inf)
        import numpy as _np
        numeric_cols = result.select_dtypes(include=[_np.floating]).columns
        if len(numeric_cols) > 0:
            result[numeric_cols] = result[numeric_cols].replace(
                [_np.inf, -_np.inf], _np.nan
            )

        return result

class FactorRunner:
    """因子批量计算 Runner。"""

    def __init__(self, db: Database):
        self.db = db

    def compute_all(self, symbols: List[str] = None,
                    start_date: str = None) -> pd.DataFrame:
        """
        计算并保存所有因子。
        - symbols: 指定股票列表,为 None 时取全量
        - start_date: 起始日期,为了保证 rolling 窗口,计算起点会向前多取约 40 日
        """
        bars = self.db.get_daily_bars(start_date=start_date)
        if symbols:
            bars = bars[bars["symbol"].isin(symbols)]

        if bars is None or bars.empty:
            print("无 K 线数据,请先运行 fetch_data.py")
            return pd.DataFrame()

        factors = FactorCalculator.compute(bars)
        if factors.empty:
            return factors

        # 如果指定了 start_date,保存时按此截断(前面多取的预热数据不入库)
        if start_date:
            start_ts = pd.Timestamp(start_date)
            date_col = factors["date"]
            # date 列可能是 datetime64[ns] 或 python date,统一成 Timestamp 再比较
            if pd.api.types.is_datetime64_any_dtype(date_col):
                factors = factors[date_col >= start_ts]
            else:
                factors = factors[pd.to_datetime(date_col) >= start_ts]

        self.db.save_factors(factors)
        print(f"因子计算完成: {len(factors)} 条记录")
        return factors