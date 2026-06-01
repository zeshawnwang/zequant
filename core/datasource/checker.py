"""
数据质量检查器(Data Quality Checker)
检查并修复常见数据质量问题。
"""
import pandas as pd
from typing import List, Optional


class DataQualityChecker:
    """必须检查的数据质量问题。"""

    def __init__(self, db=None):
        self.db = db

    @staticmethod
    def check(df: pd.DataFrame, symbol: str = None) -> List[str]:
        issues = []

        if df is None or df.empty:
            return ["数据为空"]

        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                continue
            missing = df[col].isnull().sum()
            if missing > 0:
                issues.append(f"{col}缺失值: {missing}条")

        if all(c in df.columns for c in ["high", "low", "close", "open"]):
            invalid = df[
                (df["high"] < df["low"]) |
                (df["high"] < df["close"]) |
                (df["high"] < df["open"]) |
                (df["low"] > df["close"]) |
                (df["low"] > df["open"])
            ]
            if len(invalid) > 0:
                issues.append(f"OHLC逻辑错误: {len(invalid)}条")

        if "volume" in df.columns and "pct_change" in df.columns:
            zero_vol = df[(df["volume"] == 0) & (df["pct_change"] != 0)]
            if len(zero_vol) > 0:
                issues.append(f"零成交异常: {len(zero_vol)}条")

        if "pct_change" in df.columns:
            limit_up = df[df["pct_change"] > 0.11]
            limit_down = df[df["pct_change"] < -0.11]
            if len(limit_up) > 0:
                issues.append(f"异常涨停: {len(limit_up)}条")
            if len(limit_down) > 0:
                issues.append(f"异常跌停: {len(limit_down)}条")

        if "pct_change" in df.columns:
            jump = df[abs(df["pct_change"]) > 0.20]
            if len(jump) > 0:
                issues.append(f"价格断裂>20%: {len(jump)}条")

        return issues

    def check_all(self, start: Optional[str] = None) -> List[str]:
        """批量检查所有活跃标的的数据质量。"""
        if self.db is None:
            return []

        issues = []
        rows = self.db.conn.execute(
            "SELECT DISTINCT symbol FROM daily_bars ORDER BY symbol"
        ).fetchall()
        symbols = [r[0] for r in rows]

        cond = f"WHERE date >= '{start}'" if start else ""
        for sym in symbols[:500]:
            df = self.db.conn.execute(
                f"SELECT * FROM daily_bars WHERE symbol = ? {cond} ORDER BY date",
                [sym]
            ).fetchdf()
            sym_issues = self.check(df, sym)
            for msg in sym_issues:
                issues.append(f"{sym}: {msg}")

        return issues

    @staticmethod
    def clean(df: pd.DataFrame) -> pd.DataFrame:
        """清洗数据，修复可修复的问题。"""
        df = df.copy()

        if all(c in df.columns for c in ["high", "low", "close", "open"]):
            df["high"] = df[["high", "open", "close"]].max(axis=1)
            df["low"] = df[["low", "open", "close"]].min(axis=1)

        return df
