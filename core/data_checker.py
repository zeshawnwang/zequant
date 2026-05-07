"""
Data Quality Checker
检查并修复常见数据质量问题。
"""
import pandas as pd
from typing import List


class DataQualityChecker:
    """必须检查的数据质量问题。"""

    @staticmethod
    def check(df: pd.DataFrame, symbol: str = None) -> List[str]:
        issues = []

        if df is None or df.empty:
            return ["数据为空"]

        # 1. 缺失值检查
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                continue
            missing = df[col].isnull().sum()
            if missing > 0:
                issues.append(f"{col}缺失值: {missing}条")

        # 2. OHLC逻辑错误
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

        # 3. 零成交（排除停牌）
        if "volume" in df.columns and "pct_change" in df.columns:
            zero_vol = df[(df["volume"] == 0) & (df["pct_change"] != 0)]
            if len(zero_vol) > 0:
                issues.append(f"零成交异常: {len(zero_vol)}条")

        # 4. 涨跌停检测
        if "pct_change" in df.columns:
            limit_up = df[df["pct_change"] > 0.11]
            limit_down = df[df["pct_change"] < -0.11]
            if len(limit_up) > 0:
                issues.append(f"异常涨停: {len(limit_up)}条")
            if len(limit_down) > 0:
                issues.append(f"异常跌停: {len(limit_down)}条")

        # 5. 价格断裂（单日涨跌幅>20%）
        if "pct_change" in df.columns:
            jump = df[abs(df["pct_change"]) > 0.20]
            if len(jump) > 0:
                issues.append(f"价格断裂>20%: {len(jump)}条")

        return issues

    @staticmethod
    def clean(df: pd.DataFrame) -> pd.DataFrame:
        """清洗数据，修复可修复的问题。"""
        df = df.copy()

        # 修复OHLC逻辑
        if all(c in df.columns for c in ["high", "low", "close", "open"]):
            df["high"] = df[["high", "open", "close"]].max(axis=1)
            df["low"] = df[["low", "open", "close"]].min(axis=1)

        return df
