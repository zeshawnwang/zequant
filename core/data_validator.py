"""
数据验证层 (Data Validator)

提供价格校验、缺失数据检测、数据质量指标统计，以及清洗修复能力。
支持传入 Database 实例或 DataFrame。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .database import Database

logger = logging.getLogger(__name__)

# A 股常规交易日历(粗略:周一到周五,不含节假日)
# 如需精确节假日,可后续接入 exchange_calendars
WEEKDAYS = {0, 1, 2, 3, 4}


@dataclass
class ValidationReport:
    """单 symbol 或全局验证报告。"""

    symbol: Optional[str] = None
    passed: bool = True
    issues: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    def add_issue(self, msg: str):
        self.issues.append(msg)
        self.passed = False

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "passed": self.passed,
            "issues": self.issues,
            "metrics": self.metrics,
        }


class DataValidator:
    """日线数据验证器。

    用法示例
    --------
    >>> validator = DataValidator(db)
    >>> report = validator.validate_symbol("000001")
    >>> reports = validator.validate_all()
    >>> clean_df = validator.clean(df)
    """

    # 默认价格/成交量关键列
    PRICE_COLS = ["open", "high", "low", "close"]
    CRITICAL_COLS = ["open", "high", "low", "close", "volume"]

    def __init__(
        self,
        source: Union[Database, pd.DataFrame, None] = None,
        jump_threshold: float = 0.20,
    ):
        """
        Args:
            source: Database 实例或 DataFrame;None 时后续通过 validate(df=...) 传入
            jump_threshold: 单日涨跌幅超过该阈值视为异常跳价(默认 20%)
        """
        self.source = source
        self.jump_threshold = jump_threshold

    # ------------------------------------------------------------------
    # 对外 API
    # ------------------------------------------------------------------

    def validate_symbol(
        self,
        symbol: str,
        df: Optional[pd.DataFrame] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> ValidationReport:
        """验证单只股票的日线数据。"""
        report = ValidationReport(symbol=symbol)

        data = self._resolve_data(df, symbols=[symbol], start_date=start_date, end_date=end_date)
        if data is None or data.empty:
            report.add_issue("数据为空")
            return report

        data = data.copy()
        data["date"] = pd.to_datetime(data["date"])
        data = data.sort_values("date")

        # 1) 价格校验
        self._check_prices(data, report)
        # 2) 缺失检测
        self._check_missing(data, report, symbol)
        # 3) 质量指标
        self._compute_metrics(data, report, symbol)

        if not report.passed:
            logger.warning("[%s] 验证未通过: %s", symbol, "; ".join(report.issues))
        else:
            logger.info("[%s] 验证通过", symbol)

        return report

    def validate_all(
        self,
        df: Optional[pd.DataFrame] = None,
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, ValidationReport]:
        """批量验证,返回 {symbol: report}。"""
        data = self._resolve_data(df, symbols=symbols, start_date=start_date, end_date=end_date)
        if data is None or data.empty:
            logger.warning("validate_all: 无数据")
            return {}

        if "symbol" not in data.columns:
            logger.error("validate_all: 缺少 symbol 列")
            return {}

        reports = {}
        for sym, grp in data.groupby("symbol"):
            reports[sym] = self.validate_symbol(sym, df=grp)
        return reports

    def validate_df(self, df: pd.DataFrame, symbol: Optional[str] = None) -> ValidationReport:
        """直接对 DataFrame 做验证(不依赖 DB),返回单份报告。"""
        if df is None or df.empty:
            report = ValidationReport(symbol=symbol)
            report.add_issue("数据为空")
            return report
        return self.validate_symbol(symbol or "unknown", df=df.copy())

    # ------------------------------------------------------------------
    # 清洗 / 修复
    # ------------------------------------------------------------------

    @staticmethod
    def clean(df: pd.DataFrame) -> pd.DataFrame:
        """清洗常见可修复问题。

        - 修复 OHLC 逻辑: high = max(O,H,C), low = min(O,L,C)
        - 将负价格置为 NaN(不可盲目修复,标记后由上游决定)
        - 去重
        """
        if df is None or df.empty:
            return df

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])

        # 去重
        if "symbol" in df.columns:
            df = df.drop_duplicates(subset=["symbol", "date"])
        else:
            df = df.drop_duplicates(subset=["date"])

        # 修复 OHLC 逻辑
        ohlc = ["open", "high", "low", "close"]
        if all(c in df.columns for c in ohlc):
            df["high"] = df[["open", "high", "close"]].max(axis=1)
            df["low"] = df[["open", "low", "close"]].min(axis=1)

        # 负价格标记为 NaN
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                neg = df[col] < 0
                if neg.any():
                    logger.warning("clean: %s 发现 %d 条负价格,已置 NaN", col, neg.sum())
                    df.loc[neg, col] = np.nan

        return df

    @staticmethod
    def fill_missing_trading_days(
        df: pd.DataFrame,
        freq: str = "B",
    ) -> pd.DataFrame:
        """为每个 symbol 补齐缺失的交易日(价格留 NaN,由下游插值决定)。

        Args:
            df: 必须含 symbol, date, [open, high, low, close, volume]
            freq: 重采样频率,默认 'B' 工作日
        """
        if df is None or df.empty or "symbol" not in df.columns:
            return df

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

        filled = []
        for sym, grp in df.groupby("symbol"):
            grp = grp.sort_index()
            idx = pd.date_range(grp.index.min(), grp.index.max(), freq=freq)
            grp = grp.reindex(idx)
            grp["symbol"] = sym
            filled.append(grp.reset_index().rename(columns={"index": "date"}))

        return pd.concat(filled, ignore_index=True)

    # ------------------------------------------------------------------
    # 内部校验逻辑
    # ------------------------------------------------------------------

    def _check_prices(self, df: pd.DataFrame, report: ValidationReport):
        """价格相关校验。"""
        # 负价格
        for col in self.PRICE_COLS:
            if col not in df.columns:
                continue
            neg = (df[col] < 0).sum()
            if neg:
                report.add_issue(f"负{col}: {neg}条")

        # 零成交量但价格变动非零
        if "volume" in df.columns and "pct_change" in df.columns:
            zero_vol = df[(df["volume"] == 0) & (df["pct_change"] != 0)]
            if len(zero_vol):
                report.add_issue(f"零成交但价格变动: {len(zero_vol)}条")

        # OHLC 一致性
        if all(c in df.columns for c in ["high", "low", "close", "open"]):
            invalid = df[
                (df["high"] < df["low"])
                | (df["high"] < df["close"])
                | (df["high"] < df["open"])
                | (df["low"] > df["close"])
                | (df["low"] > df["open"])
            ]
            if len(invalid):
                report.add_issue(f"OHLC逻辑错误: {len(invalid)}条")

        # 价格跳价
        if "pct_change" in df.columns:
            jump = df[df["pct_change"].abs() > self.jump_threshold]
            if len(jump):
                report.add_issue(f"价格跳变>{self.jump_threshold:.0%}: {len(jump)}条")

    def _check_missing(
        self,
        df: pd.DataFrame,
        report: ValidationReport,
        symbol: str,
    ):
        """缺失数据检测。"""
        # NaN 值
        for col in self.CRITICAL_COLS:
            if col not in df.columns:
                continue
            missing = df[col].isnull().sum()
            if missing:
                report.add_issue(f"{col}缺失值: {missing}条")

        # 缺失交易日(仅粗略检查:工作日连续性)
        if len(df) >= 2:
            expected = pd.date_range(df["date"].min(), df["date"].max(), freq="B")
            actual = pd.to_datetime(df["date"]).sort_values().unique()
            missing_days = len(expected) - len(actual)
            if missing_days > 0:
                report.add_issue(f"缺失交易日: {missing_days}天")
            report.metrics["coverage_ratio"] = len(actual) / len(expected) if len(expected) else 1.0
        else:
            report.metrics["coverage_ratio"] = 1.0 if len(df) else 0.0

    def _compute_metrics(self, df: pd.DataFrame, report: ValidationReport, symbol: str):
        """统计质量指标。"""
        total = len(df)
        report.metrics["total_rows"] = total

        # 异常值计数(价格跳变 + OHLC 错误 + 负价格)
        outlier = 0
        if "pct_change" in df.columns:
            outlier += (df["pct_change"].abs() > self.jump_threshold).sum()
        if all(c in df.columns for c in ["high", "low", "close", "open"]):
            outlier += (
                (df["high"] < df["low"])
                | (df["high"] < df["close"])
                | (df["high"] < df["open"])
                | (df["low"] > df["close"])
                | (df["low"] > df["open"])
            ).sum()
        for col in self.PRICE_COLS:
            if col in df.columns:
                outlier += (df[col] < 0).sum()
        report.metrics["outlier_count"] = int(outlier)

        # 缺失值计数
        missing = 0
        for col in self.CRITICAL_COLS:
            if col in df.columns:
                missing += df[col].isnull().sum()
        report.metrics["missing_count"] = int(missing)

    # ------------------------------------------------------------------
    # 数据解析
    # ------------------------------------------------------------------

    def _resolve_data(
        self,
        df: Optional[pd.DataFrame],
        symbols: Optional[List[str]],
        start_date: Optional[str],
        end_date: Optional[str],
    ) -> Optional[pd.DataFrame]:
        if df is not None:
            return df
        if isinstance(self.source, pd.DataFrame):
            return self.source
        if isinstance(self.source, Database):
            return self.source.get_daily_bars(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
            )
        return None


# ------------------------------------------------------------------
# 便捷函数(兼容旧 DataQualityChecker 调用风格)
# ------------------------------------------------------------------

def validate_data(
    source: Union[Database, pd.DataFrame],
    symbols: Optional[List[str]] = None,
    jump_threshold: float = 0.20,
) -> Dict[str, ValidationReport]:
    """一键验证入口。"""
    validator = DataValidator(source=source, jump_threshold=jump_threshold)
    if isinstance(source, pd.DataFrame) and (symbols is None or "symbol" not in source.columns):
        # 单 DataFrame 无 symbol 列 -> 单份报告
        report = validator.validate_df(source)
        return {"_single": report}
    return validator.validate_all(symbols=symbols)
