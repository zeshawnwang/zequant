"""
GTJA 191 因子算子库

国泰君安 191 Alpha 因子的计算所需的基础算子，基于 pandas DataFrame（宽表格式）。

每个算子接收一个或多个 DataFrame（index=date, columns=symbol），
返回相同形状的 DataFrame。

与 alpha101_ops.py 的主要区别：
- gtja191 使用 mfirst 而不是 mlag（语义相同）
- gtja191 的 rowRank 使用 pct 参数控制返回形式
"""
from typing import Union
import numpy as np
import pandas as pd


def _to_df(x) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x
    return pd.DataFrame(x)


def rank(x: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
    return _to_df(x).rank(axis=1, pct=True, method="average")


def ts_rank(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return _to_df(x).rank(axis=0, pct=True, method="average").rolling(window).apply(lambda s: s.iloc[-1], raw=False)


def delay(x: pd.DataFrame, period: int = 1) -> pd.DataFrame:
    return _to_df(x).shift(period)


def delta(x: pd.DataFrame, period: int = 1) -> pd.DataFrame:
    return _to_df(x).diff(period)


def ts_sum(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return _to_df(x).rolling(window).sum()


def ts_mean(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return _to_df(x).rolling(window).mean()


def ts_std(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return _to_df(x).rolling(window).std()


def ts_max(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return _to_df(x).rolling(window).max()


def ts_min(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return _to_df(x).rolling(window).min()


def ts_argmax(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return _to_df(x).rolling(window).apply(lambda s: s.argmax() + 1, raw=False)


def ts_argmin(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return _to_df(x).rolling(window).apply(lambda s: s.argmin() + 1, raw=False)


def correlation(x: pd.DataFrame, y: pd.DataFrame, window: int) -> pd.DataFrame:
    return _to_df(x).rolling(window).corr(_to_df(y))


def covariance(x: pd.DataFrame, y: pd.DataFrame, window: int) -> pd.DataFrame:
    return _to_df(x).rolling(window).cov(_to_df(y))


def scale(x: pd.DataFrame, a: float = 1.0) -> pd.DataFrame:
    df = _to_df(x)
    return df.apply(lambda col: col / col.abs().sum() * a, axis=1)


def signed_power(x: pd.DataFrame, a: float) -> pd.DataFrame:
    return np.sign(_to_df(x)) * np.power(_to_df(x).abs(), a)


def decay_linear(x: pd.DataFrame, window: int) -> pd.DataFrame:
    df = _to_df(x)
    weights = np.arange(1, window + 1)
    return df.rolling(window).apply(
        lambda s: np.dot(s, weights) / weights.sum() if not np.isnan(s).any() else np.nan,
        raw=True
    )


def sma(x: pd.DataFrame, window: int, n: int) -> pd.DataFrame:
    alpha = n / window
    return _to_df(x).ewm(alpha=alpha, adjust=False).mean()


def wma(x: pd.DataFrame, window: int) -> pd.DataFrame:
    df = _to_df(x)
    weights = np.arange(1, window + 1)
    return df.rolling(window).apply(
        lambda s: np.dot(s, weights) / weights.sum() if not np.isnan(s).any() else np.nan,
        raw=True
    )


def ts_correlation(x: pd.DataFrame, y: pd.DataFrame, window: int) -> pd.DataFrame:
    return correlation(x, y, window)


def row_rank(x: pd.DataFrame, pct: bool = True) -> pd.DataFrame:
    return _to_df(x).rank(axis=1, pct=pct, method="average")


def row_max(x: pd.DataFrame) -> pd.DataFrame:
    return _to_df(x).max(axis=1)


def row_min(x: pd.DataFrame) -> pd.DataFrame:
    return _to_df(x).min(axis=1)


def row_sum(x: pd.DataFrame) -> pd.DataFrame:
    return _to_df(x).sum(axis=1)


def row_mean(x: pd.DataFrame) -> pd.DataFrame:
    return _to_df(x).mean(axis=1)


def sumac(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return _to_df(x).rolling(window).sum()


def _iif(condition: pd.DataFrame, true_val: Union[pd.DataFrame, float],
         false_val: Union[pd.DataFrame, float]) -> pd.DataFrame:
    cond = _to_df(condition)
    true_v = _to_df(true_val) if isinstance(true_val, (pd.DataFrame, pd.Series)) else true_val
    false_v = _to_df(false_val) if isinstance(false_val, (pd.DataFrame, pd.Series)) else false_val
    result = cond.astype(float).where(cond, false_v)
    result = result.where(cond, true_v)
    return result


def _max(a: Union[pd.DataFrame, float], b: Union[pd.DataFrame, float]) -> pd.DataFrame:
    a_df = _to_df(a) if isinstance(a, (pd.DataFrame, pd.Series)) else pd.DataFrame(a)
    b_df = _to_df(b) if isinstance(b, (pd.DataFrame, pd.Series)) else pd.DataFrame(b)
    return a_df.bfill().fillna(a_df).where(a_df > b_df, b_df)


def _min(a: Union[pd.DataFrame, float], b: Union[pd.DataFrame, float]) -> pd.DataFrame:
    a_df = _to_df(a) if isinstance(a, (pd.DataFrame, pd.Series)) else pd.DataFrame(a)
    b_df = _to_df(b) if isinstance(b, (pd.DataFrame, pd.Series)) else pd.DataFrame(b)
    return a_df.bfill().fillna(a_df).where(a_df < b_df, b_df)


def _abs(x: pd.DataFrame) -> pd.DataFrame:
    return _to_df(x).abs()


def log(x: pd.DataFrame) -> pd.DataFrame:
    return np.log(_to_df(x))


def _log(x: pd.DataFrame) -> pd.DataFrame:
    return log(x)


def _sign(x: pd.DataFrame) -> pd.DataFrame:
    return np.sign(_to_df(x))


def _pow(x: pd.DataFrame, p: float) -> pd.DataFrame:
    return _to_df(x).apply(lambda col: np.power(col, p))


def _sqrt(x: pd.DataFrame) -> pd.DataFrame:
    return _to_df(x).apply(lambda col: np.sqrt(col))


def _stddev(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return ts_std(x, window)
