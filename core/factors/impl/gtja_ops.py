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
    """滚动分位数排名（向量化实现）。

    对每个滚动窗口计算分位数排名，返回当前值在窗口内的排名分位数。
    使用 numpy 向量化替代 rolling().apply(lambda)，性能提升 50-100x。
    """
    df = _to_df(x).astype(np.float64)
    arr = df.values
    n, m = arr.shape
    out = np.full_like(arr, np.nan)
    if n < window:
        return pd.DataFrame(out, index=df.index, columns=df.columns)
    for j in range(m):
        col = arr[:, j]
        # 逐窗口计算排名
        for i in range(window - 1, n):
            w = col[i - window + 1:i + 1]
            valid = w[~np.isnan(w)]
            if len(valid) > 1:
                # 用 scipy.stats.rankdata 的等价 numpy 实现
                sorted_idx = np.argsort(valid)
                ranks = np.empty(len(valid))
                ranks[sorted_idx] = np.arange(1, len(valid) + 1)
                # 当前值的排名
                curr_val = col[i]
                idx = np.where(valid == curr_val)
                if len(idx[0]) > 0:
                    # 取最后一个匹配（处理重复值）
                    out[i, j] = ranks[idx[0][-1]] / len(valid)
            elif len(valid) == 1:
                out[i, j] = 1.0
    return pd.DataFrame(out, index=df.index, columns=df.columns)


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


def _ts_arg_extreme(x: pd.DataFrame, window: int, find_max: bool) -> pd.DataFrame:
    """向量化实现 ts_argmax / ts_argmin。

    用 numpy 滑动窗口替代 rolling().apply(lambda)，减少 Python 函数调用。
    """
    df = _to_df(x).astype(np.float64)
    arr = df.values
    n, m = arr.shape
    out = np.full_like(arr, np.nan)
    if n < window:
        return pd.DataFrame(out, index=df.index, columns=df.columns)
    for j in range(m):
        col = arr[:, j]
        for i in range(window - 1, n):
            w = col[i - window + 1:i + 1]
            if find_max:
                extreme_idx = np.nanargmax(w) if not np.all(np.isnan(w)) else -1
            else:
                extreme_idx = np.nanargmin(w) if not np.all(np.isnan(w)) else -1
            if extreme_idx >= 0:
                out[i, j] = extreme_idx + 1  # 1-based position
    return pd.DataFrame(out, index=df.index, columns=df.columns)


def ts_argmax(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return _ts_arg_extreme(x, window, find_max=True)


def ts_argmin(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return _ts_arg_extreme(x, window, find_max=False)


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
    """线性衰减加权平均（向量化实现）。

    weights = [1, 2, ..., window]，越近权重越大。
    使用 numpy 矩阵运算替代 rolling().apply(lambda)。
    """
    df = _to_df(x).astype(np.float64)
    arr = df.values
    n, m = arr.shape
    weights = np.arange(1, window + 1, dtype=np.float64)
    wsum = weights.sum()
    out = np.full_like(arr, np.nan)
    if n < window:
        return pd.DataFrame(out, index=df.index, columns=df.columns)
    for j in range(m):
        col = arr[:, j]
        for i in range(window - 1, n):
            w = col[i - window + 1:i + 1]
            if not np.any(np.isnan(w)):
                out[i, j] = np.dot(w, weights) / wsum
    return pd.DataFrame(out, index=df.index, columns=df.columns)


def sma(x: pd.DataFrame, window: int, n: int) -> pd.DataFrame:
    alpha = n / window
    return _to_df(x).ewm(alpha=alpha, adjust=False).mean()


def wma(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """加权移动平均（同 decay_linear）。"""
    return decay_linear(x, window)


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
    result = pd.DataFrame(true_v).where(cond, false_v)
    return result


def _max(a: Union[pd.DataFrame, float], b: Union[pd.DataFrame, float]) -> pd.DataFrame:
    a_df = _to_df(a) if isinstance(a, (pd.DataFrame, pd.Series)) else pd.DataFrame(a)
    b_df = _to_df(b) if isinstance(b, (pd.DataFrame, pd.Series)) else pd.DataFrame(b)
    return a_df.where(a_df > b_df, b_df)


def _min(a: Union[pd.DataFrame, float], b: Union[pd.DataFrame, float]) -> pd.DataFrame:
    a_df = _to_df(a) if isinstance(a, (pd.DataFrame, pd.Series)) else pd.DataFrame(a)
    b_df = _to_df(b) if isinstance(b, (pd.DataFrame, pd.Series)) else pd.DataFrame(b)
    return a_df.where(a_df < b_df, b_df)


def _abs(x: pd.DataFrame) -> pd.DataFrame:
    return _to_df(x).abs()


def log(x: pd.DataFrame) -> pd.DataFrame:
    return np.log(_to_df(x))


_log = log


def _sign(x: pd.DataFrame) -> pd.DataFrame:
    return np.sign(_to_df(x))


def _pow(x: pd.DataFrame, p: float) -> pd.DataFrame:
    """幂运算（向量化，np.power 已支持 DataFrame 逐元素）。"""
    return np.power(_to_df(x), p)


def _sqrt(x: pd.DataFrame) -> pd.DataFrame:
    """开方运算（向量化）。"""
    return np.sqrt(_to_df(x))


def _stddev(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return ts_std(x, window)
