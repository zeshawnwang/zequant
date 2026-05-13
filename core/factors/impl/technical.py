"""
传统技术因子(配合 FactorHub 注册中心)

把原 core/factor.py:FactorCalculator.compute 一次性算 13 列的「单体引擎」拆解为
13 个独立的 @register_factor 函数,接入 FactorHub 单一注册体系。

设计要点:
- 每个因子函数签名: (ctx: FactorContext) -> wide DataFrame (index=date, columns=symbol)
- 输入字段最小化(close/volume),pivot 在 FactorHub._build_context 中一次性完成
- 复合因子(MACD 三件套、布林三件套)间通过 ctx.cache 共享中间结果(EMA/MA/STD),
  保证一次 compute_all 调用内不重复计算
- 所有 rolling/EMA 直接走 pandas 列向量,因为 ctx.* 已经按 symbol 列 pivot 过,
  自然实现「按股票独立计算」的语义,不再需要 .over("symbol")

调用:
    import factors.technical               # 触发 @register_factor 副作用
    from core.factor_hub import FactorHub
    long_df = FactorHub.compute_all(bars, names=FactorHub.list_by_category("technical"))
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from core.factors.base.factor_hub import register_factor, FactorContext


# ===================================================================
#  共享中间量:走 ctx.cache,避免重复计算
# ===================================================================

def _ema(df: pd.DataFrame, span: int) -> pd.DataFrame:
    """按 symbol 独立的 EMA(ctx.* 已是 wide 表,直接列向量计算即可)。"""
    return df.ewm(span=span, adjust=False).mean()


def _get_ema(ctx: FactorContext, span: int) -> pd.DataFrame:
    key = f"ema_close_{span}"
    if key not in ctx.cache:
        ctx.cache[key] = _ema(ctx.close, span)
    return ctx.cache[key]


def _get_ma(ctx: FactorContext, window: int, source: str = "close") -> pd.DataFrame:
    key = f"ma_{source}_{window}"
    if key not in ctx.cache:
        src = getattr(ctx, source)
        ctx.cache[key] = src.rolling(window=window, min_periods=1).mean()
    return ctx.cache[key]


def _get_std(ctx: FactorContext, window: int, source: str = "close",
             min_periods: int = 2) -> pd.DataFrame:
    key = f"std_{source}_{window}"
    if key not in ctx.cache:
        src = getattr(ctx, source)
        ctx.cache[key] = src.rolling(window=window, min_periods=min_periods).std()
    return ctx.cache[key]


def _get_returns(ctx: FactorContext) -> pd.DataFrame:
    if "returns_d1" not in ctx.cache:
        ctx.cache["returns_d1"] = ctx.close.pct_change()
    return ctx.cache["returns_d1"]


# ===================================================================
#  收益与动量
# ===================================================================

@register_factor("returns", category="technical", requires=["close"],
                 description="日收益率 = close.pct_change()")
def f_returns(ctx: FactorContext) -> pd.DataFrame:
    return _get_returns(ctx)


@register_factor("momentum_5", category="technical", requires=["close"],
                 description="5 日动量 = close.pct_change(5)")
def f_momentum_5(ctx: FactorContext) -> pd.DataFrame:
    return ctx.close.pct_change(5)


@register_factor("momentum_20", category="technical", requires=["close"],
                 description="20 日动量 = close.pct_change(20)")
def f_momentum_20(ctx: FactorContext) -> pd.DataFrame:
    return ctx.close.pct_change(20)


# ===================================================================
#  RSI(14)
# ===================================================================

@register_factor("rsi_14", category="technical", requires=["close"],
                 description="经典 RSI(14):用 14 日均值的涨跌幅之比")
def f_rsi_14(ctx: FactorContext) -> pd.DataFrame:
    diff = ctx.close.diff()
    gain = diff.clip(lower=0.0)
    loss = (-diff).clip(lower=0.0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean().replace(0, 1e-12)
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


# ===================================================================
#  MACD 三件套(12, 26, 9)
# ===================================================================

def _macd_line(ctx: FactorContext) -> pd.DataFrame:
    if "macd_line" not in ctx.cache:
        ctx.cache["macd_line"] = _get_ema(ctx, 12) - _get_ema(ctx, 26)
    return ctx.cache["macd_line"]


def _macd_signal_line(ctx: FactorContext) -> pd.DataFrame:
    if "macd_signal" not in ctx.cache:
        ctx.cache["macd_signal"] = _macd_line(ctx).ewm(span=9, adjust=False).mean()
    return ctx.cache["macd_signal"]


@register_factor("macd", category="technical", requires=["close"],
                 description="MACD = EMA(close,12) - EMA(close,26)")
def f_macd(ctx: FactorContext) -> pd.DataFrame:
    return _macd_line(ctx)


@register_factor("macd_signal", category="technical", requires=["close"],
                 description="MACD signal = EMA(MACD, 9)")
def f_macd_signal(ctx: FactorContext) -> pd.DataFrame:
    return _macd_signal_line(ctx)


@register_factor("macd_hist", category="technical", requires=["close"],
                 description="MACD 柱 = MACD - signal")
def f_macd_hist(ctx: FactorContext) -> pd.DataFrame:
    return _macd_line(ctx) - _macd_signal_line(ctx)


# ===================================================================
#  Bollinger(20, 2)
# ===================================================================

@register_factor("boll_middle", category="technical", requires=["close"],
                 description="布林带中轨 = MA(close,20)")
def f_boll_middle(ctx: FactorContext) -> pd.DataFrame:
    return _get_ma(ctx, 20)


@register_factor("boll_upper", category="technical", requires=["close"],
                 description="布林带上轨 = MA(close,20) + 2*STD(close,20)")
def f_boll_upper(ctx: FactorContext) -> pd.DataFrame:
    return _get_ma(ctx, 20) + 2 * _get_std(ctx, 20)


@register_factor("boll_lower", category="technical", requires=["close"],
                 description="布林带下轨 = MA(close,20) - 2*STD(close,20)")
def f_boll_lower(ctx: FactorContext) -> pd.DataFrame:
    return _get_ma(ctx, 20) - 2 * _get_std(ctx, 20)


@register_factor("boll_position", category="technical", requires=["close"],
                 description="价格在布林带的相对位置 ∈ [0,1]")
def f_boll_position(ctx: FactorContext) -> pd.DataFrame:
    mid = _get_ma(ctx, 20)
    std = _get_std(ctx, 20)
    upper = mid + 2 * std
    lower = mid - 2 * std
    width = (upper - lower).replace(0, np.nan)
    return (ctx.close - lower) / width


# ===================================================================
#  量价
# ===================================================================

@register_factor("volume_ratio", category="technical", requires=["volume"],
                 description="量比 = volume / MA(volume, 20)")
def f_volume_ratio(ctx: FactorContext) -> pd.DataFrame:
    avg_vol = _get_ma(ctx, 20, source="volume")
    return ctx.volume / avg_vol.replace(0, np.nan)


# ===================================================================
#  波动率
# ===================================================================

@register_factor("volatility_20", category="technical", requires=["close"],
                 description="20 日收益率标准差")
def f_volatility_20(ctx: FactorContext) -> pd.DataFrame:
    ret = _get_returns(ctx)
    return ret.rolling(window=20, min_periods=5).std()