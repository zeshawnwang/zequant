"""
职业投资者技术分析方法论因子库(配合 FactorHub 注册中心)

根据知乎文章《职业投资者如何分析股票选股》的技术分析方法论，实现以下因子类别:
  1. 均线相关因子: MA5/20/60/120, 多头排列, 斜率角度, 粘合度等
  2. 量价因子: 突破日量比, 缩量程度, 持续放量, 量价背离
  3. K线形态因子: 大阳线, 大阴线, 长上/下影线, 十字星等
  4. MACD形态因子: 零轴位置, 金叉/死叉, 圆弧底
  5. 箱体突破因子: 箱体宽度, 横盘天数, 突破信号, 突破强度
  6. 筹码集中度因子: 换手率, 量能萎缩, 筹码集中度
  7. β系数因子: 20日/60日β系数

调用:
    import factors.technical_analysis  # 触发 @register_factor 副作用
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.factor_hub import register_factor, FactorContext


# ===================================================================
#  共享中间量
# ===================================================================

def _ema(df: pd.DataFrame, span: int) -> pd.DataFrame:
    """EMA 计算。"""
    return df.ewm(span=span, adjust=False).mean()


def _get_ma(ctx: FactorContext, window: int, source: str = "close") -> pd.DataFrame:
    """获取或缓存 MA。"""
    key = f"ma_{source}_{window}"
    if key not in ctx.cache:
        src = getattr(ctx, source)
        ctx.cache[key] = src.rolling(window=window, min_periods=1).mean()
    return ctx.cache[key]


def _get_std(ctx: FactorContext, window: int, source: str = "close",
             min_periods: int = 2) -> pd.DataFrame:
    """获取或缓存 STD。"""
    key = f"std_{source}_{window}"
    if key not in ctx.cache:
        src = getattr(ctx, source)
        ctx.cache[key] = src.rolling(window=window, min_periods=min_periods).std()
    return ctx.cache[key]


def _get_returns(ctx: FactorContext) -> pd.DataFrame:
    """获取或缓存日收益率。"""
    if "returns_d1" not in ctx.cache:
        ctx.cache["returns_d1"] = ctx.close.pct_change()
    return ctx.cache["returns_d1"]


def _macd_line(ctx: FactorContext) -> pd.DataFrame:
    """MACD 快线。"""
    if "macd_line_ta" not in ctx.cache:
        ctx.cache["macd_line_ta"] = _ema(ctx.close, 12) - _ema(ctx.close, 26)
    return ctx.cache["macd_line_ta"]


def _macd_signal(ctx: FactorContext) -> pd.DataFrame:
    """MACD 信号线。"""
    if "macd_signal_ta" not in ctx.cache:
        ctx.cache["macd_signal_ta"] = _macd_line(ctx).ewm(span=9, adjust=False).mean()
    return ctx.cache["macd_signal_ta"]


def _linear_regression_slope(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """用 rolling apply 计算线性回归斜率，避免循环。"""
    def _slope(values: np.ndarray) -> float:
        values = np.asarray(values, dtype=float)
        valid = ~np.isnan(values)
        if valid.sum() < 3:
            return np.nan
        x = np.arange(len(values))[valid]
        y = values[valid]
        n = len(x)
        x_mean = x.mean()
        y_mean = y.mean()
        ss_xx = ((x - x_mean) ** 2).sum()
        if ss_xx == 0:
            return 0.0
        return ((x - x_mean) * (y - y_mean)).sum() / ss_xx

    return df.rolling(window=window, min_periods=max(3, window // 2)).apply(
        _slope, raw=True
    )


def _market_proxy(ctx: FactorContext) -> pd.DataFrame:
    """市场代理: 用截面收盘价均值作为市场指数。"""
    if "market_proxy" not in ctx.cache:
        ctx.cache["market_proxy"] = ctx.close.mean(axis=1)
    return ctx.cache["market_proxy"]


# ===================================================================
#  1. 均线相关因子
# ===================================================================

@register_factor("ma5", category="technical", requires=["close"],
                 description="5日均线")
def f_ma5(ctx: FactorContext) -> pd.DataFrame:
    """5日简单移动平均线。"""
    return _get_ma(ctx, 5)


@register_factor("ma20", category="technical", requires=["close"],
                 description="20日均线")
def f_ma20(ctx: FactorContext) -> pd.DataFrame:
    """20日简单移动平均线。"""
    return _get_ma(ctx, 20)


@register_factor("ma60", category="technical", requires=["close"],
                 description="60日均线")
def f_ma60(ctx: FactorContext) -> pd.DataFrame:
    """60日简单移动平均线。"""
    return _get_ma(ctx, 60)


@register_factor("ma120", category="technical", requires=["close"],
                 description="120日均线")
def f_ma120(ctx: FactorContext) -> pd.DataFrame:
    """120日简单移动平均线。"""
    return _get_ma(ctx, 120)


@register_factor("ma_alignment_score", category="technical", requires=["close"],
                 description="均线多头排列得分")
def ma_alignment_score(ctx: FactorContext) -> pd.DataFrame:
    """均线多头排列得分: MA5>MA20>MA60 为1, 反之为-1, 部分满足为0。"""
    close = ctx.close
    ma5 = _get_ma(ctx, 5)
    ma20 = _get_ma(ctx, 20)
    ma60 = _get_ma(ctx, 60)

    score = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    score[(ma5 > ma20) & (ma20 > ma60)] = 1.0
    score[(ma5 < ma20) & (ma20 < ma60)] = -1.0
    return score


@register_factor("ma_angle_20", category="technical", requires=["close"],
                 description="MA20的斜率角度(度)")
def ma_angle_20(ctx: FactorContext) -> pd.DataFrame:
    """MA20的斜率角度: 用最近20日线性回归斜率取反正切近似。"""
    ma20 = _get_ma(ctx, 20)
    slope = _linear_regression_slope(ma20, 20)
    return np.degrees(np.arctan(slope / ma20.replace(0, np.nan)))


@register_factor("ma_angle_60", category="technical", requires=["close"],
                 description="MA60的斜率角度(度)")
def ma_angle_60(ctx: FactorContext) -> pd.DataFrame:
    """MA60的斜率角度: 用最近60日线性回归斜率取反正切近似。"""
    ma60 = _get_ma(ctx, 60)
    slope = _linear_regression_slope(ma60, 60)
    return np.degrees(np.arctan(slope / ma60.replace(0, np.nan)))


@register_factor("ma_convergence", category="technical", requires=["close"],
                 description="均线粘合度")
def ma_convergence(ctx: FactorContext) -> pd.DataFrame:
    """均线粘合度: MA5/MA20/MA60的标准差除以均值, 越小越粘合。"""
    ma5 = _get_ma(ctx, 5)
    ma20 = _get_ma(ctx, 20)
    ma60 = _get_ma(ctx, 60)

    ma_stack = pd.concat([ma5, ma20, ma60], axis=0, keys=["ma5", "ma20", "ma60"])
    mean_val = ma_stack.groupby(level=1).mean()
    std_val = ma_stack.groupby(level=1).std()
    return std_val / mean_val.replace(0, np.nan)


def close_index_for(ctx: FactorContext) -> pd.DatetimeIndex:
    """获取 close 的 index。"""
    return ctx.close.index


@register_factor("price_above_ma60", category="technical", requires=["close"],
                 description="股价是否在60日均线上方")
def price_above_ma60(ctx: FactorContext) -> pd.DataFrame:
    """股价是否在60日均线上方, 1表示上方, 0表示下方。"""
    close = ctx.close
    ma60 = _get_ma(ctx, 60)
    return (close > ma60).astype(float)


@register_factor("ma60_trend", category="technical", requires=["close"],
                 description="MA60的趋势方向")
def ma60_trend(ctx: FactorContext) -> pd.DataFrame:
    """MA60的趋势方向: 斜率为正返回1, 为负返回-1。"""
    ma60 = _get_ma(ctx, 60)
    slope = _linear_regression_slope(ma60, 20)
    result = pd.DataFrame(0.0, index=ma60.index, columns=ma60.columns)
    result[slope > 0] = 1.0
    result[slope < 0] = -1.0
    return result


@register_factor("ma120_trend", category="technical", requires=["close"],
                 description="MA120的趋势方向")
def ma120_trend(ctx: FactorContext) -> pd.DataFrame:
    """MA120的趋势方向: 斜率为正返回1, 为负返回-1。"""
    ma120 = _get_ma(ctx, 120)
    slope = _linear_regression_slope(ma120, 20)
    result = pd.DataFrame(0.0, index=ma120.index, columns=ma120.columns)
    result[slope > 0] = 1.0
    result[slope < 0] = -1.0
    return result


# ===================================================================
#  2. 量价因子
# ===================================================================

@register_factor("volume_breakout_ratio", category="technical", requires=["volume"],
                 description="突破日量比")
def volume_breakout_ratio(ctx: FactorContext) -> pd.DataFrame:
    """突破日量比: 当日成交量 / 前20日平均成交量, 越大说明放量越明显。"""
    volume = ctx.volume
    avg_vol_20 = volume.rolling(20, min_periods=1).mean()
    return volume / avg_vol_20.replace(0, np.nan)


@register_factor("volume_shrink_ratio", category="technical", requires=["volume"],
                 description="缩量程度")
def volume_shrink_ratio(ctx: FactorContext) -> pd.DataFrame:
    """缩量程度: 当日成交量 / 前5日最大成交量, 越小说明缩量越厉害。"""
    volume = ctx.volume
    max_vol_5 = volume.rolling(5, min_periods=1).max()
    return volume / max_vol_5.replace(0, np.nan)


@register_factor("volume_sustained_increase", category="technical", requires=["volume"],
                 description="持续放量天数")
def volume_sustained_increase(ctx: FactorContext) -> pd.DataFrame:
    """持续放量天数: 连续成交量大于前20日均量的天数。"""
    volume = ctx.volume
    avg_vol_20 = volume.rolling(20, min_periods=1).mean()
    is_above = (volume > avg_vol_20).astype(int)

    def _count_consecutive(s: pd.Series) -> pd.Series:
        result = pd.Series(0, index=s.index)
        count = 0
        for i, val in s.items():
            if val == 1:
                count += 1
            else:
                count = 0
            result[i] = count
        return result

    return is_above.apply(_count_consecutive, axis=0)


@register_factor("volume_price_divergence", category="technical", requires=["volume", "close"],
                 description="量价背离程度")
def volume_price_divergence(ctx: FactorContext) -> pd.DataFrame:
    """量价背离程度: 价格变化方向与成交量变化方向不一致的程度。

    计算逻辑:
    - 价格上涨但成交量下降(背离), 返回负值
    - 价格下跌但成交量上升(背离), 返回负值
    - 价量同向, 返回正值
    - 值域 [-1, 1], 越小说明背离越严重
    """
    close = ctx.close
    volume = ctx.volume

    price_change = close.pct_change()
    volume_change = volume.pct_change()

    price_dir = price_change.apply(np.sign)
    volume_dir = volume_change.apply(np.sign)

    divergence = price_dir * volume_dir
    return divergence.fillna(0)


# ===================================================================
#  3. K线形态因子
# ===================================================================

@register_factor("big_bullish", category="technical", requires=["open", "close"],
                 description="大阳线实体")
def big_bullish(ctx: FactorContext) -> pd.DataFrame:
    """大阳线实体: (收盘价 - 开盘价) / 开盘价 > 3%。"""
    close = ctx.close
    open_ = ctx.open
    body_ratio = (close - open_) / open_.replace(0, np.nan)
    return (body_ratio > 0.03).astype(float)


@register_factor("big_bearish", category="technical", requires=["open", "close"],
                 description="大阴线实体")
def big_bearish(ctx: FactorContext) -> pd.DataFrame:
    """大阴线实体: (开盘价 - 收盘价) / 开盘价 > 3%。"""
    close = ctx.close
    open_ = ctx.open
    body_ratio = (open_ - close) / open_.replace(0, np.nan)
    return (body_ratio > 0.03).astype(float)


@register_factor("long_upper_shadow", category="technical",
                 requires=["open", "close", "high"],
                 description="长上影线")
def long_upper_shadow(ctx: FactorContext) -> pd.DataFrame:
    """长上影线: (最高价 - 收盘价) / |收盘价 - 开盘价| > 2。"""
    close = ctx.close
    open_ = ctx.open
    high = ctx.high

    body = (close - open_).abs().replace(0, np.nan)
    upper_shadow = (high - close) / body
    return (upper_shadow > 2).astype(float)


@register_factor("long_lower_shadow", category="technical",
                 requires=["open", "close", "low"],
                 description="长下影线")
def long_lower_shadow(ctx: FactorContext) -> pd.DataFrame:
    """长下影线: (最低价 - 收盘价) / |开盘价 - 收盘价| > 2, 且收盘低于开盘。"""
    close = ctx.close
    open_ = ctx.open
    low = ctx.low

    body = (open_ - close).replace(0, np.nan)
    lower_shadow = (open_ - low) / body
    return (lower_shadow > 2).astype(float)


@register_factor("low_open_bull", category="technical",
                 requires=["open", "close"],
                 description="低开大阳(送礼线)")
def low_open_bull(ctx: FactorContext) -> pd.DataFrame:
    """低开大阳(送礼线): 开盘价大幅低开(较昨日收盘跌>2%), 收盘大涨(较开盘涨>2%)。"""
    close = ctx.close
    open_ = ctx.open
    prev_close = close.shift(1)

    gap_down = (open_ - prev_close) / prev_close.replace(0, np.nan) < -0.02
    big_rise = (close - open_) / open_.replace(0, np.nan) > 0.02
    return (gap_down & big_rise).astype(float)


@register_factor("high_open_bear", category="technical",
                 requires=["open", "close"],
                 description="高开大阴(墓碑线)")
def high_open_bear(ctx: FactorContext) -> pd.DataFrame:
    """高开大阴(墓碑线): 开盘价大幅高开(较昨日收盘涨>2%), 收盘大跌(较开盘跌>2%)。"""
    close = ctx.close
    open_ = ctx.open
    prev_close = close.shift(1)

    gap_up = (open_ - prev_close) / prev_close.replace(0, np.nan) > 0.02
    big_drop = (open_ - close) / open_.replace(0, np.nan) > 0.02
    return (gap_up & big_drop).astype(float)


@register_factor("doji", category="technical",
                 requires=["open", "close", "high", "low"],
                 description="十字星")
def doji(ctx: FactorContext) -> pd.DataFrame:
    """十字星: |收盘价 - 开盘价| < (最高价 - 最低价) * 0.1。"""
    close = ctx.close
    open_ = ctx.open
    high = ctx.high
    low = ctx.low

    body = (close - open_).abs()
    range_ = high - low
    return (body < range_ * 0.1).astype(float)


# ===================================================================
#  4. MACD形态因子
# ===================================================================

@register_factor("macd_above_zero", category="technical", requires=["close"],
                 description="MACD是否在零轴上方")
def macd_above_zero(ctx: FactorContext) -> pd.DataFrame:
    """MACD是否在零轴上方, 1表示在上方, 0表示在下方。"""
    return (_macd_line(ctx) > 0).astype(float)


@register_factor("macd_golden_cross", category="technical", requires=["close"],
                 description="MACD金叉信号")
def macd_golden_cross(ctx: FactorContext) -> pd.DataFrame:
    """MACD金叉: MACD线上穿信号线, 发生当日返回1, 否则返回0。"""
    macd = _macd_line(ctx)
    signal = _macd_signal(ctx)

    prev_diff = (macd - signal).shift(1)
    curr_diff = macd - signal

    golden = (prev_diff < 0) & (curr_diff > 0)
    return golden.astype(float)


@register_factor("macd_dead_cross", category="technical", requires=["close"],
                 description="MACD死叉信号")
def macd_dead_cross(ctx: FactorContext) -> pd.DataFrame:
    """MACD死叉: MACD线下穿信号线, 发生当日返回1, 否则返回0。"""
    macd = _macd_line(ctx)
    signal = _macd_signal(ctx)

    prev_diff = (macd - signal).shift(1)
    curr_diff = macd - signal

    dead = (prev_diff > 0) & (curr_diff < 0)
    return dead.astype(float)


@register_factor("macd_arc_bottom", category="technical", requires=["close"],
                 description="MACD圆弧底得分")
def macd_arc_bottom(ctx: FactorContext) -> pd.DataFrame:
    """MACD圆弧底得分: 通过MACD曲线曲率判断底部形态。

    判断逻辑:
    - 计算MACD的二阶差分(曲率近似)
    - 曲率由负转正(凹变凸)且MACD在低位, 得分高
    - 得分范围 [0, 1], 越高越像圆弧底
    """
    macd = _macd_line(ctx)

    first_diff = macd.diff()
    second_diff = first_diff.diff()

    curv_turning = (second_diff > 0) & (first_diff.shift(1) < 0)

    macd_low = macd < macd.rolling(60, min_periods=1).quantile(0.3)

    score = pd.DataFrame(0.0, index=macd.index, columns=macd.columns)
    score[curv_turning & macd_low] = 1.0

    curv_positive = second_diff.rolling(5, min_periods=1).sum() > 0
    score[curv_positive & macd_low] = 0.5

    return score


# ===================================================================
#  5. 箱体突破因子
# ===================================================================

@register_factor("box_range", category="technical",
                 requires=["close", "high", "low"],
                 description="箱体宽度")
def box_range(ctx: FactorContext) -> pd.DataFrame:
    """箱体宽度: 最近60日最高价与最低价的比值, 越小说明横盘越窄。"""
    high = ctx.high
    low = ctx.low

    window = 60
    highest = high.rolling(window, min_periods=1).max()
    lowest = low.rolling(window, min_periods=1).min()

    return highest / lowest.replace(0, np.nan)


@register_factor("box_duration", category="technical",
                 requires=["close"],
                 description="横盘天数")
def box_duration(ctx: FactorContext) -> pd.DataFrame:
    """横盘天数: 价格在最近20日均值上下2%区间内的连续天数。"""
    close = ctx.close
    ma20 = _get_ma(ctx, 20)

    upper = ma20 * 1.02
    lower = ma20 * 0.98
    inside = (close >= lower) & (close <= upper)

    def _count_consecutive(s: pd.Series) -> pd.Series:
        result = pd.Series(0, index=s.index)
        count = 0
        for i, val in s.items():
            if val:
                count += 1
            else:
                count = 0
            result[i] = count
        return result

    return inside.apply(_count_consecutive, axis=0)


@register_factor("box_breakout", category="technical",
                 requires=["close", "high", "low"],
                 description="箱体突破信号")
def box_breakout(ctx: FactorContext) -> pd.DataFrame:
    """箱体突破信号: 收盘价突破最近20日箱体上轨。"""
    close = ctx.close
    high = ctx.high

    box_upper = high.rolling(20, min_periods=1).max()
    prev_box_upper = box_upper.shift(1)

    breakout = (close > prev_box_upper) & (close.shift(1) <= prev_box_upper.shift(1))
    return breakout.astype(float)


@register_factor("breakout_strength", category="technical",
                 requires=["close", "high", "low", "volume"],
                 description="突破强度")
def breakout_strength(ctx: FactorContext) -> pd.DataFrame:
    """突破强度: 突破幅度 * 放量倍数。

    突破幅度 = (收盘价 - 箱体上轨) / 箱体上轨
    放量倍数 = 当日成交量 / 前20日均量
    强度 = 突破幅度 * 放量倍数, 越大说明突破越有效
    """
    close = ctx.close
    high = ctx.high
    volume = ctx.volume

    box_upper = high.rolling(20, min_periods=1).max().shift(1)
    breakout_pct = (close - box_upper) / box_upper.replace(0, np.nan)
    breakout_pct = breakout_pct.clip(lower=0)

    avg_vol_20 = volume.rolling(20, min_periods=1).mean()
    vol_ratio = volume / avg_vol_20.replace(0, np.nan)

    return breakout_pct * vol_ratio


# ===================================================================
#  6. 筹码集中度因子
# ===================================================================

@register_factor("turnover_rate_proxy", category="technical",
                 requires=["volume", "amount"],
                 description="换手率代理")
def turnover_rate_proxy(ctx: FactorContext) -> pd.DataFrame:
    """换手率代理: 成交量 / (成交金额 / 均价) 的倒数, 近似换手率。

    由于缺少流通股本数据, 用 volume / (amount / close) = volume * close / amount
    作为换手率的代理指标。
    """
    volume = ctx.volume
    amount = ctx.amount
    close = ctx.close

    return (volume * close) / amount.replace(0, np.nan)


@register_factor("volume_contraction", category="technical", requires=["volume"],
                 description="量能萎缩程度")
def volume_contraction(ctx: FactorContext) -> pd.DataFrame:
    """量能萎缩程度: 最近5日均量 / 前20日均量, 越小说明量能收缩越厉害。"""
    volume = ctx.volume

    avg_vol_5 = volume.rolling(5, min_periods=1).mean()
    avg_vol_20 = volume.rolling(20, min_periods=1).mean()

    return avg_vol_5 / avg_vol_20.replace(0, np.nan)


@register_factor("chip_concentration", category="technical",
                 requires=["close"],
                 description="筹码集中度")
def chip_concentration(ctx: FactorContext) -> pd.DataFrame:
    """筹码集中度: 最近60日收盘价分布的标准差除以均值。

    该值越小说明筹码越集中, 价格波动区间越窄;
    值越大说明筹码越分散, 价格波动区间越宽。
    """
    close = ctx.close
    std_60 = _get_std(ctx, 60, min_periods=10)
    mean_60 = close.rolling(60, min_periods=10).mean()

    return std_60 / mean_60.replace(0, np.nan)


# ===================================================================
#  7. β系数因子
# ===================================================================

@register_factor("beta_20", category="technical", requires=["close"],
                 description="20日β系数")
def beta_20(ctx: FactorContext) -> pd.DataFrame:
    """20日β系数: 个股收益率 vs 市场收益率的回归斜率。

    市场代理: 用截面收盘价均值作为市场指数。
    β > 1 表示个股波动大于市场, β < 1 表示个股波动小于市场。
    """
    returns = _get_returns(ctx)
    market_returns = _market_proxy(ctx).pct_change()

    window = 20
    min_periods = max(5, window // 2)

    def _beta_calc(row: pd.Series) -> pd.Series:
        result = pd.Series(np.nan, index=row.index)
        for i in range(min_periods, len(row) + 1):
            stock_ret = row.iloc[i - window:i]
            mkt_ret = market_returns.iloc[i - window:i]

            valid = stock_ret.notna() & mkt_ret.notna()
            if valid.sum() < min_periods:
                continue

            s = stock_ret[valid]
            m = mkt_ret[valid]

            cov = ((s - s.mean()) * (m - m.mean())).sum() / (len(s) - 1)
            var = ((m - m.mean()) ** 2).sum() / (len(m) - 1)

            if var != 0:
                result.iloc[i - 1] = cov / var
        return result

    return returns.apply(_beta_calc, axis=0)


@register_factor("beta_60", category="technical", requires=["close"],
                 description="60日β系数")
def beta_60(ctx: FactorContext) -> pd.DataFrame:
    """60日β系数: 个股收益率 vs 市场收益率的回归斜率。

    市场代理: 用截面收盘价均值作为市场指数。
    窗口期为60日, 比beta_20更长期。
    """
    returns = _get_returns(ctx)
    market_returns = _market_proxy(ctx).pct_change()

    window = 60
    min_periods = max(20, window // 2)

    def _beta_calc(row: pd.Series) -> pd.Series:
        result = pd.Series(np.nan, index=row.index)
        for i in range(min_periods, len(row) + 1):
            stock_ret = row.iloc[i - window:i]
            mkt_ret = market_returns.iloc[i - window:i]

            valid = stock_ret.notna() & mkt_ret.notna()
            if valid.sum() < min_periods:
                continue

            s = stock_ret[valid]
            m = mkt_ret[valid]

            cov = ((s - s.mean()) * (m - m.mean())).sum() / (len(s) - 1)
            var = ((m - m.mean()) ** 2).sum() / (len(m) - 1)

            if var != 0:
                result.iloc[i - 1] = cov / var
        return result

    return returns.apply(_beta_calc, axis=0)
