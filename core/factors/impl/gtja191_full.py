"""
GTJA 191 因子实现 (因子 1-70)
国泰君安《基于短周期价量特征的多因子选股体系》
"""
from typing import Dict
import numpy as np
import pandas as pd
from core.factors.base.factor_hub import register_factor, FactorContext
from . import gtja_ops as op


@register_factor("gtja1", category="gtja191",
                 requires=["open", "close", "volume"],
                 description="Alpha 1: (-1 * CORR(RANK(DELTA(LOG(VOLUME), 1)), RANK(((CLOSE - OPEN) / OPEN)), 6))")
def gtja1(ctx: FactorContext) -> pd.DataFrame:
    log_vol = op.log(ctx.volume)
    x = op.rank(op.delta(log_vol, 1))
    y = op.rank((ctx.close - ctx.open) / ctx.open.replace(0, np.nan))
    return -op.correlation(x, y, 6)


@register_factor("gtja2", category="gtja191",
                 requires=["close", "high", "low"],
                 description="Alpha 2: (-1 * DELTA((((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW)), 1))")
def gtja2(ctx: FactorContext) -> pd.DataFrame:
    numerator = (ctx.close - ctx.low) - (ctx.high - ctx.close)
    denominator = (ctx.high - ctx.low).replace(0, np.nan)
    ratio = numerator / denominator
    return -op.delta(ratio, 1)


@register_factor("gtja3", category="gtja191",
                 requires=["close", "high", "low"],
                 description="Alpha 3: SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6)")
def gtja3(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    delayed_close = op.delay(close, 1)
    delayed_low = op.delay(ctx.low, 1)
    delayed_high = op.delay(ctx.high, 1)

    cond_equal = (close == delayed_close)
    cond_up = (close > delayed_close)

    price_ref = delayed_low.where(cond_up, delayed_high)
    diff = close - price_ref
    result = diff.where(~cond_equal, 0.0)
    return op.ts_sum(result, 6)


@register_factor("gtja4", category="gtja191",
                 requires=["close", "volume"],
                 description="Alpha 4: complex conditional formula using sums and std")
def gtja4(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    vol = ctx.volume

    sum8 = op.ts_sum(close, 8)
    sum2 = op.ts_sum(close, 2)
    mean8 = sum8 / 8.0
    mean2 = sum2 / 2.0
    std8 = op.ts_std(close, 8)

    cond1 = (mean8 + std8) < mean2
    cond2 = mean2 < (mean8 - std8)
    vol_ratio = vol / op.ts_mean(vol, 20).replace(0, np.nan)
    cond3 = (vol_ratio >= 1) | (vol_ratio == 1)

    result = pd.DataFrame(-1.0, index=close.index, columns=close.columns)
    result = result.mask(cond3, 1.0)
    result = result.mask(cond2, 1.0)
    result = result.mask(cond1, -1.0)
    return result


@register_factor("gtja5", category="gtja191",
                 requires=["high", "volume"],
                 description="Alpha 5: (-1 * TSMAX(CORR(TSRANK(VOLUME, 5), TSRANK(HIGH, 5), 5), 3))")
def gtja5(ctx: FactorContext) -> pd.DataFrame:
    vol_rank = op.ts_rank(ctx.volume, 5)
    high_rank = op.ts_rank(ctx.high, 5)
    corr = op.correlation(vol_rank, high_rank, 5)
    return -op.ts_max(corr, 3)


@register_factor("gtja6", category="gtja191",
                 requires=["open", "high"],
                 description="Alpha 6: (RANK(SIGN(DELTA((((OPEN * 0.85) + (HIGH * 0.15))), 4)))* -1)")
def gtja6(ctx: FactorContext) -> pd.DataFrame:
    weighted = (ctx.open * 0.85) + (ctx.high * 0.15)
    delta_val = op.delta(weighted, 4)
    return op.rank(np.sign(delta_val)) * -1


@register_factor("gtja7", category="gtja191",
                 requires=["close", "volume", "vwap"],
                 description="Alpha 7: ((RANK(MAX((VWAP - CLOSE), 3)) + RANK(MIN((VWAP - CLOSE), 3))) * RANK(DELTA(VOLUME, 3)))")
def gtja7(ctx: FactorContext) -> pd.DataFrame:
    diff = ctx.vwap - ctx.close
    rank_max = op.rank(op.ts_max(diff, 3))
    rank_min = op.rank(op.ts_min(diff, 3))
    rank_delta_vol = op.rank(op.delta(ctx.volume, 3))
    return (rank_max + rank_min) * rank_delta_vol


@register_factor("gtja8", category="gtja191",
                 requires=["high", "low", "vwap"],
                 description="Alpha 8: RANK(DELTA(((((HIGH + LOW) / 2) * 0.2) + (VWAP * 0.8)), 4) * -1)")
def gtja8(ctx: FactorContext) -> pd.DataFrame:
    mid = (ctx.high + ctx.low) / 2.0
    weighted = (mid * 0.2) + (ctx.vwap * 0.8)
    delta_val = op.delta(weighted, 4)
    return op.rank(delta_val) * -1


@register_factor("gtja9", category="gtja191",
                 requires=["high", "low", "volume"],
                 description="Alpha 9: SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME,7,2)")
def gtja9(ctx: FactorContext) -> pd.DataFrame:
    mid = (ctx.high + ctx.low) / 2.0
    delayed_mid = (op.delay(ctx.high, 1) + op.delay(ctx.low, 1)) / 2.0
    diff = mid - delayed_mid
    hl = ctx.high - ctx.low
    vol = ctx.volume.replace(0, np.nan)
    result = diff * hl / vol
    return op.sma(result, 7, 2)


@register_factor("gtja10", category="gtja191",
                 requires=["close"],
                 description="Alpha 10: (RANK(MAX(((RET < 0) ? STD(RET, 20) : CLOSE)^2),5))")
def gtja10(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    ret = close.pct_change()
    cond = ret < 0
    std_ret = op.ts_std(ret, 20)
    base = std_ret.where(cond, close)
    squared = base ** 2
    return op.rank(op.ts_max(squared, 5))


@register_factor("gtja11", category="gtja191",
                 requires=["close", "high", "low", "volume"],
                 description="Alpha 11: SUM(((CLOSE-LOW)-(HIGH-CLOSE))./(HIGH-LOW).*VOLUME,6)")
def gtja11(ctx: FactorContext) -> pd.DataFrame:
    numerator = (ctx.close - ctx.low) - (ctx.high - ctx.close)
    denominator = (ctx.high - ctx.low).replace(0, np.nan)
    ratio = numerator / denominator
    result = ratio * ctx.volume
    return op.ts_sum(result, 6)


@register_factor("gtja12", category="gtja191",
                 requires=["open", "close", "vwap"],
                 description="Alpha 12: (RANK((OPEN - (SUM(VWAP, 10) / 10)))) * (-1 * (RANK(ABS((CLOSE - VWAP)))))")
def gtja12(ctx: FactorContext) -> pd.DataFrame:
    rank_open = op.rank(ctx.open - op.ts_sum(ctx.vwap, 10) / 10.0)
    rank_close = op.rank((ctx.close - ctx.vwap).abs()) * -1
    return rank_open * rank_close


@register_factor("gtja13", category="gtja191",
                 requires=["high", "low", "vwap"],
                 description="Alpha 13: (((HIGH * LOW)^0.5) - VWAP)")
def gtja13(ctx: FactorContext) -> pd.DataFrame:
    return np.sqrt(ctx.high * ctx.low) - ctx.vwap


@register_factor("gtja14", category="gtja191",
                 requires=["close"],
                 description="Alpha 14: CLOSE-DELAY(CLOSE,5)")
def gtja14(ctx: FactorContext) -> pd.DataFrame:
    return ctx.close - op.delay(ctx.close, 5)


@register_factor("gtja15", category="gtja191",
                 requires=["open", "close"],
                 description="Alpha 15: OPEN/DELAY(CLOSE,1)-1")
def gtja15(ctx: FactorContext) -> pd.DataFrame:
    return ctx.open / op.delay(ctx.close, 1).replace(0, np.nan) - 1


@register_factor("gtja16", category="gtja191",
                 requires=["volume", "vwap"],
                 description="Alpha 16: (-1 * TSMAX(RANK(CORR(RANK(VOLUME), RANK(VWAP), 5)), 5))")
def gtja16(ctx: FactorContext) -> pd.DataFrame:
    corr_val = op.correlation(op.rank(ctx.volume), op.rank(ctx.vwap), 5)
    rank_corr = op.rank(corr_val)
    return -op.ts_max(rank_corr, 5)


@register_factor("gtja17", category="gtja191",
                 requires=["close", "vwap"],
                 description="Alpha 17: RANK((VWAP - MAX(VWAP, 15)))^DELTA(CLOSE, 5)")
def gtja17(ctx: FactorContext) -> pd.DataFrame:
    rank_val = op.rank(ctx.vwap - op.ts_max(ctx.vwap, 15))
    delta_close = op.delta(ctx.close, 5)
    return rank_val ** delta_close


@register_factor("gtja18", category="gtja191",
                 requires=["close"],
                 description="Alpha 18: CLOSE/DELAY(CLOSE,5)")
def gtja18(ctx: FactorContext) -> pd.DataFrame:
    return ctx.close / op.delay(ctx.close, 5).replace(0, np.nan)


@register_factor("gtja19", category="gtja191",
                 requires=["close"],
                 description="Alpha 19: complex conditional return formula")
def gtja19(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    delayed = op.delay(close, 5)

    cond_equal = close == delayed
    cond_less = close < delayed

    ratio_to_delay = (close - delayed) / delayed.replace(0, np.nan)
    ratio_to_close = (close - delayed) / close.replace(0, np.nan)

    result = ratio_to_delay.where(cond_less, ratio_to_close).where(~cond_equal, 0.0)
    return result


@register_factor("gtja20", category="gtja191",
                 requires=["close"],
                 description="Alpha 20: (CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100")
def gtja20(ctx: FactorContext) -> pd.DataFrame:
    delayed = op.delay(ctx.close, 6)
    return (ctx.close - delayed) / delayed.replace(0, np.nan) * 100


@register_factor("gtja21", category="gtja191",
                 requires=["close"],
                 description="Alpha 21: REGBETA(MEAN(CLOSE,6),SEQUENCE(6)) - linear regression slope")
def gtja21(ctx: FactorContext) -> pd.DataFrame:
    mean6 = op.ts_mean(ctx.close, 6)
    window = 6
    n = window
    x = np.arange(1, n + 1).astype(float)
    x_mean = x.mean()
    x_var = np.sum((x - x_mean) ** 2)

    def calc_slope(series):
        y = series.values
        if np.isnan(y).any():
            return np.nan
        y_mean = y.mean()
        cov = np.sum((x - x_mean) * (y - y_mean))
        return cov / x_var

    result = mean6.rolling(window).apply(calc_slope, raw=False)
    return result


@register_factor("gtja22", category="gtja191",
                 requires=["close"],
                 description="Alpha 22: SMEAN - EMA of (close-mean(close,6))/mean(close,6)")
def gtja22(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    mean6 = op.ts_mean(close, 6)
    ratio = (close - mean6) / mean6.replace(0, np.nan)
    delayed_ratio = op.delay(ratio, 3)
    diff = ratio - delayed_ratio
    return op.sma(diff, 12, 1)


@register_factor("gtja23", category="gtja191",
                 requires=["close"],
                 description="Alpha 23: SMA-based up/down strength indicator")
def gtja23(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    ret = close.pct_change()
    cond_up = ret > 0
    std20 = op.ts_std(close, 20)
    up_val = std20.where(cond_up, 0.0)
    down_val = std20.where(~cond_up, 0.0)
    sma_up = op.sma(up_val, 20, 1)
    sma_down = op.sma(down_val, 20, 1)
    denominator = (sma_up + sma_down).replace(0, np.nan)
    return (sma_up / denominator) * 100


@register_factor("gtja24", category="gtja191",
                 requires=["close"],
                 description="Alpha 24: SMA(CLOSE-DELAY(CLOSE,5),5,1)")
def gtja24(ctx: FactorContext) -> pd.DataFrame:
    diff = ctx.close - op.delay(ctx.close, 5)
    return op.sma(diff, 5, 1)


@register_factor("gtja25", category="gtja191",
                 requires=["close", "volume"],
                 description="Alpha 25: complex momentum-volume formula")
def gtja25(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    vol = ctx.volume

    ret = close.pct_change()
    vol_ratio = vol / op.ts_mean(vol, 20).replace(0, np.nan)
    decay_vol = op.decay_linear(vol_ratio, 9)

    delta7 = op.delta(close, 7)
    rank_delta = op.rank(delta7)
    rank_decay = op.rank(decay_vol)

    part1 = -rank_delta * (1 - rank_decay)
    sum_ret = op.ts_sum(ret, 250)
    rank_sum_ret = op.rank(sum_ret)

    return part1 * (1 + rank_sum_ret)


@register_factor("gtja26", category="gtja191",
                 requires=["close", "vwap"],
                 description="Alpha 26: (SUM(CLOSE, 7) / 7) - CLOSE + CORR(VWAP, DELAY(CLOSE, 5), 230)")
def gtja26(ctx: FactorContext) -> pd.DataFrame:
    mean7 = op.ts_sum(ctx.close, 7) / 7.0
    corr_val = op.correlation(ctx.vwap, op.delay(ctx.close, 5), 230)
    return mean7 - ctx.close + corr_val


@register_factor("gtja27", category="gtja191",
                 requires=["close"],
                 description="Alpha 27: WMA of dual-period returns")
def gtja27(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close

    ret3 = (close - op.delay(close, 3)) / op.delay(close, 3).replace(0, np.nan)
    ret6 = (close - op.delay(close, 6)) / op.delay(close, 6).replace(0, np.nan)
    combined = ret3 + ret6
    return op.wma(combined, 12)


@register_factor("gtja28", category="gtja191",
                 requires=["close", "high", "low"],
                 description="Alpha 28: Stochastic RSI-like formula (3*SMA - 2*SMA(SMA))")
def gtja28(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    low9 = op.ts_min(ctx.low, 9)
    high9 = op.ts_max(ctx.high, 9)
    stoch = (close - low9) / (high9 - low9).replace(0, np.nan) * 100

    sma1 = op.sma(stoch, 3, 1)
    sma2 = op.sma(sma1, 3, 1)

    return 3 * sma1 - 2 * sma2


@register_factor("gtja29", category="gtja191",
                 requires=["close", "volume"],
                 description="Alpha 29: (CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*VOLUME")
def gtja29(ctx: FactorContext) -> pd.DataFrame:
    delayed = op.delay(ctx.close, 6)
    ratio = (ctx.close - delayed) / delayed.replace(0, np.nan)
    return ratio * ctx.volume


@register_factor("gtja30", category="gtja191",
                 requires=["close"],
                 description="Alpha 30: [SKIP - requires Fama-French factors]")
def gtja30(ctx: FactorContext) -> pd.DataFrame:
    return pd.DataFrame(0, index=ctx.close.index, columns=ctx.close.columns)


@register_factor("gtja31", category="gtja191",
                 requires=["close"],
                 description="Alpha 31: (CLOSE-MEAN(CLOSE,12))/MEAN(CLOSE,12)*100")
def gtja31(ctx: FactorContext) -> pd.DataFrame:
    mean12 = op.ts_mean(ctx.close, 12)
    return (ctx.close - mean12) / mean12.replace(0, np.nan) * 100


@register_factor("gtja32", category="gtja191",
                 requires=["high", "volume"],
                 description="Alpha 32: (-1 * SUM(RANK(CORR(RANK(HIGH), RANK(VOLUME), 3)), 3))")
def gtja32(ctx: FactorContext) -> pd.DataFrame:
    corr_val = op.correlation(op.rank(ctx.high), op.rank(ctx.volume), 3)
    ranked = op.rank(corr_val)
    return -op.ts_sum(ranked, 3)


@register_factor("gtja33", category="gtja191",
                 requires=["close", "low", "volume"],
                 description="Alpha 33: complex formula with TSMIN and returns")
def gtja33(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    low = ctx.low
    vol = ctx.volume

    ret = close.pct_change()
    tsmin_low5 = op.ts_min(low, 5)
    delayed_tsmin = op.delay(tsmin_low5, 5)

    delta_tsmin = tsmin_low5 - delayed_tsmin
    sum_ret = op.ts_sum(ret, 240)
    sum_ret20 = op.ts_sum(ret, 20)
    ret_diff = (sum_ret - sum_ret20) / 220.0
    rank_ret = op.rank(ret_diff)
    tsrank_vol = op.ts_rank(vol, 5)

    return -delta_tsmin * rank_ret * tsrank_vol


@register_factor("gtja34", category="gtja191",
                 requires=["close"],
                 description="Alpha 34: MEAN(CLOSE,12)/CLOSE")
def gtja34(ctx: FactorContext) -> pd.DataFrame:
    return op.ts_mean(ctx.close, 12) / ctx.close.replace(0, np.nan)


@register_factor("gtja35", category="gtja191",
                 requires=["open", "volume"],
                 description="Alpha 35: MIN(RANK(...), RANK(...)) * -1")
def gtja35(ctx: FactorContext) -> pd.DataFrame:
    delta_open = op.delta(ctx.open, 1)
    rank1 = op.rank(op.decay_linear(delta_open, 15))

    corr_val = op.correlation(ctx.volume, ctx.open * 0.65 + ctx.open * 0.35, 17)
    rank2 = op.rank(op.decay_linear(corr_val, 7))

    return op._min(rank1, rank2) * -1


@register_factor("gtja36", category="gtja191",
                 requires=["volume", "vwap"],
                 description="Alpha 36: RANK(SUM(CORR(RANK(VOLUME), RANK(VWAP), 6), 2))")
def gtja36(ctx: FactorContext) -> pd.DataFrame:
    corr_val = op.correlation(op.rank(ctx.volume), op.rank(ctx.vwap), 6)
    sum_corr = op.ts_sum(corr_val, 2)
    return op.rank(sum_corr)


@register_factor("gtja37", category="gtja191",
                 requires=["open", "close"],
                 description="Alpha 37: (-1 * RANK(((SUM(OPEN, 5) * SUM(RET, 5)) - DELAY(...))))")
def gtja37(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    ret = close.pct_change()

    sum_open = op.ts_sum(ctx.open, 5)
    sum_ret = op.ts_sum(ret, 5)
    product = sum_open * sum_ret
    delayed = op.delay(product, 10)

    return -op.rank(product - delayed)


@register_factor("gtja38", category="gtja191",
                 requires=["high"],
                 description="Alpha 38: ((SUM(HIGH, 20) / 20) < HIGH) ? (-1 * DELTA(HIGH, 2)) : 0")
def gtja38(ctx: FactorContext) -> pd.DataFrame:
    cond = (op.ts_sum(ctx.high, 20) / 20.0) < ctx.high
    return (-op.delta(ctx.high, 2)).where(cond, 0.0)


@register_factor("gtja39", category="gtja191",
                 requires=["open", "close", "volume", "vwap"],
                 description="Alpha 39: decay_linear correlation formula")
def gtja39(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    vol = ctx.volume
    vwap = ctx.vwap
    open_val = ctx.open

    delta_close = op.delta(close, 2)
    rank1 = op.rank(op.decay_linear(delta_close, 8))

    mean_vol = op.ts_mean(vol, 180)
    mix_price = vwap * 0.3 + open_val * 0.7
    corr_val = op.correlation(mix_price, op.ts_sum(mean_vol, 37), 14)
    rank2 = op.rank(op.decay_linear(corr_val, 12))

    return (rank1 - rank2) * -1


@register_factor("gtja40", category="gtja191",
                 requires=["close", "volume"],
                 description="Alpha 40: up/down volume ratio indicator")
def gtja40(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    vol = ctx.volume

    cond_up = close > op.delay(close, 1)
    cond_down = close <= op.delay(close, 1)

    up_vol = vol.where(cond_up, 0.0)
    down_vol = vol.where(cond_down, 0.0)

    sum_up = op.ts_sum(up_vol, 26)
    sum_down = op.ts_sum(down_vol, 26)

    return (sum_up / sum_down.replace(0, np.nan)) * 100


@register_factor("gtja41", category="gtja191",
                 requires=["vwap"],
                 description="Alpha 41: (RANK(MAX(DELTA((VWAP), 3), 5))* -1)")
def gtja41(ctx: FactorContext) -> pd.DataFrame:
    delta_vwap = op.delta(ctx.vwap, 3)
    max_val = op.ts_max(delta_vwap, 5)
    return op.rank(max_val) * -1


@register_factor("gtja42", category="gtja191",
                 requires=["high", "volume"],
                 description="Alpha 42: ((-1 * RANK(STD(HIGH, 10))) * CORR(HIGH, VOLUME, 10))")
def gtja42(ctx: FactorContext) -> pd.DataFrame:
    std_high = op.ts_std(ctx.high, 10)
    corr_val = op.correlation(ctx.high, ctx.volume, 10)
    return -op.rank(std_high) * corr_val


@register_factor("gtja43", category="gtja191",
                 requires=["close", "volume"],
                 description="Alpha 43: net volume flow indicator")
def gtja43(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    vol = ctx.volume

    cond_up = close > op.delay(close, 1)
    cond_down = close < op.delay(close, 1)

    net_vol = vol.where(cond_up, 0.0) + (-vol).where(cond_down, 0.0)

    return op.ts_sum(net_vol, 6)


@register_factor("gtja44", category="gtja191",
                 requires=["low", "volume", "vwap"],
                 description="Alpha 44: TSRANK sum of decay_linear formulas")
def gtja44(ctx: FactorContext) -> pd.DataFrame:
    low = ctx.low
    vol = ctx.volume
    vwap = ctx.vwap

    mean_vol10 = op.ts_mean(vol, 10)
    corr_low = op.correlation(low, mean_vol10, 7)
    decay_corr = op.decay_linear(corr_low, 6)
    tsrank1 = op.ts_rank(decay_corr, 4)

    delta_vwap = op.delta(vwap, 3)
    decay_delta = op.decay_linear(delta_vwap, 10)
    tsrank2 = op.ts_rank(decay_delta, 15)

    return tsrank1 + tsrank2


@register_factor("gtja45", category="gtja191",
                 requires=["open", "close", "volume", "vwap"],
                 description="Alpha 45: RANK(DELTA(...)) * RANK(CORR(VWAP, MEAN(VOLUME,150), 15))")
def gtja45(ctx: FactorContext) -> pd.DataFrame:
    weighted_price = ctx.close * 0.6 + ctx.open * 0.4
    delta_price = op.delta(weighted_price, 1)
    rank1 = op.rank(delta_price)

    mean_vol150 = op.ts_mean(ctx.volume, 150)
    corr_val = op.correlation(ctx.vwap, mean_vol150, 15)
    rank2 = op.rank(corr_val)

    return rank1 * rank2


@register_factor("gtja46", category="gtja191",
                 requires=["close"],
                 description="Alpha 46: (MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/(4*CLOSE)")
def gtja46(ctx: FactorContext) -> pd.DataFrame:
    mean3 = op.ts_mean(ctx.close, 3)
    mean6 = op.ts_mean(ctx.close, 6)
    mean12 = op.ts_mean(ctx.close, 12)
    mean24 = op.ts_mean(ctx.close, 24)
    return (mean3 + mean6 + mean12 + mean24) / (4.0 * ctx.close.replace(0, np.nan))


@register_factor("gtja47", category="gtja191",
                 requires=["close", "high", "low"],
                 description="Alpha 47: SMA of stochastic value")
def gtja47(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    low9 = op.ts_min(ctx.low, 9)
    high6 = op.ts_max(ctx.high, 6)
    low6 = op.ts_min(ctx.low, 6)

    stoch = (close - low9) / (high6 - low9).replace(0, np.nan) * 100
    return op.sma(stoch, 9, 1)


@register_factor("gtja48", category="gtja191",
                 requires=["close", "volume"],
                 description="Alpha 48: trend strength indicator")
def gtja48(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    vol = ctx.volume

    sign1 = np.sign(op.delta(close, 1))
    sign2 = np.sign(op.delta(op.delay(close, 1), 1))
    sign3 = np.sign(op.delta(op.delay(close, 2), 1))
    sign_sum = sign1 + sign2 + sign3

    numerator = -op.rank(sign_sum) * op.ts_sum(vol, 5)
    denominator = op.ts_sum(vol, 20).replace(0, np.nan)

    return numerator / denominator


@register_factor("gtja49", category="gtja191",
                 requires=["high", "low"],
                 description="Alpha 49: high-low range momentum ratio")
def gtja49(ctx: FactorContext) -> pd.DataFrame:
    high = ctx.high
    low = ctx.low

    cond_hl = (high + low) >= (op.delay(high, 1) + op.delay(low, 1))
    abs_high = (high - op.delay(high, 1)).abs()
    abs_low = (low - op.delay(low, 1)).abs()
    max_diff = np.maximum(abs_high, abs_low).where(~cond_hl, 0.0)

    cond_hl2 = (high + low) <= (op.delay(high, 1) + op.delay(low, 1))
    max_diff2 = np.maximum(abs_high, abs_low).where(~cond_hl2, 0.0)

    sum1 = op.ts_sum(max_diff, 12)
    sum2 = op.ts_sum(max_diff2, 12)

    return sum1 / (sum1 + sum2.replace(0, np.nan))


@register_factor("gtja50", category="gtja191",
                 requires=["high", "low"],
                 description="Alpha 50: [Similar to Alpha 49] - SKIP for now")
def gtja50(ctx: FactorContext) -> pd.DataFrame:
    return pd.DataFrame(0, index=ctx.high.index, columns=ctx.high.columns)


@register_factor("gtja51", category="gtja191",
                 requires=["high", "low"],
                 description="Alpha 51: [Similar to Alpha 49] - SKIP for now")
def gtja51(ctx: FactorContext) -> pd.DataFrame:
    return pd.DataFrame(0, index=ctx.high.index, columns=ctx.high.columns)


@register_factor("gtja52", category="gtja191",
                 requires=["close", "high", "low"],
                 description="Alpha 52: typical price momentum")
def gtja52(ctx: FactorContext) -> pd.DataFrame:
    typical = (ctx.high + ctx.low + ctx.close) / 3.0
    delayed_typical = op.delay(typical, 1)

    up_val = (typical - delayed_typical).clip(lower=0)
    down_val = (delayed_typical - ctx.low).clip(lower=0)

    sum_up = op.ts_sum(up_val, 26)
    sum_down = op.ts_sum(down_val, 26).replace(0, np.nan)

    return (sum_up / sum_down) * 100


@register_factor("gtja53", category="gtja191",
                 requires=["close"],
                 description="Alpha 53: COUNT(CLOSE>DELAY(CLOSE,1),12)/12*100")
def gtja53(ctx: FactorContext) -> pd.DataFrame:
    cond = ctx.close > op.delay(ctx.close, 1)
    count = cond.rolling(12).sum()
    return (count / 12.0) * 100


@register_factor("gtja54", category="gtja191",
                 requires=["open", "close"],
                 description="Alpha 54: (-1 * RANK((STD(ABS(CLOSE - OPEN)) + (CLOSE - OPEN)) + CORR(CLOSE, OPEN,10)))")
def gtja54(ctx: FactorContext) -> pd.DataFrame:
    diff = ctx.close - ctx.open
    std_diff = op.ts_std(diff.abs(), 10)
    corr_val = op.correlation(ctx.close, ctx.open, 10)
    return -op.rank(std_diff + diff + corr_val)


@register_factor("gtja55", category="gtja191",
                 requires=["open", "close", "high", "low"],
                 description="Alpha 55: [Complex formula] - simplified version returning 0")
def gtja55(ctx: FactorContext) -> pd.DataFrame:
    return pd.DataFrame(0, index=ctx.close.index, columns=ctx.close.columns)


@register_factor("gtja56", category="gtja191",
                 requires=["open", "high", "low", "volume"],
                 description="Alpha 56: RANK comparison formula")
def gtja56(ctx: FactorContext) -> pd.DataFrame:
    open_val = ctx.open
    high = ctx.high
    low = ctx.low
    vol = ctx.volume

    rank1 = op.rank(open_val - op.ts_min(open_val, 12))
    mid = (high + low) / 2.0
    mean_vol40 = op.ts_mean(vol, 40)
    corr_val = op.correlation(op.ts_sum(mid, 19), op.ts_sum(mean_vol40, 19), 13)
    power5 = op.rank(corr_val) ** 5
    rank2 = op.rank(power5)

    return (rank1 < rank2).astype(float)


@register_factor("gtja57", category="gtja191",
                 requires=["close", "high", "low"],
                 description="Alpha 57: SMA of stochastic value")
def gtja57(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    low9 = op.ts_min(ctx.low, 9)
    high9 = op.ts_max(ctx.high, 9)

    stoch = (close - low9) / (high9 - low9).replace(0, np.nan) * 100
    return op.sma(stoch, 3, 1)


@register_factor("gtja58", category="gtja191",
                 requires=["close"],
                 description="Alpha 58: COUNT(CLOSE>DELAY(CLOSE,1),20)/20*100")
def gtja58(ctx: FactorContext) -> pd.DataFrame:
    cond = ctx.close > op.delay(ctx.close, 1)
    count = cond.rolling(20).sum()
    return (count / 20.0) * 100


@register_factor("gtja59", category="gtja191",
                 requires=["close", "high", "low"],
                 description="Alpha 59: SUM of conditional price changes")
def gtja59(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    delayed_close = op.delay(close, 1)
    delayed_low = op.delay(ctx.low, 1)
    delayed_high = op.delay(ctx.high, 1)

    cond_equal = close == delayed_close
    cond_up = close > delayed_close

    price_ref = delayed_low.where(cond_up, delayed_high)
    diff = close - price_ref
    result = diff.where(~cond_equal, 0.0)

    return op.ts_sum(result, 20)


@register_factor("gtja60", category="gtja191",
                 requires=["close", "high", "low", "volume"],
                 description="Alpha 60: SUM of volume-weighted price position")
def gtja60(ctx: FactorContext) -> pd.DataFrame:
    numerator = (ctx.close - ctx.low) - (ctx.high - ctx.close)
    denominator = (ctx.high - ctx.low).replace(0, np.nan)
    ratio = numerator / denominator
    result = ratio * ctx.volume

    return op.ts_sum(result, 20)


@register_factor("gtja61", category="gtja191",
                 requires=["low", "volume", "vwap"],
                 description="Alpha 61: MAX(RANK(...), RANK(...)) * -1")
def gtja61(ctx: FactorContext) -> pd.DataFrame:
    low = ctx.low
    vol = ctx.volume
    vwap = ctx.vwap

    delta_vwap = op.delta(vwap, 1)
    rank1 = op.rank(op.decay_linear(delta_vwap, 12))

    mean_vol80 = op.ts_mean(vol, 80)
    corr_low = op.correlation(low, mean_vol80, 8)
    rank2 = op.rank(op.decay_linear(corr_low, 17))

    return op._max(rank1, rank2) * -1


@register_factor("gtja62", category="gtja191",
                 requires=["high", "volume"],
                 description="Alpha 62: (-1 * CORR(HIGH, RANK(VOLUME), 5))")
def gtja62(ctx: FactorContext) -> pd.DataFrame:
    return -op.correlation(ctx.high, op.rank(ctx.volume), 5)


@register_factor("gtja63", category="gtja191",
                 requires=["close"],
                 description="Alpha 63: SMA(MAX(CLOSE-DELAY(CLOSE,1),0),6,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),6,1)*100")
def gtja63(ctx: FactorContext) -> pd.DataFrame:
    delta_close = op.delta(ctx.close, 1)
    up_val = delta_close.clip(lower=0)
    abs_val = delta_close.abs()

    sma_up = op.sma(up_val, 6, 1)
    sma_abs = op.sma(abs_val, 6, 1).replace(0, np.nan)

    return (sma_up / sma_abs) * 100


@register_factor("gtja64", category="gtja191",
                 requires=["close", "volume", "vwap"],
                 description="Alpha 64: MAX(RANK(...), RANK(...)) * -1")
def gtja64(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    vol = ctx.volume
    vwap = ctx.vwap

    corr1 = op.correlation(op.rank(vwap), op.rank(vol), 4)
    rank1 = op.rank(op.decay_linear(corr1, 4))

    mean_vol60 = op.ts_mean(vol, 60)
    corr2 = op.correlation(op.rank(close), op.rank(mean_vol60), 4)
    rank2 = op.rank(op.decay_linear(corr2.clip(-1, 1), 14))

    return op._max(rank1, rank2) * -1


@register_factor("gtja65", category="gtja191",
                 requires=["close"],
                 description="Alpha 65: MEAN(CLOSE,6)/CLOSE")
def gtja65(ctx: FactorContext) -> pd.DataFrame:
    return op.ts_mean(ctx.close, 6) / ctx.close.replace(0, np.nan)


@register_factor("gtja66", category="gtja191",
                 requires=["close"],
                 description="Alpha 66: (CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)*100")
def gtja66(ctx: FactorContext) -> pd.DataFrame:
    mean6 = op.ts_mean(ctx.close, 6)
    return (ctx.close - mean6) / mean6.replace(0, np.nan) * 100


@register_factor("gtja67", category="gtja191",
                 requires=["close"],
                 description="Alpha 67: SMA(MAX(CLOSE-DELAY(CLOSE,1),0),24,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),24,1)*100")
def gtja67(ctx: FactorContext) -> pd.DataFrame:
    delta_close = op.delta(ctx.close, 1)
    up_val = delta_close.clip(lower=0)
    abs_val = delta_close.abs()

    sma_up = op.sma(up_val, 24, 1)
    sma_abs = op.sma(abs_val, 24, 1).replace(0, np.nan)

    return (sma_up / sma_abs) * 100


@register_factor("gtja68", category="gtja191",
                 requires=["high", "low", "volume"],
                 description="Alpha 68: SMA of typical price momentum")
def gtja68(ctx: FactorContext) -> pd.DataFrame:
    mid = (ctx.high + ctx.low) / 2.0
    delayed_mid = (op.delay(ctx.high, 1) + op.delay(ctx.low, 1)) / 2.0
    diff = mid - delayed_mid
    hl = ctx.high - ctx.low
    vol = ctx.volume.replace(0, np.nan)
    result = diff * hl / vol

    return op.sma(result, 15, 2)


@register_factor("gtja69", category="gtja191",
                 requires=["open", "high", "low"],
                 description="Alpha 69: DTM/DBM comparison indicator")
def gtja69(ctx: FactorContext) -> pd.DataFrame:
    open_val = ctx.open
    high = ctx.high
    low = ctx.low

    cond_dtm = (high > op.delay(high, 1)) & (open_val >= op.delay(open_val, 1))
    cond_dbm = (low < op.delay(low, 1)) & (open_val <= op.delay(open_val, 1))

    dtm = np.maximum(high - open_val, open_val - low).where(cond_dtm, 0.0)
    dbm = np.maximum(open_val - low, high - open_val).where(cond_dbm, 0.0)

    sum_dtm = op.ts_sum(dtm, 20)
    sum_dbm = op.ts_sum(dbm, 20)

    cond_equal = sum_dtm == sum_dbm
    cond_dtm_gt = sum_dtm > sum_dbm

    result = (
        ((sum_dtm - sum_dbm) / sum_dtm.replace(0, np.nan)).where(cond_dtm_gt, (sum_dtm - sum_dbm) / sum_dbm.replace(0, np.nan))
        .where(~cond_equal, 0.0)
    )

    return result


@register_factor("gtja70", category="gtja191",
                 requires=["volume", "vwap"],
                 description="Alpha 70: STD(AMOUNT,6) where AMOUNT = volume * vwap")
def gtja70(ctx: FactorContext) -> pd.DataFrame:
    amount = ctx.volume * ctx.vwap
    return op.ts_std(amount, 6)


@register_factor("gtja71", category="gtja191",
                 requires=["close"],
                 description="Alpha 71: (close - mean(close, 24)) / mean(close, 24) * 100")
def gtja71(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    mean_close = op.ts_mean(close, 24)
    return (close - mean_close) / mean_close * 100


@register_factor("gtja72", category="gtja191",
                 requires=["close", "high", "low"],
                 description="Alpha 72: SMA((tsmax(high,6)-close)/(tsmax(high,6)-tsmin(low,6))*100, 15, 1)")
def gtja72(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    high = ctx.high
    low = ctx.low
    ts_max_high = op.ts_max(high, 6)
    ts_min_low = op.ts_min(low, 6)
    numerator = ts_max_high - close
    denominator = (ts_max_high - ts_min_low).replace(0, np.nan)
    result = (numerator / denominator) * 100
    return op.sma(result, 15, 1)


@register_factor("gtja73", category="gtja191",
                 requires=["close", "volume", "vwap"],
                 description="Alpha 73: tsrank(decay_linear(decay_linear(corr(close, volume, 10), 16), 4), 5) - rank(decay_linear(corr(vwap, mean(volume,30), 4),3)) * -1")
def gtja73(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    volume = ctx.volume
    vwap = ctx.vwap
    mean_vol_30 = op.ts_mean(volume, 30)
    corr_close_vol = op.correlation(close, volume, 10)
    dec_corr_16 = op.decay_linear(corr_close_vol, 16)
    dec_corr_16_4 = op.decay_linear(dec_corr_16, 4)
    ts_rank_dec = op.ts_rank(dec_corr_16_4, 5)
    corr_vwap_vol = op.correlation(vwap, mean_vol_30, 4)
    dec_corr_vwap = op.decay_linear(corr_vwap_vol, 3)
    rank_dec_corr = op.rank(dec_corr_vwap)
    return (ts_rank_dec - rank_dec_corr) * -1


@register_factor("gtja74", category="gtja191",
                 requires=["low", "volume", "vwap"],
                 description="Alpha 74: rank(corr(sum(((low * 0.35) + (vwap * 0.65)), 20), sum(mean(volume,40), 20), 7)) + rank(corr(rank(vwap), rank(volume), 6))")
def gtja74(ctx: FactorContext) -> pd.DataFrame:
    low = ctx.low
    vwap = ctx.vwap
    volume = ctx.volume
    price_sum = op.ts_sum(low * 0.35 + vwap * 0.65, 20)
    vol_sum = op.ts_sum(op.ts_mean(volume, 40), 20)
    corr1 = op.correlation(price_sum, vol_sum, 7)
    rank1 = op.rank(corr1)
    rank2 = op.rank(op.correlation(op.rank(vwap), op.rank(volume), 6))
    return rank1 + rank2


@register_factor("gtja75", category="gtja191",
                 requires=["close"],
                 description="Alpha 75: [SKIP - requires index] - return 0")
def gtja75(ctx: FactorContext) -> pd.DataFrame:
    return pd.DataFrame(0, index=ctx.close.index, columns=ctx.close.columns)


@register_factor("gtja76", category="gtja191",
                 requires=["close", "volume"],
                 description="Alpha 76: ts_std(abs((close/delay(close,1)-1))/volume, 20) / mean(abs((close/delay(close,1)-1))/volume, 20)")
def gtja76(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    volume = ctx.volume
    delay_close = op.delay(close, 1)
    returns = (close / delay_close - 1).abs()
    ratio = returns / volume.replace(0, np.nan)
    ts_std_ratio = op.ts_std(ratio, 20)
    mean_ratio = op.ts_mean(ratio, 20)
    return ts_std_ratio / mean_ratio


@register_factor("gtja77", category="gtja191",
                 requires=["high", "low", "volume", "vwap"],
                 description="Alpha 77: min(rank(decay_linear(((((high + low) / 2) + high) - (vwap + high)), 20)), rank(decay_linear(corr(((high + low) / 2), mean(volume,40), 3), 6)))")
def gtja77(ctx: FactorContext) -> pd.DataFrame:
    high = ctx.high
    low = ctx.low
    volume = ctx.volume
    vwap = ctx.vwap
    mid = (high + low) / 2.0
    term1 = ((mid + high) - (vwap + high))
    dec1 = op.decay_linear(term1, 20)
    rank1 = op.rank(dec1)
    corr_val = op.correlation(mid, op.ts_mean(volume, 40), 3)
    dec2 = op.decay_linear(corr_val, 6)
    rank2 = op.rank(dec2)
    return np.minimum(rank1, rank2)


@register_factor("gtja78", category="gtja191",
                 requires=["close", "high", "low"],
                 description="Alpha 78: ((high+low+close)/3 - mean((high+low+close)/3, 12)) / (0.015 * mean(abs(close - mean((high+low+close)/3, 12)), 12))")
def gtja78(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    high = ctx.high
    low = ctx.low
    typical = (high + low + close) / 3.0
    mean_typical = op.ts_mean(typical, 12)
    numerator = typical - mean_typical
    denominator = 0.015 * op.ts_mean((close - mean_typical).abs(), 12)
    return numerator / denominator.replace(0, np.nan)


@register_factor("gtja79", category="gtja191",
                 requires=["close"],
                 description="Alpha 79: sma(max(close-delay(close,1), 0), 12, 1) / sma(abs(close-delay(close,1)), 12, 1) * 100")
def gtja79(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    delay_close = op.delay(close, 1)
    diff = close - delay_close
    pos_diff = diff.clip(lower=0)
    numerator = op.sma(pos_diff, 12, 1)
    denominator = op.sma(diff.abs(), 12, 1)
    return (numerator / denominator.replace(0, np.nan)) * 100


@register_factor("gtja80", category="gtja191",
                 requires=["volume"],
                 description="Alpha 80: (volume - delay(volume, 5)) / delay(volume, 5) * 100")
def gtja80(ctx: FactorContext) -> pd.DataFrame:
    volume = ctx.volume
    delay_vol = op.delay(volume, 5)
    return ((volume - delay_vol) / delay_vol.replace(0, np.nan)) * 100


@register_factor("gtja81", category="gtja191",
                 requires=["volume"],
                 description="Alpha 81: sma(volume, 21, 2)")
def gtja81(ctx: FactorContext) -> pd.DataFrame:
    volume = ctx.volume
    return op.sma(volume, 21, 2)


@register_factor("gtja82", category="gtja191",
                 requires=["close", "high", "low"],
                 description="Alpha 82: SMA((tsmax(high,6)-close)/(tsmax(high,6)-tsmin(low,6))*100, 20, 1)")
def gtja82(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    high = ctx.high
    low = ctx.low
    ts_max_high = op.ts_max(high, 6)
    ts_min_low = op.ts_min(low, 6)
    numerator = ts_max_high - close
    denominator = (ts_max_high - ts_min_low).replace(0, np.nan)
    result = (numerator / denominator) * 100
    return op.sma(result, 20, 1)


@register_factor("gtja83", category="gtja191",
                 requires=["high", "volume"],
                 description="Alpha 83: -rank(covariance(rank(high), rank(volume), 5))")
def gtja83(ctx: FactorContext) -> pd.DataFrame:
    high = ctx.high
    volume = ctx.volume
    rank_high = op.rank(high)
    rank_vol = op.rank(volume)
    cov_matrix = rank_high.rolling(5).cov(rank_vol)
    return -op.rank(cov_matrix)


@register_factor("gtja84", category="gtja191",
                 requires=["close", "volume"],
                 description="Alpha 84: sum((close>delay(close,1)?volume:(close<delay(close,1)?-volume:0)), 20)")
def gtja84(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    volume = ctx.volume
    delay_close = op.delay(close, 1)
    cond_up = close > delay_close
    cond_down = close < delay_close
    result = volume.where(cond_up, (-volume).where(cond_down, 0.0))
    return op.ts_sum(result, 20)


@register_factor("gtja85", category="gtja191",
                 requires=["close", "volume"],
                 description="Alpha 85: tsrank(volume/mean(volume,20), 20) * tsrank(-delta(close, 7), 8)")
def gtja85(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    volume = ctx.volume
    mean_vol = op.ts_mean(volume, 20)
    vol_ratio = volume / mean_vol.replace(0, np.nan)
    ts_rank_vol = op.ts_rank(vol_ratio, 20)
    delta_close = op.delta(close, 7)
    ts_rank_delta = op.ts_rank(-delta_close, 8)
    return ts_rank_vol * ts_rank_delta


@register_factor("gtja86", category="gtja191",
                 requires=["close"],
                 description="Alpha 86: complex conditional formula - return 0 as simplified")
def gtja86(ctx: FactorContext) -> pd.DataFrame:
    return pd.DataFrame(0, index=ctx.close.index, columns=ctx.close.columns)


@register_factor("gtja87", category="gtja191",
                 requires=["open", "high", "low", "vwap"],
                 description="Alpha 87: rank(decay_linear(delta(vwap, 4), 7)) + tsrank(decay_linear(((low * 0.9 + low * 0.1 - vwap) / (open - (high + low)/2), 11), 7)) * -1")
def gtja87(ctx: FactorContext) -> pd.DataFrame:
    open_val = ctx.open
    high = ctx.high
    low = ctx.low
    vwap = ctx.vwap
    delta_vwap = op.delta(vwap, 4)
    dec_delta = op.decay_linear(delta_vwap, 7)
    rank_dec = op.rank(dec_delta)
    term = ((low * 0.9 + low * 0.1 - vwap) / (open_val - (high + low) / 2.0)).replace(0, np.nan)
    dec_term = op.decay_linear(term, 11)
    ts_rank_term = op.ts_rank(dec_term, 7)
    return rank_dec + ts_rank_term * -1


@register_factor("gtja88", category="gtja191",
                 requires=["close"],
                 description="Alpha 88: (close - delay(close, 20)) / delay(close, 20) * 100")
def gtja88(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    delay_close = op.delay(close, 20)
    return ((close - delay_close) / delay_close.replace(0, np.nan)) * 100


@register_factor("gtja89", category="gtja191",
                 requires=["close"],
                 description="Alpha 89: 2*(sma(close,13,2)-sma(close,27,2)-sma(sma(close,13,2)-sma(close,27,2),10,2))")
def gtja89(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    sma_13 = op.sma(close, 13, 2)
    sma_27 = op.sma(close, 27, 2)
    diff = sma_13 - sma_27
    sma_diff = op.sma(diff, 10, 2)
    return 2 * (diff - sma_diff)


@register_factor("gtja90", category="gtja191",
                 requires=["volume", "vwap"],
                 description="Alpha 90: rank(corr(rank(vwap), rank(volume), 5)) * -1")
def gtja90(ctx: FactorContext) -> pd.DataFrame:
    vwap = ctx.vwap
    volume = ctx.volume
    corr_val = op.correlation(op.rank(vwap), op.rank(volume), 5)
    return op.rank(corr_val) * -1


@register_factor("gtja91", category="gtja191",
                 requires=["close", "low", "volume"],
                 description="Alpha 91: rank(close - max(close, 5)) * rank(corr(mean(volume,40), low, 5)) * -1")
def gtja91(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    low = ctx.low
    volume = ctx.volume
    max_close_5 = close.where(close > 5, 5.0)
    rank1 = op.rank(close - max_close_5)
    mean_vol = op.ts_mean(volume, 40)
    corr_val = op.correlation(mean_vol, low, 5)
    rank2 = op.rank(corr_val)
    return rank1 * rank2 * -1


@register_factor("gtja92", category="gtja191",
                 requires=["close", "volume", "vwap"],
                 description="Alpha 92: max(rank(decay_linear(delta((close * 0.35 + vwap * 0.65), 2), 3)), tsrank(decay_linear(abs(corr(mean(volume,180), close, 13)), 5), 15)) * -1")
def gtja92(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    volume = ctx.volume
    vwap = ctx.vwap
    delta_val = op.delta(close * 0.35 + vwap * 0.65, 2)
    dec_delta = op.decay_linear(delta_val, 3)
    rank1 = op.rank(dec_delta)
    corr_val = op.correlation(op.ts_mean(volume, 180), close, 13)
    dec_corr = op.decay_linear(corr_val.abs(), 5)
    ts_rank_corr = op.ts_rank(dec_corr, 15)
    result = np.maximum(rank1, ts_rank_corr)
    return result * -1


@register_factor("gtja93", category="gtja191",
                 requires=["open", "low"],
                 description="Alpha 93: sum((open>=delay(open,1)?0:max(open-low, open-delay(open,1))), 20)")
def gtja93(ctx: FactorContext) -> pd.DataFrame:
    open_val = ctx.open
    low = ctx.low
    delay_open = op.delay(open_val, 1)
    cond_down = open_val < delay_open
    term1 = open_val - low
    term2 = open_val - delay_open
    max_val = np.maximum(term1, term2)
    result = max_val.where(cond_down, 0.0)
    return op.ts_sum(result, 20)


@register_factor("gtja94", category="gtja191",
                 requires=["close", "volume"],
                 description="Alpha 94: sum((close>delay(close,1)?volume:(close<delay(close,1)?-volume:0)), 30)")
def gtja94(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    volume = ctx.volume
    delay_close = op.delay(close, 1)
    cond_up = close > delay_close
    cond_down = close < delay_close
    result = volume.where(cond_up, (-volume).where(cond_down, 0.0))
    return op.ts_sum(result, 30)


@register_factor("gtja95", category="gtja191",
                 requires=["volume", "vwap"],
                 description="Alpha 95: ts_std(volume * vwap, 20)")
def gtja95(ctx: FactorContext) -> pd.DataFrame:
    amount = ctx.volume * ctx.vwap
    return op.ts_std(amount, 20)


@register_factor("gtja96", category="gtja191",
                 requires=["close", "high", "low"],
                 description="Alpha 96: sma(sma((close-tsmin(low,9))/(tsmax(high,9)-tsmin(low,9))*100, 3, 1), 3, 1)")
def gtja96(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    high = ctx.high
    low = ctx.low
    ts_min_low = op.ts_min(low, 9)
    ts_max_high = op.ts_min(high, 9)
    numerator = close - ts_min_low
    denominator = (ts_max_high - ts_min_low).replace(0, np.nan)
    result = (numerator / denominator) * 100
    inner_sma = op.sma(result, 3, 1)
    return op.sma(inner_sma, 3, 1)


@register_factor("gtja97", category="gtja191",
                 requires=["volume"],
                 description="Alpha 97: ts_std(volume, 10)")
def gtja97(ctx: FactorContext) -> pd.DataFrame:
    return op.ts_std(ctx.volume, 10)


@register_factor("gtja98", category="gtja191",
                 requires=["close"],
                 description="Alpha 98: complex conditional - return -delta(close, 3) as approximation")
def gtja98(ctx: FactorContext) -> pd.DataFrame:
    return -op.delta(ctx.close, 3)


@register_factor("gtja99", category="gtja191",
                 requires=["close", "volume"],
                 description="Alpha 99: -rank(covariance(rank(close), rank(volume), 5))")
def gtja99(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    volume = ctx.volume
    rank_close = op.rank(close)
    rank_vol = op.rank(volume)
    cov_matrix = rank_close.rolling(5).cov(rank_vol)
    return -op.rank(cov_matrix)


@register_factor("gtja100", category="gtja191",
                 requires=["volume"],
                 description="Alpha 100: ts_std(volume, 20)")
def gtja100(ctx: FactorContext) -> pd.DataFrame:
    return op.ts_std(ctx.volume, 20)


@register_factor("gtja101", category="gtja191",
                 requires=["close", "high", "volume", "vwap"],
                 description="Alpha 101: (rank(corr(close, sum(mean(volume,30), 37), 15)) < rank(corr(rank(high * 0.1 + vwap * 0.9), rank(volume), 11))) * -1")
def gtja101(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    high = ctx.high
    volume = ctx.volume
    vwap = ctx.vwap
    sum_vol = op.ts_sum(op.ts_mean(volume, 30), 37)
    corr1 = op.correlation(close, sum_vol, 15)
    rank1 = op.rank(corr1)
    rank2 = op.correlation(op.rank(high * 0.1 + vwap * 0.9), op.rank(volume), 11)
    result = (rank1 < rank2).astype(float)
    return result * -1


@register_factor("gtja102", category="gtja191",
                 requires=["volume"],
                 description="Alpha 102: sma(max(volume-delay(volume,1), 0), 6, 1) / sma(abs(volume-delay(volume,1)), 6, 1) * 100")
def gtja102(ctx: FactorContext) -> pd.DataFrame:
    volume = ctx.volume
    delay_vol = op.delay(volume, 1)
    diff = volume - delay_vol
    pos_diff = diff.clip(lower=0)
    numerator = op.sma(pos_diff, 6, 1)
    denominator = op.sma(diff.abs(), 6, 1)
    return (numerator / denominator.replace(0, np.nan)) * 100


@register_factor("gtja103", category="gtja191",
                 requires=["low"],
                 description="Alpha 103: (20 - lowday(low, 20)) / 20 * 100")
def gtja103(ctx: FactorContext) -> pd.DataFrame:
    low = ctx.low
    lowday = low.rolling(20).apply(lambda x: (19 - np.argmax(x[::-1])) if len(x) == 20 else np.nan, raw=True)
    return (20 - lowday) / 20 * 100


@register_factor("gtja104", category="gtja191",
                 requires=["close", "high", "volume"],
                 description="Alpha 104: -delta(corr(high, volume, 5), 5) * rank(ts_std(close, 20))")
def gtja104(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    high = ctx.high
    volume = ctx.volume
    corr_val = op.correlation(high, volume, 5)
    delta_corr = op.delta(corr_val, 5)
    ts_std_close = op.ts_std(close, 20)
    rank_std = op.rank(ts_std_close)
    return -delta_corr * rank_std


@register_factor("gtja105", category="gtja191",
                 requires=["open", "volume"],
                 description="Alpha 105: -corr(rank(open), rank(volume), 10)")
def gtja105(ctx: FactorContext) -> pd.DataFrame:
    open_val = ctx.open
    volume = ctx.volume
    corr_val = op.correlation(op.rank(open_val), op.rank(volume), 10)
    return -corr_val


@register_factor("gtja106", category="gtja191",
                 requires=["close"],
                 description="Alpha 106: close - delay(close, 20)")
def gtja106(ctx: FactorContext) -> pd.DataFrame:
    return ctx.close - op.delay(ctx.close, 20)


@register_factor("gtja107", category="gtja191",
                 requires=["open", "close", "high", "low"],
                 description="Alpha 107: -rank(open - delay(high, 1)) * rank(open - delay(close, 1)) * rank(open - delay(low, 1))")
def gtja107(ctx: FactorContext) -> pd.DataFrame:
    open_val = ctx.open
    close = ctx.close
    high = ctx.high
    low = ctx.low
    rank1 = op.rank(open_val - op.delay(high, 1))
    rank2 = op.rank(open_val - op.delay(close, 1))
    rank3 = op.rank(open_val - op.delay(low, 1))
    return -rank1 * rank2 * rank3


@register_factor("gtja108", category="gtja191",
                 requires=["high", "volume", "vwap"],
                 description="Alpha 108: rank(high - min(high, 2)) ** rank(corr(vwap, mean(volume,120), 6)) * -1")
def gtja108(ctx: FactorContext) -> pd.DataFrame:
    high = ctx.high
    volume = ctx.volume
    vwap = ctx.vwap
    min_high_2 = high.where(high < 2, 2.0)
    rank1 = op.rank(high - min_high_2)
    corr_val = op.correlation(vwap, op.ts_mean(volume, 120), 6)
    rank2 = op.rank(corr_val)
    return (rank1 ** rank2) * -1


@register_factor("gtja109", category="gtja191",
                 requires=["high", "low"],
                 description="Alpha 109: sma(high-low, 10, 2) / sma(sma(high-low, 10, 2), 10, 2)")
def gtja109(ctx: FactorContext) -> pd.DataFrame:
    hl = ctx.high - ctx.low
    sma1 = op.sma(hl, 10, 2)
    sma2 = op.sma(sma1, 10, 2)
    return sma1 / sma2.replace(0, np.nan)


@register_factor("gtja110", category="gtja191",
                 requires=["close", "high", "low"],
                 description="Alpha 110: sum(max(0, high-delay(close,1)), 20) / sum(max(0, delay(close,1)-low), 20) * 100")
def gtja110(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    high = ctx.high
    low = ctx.low
    delay_close = op.delay(close, 1)
    up_sum = op.ts_sum((high - delay_close).clip(lower=0), 20)
    down_sum = op.ts_sum((delay_close - low).clip(lower=0), 20)
    return (up_sum / down_sum.replace(0, np.nan)) * 100


@register_factor("gtja111", category="gtja191",
                 requires=["close", "high", "low", "volume"],
                 description="Alpha 111: sma(volume * ((close-low)-(high-close))/(high-low), 11, 2) - sma(volume * ((close-low)-(high-close))/(high-low), 4, 2)")
def gtja111(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    high = ctx.high
    low = ctx.low
    volume = ctx.volume
    hl = high - low
    term = ((close - low) - (high - close)) / hl.replace(0, np.nan)
    result = volume * term
    return op.sma(result, 11, 2) - op.sma(result, 4, 2)


@register_factor("gtja112", category="gtja191",
                 requires=["close"],
                 description="Alpha 112: (sum(max(0, close-delay(close,1)), 12) - sum(min(0, close-delay(close,1)), 12)) / (sum(max(0, close-delay(close,1)), 12) + sum(min(0, close-delay(close,1)), 12)) * 100")
def gtja112(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    delay_close = op.delay(close, 1)
    diff = close - delay_close
    up_sum = op.ts_sum(diff.clip(lower=0), 12)
    down_sum = op.ts_sum(diff.clip(upper=0).abs(), 12)
    return ((up_sum - down_sum) / (up_sum + down_sum.replace(0, np.nan))) * 100


@register_factor("gtja113", category="gtja191",
                 requires=["close", "volume"],
                 description="Alpha 113: -rank((sum(delay(close,5), 20)/20) * corr(close, volume, 2) * rank(corr(sum(close,5), sum(close,20), 2)))")
def gtja113(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    volume = ctx.volume
    sum_delay = op.ts_sum(op.delay(close, 5), 20) / 20
    corr1 = op.correlation(close, volume, 2)
    corr2 = op.correlation(op.ts_sum(close, 5), op.ts_sum(close, 20), 2)
    rank_corr = op.rank(corr2)
    return -op.rank(sum_delay * corr1 * rank_corr)


@register_factor("gtja114", category="gtja191",
                 requires=["close", "high", "low", "volume", "vwap"],
                 description="Alpha 114: rank(delay((high-low)/(sum(close,5)/5), 2)) * rank(rank(volume)) / ((high-low)/(sum(close,5)/5)/(vwap-close))")
def gtja114(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    high = ctx.high
    low = ctx.low
    volume = ctx.volume
    vwap = ctx.vwap
    hl = high - low
    mean_close = op.ts_sum(close, 5) / 5
    ratio = hl / mean_close
    delay_ratio = op.delay(ratio, 2)
    rank1 = op.rank(delay_ratio)
    rank2 = op.rank(op.rank(volume))
    denominator = ratio / (vwap - close).replace(0, np.nan)
    return (rank1 * rank2) / denominator


@register_factor("gtja115", category="gtja191",
                 requires=["close", "high", "low", "volume"],
                 description="Alpha 115: rank(corr(high * 0.9 + close * 0.1, mean(volume,30), 10)) ** rank(corr(tsrank((high+low)/2, 4), tsrank(volume, 10), 7))")
def gtja115(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    high = ctx.high
    low = ctx.low
    volume = ctx.volume
    price = high * 0.9 + close * 0.1
    corr1 = op.correlation(price, op.ts_mean(volume, 30), 10)
    rank1 = op.rank(corr1)
    mid = (high + low) / 2.0
    ts_rank_mid = op.ts_rank(mid, 4)
    ts_rank_vol = op.ts_rank(volume, 10)
    corr2 = op.correlation(ts_rank_mid, ts_rank_vol, 7)
    rank2 = op.rank(corr2)
    return rank1 ** rank2


@register_factor("gtja116", category="gtja191",
                 requires=["close"],
                 description="Alpha 116: Linear regression slope of close over 20 days - simplified as delta/19")
def gtja116(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    return op.delta(close, 19) / 19


@register_factor("gtja117", category="gtja191",
                 requires=["close", "high", "low", "volume"],
                 description="Alpha 117: tsrank(volume, 32) * (1 - tsrank((close+high)-low, 16)) * (1 - tsrank(returns, 32))")
def gtja117(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    high = ctx.high
    low = ctx.low
    volume = ctx.volume
    ts_rank_vol = op.ts_rank(volume, 32)
    ts_rank_price = op.ts_rank((close + high) - low, 16)
    returns = (close / op.delay(close, 1) - 1).replace(0, np.nan)
    ts_rank_ret = op.ts_rank(returns, 32)
    return ts_rank_vol * (1 - ts_rank_price) * (1 - ts_rank_ret)


@register_factor("gtja118", category="gtja191",
                 requires=["open", "high", "low"],
                 description="Alpha 118: sum(high-open, 20) / sum(open-low, 20) * 100")
def gtja118(ctx: FactorContext) -> pd.DataFrame:
    open_val = ctx.open
    high = ctx.high
    low = ctx.low
    up_sum = op.ts_sum(high - open_val, 20)
    down_sum = op.ts_sum(open_val - low, 20)
    return (up_sum / down_sum.replace(0, np.nan)) * 100


@register_factor("gtja119", category="gtja191",
                 requires=["open", "volume", "vwap"],
                 description="Alpha 119: rank(decay_linear(corr(vwap, sum(mean(volume,5), 26), 5), 7)) - rank(decay_linear(tsrank(min(corr(rank(open), rank(mean(volume,15)), 21), 9), 7), 8))")
def gtja119(ctx: FactorContext) -> pd.DataFrame:
    open_val = ctx.open
    volume = ctx.volume
    vwap = ctx.vwap
    sum_vol = op.ts_sum(op.ts_mean(volume, 5), 26)
    corr1 = op.correlation(vwap, sum_vol, 5)
    dec1 = op.decay_linear(corr1, 7)
    rank1 = op.rank(dec1)
    mean_vol = op.ts_mean(volume, 15)
    corr2 = op.correlation(op.rank(open_val), op.rank(mean_vol), 21)
    min_corr = corr2.clip(upper=9)
    ts_rank_min = op.ts_rank(min_corr, 7)
    dec2 = op.decay_linear(ts_rank_min, 8)
    rank2 = op.rank(dec2)
    return rank1 - rank2


@register_factor("gtja120", category="gtja191",
                 requires=["close", "vwap"],
                 description="Alpha 120: rank(vwap - close) / rank(vwap + close)")
def gtja120(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    vwap = ctx.vwap
    numerator = op.rank(vwap - close)
    denominator = op.rank(vwap + close)
    return numerator / denominator.replace(0, np.nan)


@register_factor("gtja121", category="gtja191",
                 requires=["volume", "vwap"],
                 description="Alpha 121: rank(vwap - min(vwap, 12)) ** tsrank(corr(tsrank(vwap, 20), tsrank(mean(volume,60), 2), 18), 3) * -1")
def gtja121(ctx: FactorContext) -> pd.DataFrame:
    volume = ctx.volume
    vwap = ctx.vwap
    min_vwap_12 = vwap.where(vwap < 12, 12.0)
    rank1 = op.rank(vwap - min_vwap_12)
    ts_rank_vwap = op.ts_rank(vwap, 20)
    ts_rank_vol = op.ts_rank(op.ts_mean(volume, 60), 2)
    corr_val = op.correlation(ts_rank_vwap, ts_rank_vol, 18)
    ts_rank_corr = op.ts_rank(corr_val, 3)
    return (rank1 ** ts_rank_corr) * -1


@register_factor("gtja122", category="gtja191",
                 requires=["close"],
                 description="Alpha 122: Triple SMA of log(close) - simplified as delta(log(close), 1)")
def gtja122(ctx: FactorContext) -> pd.DataFrame:
    log_close = np.log(ctx.close.replace(0, np.nan))
    return op.delta(log_close, 1)


@register_factor("gtja123", category="gtja191",
                 requires=["high", "low", "volume"],
                 description="Alpha 123: (rank(corr(sum((high+low)/2, 20), sum(mean(volume,60), 20), 9)) < rank(corr(low, volume, 6))) * -1")
def gtja123(ctx: FactorContext) -> pd.DataFrame:
    high = ctx.high
    low = ctx.low
    volume = ctx.volume
    mid = (high + low) / 2.0
    sum_mid = op.ts_sum(mid, 20)
    sum_vol = op.ts_sum(op.ts_mean(volume, 60), 20)
    corr1 = op.correlation(sum_mid, sum_vol, 9)
    rank1 = op.rank(corr1)
    corr2 = op.correlation(low, volume, 6)
    rank2 = op.rank(corr2)
    result = (rank1 < rank2).astype(float)
    return result * -1


@register_factor("gtja124", category="gtja191",
                 requires=["close", "vwap"],
                 description="Alpha 124: (close - vwap) / decay_linear(rank(tsmax(close, 30)), 2)")
def gtja124(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    vwap = ctx.vwap
    ts_max_close = op.ts_max(close, 30)
    rank_ts_max = op.rank(ts_max_close)
    decay_val = op.decay_linear(rank_ts_max, 2)
    return (close - vwap) / decay_val.replace(0, np.nan)


@register_factor("gtja125", category="gtja191",
                 requires=["close", "volume", "vwap"],
                 description="Alpha 125: rank(decay_linear(corr(vwap, mean(volume,80), 17), 20)) / rank(decay_linear(delta(close * 0.5 + vwap * 0.5, 3), 16))")
def gtja125(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    volume = ctx.volume
    vwap = ctx.vwap
    corr_val = op.correlation(vwap, op.ts_mean(volume, 80), 17)
    dec1 = op.decay_linear(corr_val, 20)
    rank1 = op.rank(dec1)
    delta_val = op.delta(close * 0.5 + vwap * 0.5, 3)
    dec2 = op.decay_linear(delta_val, 16)
    rank2 = op.rank(dec2)
    return rank1 / rank2.replace(0, np.nan)


@register_factor("gtja126", category="gtja191",
                 requires=["close", "high", "low"],
                 description="Alpha 126: (close + high + low) / 3")
def gtja126(ctx: FactorContext) -> pd.DataFrame:
    return (ctx.close + ctx.high + ctx.low) / 3.0


@register_factor("gtja127", category="gtja191",
                 requires=["close"],
                 description="Alpha 127: sqrt(mean((100 * (close - max(close,12)) / max(close,12)) ** 2, 12))")
def gtja127(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    max_close_12 = close.where(close > 12, 12.0)
    ratio = (close - max_close_12) / max_close_12
    pct_change = ratio * 100
    squared = pct_change ** 2
    return np.sqrt(op.ts_mean(squared, 12))


@register_factor("gtja128", category="gtja191",
                 requires=["close", "high", "low", "volume"],
                 description="Alpha 128: 100 - 100/(1 + sum((tp>delay(tp,1)?tp*volume:0), 14)/sum((tp<delay(tp,1)?tp*volume:0), 14)) where tp=(high+low+close)/3")
def gtja128(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    high = ctx.high
    low = ctx.low
    volume = ctx.volume
    tp = (high + low + close) / 3.0
    delay_tp = op.delay(tp, 1)
    cond_up = tp > delay_tp
    cond_down = tp < delay_tp
    up_val = (tp * volume).where(cond_up, 0.0)
    down_val = (tp * volume).where(cond_down, 0.0)
    sum_up = op.ts_sum(up_val, 14)
    sum_down = op.ts_sum(down_val, 14)
    return 100 - 100 / (1 + sum_up / sum_down.replace(0, np.nan))


@register_factor("gtja129", category="gtja191",
                 requires=["close"],
                 description="Alpha 129: sum(abs(close - delay(close,1)) where close < delay(close,1), 12)")
def gtja129(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    delay_close = op.delay(close, 1)
    cond_down = close < delay_close
    diff = (close - delay_close).abs()
    result = diff.where(cond_down, 0.0)
    return op.ts_sum(result, 12)


@register_factor("gtja130", category="gtja191",
                 requires=["high", "low", "volume", "vwap"],
                 description="Alpha 130: rank(decay_linear(corr((high+low)/2, mean(volume,40), 9), 10)) / rank(decay_linear(corr(rank(vwap), rank(volume), 7), 3))")
def gtja130(ctx: FactorContext) -> pd.DataFrame:
    high = ctx.high
    low = ctx.low
    volume = ctx.volume
    vwap = ctx.vwap
    mid = (high + low) / 2.0
    corr1 = op.correlation(mid, op.ts_mean(volume, 40), 9)
    dec1 = op.decay_linear(corr1, 10)
    rank1 = op.rank(dec1)
    corr2 = op.correlation(op.rank(vwap), op.rank(volume), 7)
    dec2 = op.decay_linear(corr2, 3)
    rank2 = op.rank(dec2)
    return rank1 / rank2.replace(0, np.nan)


@register_factor("gtja131", category="gtja191",
                 requires=["close", "volume", "vwap"],
                 description="Alpha 131: rank(delta(vwap, 1)) ** tsrank(corr(close, mean(volume,50), 18), 18)")
def gtja131(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    volume = ctx.volume
    vwap = ctx.vwap
    delta_vwap = op.delta(vwap, 1)
    rank_delta = op.rank(delta_vwap)
    corr_val = op.correlation(close, op.ts_mean(volume, 50), 18)
    ts_rank_corr = op.ts_rank(corr_val, 18)
    return rank_delta ** ts_rank_corr


@register_factor("gtja132", category="gtja191",
                 requires=["volume", "vwap"],
                 description="Alpha 132: mean(volume * vwap, 20)")
def gtja132(ctx: FactorContext) -> pd.DataFrame:
    return op.ts_mean(ctx.volume * ctx.vwap, 20)


@register_factor("gtja133", category="gtja191",
                 requires=["high", "low"],
                 description="Alpha 133: (20 - highday(high, 20))/20*100 - (20 - lowday(low, 20))/20*100")
def gtja133(ctx: FactorContext) -> pd.DataFrame:
    high = ctx.high
    low = ctx.low
    highday = high.rolling(20).apply(lambda x: (19 - np.argmax(x[::-1])) if len(x) == 20 else np.nan, raw=True)
    lowday = low.rolling(20).apply(lambda x: (19 - np.argmin(x[::-1])) if len(x) == 20 else np.nan, raw=True)
    return ((20 - highday) / 20 * 100) - ((20 - lowday) / 20 * 100)


@register_factor("gtja134", category="gtja191",
                 requires=["close", "volume"],
                 description="Alpha 134: (close - delay(close, 12)) / delay(close, 12) * volume")
def gtja134(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    volume = ctx.volume
    delay_close = op.delay(close, 12)
    return ((close - delay_close) / delay_close.replace(0, np.nan)) * volume


@register_factor("gtja135", category="gtja191",
                 requires=["close"],
                 description="Alpha 135: sma(delay(close/delay(close,20), 1), 20, 1)")
def gtja135(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    delay_close_20 = op.delay(close, 20)
    ratio = close / delay_close_20.replace(0, np.nan)
    delayed_ratio = op.delay(ratio, 1)
    return op.sma(delayed_ratio, 20, 1)


@register_factor("gtja136", category="gtja191",
                 requires=["open", "close", "volume"],
                 description="Alpha 136: -rank(delta(returns, 3)) * corr(open, volume, 10)")
def gtja136(ctx: FactorContext) -> pd.DataFrame:
    open_val = ctx.open
    close = ctx.close
    volume = ctx.volume
    returns = close / op.delay(close, 1).replace(0, np.nan) - 1
    delta_ret = op.delta(returns, 3)
    rank_delta = op.rank(delta_ret)
    corr_val = op.correlation(open_val, volume, 10)
    return -rank_delta * corr_val


@register_factor("gtja137", category="gtja191",
                 requires=["open", "close", "high", "low"],
                 description="Alpha 137: Complex formula - simplified as -delta(close, 1)")
def gtja137(ctx: FactorContext) -> pd.DataFrame:
    return -op.delta(ctx.close, 1)


@register_factor("gtja138", category="gtja191",
                 requires=["low", "volume", "vwap"],
                 description="Alpha 138: (rank(decay_linear(delta(low * 0.7 + vwap * 0.3, 3), 20)) - tsrank(decay_linear(tsrank(corr(tsrank(low, 8), tsrank(mean(volume,60), 17), 5), 19), 16), 7)) * -1")
def gtja138(ctx: FactorContext) -> pd.DataFrame:
    low = ctx.low
    volume = ctx.volume
    vwap = ctx.vwap
    mixed = low * 0.7 + vwap * 0.3
    delta_mixed = op.delta(mixed, 3)
    dec_delta = op.decay_linear(delta_mixed, 20)
    rank1 = op.rank(dec_delta)
    ts_rank_low = op.ts_rank(low, 8)
    ts_rank_vol = op.ts_rank(op.ts_mean(volume, 60), 17)
    corr_val = op.correlation(ts_rank_low, ts_rank_vol, 5)
    ts_rank_corr = op.ts_rank(corr_val, 19)
    dec_ts_rank = op.decay_linear(ts_rank_corr, 16)
    ts_rank_dec = op.ts_rank(dec_ts_rank, 7)
    return (rank1 - ts_rank_dec) * -1


@register_factor("gtja139", category="gtja191",
                 requires=["open", "volume"],
                 description="Alpha 139: -corr(open, volume, 10)")
def gtja139(ctx: FactorContext) -> pd.DataFrame:
    return -op.correlation(ctx.open, ctx.volume, 10)


@register_factor("gtja140", category="gtja191",
                 requires=["open", "close", "high", "low", "volume"],
                 description="Alpha 140: min(rank(decay_linear((rank(open) + rank(low)) - (rank(high) + rank(close)), 8)), tsrank(decay_linear(corr(tsrank(close, 8), tsrank(mean(volume,60), 20), 8), 7), 3))")
def gtja140(ctx: FactorContext) -> pd.DataFrame:
    open_val = ctx.open
    close = ctx.close
    high = ctx.high
    low = ctx.low
    volume = ctx.volume
    term = (op.rank(open_val) + op.rank(low)) - (op.rank(high) + op.rank(close))
    dec_term = op.decay_linear(term, 8)
    rank1 = op.rank(dec_term)
    ts_rank_close = op.ts_rank(close, 8)
    ts_rank_vol = op.ts_rank(op.ts_mean(volume, 60), 20)
    corr_val = op.correlation(ts_rank_close, ts_rank_vol, 8)
    dec_corr = op.decay_linear(corr_val, 7)
    ts_rank_corr = op.ts_rank(dec_corr, 3)
    return np.minimum(rank1, ts_rank_corr)


@register_factor("gtja141", category="gtja191",
                 requires=["high", "volume"],
                 description="Alpha 141: rank(corr(rank(high), rank(mean(volume,15)), 9)) * -1")
def gtja141(ctx: FactorContext) -> pd.DataFrame:
    high = ctx.high
    volume = ctx.volume
    corr_val = op.correlation(op.rank(high), op.rank(op.ts_mean(volume, 15)), 9)
    return op.rank(corr_val) * -1


@register_factor("gtja142", category="gtja191",
                 requires=["close", "volume"],
                 description="Alpha 142: -rank(tsrank(close, 10)) * rank(delta(delta(close, 1), 1)) * rank(tsrank(volume/mean(volume,20), 5))")
def gtja142(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    volume = ctx.volume
    ts_rank_close = op.ts_rank(close, 10)
    delta1 = op.delta(close, 1)
    delta2 = op.delta(delta1, 1)
    vol_ratio = volume / op.ts_mean(volume, 20).replace(0, np.nan)
    ts_rank_vol = op.ts_rank(vol_ratio, 5)
    return -op.rank(ts_rank_close) * op.rank(delta2) * op.rank(ts_rank_vol)


@register_factor("gtja143", category="gtja191",
                 requires=["close"],
                 description="Alpha 143: cumulative return - simplified as (close/delay(close,20) - 1) * 100")
def gtja143(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    delay_close = op.delay(close, 20)
    return (close / delay_close.replace(0, np.nan) - 1) * 100


@register_factor("gtja144", category="gtja191",
                 requires=["close", "volume", "vwap"],
                 description="Alpha 144: sum(abs(close/delay(close,1)-1)/amount where close<delay(close,1), 20) / count(close<delay(close,1), 20) where amount=volume*vwap")
def gtja144(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    volume = ctx.volume
    vwap = ctx.vwap
    delay_close = op.delay(close, 1)
    cond_down = close < delay_close
    amount = volume * vwap
    returns = (close / delay_close.replace(0, np.nan) - 1).abs()
    ratio = returns / amount.replace(0, np.nan)
    result = ratio.where(cond_down, 0.0)
    sum_ratio = op.ts_sum(result, 20)
    count = cond_down.rolling(20).sum()
    return sum_ratio / count.replace(0, np.nan)


@register_factor("gtja145", category="gtja191",
                 requires=["volume"],
                 description="Alpha 145: (mean(volume,9) - mean(volume,26)) / mean(volume,12) * 100")
def gtja145(ctx: FactorContext) -> pd.DataFrame:
    volume = ctx.volume
    mean_vol_9 = op.ts_mean(volume, 9)
    mean_vol_26 = op.ts_mean(volume, 26)
    mean_vol_12 = op.ts_mean(volume, 12)
    return ((mean_vol_9 - mean_vol_26) / mean_vol_12.replace(0, np.nan)) * 100


@register_factor("gtja146", category="gtja191",
                 requires=["close"],
                 description="Alpha 146: complex - return 0")
def gtja146(ctx: FactorContext) -> pd.DataFrame:
    return pd.DataFrame(0, index=ctx.close.index, columns=ctx.close.columns)


@register_factor("gtja147", category="gtja191",
                 requires=["close"],
                 description="Alpha 147: Linear regression slope of mean(close,12) over 12 days")
def gtja147(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    mean_close = op.ts_mean(close, 12)
    return op.delta(mean_close, 11) / 11


@register_factor("gtja148", category="gtja191",
                 requires=["open", "volume"],
                 description="Alpha 148: (rank(corr(open, sum(mean(volume,60), 9), 6)) < rank(open - tsmin(open, 14))) * -1")
def gtja148(ctx: FactorContext) -> pd.DataFrame:
    open_val = ctx.open
    volume = ctx.volume
    sum_vol = op.ts_sum(op.ts_mean(volume, 60), 9)
    corr_val = op.correlation(open_val, sum_vol, 6)
    rank1 = op.rank(corr_val)
    rank2 = op.rank(open_val - op.ts_min(open_val, 14))
    result = (rank1 < rank2).astype(float)
    return result * -1


@register_factor("gtja149", category="gtja191",
                 requires=["close"],
                 description="Alpha 149: [SKIP - requires index] return 0")
def gtja149(ctx: FactorContext) -> pd.DataFrame:
    return pd.DataFrame(0, index=ctx.close.index, columns=ctx.close.columns)


@register_factor("gtja150", category="gtja191",
                 requires=["close", "high", "low", "volume"],
                 description="Alpha 150: (close + high + low) / 3 * volume")
def gtja150(ctx: FactorContext) -> pd.DataFrame:
    return ((ctx.close + ctx.high + ctx.low) / 3.0) * ctx.volume


@register_factor("gtja151", category="gtja191",
                 requires=["close"],
                 description="Alpha 151: sma(close - delay(close, 20), 20, 1)")
def gtja151(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    delay_close = op.delay(close, 20)
    return op.sma(close - delay_close, 20, 1)


@register_factor("gtja152", category="gtja191",
                 requires=["close"],
                 description="Alpha 152: sma(mean(delay(sma(delay(close/delay(close,9), 1), 9), 1), 12) - mean(delay(sma(delay(close/delay(close,9), 1), 9), 1), 26), 9, 1)")
def gtja152(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    ratio = close / op.delay(close, 9).replace(0, np.nan)
    delayed_ratio = op.delay(ratio, 1)
    ema = op.sma(delayed_ratio, 9, 1)
    delayed_ema = op.delay(ema, 1)
    mean_12 = op.ts_mean(delayed_ema, 12)
    mean_26 = op.ts_mean(delayed_ema, 26)
    return op.sma(mean_12 - mean_26, 9, 1)


@register_factor("gtja153", category="gtja191",
                 requires=["close"],
                 description="Alpha 153: (mean(close,3) + mean(close,6) + mean(close,12) + mean(close,24)) / 4")
def gtja153(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    mean_3 = op.ts_mean(close, 3)
    mean_6 = op.ts_mean(close, 6)
    mean_12 = op.ts_mean(close, 12)
    mean_24 = op.ts_mean(close, 24)
    return (mean_3 + mean_6 + mean_12 + mean_24) / 4.0


@register_factor("gtja154", category="gtja191",
                 requires=["volume", "vwap"],
                 description="Alpha 154: placeholder returning zeros")
def gtja154(ctx: FactorContext) -> pd.DataFrame:
    return pd.DataFrame(0, index=ctx.volume.index, columns=ctx.volume.columns)


@register_factor("gtja155", category="gtja191",
                 requires=["volume"],
                 description="Alpha 155: sma(volume, 13, 2) - sma(volume, 27, 2) - sma(sma(volume,13,2)-sma(volume,27,2), 10, 2)")
def gtja155(ctx: FactorContext) -> pd.DataFrame:
    volume = ctx.volume
    sma_13 = op.sma(volume, 13, 2)
    sma_27 = op.sma(volume, 27, 2)
    diff = sma_13 - sma_27
    sma_diff = op.sma(diff, 10, 2)
    return diff - sma_diff


@register_factor("gtja156", category="gtja191",
                 requires=["open", "low", "vwap"],
                 description="Alpha 156: (max(rank(decay_linear(delta(vwap, 5), 3)), rank(decay_linear((delta(open*0.15+low*0.85, 2)/(open*0.15+low*0.85)*-1), 3))) * -1")
def gtja156(ctx: FactorContext) -> pd.DataFrame:
    open_val = ctx.open
    low = ctx.low
    vwap = ctx.vwap
    delta_vwap = op.delta(vwap, 5)
    dec_vwap = op.decay_linear(delta_vwap, 3)
    rank1 = op.rank(dec_vwap)
    mixed = open_val * 0.15 + low * 0.85
    delta_mixed = op.delta(mixed, 2)
    term = (delta_mixed / mixed.replace(0, np.nan)) * -1
    dec_term = op.decay_linear(term, 3)
    rank2 = op.rank(dec_term)
    return np.maximum(rank1, rank2) * -1


@register_factor("gtja157", category="gtja191",
                 requires=["close"],
                 description="Alpha 157: [Complex - return 0]")
def gtja157(ctx: FactorContext) -> pd.DataFrame:
    return pd.DataFrame(0, index=ctx.close.index, columns=ctx.close.columns)


@register_factor("gtja158", category="gtja191",
                 requires=["close", "high", "low"],
                 description="Alpha 158: ((high - sma(close, 15, 2)) - (low - sma(close, 15, 2))) / close")
def gtja158(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    high = ctx.high
    low = ctx.low
    sma_close = op.sma(close, 15, 2)
    return ((high - sma_close) - (low - sma_close)) / close.replace(0, np.nan)


@register_factor("gtja159", category="gtja191",
                 requires=["close", "high", "low"],
                 description="Alpha 159: Weighted price oscillator - simplified as typical price")
def gtja159(ctx: FactorContext) -> pd.DataFrame:
    return (ctx.high + ctx.low + ctx.close) / 3.0


@register_factor("gtja160", category="gtja191",
                 requires=["close"],
                 description="Alpha 160: sma((close<=delay(close,1)?ts_std(close,20):0), 20, 1)")
def gtja160(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    delay_close = op.delay(close, 1)
    cond = close <= delay_close
    ts_std = op.ts_std(close, 20)
    result = ts_std.where(cond, 0.0)
    return op.sma(result, 20, 1)


@register_factor("gtja161", category="gtja191",
                 requires=["close", "high", "low"],
                 description="Alpha 161: mean(max(max(high-low, abs(delay(close,1)-high)), abs(delay(close,1)-low)), 12)")
def gtja161(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    high = ctx.high
    low = ctx.low
    delay_close = op.delay(close, 1)
    hl = high - low
    abs_diff_high = (delay_close - high).abs()
    abs_diff_low = (delay_close - low).abs()
    max1 = np.maximum(hl, abs_diff_high)
    max_val = np.maximum(max1, abs_diff_low)
    return op.ts_mean(max_val, 12)


@register_factor("gtja162", category="gtja191",
                 requires=["close"],
                 description="Alpha 162: complex RSI variant - simplified as delta(close, 1)")
def gtja162(ctx: FactorContext) -> pd.DataFrame:
    return op.delta(ctx.close, 1)


@register_factor("gtja163", category="gtja191",
                 requires=["close", "high", "volume", "vwap"],
                 description="Alpha 163: rank(-returns * mean(volume,20) * vwap * (high - close))")
def gtja163(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    high = ctx.high
    volume = ctx.volume
    vwap = ctx.vwap
    returns = close / op.delay(close, 1).replace(0, np.nan) - 1
    mean_vol = op.ts_mean(volume, 20)
    term = -returns * mean_vol * vwap * (high - close)
    return op.rank(term)


@register_factor("gtja164", category="gtja191",
                 requires=["close", "high", "low"],
                 description="Alpha 164: sma((1/(close-delay(close,1))-min(1/(close-delay(close,1)), 12))/(high-low)*100, 13, 2)")
def gtja164(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    high = ctx.high
    low = ctx.low
    delay_close = op.delay(close, 1)
    diff = close - delay_close
    inv_diff = 1 / diff.replace(0, np.nan)
    term1 = inv_diff - inv_diff.clip(upper=12)
    term2 = (high - low).replace(0, np.nan)
    result = (term1 / term2) * 100
    return op.sma(result, 13, 2)


@register_factor("gtja165", category="gtja191",
                 requires=["close"],
                 description="Alpha 165: max(sumac(close-mean(close,48))) - min(sumac(close-mean(close,48))) / ts_std(close, 48)")
def gtja165(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    mean_close = op.ts_mean(close, 48)
    diff = close - mean_close
    sumac_val = op.sumac(diff, 48)
    max_sumac = sumac_val.rolling(48).max()
    min_sumac = sumac_val.rolling(48).min()
    ts_std_close = op.ts_std(close, 48)
    return (max_sumac - min_sumac) / ts_std_close.replace(0, np.nan)


@register_factor("gtja166", category="gtja191",
                 requires=["close"],
                 description="Alpha 166: Skewness of returns - complex simplified as 0")
def gtja166(ctx: FactorContext) -> pd.DataFrame:
    return pd.DataFrame(0, index=ctx.close.index, columns=ctx.close.columns)


@register_factor("gtja167", category="gtja191",
                 requires=["close"],
                 description="Alpha 167: sum(max(0, close-delay(close,1)), 12)")
def gtja167(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    delay_close = op.delay(close, 1)
    diff = close - delay_close
    return op.ts_sum(diff.clip(lower=0), 12)


@register_factor("gtja168", category="gtja191",
                 requires=["volume"],
                 description="Alpha 168: -volume / mean(volume, 20)")
def gtja168(ctx: FactorContext) -> pd.DataFrame:
    volume = ctx.volume
    mean_vol = op.ts_mean(volume, 20)
    return -(volume / mean_vol.replace(0, np.nan))


@register_factor("gtja169", category="gtja191",
                 requires=["close"],
                 description="Alpha 169: sma(mean(delay(sma(close-delay(close,1), 9), 1), 12) - mean(delay(sma(close-delay(close,1), 9), 1), 26), 10, 1)")
def gtja169(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    diff = close - op.delay(close, 1)
    ema = op.sma(diff, 9, 1)
    delayed_ema = op.delay(ema, 1)
    mean_12 = op.ts_mean(delayed_ema, 12)
    mean_26 = op.ts_mean(delayed_ema, 26)
    return op.sma(mean_12 - mean_26, 10, 1)


@register_factor("gtja170", category="gtja191",
                 requires=["close", "high", "volume", "vwap"],
                 description="Alpha 170: rank(1/close) * volume/mean(volume,20) * high * rank(high-close) / (sum(high,5)/5) - rank(vwap-delay(vwap,5))")
def gtja170(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    high = ctx.high
    volume = ctx.volume
    vwap = ctx.vwap
    rank1 = op.rank(1 / close.replace(0, np.nan))
    vol_ratio = volume / op.ts_mean(volume, 20).replace(0, np.nan)
    rank2 = op.rank(high - close)
    mean_high_5 = op.ts_sum(high, 5) / 5
    sum_term = rank1 * vol_ratio * high * rank2 / mean_high_5.replace(0, np.nan)
    rank3 = op.rank(vwap - op.delay(vwap, 5))
    return sum_term - rank3


@register_factor("gtja171", category="gtja191",
                 requires=["open", "close", "high", "low"],
                 description="Alpha 171: -((low - close) * open**5) / ((close - high) * close**5)")
def gtja171(ctx: FactorContext) -> pd.DataFrame:
    open_val = ctx.open
    close = ctx.close
    high = ctx.high
    low = ctx.low
    numerator = -(low - close) * (open_val ** 5)
    denominator = (close - high) * (close ** 5)
    return numerator / denominator.replace(0, np.nan)


@register_factor("gtja172", category="gtja191",
                 requires=["close", "high", "low"],
                 description="Alpha 172: A/D indicator variant - simplified as (close - low) - (high - close)")
def gtja172(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    high = ctx.high
    low = ctx.low
    return (close - low) - (high - close)


@register_factor("gtja173", category="gtja191",
                 requires=["close"],
                 description="Alpha 173: Trix indicator - simplified as delta(log(close), 1)")
def gtja173(ctx: FactorContext) -> pd.DataFrame:
    log_close = np.log(ctx.close.replace(0, np.nan))
    return op.delta(log_close, 1)


@register_factor("gtja174", category="gtja191",
                 requires=["close"],
                 description="Alpha 174: sma((close>delay(close,1)?ts_std(close,20):0), 20, 1)")
def gtja174(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    delay_close = op.delay(close, 1)
    cond = close > delay_close
    ts_std = op.ts_std(close, 20)
    result = ts_std.where(cond, 0.0)
    return op.sma(result, 20, 1)


@register_factor("gtja175", category="gtja191",
                 requires=["close", "high", "low"],
                 description="Alpha 175: mean(max(max(high-low, abs(delay(close,1)-high)), abs(delay(close,1)-low)), 6)")
def gtja175(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    high = ctx.high
    low = ctx.low
    delay_close = op.delay(close, 1)
    hl = high - low
    abs_diff_high = (delay_close - high).abs()
    abs_diff_low = (delay_close - low).abs()
    max1 = np.maximum(hl, abs_diff_high)
    max_val = np.maximum(max1, abs_diff_low)
    return op.ts_mean(max_val, 6)


@register_factor("gtja176", category="gtja191",
                 requires=["close", "high", "low", "volume"],
                 description="Alpha 176: corr(rank((close-tsmin(low,12))/(tsmax(high,12)-tsmin(low,12))), rank(volume), 6)")
def gtja176(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    high = ctx.high
    low = ctx.low
    volume = ctx.volume
    ts_min_low = op.ts_min(low, 12)
    ts_max_high = op.ts_max(high, 12)
    denominator = (ts_max_high - ts_min_low).replace(0, np.nan)
    numerator = close - ts_min_low
    stoch = numerator / denominator
    rank_stoch = op.rank(stoch)
    rank_vol = op.rank(volume)
    return op.correlation(rank_stoch, rank_vol, 6)


@register_factor("gtja177", category="gtja191",
                 requires=["high"],
                 description="Alpha 177: (20 - highday(high, 20)) / 20 * 100")
def gtja177(ctx: FactorContext) -> pd.DataFrame:
    high = ctx.high
    highday = high.rolling(20).apply(lambda x: (19 - np.argmax(x[::-1])) if len(x) == 20 else np.nan, raw=True)
    return (20 - highday) / 20 * 100


@register_factor("gtja178", category="gtja191",
                 requires=["close", "volume"],
                 description="Alpha 178: (close - delay(close,1)) / delay(close,1) * volume")
def gtja178(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    volume = ctx.volume
    delay_close = op.delay(close, 1)
    return ((close / delay_close.replace(0, np.nan) - 1)) * volume


@register_factor("gtja179", category="gtja191",
                 requires=["low", "volume", "vwap"],
                 description="Alpha 179: rank(corr(vwap, volume, 4)) * rank(corr(rank(low), rank(mean(volume,50)), 12))")
def gtja179(ctx: FactorContext) -> pd.DataFrame:
    low = ctx.low
    volume = ctx.volume
    vwap = ctx.vwap
    corr1 = op.correlation(vwap, volume, 4)
    rank1 = op.rank(corr1)
    rank2 = op.correlation(op.rank(low), op.rank(op.ts_mean(volume, 50)), 12)
    rank2 = op.rank(rank2)
    return rank1 * rank2


@register_factor("gtja180", category="gtja191",
                 requires=["close", "volume"],
                 description="Alpha 180: (mean(volume,20) < volume) ? (-tsrank(abs(delta(close, 7)), 60) * sign(delta(close, 7))) : (-volume)")
def gtja180(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    volume = ctx.volume
    mean_vol = op.ts_mean(volume, 20)
    cond = mean_vol < volume
    delta_close = op.delta(close, 7)
    ts_rank_delta = op.ts_rank(delta_close.abs(), 60)
    sign_delta = np.sign(delta_close)
    term = -ts_rank_delta * sign_delta
    return term.where(cond, -volume)


@register_factor("gtja181", category="gtja191",
                 requires=["close"],
                 description="Alpha 181: [SKIP - requires index] return 0")
def gtja181(ctx: FactorContext) -> pd.DataFrame:
    return pd.DataFrame(0, index=ctx.close.index, columns=ctx.close.columns)


@register_factor("gtja182", category="gtja191",
                 requires=["close"],
                 description="Alpha 182: [SKIP - requires index] return 0")
def gtja182(ctx: FactorContext) -> pd.DataFrame:
    return pd.DataFrame(0, index=ctx.close.index, columns=ctx.close.columns)


@register_factor("gtja183", category="gtja191",
                 requires=["close"],
                 description="Alpha 183: max(sumac(close-mean(close,24))) - min(sumac(close-mean(close,24))) / ts_std(close, 24)")
def gtja183(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    mean_close = op.ts_mean(close, 24)
    diff = close - mean_close
    sumac_val = op.sumac(diff, 24)
    max_sumac = sumac_val.rolling(24).max()
    min_sumac = sumac_val.rolling(24).min()
    ts_std_close = op.ts_std(close, 24)
    return (max_sumac - min_sumac) / ts_std_close.replace(0, np.nan)


@register_factor("gtja184", category="gtja191",
                 requires=["open", "close"],
                 description="Alpha 184: rank(corr(delay(open-close, 1), close, 200)) + rank(open-close)")
def gtja184(ctx: FactorContext) -> pd.DataFrame:
    open_val = ctx.open
    close = ctx.close
    delay_diff = op.delay(open_val - close, 1)
    corr_val = op.correlation(delay_diff, close, 200)
    rank1 = op.rank(corr_val)
    rank2 = op.rank(open_val - close)
    return rank1 + rank2


@register_factor("gtja185", category="gtja191",
                 requires=["open", "close"],
                 description="Alpha 185: rank(-(1 - open/close)**2)")
def gtja185(ctx: FactorContext) -> pd.DataFrame:
    open_val = ctx.open
    close = ctx.close
    return op.rank(-((1 - open_val / close.replace(0, np.nan)) ** 2))


@register_factor("gtja186", category="gtja191",
                 requires=["close", "high", "low"],
                 description="Alpha 186: A/D with delay - simplified as (close - low) - (high - close)")
def gtja186(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    high = ctx.high
    low = ctx.low
    return (close - low) - (high - close)


@register_factor("gtja187", category="gtja191",
                 requires=["open", "high"],
                 description="Alpha 187: sum((open<=delay(open,1)?0:max(high-open, open-delay(open,1))), 20)")
def gtja187(ctx: FactorContext) -> pd.DataFrame:
    open_val = ctx.open
    high = ctx.high
    delay_open = op.delay(open_val, 1)
    cond = open_val <= delay_open
    term1 = high - open_val
    term2 = open_val - delay_open
    max_val = np.maximum(term1, term2)
    result = max_val.where(cond, 0.0)
    return op.ts_sum(result, 20)


@register_factor("gtja188", category="gtja191",
                 requires=["high", "low"],
                 description="Alpha 188: ((high-low) - sma(high-low, 11, 2)) / sma(high-low, 11, 2) * 100")
def gtja188(ctx: FactorContext) -> pd.DataFrame:
    high = ctx.high
    low = ctx.low
    hl = high - low
    sma_hl = op.sma(hl, 11, 2)
    return ((hl - sma_hl) / sma_hl.replace(0, np.nan)) * 100


@register_factor("gtja189", category="gtja191",
                 requires=["close"],
                 description="Alpha 189: mean(abs(close - mean(close, 6)), 6)")
def gtja189(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    mean_close = op.ts_mean(close, 6)
    return op.ts_mean((close - mean_close).abs(), 6)


@register_factor("gtja190", category="gtja191",
                 requires=["close"],
                 description="Alpha 190: [SKIP - too complex] return 0")
def gtja190(ctx: FactorContext) -> pd.DataFrame:
    return pd.DataFrame(0, index=ctx.close.index, columns=ctx.close.columns)


@register_factor("gtja191", category="gtja191",
                 requires=["close", "high", "low", "volume"],
                 description="Alpha 191: corr(mean(volume,20), low, 5) + (high + low)/2 - close")
def gtja191(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    high = ctx.high
    low = ctx.low
    volume = ctx.volume
    corr_val = op.correlation(op.ts_mean(volume, 20), low, 5)
    mid = (high + low) / 2.0
    return corr_val + mid - close
