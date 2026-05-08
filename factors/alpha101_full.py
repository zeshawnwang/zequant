"""
Alpha101 全量实现(配合 FactorHub 注册中心)

实现策略:
- 每个 alpha 用 `@register_factor` 注册到 FactorHub
- 公式按 WorldQuant《101 Formulaic Alphas》论文(Kakushadze 2015)
- 部分公式涉及 IndNeutralize / cap / industry,无行业数据时退化为 zscore/直接值
- 所有 alpha 函数签名统一: (ctx) -> wide DataFrame

约定:
- adv{N} = N 日均量
- vwap = amount / volume(在 ctx.vwap 中已派生)
- 暂无 cap(市值)和 industry(行业),用到时以 close/sector_proxy 占位

调用:
    import factors.alpha101_full   # 触发 @register_factor 副作用
    from core.factor_hub import FactorHub
    long_df = FactorHub.compute_all(bars, names=['a3','a13','a26','a55'])
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from core.factor_hub import register_factor, FactorContext
from . import alpha101_ops as op


# ===================================================================
#  Alpha 1-10
# ===================================================================

@register_factor("a1", category="alpha101",
                 requires=["close"],
                 description="rank(ts_argmax(signed_power(returns?std:close, 2), 5)) - 0.5")
def alpha1(ctx: FactorContext) -> pd.DataFrame:
    close = ctx.close
    ret = close.pct_change()
    std20 = op.stddev(ret, 20)
    base = ret.where(ret < 0, close)
    sp = op.signed_power(base, 2.0)
    return op.rank_(op.ts_argmax(sp, 5)) - 0.5


@register_factor("a2", category="alpha101",
                 requires=["open", "close", "volume"],
                 description="-corr(rank(Δlog_vol,2), rank((c-o)/o), 6)")
def alpha2(ctx: FactorContext) -> pd.DataFrame:
    log_vol = op.log_(ctx.volume)
    x = op.rank_(op.delta(log_vol, 2))
    y = op.rank_((ctx.close - ctx.open) / ctx.open.where(ctx.open != 0))
    return -op.correlation(x, y, 6)


@register_factor("a3", category="alpha101",
                 requires=["open", "volume"],
                 description="-corr(rank(open), rank(volume), 10)")
def alpha3(ctx: FactorContext) -> pd.DataFrame:
    return -op.correlation(op.rank_(ctx.open), op.rank_(ctx.volume), 10)


@register_factor("a4", category="alpha101", requires=["low"],
                 description="-ts_rank(rank(low), 9)")
def alpha4(ctx: FactorContext) -> pd.DataFrame:
    return -op.ts_rank(op.rank_(ctx.low), 9)


@register_factor("a5", category="alpha101",
                 requires=["open", "close", "vwap"],
                 description="rank(open - mean(vwap,10)) * -|rank(close - vwap)|")
def alpha5(ctx: FactorContext) -> pd.DataFrame:
    t1 = op.rank_(ctx.open - op.ts_mean(ctx.vwap, 10))
    t2 = -op.rank_(ctx.close - ctx.vwap).abs()
    return t1 * t2


@register_factor("a6", category="alpha101", requires=["open", "volume"],
                 description="-corr(open, volume, 10)")
def alpha6(ctx: FactorContext) -> pd.DataFrame:
    return -op.correlation(ctx.open, ctx.volume, 10)


@register_factor("a7", category="alpha101", requires=["close", "volume"],
                 description="adv20<vol ? -ts_rank(|Δclose,7|,60)*sign(Δclose,7) : -1")
def alpha7(ctx: FactorContext) -> pd.DataFrame:
    adv20 = op.adv(ctx.volume, 20)
    d7 = op.delta(ctx.close, 7)
    a = -op.ts_rank(d7.abs(), 60) * np.sign(d7)
    return a.where(adv20 < ctx.volume, -1.0)


@register_factor("a8", category="alpha101", requires=["open", "close"],
                 description="-rank(sum(open,5)*sum(ret,5) - delay(same,10))")
def alpha8(ctx: FactorContext) -> pd.DataFrame:
    ret = ctx.close.pct_change()
    s = op.sum_(ctx.open, 5) * op.sum_(ret, 5)
    return -op.rank_(s - op.delay(s, 10))


@register_factor("a9", category="alpha101", requires=["close"],
                 description="(min(Δclose,1,5)>0) ? Δclose : ((max(Δclose,1,5)<0) ? Δclose : -Δclose)")
def alpha9(ctx: FactorContext) -> pd.DataFrame:
    d1 = op.delta(ctx.close, 1)
    cond1 = op.ts_min(d1, 5) > 0
    cond2 = op.ts_max(d1, 5) < 0
    return d1.where(cond1, np.where(cond2, d1, -d1))


@register_factor("a10", category="alpha101", requires=["close"],
                 description="rank(((min(Δclose,1,4)>0) ? Δclose : ((max(...)<0)?Δclose:-Δclose)))")
def alpha10(ctx: FactorContext) -> pd.DataFrame:
    d1 = op.delta(ctx.close, 1)
    cond1 = op.ts_min(d1, 4) > 0
    cond2 = op.ts_max(d1, 4) < 0
    out = d1.where(cond1, np.where(cond2, d1, -d1))
    return op.rank_(out)


# ===================================================================
#  Alpha 11-20
# ===================================================================

@register_factor("a11", category="alpha101",
                 requires=["close", "volume", "vwap"],
                 description="(rank(max(vwap-close,3))+rank(min(vwap-close,3)))*rank(Δvolume,3)")
def alpha11(ctx: FactorContext) -> pd.DataFrame:
    diff = ctx.vwap - ctx.close
    return (op.rank_(op.ts_max(diff, 3)) + op.rank_(op.ts_min(diff, 3))) \
           * op.rank_(op.delta(ctx.volume, 3))


@register_factor("a12", category="alpha101", requires=["close", "volume"],
                 description="sign(Δvol,1) * -Δclose,1")
def alpha12(ctx: FactorContext) -> pd.DataFrame:
    return np.sign(op.delta(ctx.volume, 1)) * (-op.delta(ctx.close, 1))


@register_factor("a13", category="alpha101", requires=["close", "volume"],
                 description="-rank(cov(rank(close), rank(volume), 5))")
def alpha13(ctx: FactorContext) -> pd.DataFrame:
    return -op.rank_(op.covariance(op.rank_(ctx.close), op.rank_(ctx.volume), 5))


@register_factor("a14", category="alpha101",
                 requires=["open", "close", "volume"],
                 description="(-rank(Δreturns,3)) * corr(open, volume, 10)")
def alpha14(ctx: FactorContext) -> pd.DataFrame:
    ret = ctx.close.pct_change()
    return (-op.rank_(op.delta(ret, 3))) * op.correlation(ctx.open, ctx.volume, 10)


@register_factor("a15", category="alpha101",
                 requires=["high", "volume"],
                 description="-sum(rank(corr(rank(high), rank(volume), 3)), 3)")
def alpha15(ctx: FactorContext) -> pd.DataFrame:
    c = op.correlation(op.rank_(ctx.high), op.rank_(ctx.volume), 3)
    return -op.sum_(op.rank_(c), 3)


@register_factor("a16", category="alpha101", requires=["high", "volume"],
                 description="-rank(cov(rank(high), rank(volume), 5))")
def alpha16(ctx: FactorContext) -> pd.DataFrame:
    return -op.rank_(op.covariance(op.rank_(ctx.high), op.rank_(ctx.volume), 5))


@register_factor("a17", category="alpha101", requires=["close", "volume"],
                 description="-rank(ts_rank(close,10)) * rank(Δ(Δclose,1),1) * rank(ts_rank(vol/adv20,5))")
def alpha17(ctx: FactorContext) -> pd.DataFrame:
    adv20 = op.adv(ctx.volume, 20)
    a = -op.rank_(op.ts_rank(ctx.close, 10))
    b = op.rank_(op.delta(op.delta(ctx.close, 1), 1))
    c = op.rank_(op.ts_rank(ctx.volume / adv20.replace(0, np.nan), 5))
    return a * b * c


@register_factor("a18", category="alpha101",
                 requires=["open", "close"],
                 description="-rank((std(|c-o|,5)+(c-o)+corr(c,o,10)))")
def alpha18(ctx: FactorContext) -> pd.DataFrame:
    diff = (ctx.close - ctx.open).abs()
    return -op.rank_(op.stddev(diff, 5) + (ctx.close - ctx.open)
                     + op.correlation(ctx.close, ctx.open, 10))


@register_factor("a19", category="alpha101", requires=["close"],
                 description="-sign((c-delay(c,7))+Δc,7) * (1+rank(1+sum(returns,250)))")
def alpha19(ctx: FactorContext) -> pd.DataFrame:
    ret = ctx.close.pct_change()
    s1 = (ctx.close - op.delay(ctx.close, 7)) + op.delta(ctx.close, 7)
    s2 = 1 + op.rank_(1 + op.sum_(ret, 250))
    return -np.sign(s1) * s2


@register_factor("a20", category="alpha101",
                 requires=["open", "high", "low", "close"],
                 description="-rank(open-delay(high,1)) * rank(open-delay(close,1)) * rank(open-delay(low,1))")
def alpha20(ctx: FactorContext) -> pd.DataFrame:
    return (-op.rank_(ctx.open - op.delay(ctx.high, 1))
            * op.rank_(ctx.open - op.delay(ctx.close, 1))
            * op.rank_(ctx.open - op.delay(ctx.low, 1)))


# ===================================================================
#  Alpha 21-30
# ===================================================================

@register_factor("a21", category="alpha101", requires=["close", "volume"],
                 description="if mean(c,8)+std(c,8)<mean(c,2): -1 elif mean(c,2)<mean(c,8)-std(c,8): 1 elif vol/adv20>=1: 1 else -1")
def alpha21(ctx: FactorContext) -> pd.DataFrame:
    m8 = op.ts_mean(ctx.close, 8)
    m2 = op.ts_mean(ctx.close, 2)
    s8 = op.stddev(ctx.close, 8)
    adv20 = op.adv(ctx.volume, 20)
    cond1 = (m8 + s8) < m2
    cond2 = m2 < (m8 - s8)
    cond3 = (ctx.volume / adv20.replace(0, np.nan)) >= 1
    out = pd.DataFrame(-1.0, index=ctx.close.index, columns=ctx.close.columns)
    out = out.mask(cond3, 1.0)
    out = out.mask(cond2, 1.0)
    out = out.mask(cond1, -1.0)
    return out


@register_factor("a22", category="alpha101",
                 requires=["high", "close", "volume"],
                 description="-(Δ(corr(high,volume,5),5) * rank(std(close,20)))")
def alpha22(ctx: FactorContext) -> pd.DataFrame:
    c = op.correlation(ctx.high, ctx.volume, 5)
    return -(op.delta(c, 5) * op.rank_(op.stddev(ctx.close, 20)))


@register_factor("a23", category="alpha101", requires=["high"],
                 description="if mean(high,20)<high: -Δhigh,2 else 0")
def alpha23(ctx: FactorContext) -> pd.DataFrame:
    cond = op.ts_mean(ctx.high, 20) < ctx.high
    return (-op.delta(ctx.high, 2)).where(cond, 0.0)


@register_factor("a24", category="alpha101", requires=["close"],
                 description="if Δ(mean(c,100),100)/delay(c,100) <= 0.05: -(c-min(c,100)) else -Δc,3")
def alpha24(ctx: FactorContext) -> pd.DataFrame:
    m100 = op.ts_mean(ctx.close, 100)
    cond = (op.delta(m100, 100) / op.delay(ctx.close, 100).replace(0, np.nan)) <= 0.05
    branch1 = -(ctx.close - op.ts_min(ctx.close, 100))
    branch2 = -op.delta(ctx.close, 3)
    return branch1.where(cond, branch2)


@register_factor("a25", category="alpha101",
                 requires=["high", "close", "volume", "vwap"],
                 description="rank((-returns)*adv20*vwap*(high-close))")
def alpha25(ctx: FactorContext) -> pd.DataFrame:
    ret = ctx.close.pct_change()
    adv20 = op.adv(ctx.volume, 20)
    return op.rank_((-ret) * adv20 * ctx.vwap * (ctx.high - ctx.close))


@register_factor("a26", category="alpha101", requires=["high", "volume"],
                 description="-ts_max(corr(ts_rank(vol,5), ts_rank(high,5), 5), 3)")
def alpha26(ctx: FactorContext) -> pd.DataFrame:
    c = op.correlation(op.ts_rank(ctx.volume, 5), op.ts_rank(ctx.high, 5), 5)
    return -op.ts_max(c, 3)


@register_factor("a27", category="alpha101",
                 requires=["volume", "vwap"],
                 description="if 0.5<rank(mean(corr(rank(vol),rank(vwap),6),2)): -1 else 1")
def alpha27(ctx: FactorContext) -> pd.DataFrame:
    c = op.correlation(op.rank_(ctx.volume), op.rank_(ctx.vwap), 6)
    m = op.ts_mean(c, 2)
    cond = 0.5 < op.rank_(m)
    out = pd.DataFrame(1.0, index=ctx.vwap.index, columns=ctx.vwap.columns)
    return out.mask(cond, -1.0)


@register_factor("a28", category="alpha101",
                 requires=["high", "low", "close", "volume"],
                 description="scale(corr(adv20, low, 5) + (high+low)/2 - close)")
def alpha28(ctx: FactorContext) -> pd.DataFrame:
    adv20 = op.adv(ctx.volume, 20)
    c = op.correlation(adv20, ctx.low, 5)
    return op.scale(c + (ctx.high + ctx.low) / 2.0 - ctx.close)


@register_factor("a29", category="alpha101", requires=["close"],
                 description="min(product(rank(rank(scale(log(sum(ts_min(rank(rank(-rank(Δ(c-1,5)))),2),1)))))),5)+ts_rank(delay(-returns,6),5)")
def alpha29(ctx: FactorContext) -> pd.DataFrame:
    ret = ctx.close.pct_change()
    inner = -op.rank_(op.delta(ctx.close - 1, 5))
    a = op.ts_min(op.rank_(op.rank_(inner)), 2)
    # 原文 sum(...,1) 是恒等(等价于 a 自身),log 在小负值上会 NaN -> 用 signedlog 兜底
    b = op.signedlog(a)
    c = op.product(op.rank_(op.rank_(op.scale(b))), 1)
    p1 = op.ts_min(c, 5)
    p2 = op.ts_rank(op.delay(-ret, 6), 5)
    return p1 + p2


@register_factor("a30", category="alpha101",
                 requires=["close", "volume"],
                 description="((1.0-rank((sign(Δc,1)+sign(Δc-1,1)+sign(Δc-2,1)))) * sum(vol,5)) / sum(vol,20)")
def alpha30(ctx: FactorContext) -> pd.DataFrame:
    s1 = np.sign(op.delta(ctx.close, 1))
    s2 = np.sign(op.delta(op.delay(ctx.close, 1), 1))
    s3 = np.sign(op.delta(op.delay(ctx.close, 2), 1))
    a = 1.0 - op.rank_(s1 + s2 + s3)
    return (a * op.sum_(ctx.volume, 5)) / op.sum_(ctx.volume, 20).replace(0, np.nan)


# ===================================================================
#  Alpha 31-40
# ===================================================================

@register_factor("a31", category="alpha101",
                 requires=["low", "close", "volume"],
                 description="rank^3(decay_linear(-rank(rank(Δc,10)),10)) + rank(-Δc,3) + sign(scale(corr(adv20,low,12)))")
def alpha31(ctx: FactorContext) -> pd.DataFrame:
    adv20 = op.adv(ctx.volume, 20)
    a = op.rank_(op.rank_(op.rank_(op.decay_linear(-op.rank_(op.rank_(op.delta(ctx.close, 10))), 10))))
    b = op.rank_(-op.delta(ctx.close, 3))
    c = np.sign(op.scale(op.correlation(adv20, ctx.low, 12)))
    return a + b + c


@register_factor("a32", category="alpha101",
                 requires=["close", "vwap"],
                 description="scale(mean(c,7)-c) + 20*scale(corr(vwap, delay(c,5), 230))")
def alpha32(ctx: FactorContext) -> pd.DataFrame:
    a = op.scale(op.ts_mean(ctx.close, 7) - ctx.close)
    b = 20 * op.scale(op.correlation(ctx.vwap, op.delay(ctx.close, 5), 230))
    return a + b


@register_factor("a33", category="alpha101",
                 requires=["open", "close"],
                 description="rank(-(1 - open/close))")
def alpha33(ctx: FactorContext) -> pd.DataFrame:
    return op.rank_(-(1.0 - ctx.open / ctx.close.replace(0, np.nan)))


@register_factor("a34", category="alpha101", requires=["close"],
                 description="rank(1 - rank(std(ret,2)/std(ret,5)) + 1 - rank(Δc,1))")
def alpha34(ctx: FactorContext) -> pd.DataFrame:
    ret = ctx.close.pct_change()
    a = 1 - op.rank_(op.stddev(ret, 2) / op.stddev(ret, 5).replace(0, np.nan))
    b = 1 - op.rank_(op.delta(ctx.close, 1))
    return op.rank_(a + b)


@register_factor("a35", category="alpha101",
                 requires=["high", "low", "close", "volume"],
                 description="ts_rank(vol,32)*(1-ts_rank(c+h-l,16))*(1-ts_rank(ret,32))")
def alpha35(ctx: FactorContext) -> pd.DataFrame:
    ret = ctx.close.pct_change()
    return (op.ts_rank(ctx.volume, 32)
            * (1 - op.ts_rank(ctx.close + ctx.high - ctx.low, 16))
            * (1 - op.ts_rank(ret, 32)))


@register_factor("a36", category="alpha101",
                 requires=["open", "close", "volume", "vwap"],
                 description="2.21*rank(corr(c-o,delay(vol,1),15))+0.7*rank(o-c)+0.73*rank(ts_rank(delay(-ret,6),5))+rank(|corr(vwap,adv20,6)|)+0.6*rank((mean(c,200)-o)*(c-o))")
def alpha36(ctx: FactorContext) -> pd.DataFrame:
    ret = ctx.close.pct_change()
    adv20 = op.adv(ctx.volume, 20)
    t1 = 2.21 * op.rank_(op.correlation(ctx.close - ctx.open, op.delay(ctx.volume, 1), 15))
    t2 = 0.7 * op.rank_(ctx.open - ctx.close)
    t3 = 0.73 * op.rank_(op.ts_rank(op.delay(-ret, 6), 5))
    t4 = op.rank_(op.correlation(ctx.vwap, adv20, 6).abs())
    t5 = 0.6 * op.rank_((op.ts_mean(ctx.close, 200) - ctx.open) * (ctx.close - ctx.open))
    return t1 + t2 + t3 + t4 + t5


@register_factor("a37", category="alpha101",
                 requires=["open", "close"],
                 description="rank(corr(delay(o-c,1),c,200)) + rank(o-c)")
def alpha37(ctx: FactorContext) -> pd.DataFrame:
    return (op.rank_(op.correlation(op.delay(ctx.open - ctx.close, 1), ctx.close, 200))
            + op.rank_(ctx.open - ctx.close))


@register_factor("a38", category="alpha101",
                 requires=["open", "close"],
                 description="-rank(ts_rank(c,10)) * rank(c/o)")
def alpha38(ctx: FactorContext) -> pd.DataFrame:
    return -op.rank_(op.ts_rank(ctx.close, 10)) * op.rank_(ctx.close / ctx.open.replace(0, np.nan))


@register_factor("a39", category="alpha101",
                 requires=["close", "volume"],
                 description="-rank(Δc,7*(1-rank(decay_linear(vol/adv20,9))))*(1+rank(sum(ret,250)))")
def alpha39(ctx: FactorContext) -> pd.DataFrame:
    ret = ctx.close.pct_change()
    adv20 = op.adv(ctx.volume, 20)
    v_over = ctx.volume / adv20.replace(0, np.nan)
    a = -op.rank_(op.delta(ctx.close, 7) * (1 - op.rank_(op.decay_linear(v_over, 9))))
    b = 1 + op.rank_(op.sum_(ret, 250))
    return a * b


@register_factor("a40", category="alpha101",
                 requires=["high", "volume"],
                 description="-rank(std(high,10)) * corr(high,volume,10)")
def alpha40(ctx: FactorContext) -> pd.DataFrame:
    return -op.rank_(op.stddev(ctx.high, 10)) * op.correlation(ctx.high, ctx.volume, 10)


# ===================================================================
#  Alpha 41-50
# ===================================================================

@register_factor("a41", category="alpha101",
                 requires=["high", "low", "vwap"],
                 description="sqrt(high*low) - vwap")
def alpha41(ctx: FactorContext) -> pd.DataFrame:
    return np.sqrt(ctx.high * ctx.low) - ctx.vwap


@register_factor("a42", category="alpha101",
                 requires=["close", "vwap"],
                 description="rank(vwap-close) / rank(vwap+close)")
def alpha42(ctx: FactorContext) -> pd.DataFrame:
    num = op.rank_(ctx.vwap - ctx.close)
    den = op.rank_(ctx.vwap + ctx.close).replace(0, np.nan)
    return num / den


@register_factor("a43", category="alpha101", requires=["close", "volume"],
                 description="ts_rank(vol/adv20,20) * ts_rank(-Δc,7,8)")
def alpha43(ctx: FactorContext) -> pd.DataFrame:
    adv20 = op.adv(ctx.volume, 20)
    return (op.ts_rank(ctx.volume / adv20.replace(0, np.nan), 20)
            * op.ts_rank(-op.delta(ctx.close, 7), 8))


@register_factor("a44", category="alpha101", requires=["high", "volume"],
                 description="-corr(high, rank(volume), 5)")
def alpha44(ctx: FactorContext) -> pd.DataFrame:
    return -op.correlation(ctx.high, op.rank_(ctx.volume), 5)


@register_factor("a45", category="alpha101",
                 requires=["close", "volume"],
                 description="-rank(mean(delay(c,5),20))*corr(c,vol,2)*rank(corr(sum(c,5),sum(c,20),2))")
def alpha45(ctx: FactorContext) -> pd.DataFrame:
    a = op.rank_(op.ts_mean(op.delay(ctx.close, 5), 20))
    b = op.correlation(ctx.close, ctx.volume, 2)
    c = op.rank_(op.correlation(op.sum_(ctx.close, 5), op.sum_(ctx.close, 20), 2))
    return -(a * b * c)


@register_factor("a46", category="alpha101", requires=["close"],
                 description="分段:动量差>0.25 -> -1, <0 -> 1, else -Δc,1")
def alpha46(ctx: FactorContext) -> pd.DataFrame:
    c = ctx.close
    diff = ((op.delay(c, 20) - op.delay(c, 10)) / 10.0
            - (op.delay(c, 10) - c) / 10.0)
    out = -op.delta(c, 1)
    out = out.mask(diff > 0.25, -1.0)
    out = out.mask(diff < 0, 1.0)
    return out


@register_factor("a47", category="alpha101",
                 requires=["high", "close", "volume", "vwap"],
                 description="(rank(1/c)*vol/adv20)*(h*rank(h-c)/mean(h,5)) - rank(Δvwap,5)")
def alpha47(ctx: FactorContext) -> pd.DataFrame:
    adv20 = op.adv(ctx.volume, 20)
    a = (op.rank_(1.0 / ctx.close.replace(0, np.nan))
         * ctx.volume / adv20.replace(0, np.nan))
    b = (ctx.high * op.rank_(ctx.high - ctx.close)
         / op.ts_mean(ctx.high, 5).replace(0, np.nan))
    c = op.rank_(op.delta(ctx.vwap, 5))
    return a * b - c


@register_factor("a48", category="alpha101", requires=["close"],
                 description="简化:indneutralize(corr(Δc,delay(Δc,1),250)*Δc/c) / sum((Δc/c)²,250)")
def alpha48(ctx: FactorContext) -> pd.DataFrame:
    c = ctx.close
    num = (op.correlation(op.delta(c, 1), op.delay(op.delta(c, 1), 1), 250)
           * op.delta(c, 1) / c.replace(0, np.nan))
    sub = (op.delta(c, 1) / op.delay(c, 1).replace(0, np.nan)) ** 2
    return op.indneutralize(num) / (op.sum_(sub, 250) + 1e-9)


@register_factor("a49", category="alpha101", requires=["close"],
                 description="if 动量差<-0.1: 1 else -Δc,1")
def alpha49(ctx: FactorContext) -> pd.DataFrame:
    c = ctx.close
    diff = ((op.delay(c, 20) - op.delay(c, 10)) / 10.0
            - (op.delay(c, 10) - c) / 10.0)
    out = -op.delta(c, 1)
    return out.mask(diff < -0.1, 1.0)


@register_factor("a50", category="alpha101",
                 requires=["volume", "vwap"],
                 description="-ts_max(rank(corr(rank(vol),rank(vwap),5)),5)")
def alpha50(ctx: FactorContext) -> pd.DataFrame:
    c = op.correlation(op.rank_(ctx.volume), op.rank_(ctx.vwap), 5)
    return -op.ts_max(op.rank_(c), 5)


# ===================================================================
#  Alpha 51-60
# ===================================================================

@register_factor("a51", category="alpha101", requires=["close"],
                 description="if 动量差<-0.05: 1 else -Δc,1")
def alpha51(ctx: FactorContext) -> pd.DataFrame:
    c = ctx.close
    diff = ((op.delay(c, 20) - op.delay(c, 10)) / 10.0
            - (op.delay(c, 10) - c) / 10.0)
    out = -op.delta(c, 1)
    return out.mask(diff < -0.05, 1.0)


@register_factor("a52", category="alpha101",
                 requires=["low", "close", "volume"],
                 description="(-min(l,5)+delay(min(l,5),5))*rank((sum(ret,240)-sum(ret,20))/220)*ts_rank(vol,5)")
def alpha52(ctx: FactorContext) -> pd.DataFrame:
    ret = ctx.close.pct_change()
    a = -op.ts_min(ctx.low, 5) + op.delay(op.ts_min(ctx.low, 5), 5)
    b = op.rank_((op.sum_(ret, 240) - op.sum_(ret, 20)) / 220.0)
    c = op.ts_rank(ctx.volume, 5)
    return a * b * c


@register_factor("a53", category="alpha101",
                 requires=["high", "low", "close"],
                 description="-Δ(((c-l)-(h-c))/(c-l), 9)")
def alpha53(ctx: FactorContext) -> pd.DataFrame:
    num = (ctx.close - ctx.low) - (ctx.high - ctx.close)
    den = (ctx.close - ctx.low).replace(0, np.nan)
    return -op.delta(num / den, 9)


@register_factor("a54", category="alpha101",
                 requires=["open", "high", "low", "close"],
                 description="-((low-close)*open^5)/((low-high)*close^5)")
def alpha54(ctx: FactorContext) -> pd.DataFrame:
    num = -(ctx.low - ctx.close) * (ctx.open ** 5)
    den = ((ctx.low - ctx.high) * (ctx.close ** 5)).replace(0, np.nan)
    return num / den


@register_factor("a55", category="alpha101",
                 requires=["high", "low", "close", "volume"],
                 description="-corr(rank((c-min(l,12))/(max(h,12)-min(l,12))), rank(vol), 6)")
def alpha55(ctx: FactorContext) -> pd.DataFrame:
    ll = op.ts_min(ctx.low, 12)
    hh = op.ts_max(ctx.high, 12)
    stoch = (ctx.close - ll) / (hh - ll).replace(0, np.nan)
    return -op.correlation(op.rank_(stoch), op.rank_(ctx.volume), 6)


@register_factor("a56", category="alpha101", requires=["close"],
                 description="简化:-rank(sum(ret,10)/sum(sum(ret,2),3))(原公式要 cap)")
def alpha56(ctx: FactorContext) -> pd.DataFrame:
    ret = ctx.close.pct_change()
    num = op.sum_(ret, 10)
    den = op.sum_(op.sum_(ret, 2), 3).replace(0, np.nan)
    return -op.rank_(num / den)


@register_factor("a57", category="alpha101",
                 requires=["close", "vwap"],
                 description="-(close-vwap)/decay_linear(rank(ts_argmax(c,30)),2)")
def alpha57(ctx: FactorContext) -> pd.DataFrame:
    num = -(ctx.close - ctx.vwap)
    den = op.decay_linear(op.rank_(op.ts_argmax(ctx.close, 30)), 2).replace(0, np.nan)
    return num / den


@register_factor("a58", category="alpha101",
                 requires=["volume", "vwap"],
                 description="简化:-ts_rank(decay_linear(corr(vwap,vol,3),7),5)")
def alpha58(ctx: FactorContext) -> pd.DataFrame:
    c = op.correlation(ctx.vwap, ctx.volume, 3)
    return -op.ts_rank(op.decay_linear(c, 7), 5)


@register_factor("a59", category="alpha101",
                 requires=["close", "volume", "vwap"],
                 description="a58 变种:vwap*0.7+close*0.3")
def alpha59(ctx: FactorContext) -> pd.DataFrame:
    mix = ctx.vwap * 0.7 + ctx.close * 0.3
    c = op.correlation(mix, ctx.volume, 4)
    return -op.ts_rank(op.decay_linear(c, 16), 8)


@register_factor("a60", category="alpha101",
                 requires=["high", "low", "close", "volume"],
                 description="-(2*scale(rank(((c-l)-(h-c))/(h-l)*vol)) - scale(rank(ts_argmax(c,10))))")
def alpha60(ctx: FactorContext) -> pd.DataFrame:
    clv = ((ctx.close - ctx.low) - (ctx.high - ctx.close)) / (ctx.high - ctx.low).replace(0, np.nan)
    a = op.scale(op.rank_(clv * ctx.volume))
    b = op.scale(op.rank_(op.ts_argmax(ctx.close, 10)))
    return -(2 * a - b)


# ===================================================================
#  Alpha 61-70
# ===================================================================

@register_factor("a61", category="alpha101",
                 requires=["volume", "vwap"],
                 description="rank(vwap-min(vwap,16)) < rank(corr(vwap,adv180,18))")
def alpha61(ctx: FactorContext) -> pd.DataFrame:
    adv180 = op.adv(ctx.volume, 180)
    a = op.rank_(ctx.vwap - op.ts_min(ctx.vwap, 16))
    b = op.rank_(op.correlation(ctx.vwap, adv180, 18))
    return (a < b).astype(float) * 1.0 - 0.5


@register_factor("a62", category="alpha101",
                 requires=["open", "high", "low", "volume", "vwap"],
                 description="-(rank(corr(vwap,sum(adv20,22),10)) < rank((rank(open)*2 < rank((h+l)/2)+rank(h))))")
def alpha62(ctx: FactorContext) -> pd.DataFrame:
    adv20 = op.adv(ctx.volume, 20)
    a = op.rank_(op.correlation(ctx.vwap, op.sum_(adv20, 22), 10))
    b = op.rank_((op.rank_(ctx.open) * 2 < op.rank_((ctx.high + ctx.low) / 2.0) + op.rank_(ctx.high)).astype(float))
    return -((a < b).astype(float))


@register_factor("a63", category="alpha101",
                 requires=["open", "close", "volume", "vwap"],
                 description="简化:-(rank(decay_linear(Δc,2)) - rank(decay_linear(corr(vwap*0.18+open*0.82, sum(adv180,37), 14), 12)))")
def alpha63(ctx: FactorContext) -> pd.DataFrame:
    adv180 = op.adv(ctx.volume, 180)
    a = op.rank_(op.decay_linear(op.delta(ctx.close, 2), 8))
    mix = ctx.vwap * 0.18 + ctx.open * 0.82
    c = op.correlation(mix, op.sum_(adv180, 37), 14)
    b = op.rank_(op.decay_linear(c, 12))
    return -(a - b)


@register_factor("a64", category="alpha101",
                 requires=["open", "high", "low", "volume", "vwap"],
                 description="-(rank(corr(sum(o*0.18+l*0.82,12), sum(adv120,12), 16)) < rank(Δ((h+l)/2*0.18+vwap*0.82, 3)))")
def alpha64(ctx: FactorContext) -> pd.DataFrame:
    adv120 = op.adv(ctx.volume, 120)
    mix1 = ctx.open * 0.18 + ctx.low * 0.82
    a = op.rank_(op.correlation(op.sum_(mix1, 12), op.sum_(adv120, 12), 16))
    mix2 = (ctx.high + ctx.low) / 2.0 * 0.18 + ctx.vwap * 0.82
    b = op.rank_(op.delta(mix2, 3))
    return -((a < b).astype(float))


@register_factor("a65", category="alpha101",
                 requires=["open", "volume", "vwap"],
                 description="-(rank(corr(o*0.0073+vwap*0.99, sum(adv60,8), 6)) < rank(o-min(o,13)))")
def alpha65(ctx: FactorContext) -> pd.DataFrame:
    adv60 = op.adv(ctx.volume, 60)
    mix = ctx.open * 0.0073 + ctx.vwap * 0.9927
    a = op.rank_(op.correlation(mix, op.sum_(adv60, 8), 6))
    b = op.rank_(ctx.open - op.ts_min(ctx.open, 13))
    return -((a < b).astype(float))


@register_factor("a66", category="alpha101",
                 requires=["open", "high", "low", "vwap"],
                 description="-(rank(decay_linear(Δvwap,4)) + ts_rank(decay_linear((((l*0.96+l*0.04)-vwap)/(o-(h+l)/2)), 11), 7))")
def alpha66(ctx: FactorContext) -> pd.DataFrame:
    a = op.rank_(op.decay_linear(op.delta(ctx.vwap, 4), 7))
    num = ctx.low - ctx.vwap
    den = (ctx.open - (ctx.high + ctx.low) / 2.0).replace(0, np.nan)
    b = op.ts_rank(op.decay_linear(num / den, 11), 7)
    return -(a + b)


@register_factor("a67", category="alpha101",
                 requires=["high", "volume", "vwap"],
                 description="简化(去 indneutralize):-rank(h-min(h,2))^rank(corr(vwap,adv20,6))")
def alpha67(ctx: FactorContext) -> pd.DataFrame:
    adv20 = op.adv(ctx.volume, 20)
    a = op.rank_(ctx.high - op.ts_min(ctx.high, 2))
    b = op.rank_(op.correlation(ctx.vwap, adv20, 6))
    return -(op.signed_power(a, 1.0) * b)


@register_factor("a68", category="alpha101",
                 requires=["high", "low", "close", "volume"],
                 description="-(ts_rank(corr(rank(h),rank(adv15),9),14) < rank(Δ(c*0.518+l*0.482,1)))")
def alpha68(ctx: FactorContext) -> pd.DataFrame:
    adv15 = op.adv(ctx.volume, 15)
    a = op.ts_rank(op.correlation(op.rank_(ctx.high), op.rank_(adv15), 9), 14)
    mix = ctx.close * 0.518 + ctx.low * 0.482
    b = op.rank_(op.delta(mix, 1))
    return -((a < b).astype(float))


@register_factor("a69", category="alpha101",
                 requires=["close", "volume", "vwap"],
                 description="简化:-(rank(max(Δvwap,5)) ^ ts_rank(corr(c*0.49+vwap*0.51, adv20, 5), 9))")
def alpha69(ctx: FactorContext) -> pd.DataFrame:
    adv20 = op.adv(ctx.volume, 20)
    a = op.rank_(op.ts_max(op.delta(ctx.vwap, 3), 5))
    mix = ctx.close * 0.49 + ctx.vwap * 0.51
    b = op.ts_rank(op.correlation(mix, adv20, 5), 9)
    return -(a * b)


@register_factor("a70", category="alpha101",
                 requires=["close", "volume", "vwap"],
                 description="简化:-(rank(Δvwap,1) ^ ts_rank(corr(c, adv50, 18), 18))")
def alpha70(ctx: FactorContext) -> pd.DataFrame:
    adv50 = op.adv(ctx.volume, 50)
    a = op.rank_(op.delta(ctx.vwap, 1))
    b = op.ts_rank(op.correlation(ctx.close, adv50, 18), 18)
    return -(a * b)


# ===================================================================
#  Alpha 71-80
# ===================================================================

@register_factor("a71", category="alpha101",
                 requires=["open", "low", "close", "volume", "vwap"],
                 description="max(ts_rank(decay_linear(corr(ts_rank(c,3),ts_rank(adv180,12),18),4),16), ts_rank(decay_linear(rank((l+o-vwap*2))^2,16),4))")
def alpha71(ctx: FactorContext) -> pd.DataFrame:
    adv180 = op.adv(ctx.volume, 180)
    p1 = op.ts_rank(op.decay_linear(op.correlation(op.ts_rank(ctx.close, 3), op.ts_rank(adv180, 12), 18), 4), 16)
    inner = op.signed_power(op.rank_(ctx.low + ctx.open - ctx.vwap * 2.0), 2.0)
    p2 = op.ts_rank(op.decay_linear(inner, 16), 4)
    return op.max_(p1, p2)


@register_factor("a72", category="alpha101",
                 requires=["high", "low", "volume", "vwap"],
                 description="rank(decay_linear(corr((h+l)/2, adv40, 9), 10)) / rank(decay_linear(corr(ts_rank(vwap,4),ts_rank(vol,19),7),3))")
def alpha72(ctx: FactorContext) -> pd.DataFrame:
    adv40 = op.adv(ctx.volume, 40)
    num = op.rank_(op.decay_linear(op.correlation((ctx.high + ctx.low) / 2.0, adv40, 9), 10))
    den = op.rank_(op.decay_linear(op.correlation(op.ts_rank(ctx.vwap, 4), op.ts_rank(ctx.volume, 19), 7), 3))
    return num / den.replace(0, np.nan)


@register_factor("a73", category="alpha101",
                 requires=["open", "low", "vwap"],
                 description="-max(rank(decay_linear(Δvwap,5)), ts_rank(decay_linear((Δ(o*0.147+l*0.853)/(o*0.147+l*0.853))*-1, 3), 17))")
def alpha73(ctx: FactorContext) -> pd.DataFrame:
    a = op.rank_(op.decay_linear(op.delta(ctx.vwap, 5), 3))
    mix = ctx.open * 0.147 + ctx.low * 0.853
    inner = -op.delta(mix, 2) / mix.replace(0, np.nan)
    b = op.ts_rank(op.decay_linear(inner, 3), 17)
    return -op.max_(a, b)


@register_factor("a74", category="alpha101",
                 requires=["close", "high", "volume", "vwap"],
                 description="-(rank(corr(c,sum(adv30,37),15)) < rank(corr(rank(h*0.027+vwap*0.97), rank(vol), 11)))")
def alpha74(ctx: FactorContext) -> pd.DataFrame:
    adv30 = op.adv(ctx.volume, 30)
    a = op.rank_(op.correlation(ctx.close, op.sum_(adv30, 37), 15))
    mix = ctx.high * 0.027 + ctx.vwap * 0.973
    b = op.rank_(op.correlation(op.rank_(mix), op.rank_(ctx.volume), 11))
    return -((a < b).astype(float))


@register_factor("a75", category="alpha101",
                 requires=["low", "volume", "vwap"],
                 description="(rank(corr(vwap,vol,4)) < rank(corr(rank(l),rank(adv50),12))) ? 1 : 0")
def alpha75(ctx: FactorContext) -> pd.DataFrame:
    adv50 = op.adv(ctx.volume, 50)
    a = op.rank_(op.correlation(ctx.vwap, ctx.volume, 4))
    b = op.rank_(op.correlation(op.rank_(ctx.low), op.rank_(adv50), 12))
    return (a < b).astype(float)


@register_factor("a76", category="alpha101",
                 requires=["low", "volume", "vwap"],
                 description="-max(rank(decay_linear(Δvwap,1)), ts_rank(decay_linear(ts_rank(corr(l,adv81,8),20),17),5))")
def alpha76(ctx: FactorContext) -> pd.DataFrame:
    adv81 = op.adv(ctx.volume, 81)
    a = op.rank_(op.decay_linear(op.delta(ctx.vwap, 1), 12))
    b = op.ts_rank(op.decay_linear(op.ts_rank(op.correlation(ctx.low, adv81, 8), 20), 17), 5)
    return -op.max_(a, b)


@register_factor("a77", category="alpha101",
                 requires=["high", "low", "volume", "vwap"],
                 description="min(rank(decay_linear((h+l)/2+h-(vwap+h),20)), rank(decay_linear(corr((h+l)/2, adv40, 3), 6)))")
def alpha77(ctx: FactorContext) -> pd.DataFrame:
    adv40 = op.adv(ctx.volume, 40)
    inner1 = (ctx.high + ctx.low) / 2.0 + ctx.high - (ctx.vwap + ctx.high)
    a = op.rank_(op.decay_linear(inner1, 20))
    b = op.rank_(op.decay_linear(op.correlation((ctx.high + ctx.low) / 2.0, adv40, 3), 6))
    return op.min_(a, b)


@register_factor("a78", category="alpha101",
                 requires=["low", "volume", "vwap"],
                 description="rank(corr(sum(l*0.35+vwap*0.65, 20), sum(adv40,20), 7)) ^ rank(corr(rank(vwap), rank(vol), 6))")
def alpha78(ctx: FactorContext) -> pd.DataFrame:
    adv40 = op.adv(ctx.volume, 40)
    mix = ctx.low * 0.35 + ctx.vwap * 0.65
    a = op.rank_(op.correlation(op.sum_(mix, 20), op.sum_(adv40, 20), 7))
    b = op.rank_(op.correlation(op.rank_(ctx.vwap), op.rank_(ctx.volume), 6))
    return a * b


@register_factor("a79", category="alpha101",
                 requires=["close", "open", "volume", "vwap"],
                 description="-(rank(Δ(c*0.6+o*0.4,1)) < rank(corr(ts_rank(vwap,4),ts_rank(adv150,9),15)))")
def alpha79(ctx: FactorContext) -> pd.DataFrame:
    adv150 = op.adv(ctx.volume, 150)
    mix = ctx.close * 0.6 + ctx.open * 0.4
    a = op.rank_(op.delta(mix, 1))
    b = op.rank_(op.correlation(op.ts_rank(ctx.vwap, 4), op.ts_rank(adv150, 9), 15))
    return -((a < b).astype(float))


@register_factor("a80", category="alpha101",
                 requires=["open", "high", "volume"],
                 description="-(rank(sign(Δ(o*0.87+h*0.13,4))) ^ ts_rank(corr(h,adv10,5),6))")
def alpha80(ctx: FactorContext) -> pd.DataFrame:
    adv10 = op.adv(ctx.volume, 10)
    mix = ctx.open * 0.87 + ctx.high * 0.13
    a = op.rank_(np.sign(op.delta(mix, 4)))
    b = op.ts_rank(op.correlation(ctx.high, adv10, 5), 6)
    return -(a * b)


# ===================================================================
#  Alpha 81-90
# ===================================================================

@register_factor("a81", category="alpha101",
                 requires=["volume", "vwap"],
                 description="-(rank(log(product(rank(rank(corr(vwap,sum(adv10,49),8))^4),15))) < rank(corr(rank(vwap),rank(vol),5)))")
def alpha81(ctx: FactorContext) -> pd.DataFrame:
    adv10 = op.adv(ctx.volume, 10)
    inner = op.signed_power(op.rank_(op.rank_(op.correlation(ctx.vwap, op.sum_(adv10, 49), 8))), 4)
    a = op.rank_(op.log_(op.product(inner, 15)))
    b = op.rank_(op.correlation(op.rank_(ctx.vwap), op.rank_(ctx.volume), 5))
    return -((a < b).astype(float))


@register_factor("a82", category="alpha101",
                 requires=["open", "volume"],
                 description="-min(rank(decay_linear(Δo,2)), ts_rank(decay_linear(corr(vol,o,17),7),13))")
def alpha82(ctx: FactorContext) -> pd.DataFrame:
    a = op.rank_(op.decay_linear(op.delta(ctx.open, 2), 14))
    b = op.ts_rank(op.decay_linear(op.correlation(ctx.volume, ctx.open, 17), 7), 13)
    return -op.min_(a, b)


@register_factor("a83", category="alpha101",
                 requires=["high", "low", "close", "volume", "vwap"],
                 description="(rank(delay((h-l)/mean(c,5),2))*rank(rank(vol))) / ((h-l)/mean(c,5)/(vwap-c))")
def alpha83(ctx: FactorContext) -> pd.DataFrame:
    hl_c = (ctx.high - ctx.low) / op.ts_mean(ctx.close, 5).replace(0, np.nan)
    a = op.rank_(op.delay(hl_c, 2)) * op.rank_(op.rank_(ctx.volume))
    den = (hl_c / (ctx.vwap - ctx.close).replace(0, np.nan)).replace(0, np.nan)
    return a / den


@register_factor("a84", category="alpha101",
                 requires=["close", "vwap"],
                 description="signedlog(ts_rank(vwap-max(vwap,15),20)^Δc,6)")
def alpha84(ctx: FactorContext) -> pd.DataFrame:
    base = op.ts_rank(ctx.vwap - op.ts_max(ctx.vwap, 15), 20)
    exp = op.delta(ctx.close, 6)
    return op.signedlog(base * exp)


@register_factor("a85", category="alpha101",
                 requires=["high", "low", "close", "volume"],
                 description="rank(corr(h*0.88+c*0.12,adv30,10))^rank(corr(ts_rank((h+l)/2,4),ts_rank(vol,10),7))")
def alpha85(ctx: FactorContext) -> pd.DataFrame:
    adv30 = op.adv(ctx.volume, 30)
    mix = ctx.high * 0.88 + ctx.close * 0.12
    a = op.rank_(op.correlation(mix, adv30, 10))
    b = op.rank_(op.correlation(op.ts_rank((ctx.high + ctx.low) / 2.0, 4),
                                op.ts_rank(ctx.volume, 10), 7))
    return a * b


@register_factor("a86", category="alpha101",
                 requires=["open", "close", "volume", "vwap"],
                 description="-(ts_rank(corr(c,sum(adv20,15),6),20) < rank((o+c)-(vwap+o)))")
def alpha86(ctx: FactorContext) -> pd.DataFrame:
    adv20 = op.adv(ctx.volume, 20)
    a = op.ts_rank(op.correlation(ctx.close, op.sum_(adv20, 15), 6), 20)
    b = op.rank_((ctx.open + ctx.close) - (ctx.vwap + ctx.open))
    return -((a < b).astype(float))


@register_factor("a87", category="alpha101",
                 requires=["close", "volume", "vwap"],
                 description="-max(rank(decay_linear(Δ(c*0.37+vwap*0.63,2),3)), ts_rank(decay_linear(|corr(adv81,c,13)|,5),14))")
def alpha87(ctx: FactorContext) -> pd.DataFrame:
    adv81 = op.adv(ctx.volume, 81)
    mix = ctx.close * 0.37 + ctx.vwap * 0.63
    a = op.rank_(op.decay_linear(op.delta(mix, 2), 3))
    b = op.ts_rank(op.decay_linear(op.correlation(adv81, ctx.close, 13).abs(), 5), 14)
    return -op.max_(a, b)


@register_factor("a88", category="alpha101",
                 requires=["open", "high", "low", "close", "volume"],
                 description="min(rank(decay_linear(rank(o)+rank(l)-rank(h)-rank(c),8)), ts_rank(decay_linear(corr(ts_rank(c,8),ts_rank(adv60,21),8),7),3))")
def alpha88(ctx: FactorContext) -> pd.DataFrame:
    adv60 = op.adv(ctx.volume, 60)
    a = op.rank_(op.decay_linear(op.rank_(ctx.open) + op.rank_(ctx.low) - op.rank_(ctx.high) - op.rank_(ctx.close), 8))
    b = op.ts_rank(op.decay_linear(op.correlation(op.ts_rank(ctx.close, 8), op.ts_rank(adv60, 21), 8), 7), 3)
    return op.min_(a, b)


@register_factor("a89", category="alpha101",
                 requires=["low", "volume", "vwap"],
                 description="ts_rank(decay_linear(corr(l*0.97+vwap*0.03, adv10, 6),5),3) - ts_rank(decay_linear(Δvwap,10),15)")
def alpha89(ctx: FactorContext) -> pd.DataFrame:
    adv10 = op.adv(ctx.volume, 10)
    mix = ctx.low * 0.97 + ctx.vwap * 0.03
    a = op.ts_rank(op.decay_linear(op.correlation(mix, adv10, 6), 5), 3)
    b = op.ts_rank(op.decay_linear(op.delta(ctx.vwap, 10), 15), 15)
    return a - b


@register_factor("a90", category="alpha101",
                 requires=["close", "low", "volume"],
                 description="-(rank(c-max(c,5))^ts_rank(corr(adv40,l,5),3))")
def alpha90(ctx: FactorContext) -> pd.DataFrame:
    adv40 = op.adv(ctx.volume, 40)
    a = op.rank_(ctx.close - op.ts_max(ctx.close, 5))
    b = op.ts_rank(op.correlation(adv40, ctx.low, 5), 3)
    return -(a * b)


# ===================================================================
#  Alpha 91-101
# ===================================================================

@register_factor("a91", category="alpha101",
                 requires=["close", "volume", "vwap"],
                 description="简化(去 indneutralize):-(ts_rank(decay_linear(decay_linear(corr(c,vol,10),16),4),5) - rank(decay_linear(corr(vwap,adv30,4),3)))")
def alpha91(ctx: FactorContext) -> pd.DataFrame:
    adv30 = op.adv(ctx.volume, 30)
    a = op.ts_rank(op.decay_linear(op.decay_linear(op.correlation(ctx.close, ctx.volume, 10), 16), 4), 5)
    b = op.rank_(op.decay_linear(op.correlation(ctx.vwap, adv30, 4), 3))
    return -(a - b)


@register_factor("a92", category="alpha101",
                 requires=["open", "high", "low", "close", "volume"],
                 description="min(ts_rank(decay_linear(((h+l)/2+c)<(l+o),15),19), ts_rank(decay_linear(corr(rank(l),rank(adv30),8),7),7))")
def alpha92(ctx: FactorContext) -> pd.DataFrame:
    adv30 = op.adv(ctx.volume, 30)
    cond = (((ctx.high + ctx.low) / 2.0 + ctx.close) < (ctx.low + ctx.open)).astype(float)
    a = op.ts_rank(op.decay_linear(cond, 15), 19)
    b = op.ts_rank(op.decay_linear(op.correlation(op.rank_(ctx.low), op.rank_(adv30), 8), 7), 7)
    return op.min_(a, b)


@register_factor("a93", category="alpha101",
                 requires=["close", "volume", "vwap"],
                 description="简化:ts_rank(decay_linear(corr(vwap,adv81,17),20),8) / rank(decay_linear(Δ(c*0.524+vwap*0.476,3),16))")
def alpha93(ctx: FactorContext) -> pd.DataFrame:
    adv81 = op.adv(ctx.volume, 81)
    a = op.ts_rank(op.decay_linear(op.correlation(ctx.vwap, adv81, 17), 20), 8)
    mix = ctx.close * 0.524 + ctx.vwap * 0.476
    b = op.rank_(op.decay_linear(op.delta(mix, 3), 16))
    return a / b.replace(0, np.nan)


@register_factor("a94", category="alpha101",
                 requires=["volume", "vwap"],
                 description="-(rank(vwap-min(vwap,12))^ts_rank(corr(ts_rank(vwap,20),ts_rank(adv60,4),18),3))")
def alpha94(ctx: FactorContext) -> pd.DataFrame:
    adv60 = op.adv(ctx.volume, 60)
    a = op.rank_(ctx.vwap - op.ts_min(ctx.vwap, 12))
    b = op.ts_rank(op.correlation(op.ts_rank(ctx.vwap, 20), op.ts_rank(adv60, 4), 18), 3)
    return -(a * b)


@register_factor("a95", category="alpha101",
                 requires=["open", "high", "low", "volume"],
                 description="rank(o-min(o,12)) < ts_rank(rank(corr(sum((h+l)/2,19),sum(adv40,19),13))^5, 12)")
def alpha95(ctx: FactorContext) -> pd.DataFrame:
    adv40 = op.adv(ctx.volume, 40)
    a = op.rank_(ctx.open - op.ts_min(ctx.open, 12))
    inner = op.rank_(op.correlation(op.sum_((ctx.high + ctx.low) / 2.0, 19), op.sum_(adv40, 19), 13))
    b = op.ts_rank(op.signed_power(inner, 5), 12)
    return (a < b).astype(float)


@register_factor("a96", category="alpha101",
                 requires=["close", "volume", "vwap"],
                 description="-max(ts_rank(decay_linear(corr(rank(vwap),rank(vol),4),4),8), ts_rank(decay_linear(ts_argmax(corr(ts_rank(c,7),ts_rank(adv60,4),4),13),14),13))")
def alpha96(ctx: FactorContext) -> pd.DataFrame:
    adv60 = op.adv(ctx.volume, 60)
    a = op.ts_rank(op.decay_linear(op.correlation(op.rank_(ctx.vwap), op.rank_(ctx.volume), 4), 4), 8)
    inner = op.correlation(op.ts_rank(ctx.close, 7), op.ts_rank(adv60, 4), 4)
    b = op.ts_rank(op.decay_linear(op.ts_argmax(inner, 13), 14), 13)
    return -op.max_(a, b)


@register_factor("a97", category="alpha101",
                 requires=["low", "volume", "vwap"],
                 description="简化:-(rank(decay_linear(Δ(l*0.72+vwap*0.28,3),20)) - ts_rank(decay_linear(ts_rank(corr(ts_rank(l,8),ts_rank(adv60,17),5),19),16),7))")
def alpha97(ctx: FactorContext) -> pd.DataFrame:
    adv60 = op.adv(ctx.volume, 60)
    mix = ctx.low * 0.72 + ctx.vwap * 0.28
    a = op.rank_(op.decay_linear(op.delta(mix, 3), 20))
    inner = op.correlation(op.ts_rank(ctx.low, 8), op.ts_rank(adv60, 17), 5)
    b = op.ts_rank(op.decay_linear(op.ts_rank(inner, 19), 16), 7)
    return -(a - b)


@register_factor("a98", category="alpha101",
                 requires=["open", "volume", "vwap"],
                 description="rank(decay_linear(corr(vwap,sum(adv5,26),5),7)) - rank(decay_linear(ts_rank(ts_argmin(corr(rank(o),rank(adv15),21),9),7),8))")
def alpha98(ctx: FactorContext) -> pd.DataFrame:
    adv5 = op.adv(ctx.volume, 5)
    adv15 = op.adv(ctx.volume, 15)
    a = op.rank_(op.decay_linear(op.correlation(ctx.vwap, op.sum_(adv5, 26), 5), 7))
    inner = op.correlation(op.rank_(ctx.open), op.rank_(adv15), 21)
    b = op.rank_(op.decay_linear(op.ts_rank(op.ts_argmin(inner, 9), 7), 8))
    return a - b


@register_factor("a99", category="alpha101",
                 requires=["high", "low", "volume"],
                 description="-(rank(corr(sum((h+l)/2,20),sum(adv60,20),9)) < rank(corr(l,vol,6)))")
def alpha99(ctx: FactorContext) -> pd.DataFrame:
    adv60 = op.adv(ctx.volume, 60)
    a = op.rank_(op.correlation(op.sum_((ctx.high + ctx.low) / 2.0, 20), op.sum_(adv60, 20), 9))
    b = op.rank_(op.correlation(ctx.low, ctx.volume, 6))
    return -((a < b).astype(float))


@register_factor("a100", category="alpha101",
                 requires=["high", "low", "close", "volume"],
                 description="简化(去 indneutralize):-(1.5*scale(rank(((c-l)-(h-c))/(h-l)*vol)) - scale(rank(corr(c,rank(adv20),5))))*(vol/adv20)")
def alpha100(ctx: FactorContext) -> pd.DataFrame:
    adv20 = op.adv(ctx.volume, 20)
    clv = ((ctx.close - ctx.low) - (ctx.high - ctx.close)) / (ctx.high - ctx.low).replace(0, np.nan)
    a = op.scale(op.rank_(clv * ctx.volume))
    b = op.scale(op.rank_(op.correlation(ctx.close, op.rank_(adv20), 5)))
    return -(1.5 * a - b) * (ctx.volume / adv20.replace(0, np.nan))


@register_factor("a101", category="alpha101",
                 requires=["open", "high", "low", "close"],
                 description="(close - open) / ((high - low) + 0.001)")
def alpha101(ctx: FactorContext) -> pd.DataFrame:
    return (ctx.close - ctx.open) / ((ctx.high - ctx.low) + 1e-3)