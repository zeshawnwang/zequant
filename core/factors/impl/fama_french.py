"""
Fama-French 因子 (A股适配版)

基于 AKShare 免费数据实现的 Fama-French 三因子:
- MKT: 市场因子 (沪深300收益 - 无风险利率)
- SMB: 规模因子 (小市值组合 - 大市值组合)
- HML: 价值因子 (用 1/PB 代替账面市值比)

Fama-French 三因子模型 (1993):
    R_i - R_f = α_i + β_i·MKT + s_i·SMB + h_i·HML + ε_i

调用:
    import core.factors.impl.fama_french
    from core.factors.base.factor_hub import FactorHub
    long_df = FactorHub.compute_all(bars, names=["ff_mkt", "ff_smb", "ff_hml"])
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Any
import warnings

from core.factors.base.factor_hub import register_factor, FactorContext


class _TTLCache:
    """TTL 缓存,避免重复网络请求。

    每个条目缓存 ttl 秒,超时后自动失效。
    函数级实例而非模块级可变 dict。
    """
    def __init__(self, ttl: int = 3600):
        self._ttl = ttl
        self._data: Dict[str, Tuple[Any, float]] = {}

    def get(self, key: str):
        if key in self._data:
            value, timestamp = self._data[key]
            if time.monotonic() - timestamp < self._ttl:
                return value
            del self._data[key]
        return None

    def set(self, key: str, value):
        self._data[key] = (value, time.monotonic())

    def clear(self):
        self._data.clear()


_cache = _TTLCache()


def _get_risk_free_rate() -> pd.DataFrame:
    """获取国债收益率作为无风险利率(年化,转日频)。"""
    cached = _cache.get("_rf")
    if cached is not None:
        return cached

    try:
        import akshare as ak
        bond_df = ak.bond_china_treasury_yield()
        rf_series = bond_df.set_index("日期")["收益率"]
        rf_daily = rf_series / 252 / 100
        _cache.set("_rf", rf_daily)
        return rf_daily
    except Exception:
        warnings.warn("无法获取国债收益率,使用固定无风险利率 0.03")
        dates = pd.date_range("2019-01-01", pd.Timestamp.today().strftime("%Y-%m-%d"))
        fallback = pd.Series(0.03 / 252, index=dates)
        _cache.set("_rf", fallback)
        return fallback


def _get_market_return() -> pd.DataFrame:
    """获取沪深300日收益率。"""
    cached = _cache.get("_mkt")
    if cached is not None:
        return cached

    try:
        import akshare as ak
        hs300 = ak.stock_zh_index_daily(symbol="sh000300")
        hs300["returns"] = hs300["close"].pct_change()
        hs300.index = pd.to_datetime(hs300["date"])
        result = hs300["returns"]
        _cache.set("_mkt", result)
        return result
    except Exception as e:
        warnings.warn(f"无法获取沪深300数据: {e}")
        return pd.Series(dtype=float)


def _batch_daily_basic(symbols: list, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """批量获取 A 股每日基本面数据（单次 API 调用替代 N+1 逐股请求）。

    使用 ak.stock_zh_a_daily_basic() 不带 symbol 参数一次性获取全市场数据，
    避免 for 循环逐只股票 HTTP 请求。

    Returns:
        MultiIndex DataFrame: (date, symbol) → columns
    """
    cached = _cache.get("_batch_basic")
    if cached is not None:
        return cached

    try:
        import akshare as ak
        # 不传 symbol 拉取全市场，一次请求
        all_basic = ak.stock_zh_a_daily_basic(
            start_date=dates.min().strftime("%Y%m%d"),
            end_date=dates.max().strftime("%Y%m%d"),
        )
        if all_basic is None or all_basic.empty:
            raise ValueError("空结果")

        all_basic["date"] = pd.to_datetime(all_basic["date"])
        all_basic = all_basic[all_basic["symbol"].isin(symbols)].copy()
        all_basic.set_index(["date", "symbol"], inplace=True)

        _cache.set("_batch_basic", all_basic)
        return all_basic
    except Exception as e:
        warnings.warn(f"批量基本面数据获取失败: {e}")
        result = pd.DataFrame()
        _cache.set("_batch_basic", result)
        return result


def _get_market_capitalization(symbols: list, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """获取股票市值数据(收盘价 * 总股本)。

    Returns:
        DataFrame: index=date, columns=symbol, values=市值(亿元)
    """
    cached = _cache.get("_mcaps")
    if cached is not None:
        return cached

    all_basic = _batch_daily_basic(symbols, dates)
    if all_basic.empty:
        warnings.warn("市值数据为空,SMB因子可能不准确")
        _cache.set("_mcaps", pd.DataFrame())
        return pd.DataFrame()

    mcaps = {}
    for sym in symbols:
        if sym not in all_basic.index.get_level_values("symbol"):
            continue
        sym_data = all_basic.xs(sym, level="symbol")
        if "total_market_cap" in sym_data.columns:
            mcaps[sym] = sym_data["total_market_cap"] / 1e8
        elif "total_share" in sym_data.columns and "close" in sym_data.columns:
            mcaps[sym] = sym_data["total_share"] * sym_data["close"] / 1e8

    result = pd.DataFrame(mcaps) if mcaps else pd.DataFrame()
    _cache.set("_mcaps", result)
    return result


def _get_pb_ratio(symbols: list, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """获取PB(市净率)数据。

    Returns:
        DataFrame: index=date, columns=symbol, values=PB
    """
    cached = _cache.get("_pb")
    if cached is not None:
        return cached

    all_basic = _batch_daily_basic(symbols, dates)
    if all_basic.empty or "pb" not in all_basic.columns:
        warnings.warn("PB数据为空,HML因子可能不准确")
        _cache.set("_pb", pd.DataFrame())
        return pd.DataFrame()

    pbs = {}
    for sym in symbols:
        if sym not in all_basic.index.get_level_values("symbol"):
            continue
        sym_data = all_basic.xs(sym, level="symbol")
        pbs[sym] = sym_data["pb"]

    result = pd.DataFrame(pbs) if pbs else pd.DataFrame()
    _cache.set("_pb", result)
    return result


@register_factor(
    name="ff_mkt",
    category="fama_french",
    requires=["close"],
    description="Fama-French MKT因子: 市场超额收益 (沪深300收益 - 无风险利率)",
)
def ff_mkt(ctx: FactorContext) -> pd.DataFrame:
    """MKT 因子: 市场超额收益。

    MKT = R_market - R_f
    其中 R_market 是沪深300日收益率, R_f 是国债日化收益率。

    因子含义:
    - MKT > 0: 市场上涨日
    - MKT < 0: 市场下跌日
    - 因子值对所有股票相同(市场因子)
    """
    market_ret = _get_market_return()
    rf = _get_risk_free_rate()

    close_index = ctx.close.index
    mkt_series = market_ret.reindex(close_index).fillna(0)
    rf_series = rf.reindex(close_index).fillna(0)

    mkt_values = (mkt_series - rf_series).values
    mkt_factor = pd.DataFrame(
        np.tile(mkt_values, (len(ctx.close.columns), 1)).T,
        index=ctx.close.index,
        columns=ctx.close.columns,
    )
    return mkt_factor


@register_factor(
    name="ff_smb",
    category="fama_french",
    requires=["close"],
    description="Fama-French SMB因子: 规模因子 (小市值 - 大市值)",
)
def ff_smb(ctx: FactorContext) -> pd.DataFrame:
    """SMB 因子: 规模因子。

    SMB = R_small - R_big
    做多小市值组合,做空大市值组合。

    构建方法:
    1. 每日按市值分为3组 (小/中/大)
    2. 做多小市值组,做空大市值组
    3. 等权平均各组收益

    因子含义:
    - SMB > 0: 小市值表现好于大市值
    - SMB < 0: 大市值表现好于小市值
    """
    market_cap = _get_market_capitalization(
        symbols=ctx.close.columns.tolist(),
        dates=ctx.close.index,
    )

    if market_cap.empty:
        warnings.warn("市值数据为空,返回0因子值")
        return pd.DataFrame(0, index=ctx.close.index, columns=ctx.close.columns)

    mcap_aligned = ctx.close.align(market_cap, join="left")[1].fillna(method="ffill")

    returns = ctx.close.pct_change()

    smb_factor = pd.DataFrame(0.0, index=ctx.close.index, columns=ctx.close.columns)

    for date in ctx.close.index[20:]:
        date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
        if date_str not in mcap_aligned.index:
            continue

        mcap_row = mcap_aligned.loc[date_str]
        ret_row = returns.loc[date] if date in returns.index else pd.Series()

        valid = mcap_row.dropna()
        if len(valid) < 10:
            continue

        try:
            quant33 = valid.quantile(0.33)
            quant66 = valid.quantile(0.66)

            small_mask = valid <= quant33
            big_mask = valid >= quant66

            if small_mask.any() and big_mask.any():
                small_ret = ret_row[small_mask[valid.index]].mean()
                big_ret = ret_row[big_mask[valid.index]].mean()
                smb_val = small_ret - big_ret
                smb_factor.loc[date] = smb_val
        except Exception:
            continue

    smb_factor = smb_factor.ffill().fillna(0)
    return smb_factor


@register_factor(
    name="ff_hml",
    category="fama_french",
    requires=["close"],
    description="Fama-French HML因子: 价值因子 (高PB倒数 - 低PB倒数)",
)
def ff_hml(ctx: FactorContext) -> pd.DataFrame:
    """HML 因子: 价值因子 (用 1/PB 代替账面市值比)。

    HML = R_high - R_low
    做多高账面市值比(价值股),做空低账面市值比(成长股)。

    由于无法直接获取账面价值,这里用 1/PB 作为代理。

    构建方法:
    1. 每日按 PB 分为3组 (低/中/高)
    2. 做多高 PB 倒数(价值股),做空低 PB 倒数(成长股)
    3. 等权平均各组收益

    因子含义:
    - HML > 0: 价值股表现好于成长股
    - HML < 0: 成长股表现好于价值股
    """
    pb_ratio = _get_pb_ratio(
        symbols=ctx.close.columns.tolist(),
        dates=ctx.close.index,
    )

    if pb_ratio.empty:
        warnings.warn("PB数据为空,返回0因子值")
        return pd.DataFrame(0, index=ctx.close.index, columns=ctx.close.columns)

    pb_aligned = ctx.close.align(pb_ratio, join="left")[1].fillna(method="ffill")

    inv_pb = 1 / pb_aligned.replace(0, np.nan)
    returns = ctx.close.pct_change()

    hml_factor = pd.DataFrame(0.0, index=ctx.close.index, columns=ctx.close.columns)

    for date in ctx.close.index[20:]:
        date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
        if date_str not in inv_pb.index:
            continue

        inv_pb_row = inv_pb.loc[date_str]
        ret_row = returns.loc[date] if date in returns.index else pd.Series()

        valid = inv_pb_row.dropna()
        valid = valid[valid > 0]
        if len(valid) < 10:
            continue

        try:
            quant33 = valid.quantile(0.33)
            quant66 = valid.quantile(0.66)

            high_mask = valid >= quant66
            low_mask = valid <= quant33

            if high_mask.any() and low_mask.any():
                high_ret = ret_row[high_mask[valid.index]].mean()
                low_ret = ret_row[low_mask[valid.index]].mean()
                hml_val = high_ret - low_ret
                hml_factor.loc[date] = hml_val
        except Exception:
            continue

    hml_factor = hml_factor.ffill().fillna(0)
    return hml_factor


@register_factor(
    name="ff_size",
    category="fama_french",
    requires=["close"],
    description="Size因子: 市值对数 (直接用作横截面因子,非组合收益)",
)
def ff_size(ctx: FactorContext) -> pd.DataFrame:
    """Size 因子: 市值对数。

    这是一个横截面因子,直接反映股票市值大小:
    - 值越大: 市值越大
    - 值越小: 市值越小

    注意: 与 SMB 不同,SMB 是时序因子(组合收益差),
    Size 是横截面因子(个股相对大小)。

    因子含义:
    - Size > 0: 大市值股票
    - Size < 0: 小市值股票
    """
    market_cap = _get_market_capitalization(
        symbols=ctx.close.columns.tolist(),
        dates=ctx.close.index,
    )

    if market_cap.empty:
        warnings.warn("市值数据为空,返回市值因子0")
        return pd.DataFrame(0, index=ctx.close.index, columns=ctx.close.columns)

    size_factor = ctx.close.align(market_cap, join="left")[1].fillna(method="ffill")
    size_factor = np.log(size_factor.replace(0, np.nan) + 1)

    return size_factor.reindex(ctx.close.index).fillna(0)


@register_factor(
    name="ff_value",
    category="fama_french",
    requires=["close"],
    description="Value因子: PB倒数 (直接用作横截面因子,非组合收益)",
)
def ff_value(ctx: FactorContext) -> pd.DataFrame:
    """Value 因子: PB 倒数 (1/PB)。

    这是一个横截面因子,直接反映股票估值高低:
    - 值越大: 估值越低(价值股)
    - 值越小: 估值越高(成长股)

    注意: 与 HML 不同,HML 是时序因子(组合收益差),
    Value 是横截面因子(个股相对估值)。

    因子含义:
    - Value > 0: 价值股
    - Value < 0: 成长股
    """
    pb_ratio = _get_pb_ratio(
        symbols=ctx.close.columns.tolist(),
        dates=ctx.close.index,
    )

    if pb_ratio.empty:
        warnings.warn("PB数据为空,返回价值因子0")
        return pd.DataFrame(0, index=ctx.close.index, columns=ctx.close.columns)

    value_factor = 1 / pb_ratio.replace(0, np.nan)
    value_factor = value_factor.reindex(ctx.close.index).fillna(method="ffill")
    value_factor = value_factor.replace([np.inf, -np.inf], 0).fillna(0)

    return value_factor


def clear_cache():
    """清除因子缓存(用于测试或重置)。"""
    _cache.clear()
