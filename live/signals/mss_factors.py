"""mss_factors — split from mss_dynamic.py."""
from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd

logger = logging.getLogger('mss_live')

from .mss_state import _signal_dir, _qdb, _trading_days_between, FACTOR_NAMES



def _factors(qconn, date_str: str, names: list) -> pd.DataFrame:
    cols = ", ".join([f'"{c}"' for c in names if c != 'close'])
    df = qconn.execute(f"""
        SELECT f.date, f.symbol, b.close, {cols}
        FROM factors_wide f LEFT JOIN daily_bars b ON f.date=b.date AND f.symbol=b.symbol
        WHERE f.date='{date_str}'
    """).fetchdf()
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
    return df



def _bars(qconn, start: str, end: str) -> pd.DataFrame:
    return qconn.execute(
        "SELECT * FROM daily_bars WHERE date>=? AND date<=? ORDER BY date, symbol",
        [start, end]
    ).fetchdf()



def _weights() -> dict:
    p = "core/strategies/impl/v1_ga_rp/config.json"
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f).get("selector", {}).get("weights", {})
    return {}



def _zscore(vals: np.ndarray) -> np.ndarray:
    nz = vals[~np.isnan(vals)]
    if len(nz) < 2:
        return np.zeros_like(vals)
    lo, hi = np.percentile(nz, [1, 99])
    c = np.clip(vals, lo, hi)
    mu, sd = np.mean(c), np.std(c)
    return (c - mu) / sd if sd > 1e-10 else np.zeros_like(vals)



def mf_score(qconn, date_str: str) -> pd.DataFrame:
    w = _weights()
    available = [c for c in w.keys() if c in FACTOR_NAMES]
    cols = list(set(available + ['close']))
    df = _factors(qconn, date_str, cols)
    if df.empty:
        return df
    scores = np.zeros(len(df))
    for fn in [c for c in available if c in df.columns]:
        scores += df[fn].fillna(0).values.astype(float) * w[fn]
    df['score'] = _zscore(scores)
    return df



def chip_score(qconn, date_str: str) -> pd.DataFrame:
    df = _factors(qconn, date_str, ['volatility_20', 'momentum_20', 'close'])
    if df.empty:
        return df
    scores = np.zeros(len(df))
    if 'volatility_20' in df.columns:
        v = df['volatility_20'].fillna(0).values.astype(float)
        scores += np.where(v < 0.3, 1.0, 0.0) * 0.5
    if 'momentum_20' in df.columns:
        v = _zscore(df['momentum_20'].fillna(0).values.astype(float))
        scores += np.where(np.abs(v) < 0.3, 1.0, 0.0) * 0.3
    df['score'] = scores
    return df



def trend_factor(qconn, date_str: str) -> float:
    df = _factors(qconn, date_str, ['macd', 'macd_signal', 'momentum_5', 'momentum_20', 'rsi_14'])
    if df.empty:
        return 0.5
    sl = []
    if 'macd' in df.columns and 'macd_signal' in df.columns:
        sl.append(np.mean(df['macd'].fillna(0).values > df['macd_signal'].fillna(0).values))
    if 'momentum_5' in df.columns and 'momentum_20' in df.columns:
        m5, m20 = df['momentum_5'].fillna(0).values, df['momentum_20'].fillna(0).values
        sl.append(np.mean((m5 > 0) & (m5 > m20)))
    if 'rsi_14' in df.columns:
        r = df['rsi_14'].fillna(50).values
        sl.append(np.mean(np.where(r > 70, 0.0, np.where(r >= 50, 1.0, np.where(r >= 30, 0.5, 0.0)))))
    return np.clip(np.mean(sl) * 2.0, 0.1, 1.0) if sl else 0.5



def vol_factor(qconn, date_str: str) -> float:
    df = _factors(qconn, date_str, ['volatility_20'])
    if df.empty or 'volatility_20' not in df.columns:
        return 1.0
    return np.clip(1.0 - np.mean(df['volatility_20'].fillna(0).values > 0.05), 0.2, 1.0)



def composite_factor(qconn, date_str: str) -> float:
    """V6 composite择时: trend×60% + volatility×40%"""
    tr = trend_factor(qconn, date_str)
    vr = vol_factor(qconn, date_str)
    return np.clip(tr * 0.6 + vr * 0.4, 0.1, 1.0)



def _market_breadth(qconn, date_str: str) -> Optional[float]:
    """计算当日市场广度（涨跌比），用于二次确认市场状态"""
    df = _factors(qconn, date_str, ['close', 'returns'])
    if df.empty or 'returns' not in df.columns:
        return None
    pct = df['returns'].dropna().values.astype(float)
    pct = pct[(pct < 100) & (pct > -100) & (pct != 0)]
    if len(pct) < 50:
        return None
    return float(np.mean(pct > 0))



def market_state(qconn, date_str: str) -> Tuple[str, float]:
    bars = _bars(qconn, '2018-01-01', date_str)
    if bars.empty:
        return "oscillate", 0.3
    daily = bars.sort_values('date').groupby('date')['pct_change'].mean().fillna(0)
    p = (1 + daily).cumsum().values
    if len(p) < 200:
        return "oscillate", 0.3
    ma5 = pd.Series(p).rolling(5).mean().values
    ma20 = pd.Series(p).rolling(20).mean().values
    ma60 = pd.Series(p).rolling(60).mean().values
    ma200 = pd.Series(p).rolling(200).mean().values
    a200 = (p[-1] - ma200[-1]) / ma200[-1] if ma200[-1] > 0 else 0

    def _sl(arr, lb):
        if lb < 2 or arr[-lb] == 0:
            return 0.0
        return (arr[-1] - arr[-lb]) / arr[-lb]

    s5 = _sl(ma5, min(5, len(p) - 1))
    s20 = _sl(ma20, min(20, len(p) - 1))
    s60 = _sl(ma60, min(60, len(p) - 1))
    if a200 > 0 and s20 > -0.001:
        return "bull", min(1.0, a200 * 2 + s20 * 20)
    if a200 < 0 and s20 < 0 and s60 < 0:
        return "bear", min(1.0, abs(a200) * 2 + abs(s20) * 10 + abs(s60) * 10)
    if a200 < 0 and s5 > 0.005:
        return "recovery", s5 * 50
    sp = abs(ma5[-1] - ma20[-1]) / max(abs(ma20[-1]), 1e-10) + abs(ma20[-1] - ma60[-1]) / max(abs(ma60[-1]), 1e-10)
    if sp < 0.03:
        return "oscillate", max(0.3, 1.0 - sp * 15)
    if a200 < 0 and s5 > 0:
        return "recovery", max(0.3, s5 * 30)
    return "oscillate", 0.3



def combo_score(df_mf: pd.DataFrame, df_chip: pd.DataFrame, w_mf: float) -> pd.DataFrame:
    m = df_mf[['symbol', 'close', 'score']].rename(columns={'score': 'ms'})
    c = df_chip[['symbol', 'score']].rename(columns={'score': 'cs'})
    r = m.merge(c, on='symbol', how='left')
    r['cs'] = r['cs'].fillna(0)
    r['score'] = r['ms'] * w_mf + r['cs'] * (1 - w_mf)
    r['score'] = _zscore(r['score'].values)
    return r



def filter_buyable(df: pd.DataFrame, qconn, date_str: str,
                   exclude_prefixes: list = None,
                   enhanced_st: bool = False) -> pd.DataFrame:
    exclude_prefixes = exclude_prefixes or []
    try:
        st = set()
        try:
            st = set(qconn.execute("SELECT symbol FROM symbols WHERE UPPER(name) LIKE '%ST%'").fetchdf()['symbol'])
        except Exception:
            pass
        df = df[~df['symbol'].isin(st)]

        if enhanced_st:
            try:
                end = pd.Timestamp(date_str)
                start = end - pd.Timedelta(days=60)
                recent = qconn.execute(
                    "SELECT symbol, date, pct_change, close FROM daily_bars WHERE date>=? AND date<=? ORDER BY date",
                    [start.strftime('%Y-%m-%d'), date_str]
                ).fetchdf()
                if not recent.empty:
                    recent['pct_change'] = recent['pct_change'].astype(float)
                    recent['close'] = recent['close'].astype(float)
                    bad = set()
                    for sym in df['symbol']:
                        sb = recent[recent['symbol'] == sym].sort_values('date')
                        if len(sb) < 5:
                            continue
                        pcts = sb['pct_change'].values
                        for j in range(len(pcts) - 1):
                            if pcts[j] < -9.5 and pcts[j + 1] < -9.5:
                                bad.add(sym)
                                break
                        if sym in bad:
                            continue
                        closes = sb['close'].values[-5:]
                        lp = sb['pct_change'].values[-5:].astype(float)
                        if np.mean(closes) < 3.0 and np.mean(lp) < -2.0:
                            bad.add(sym)
                    if bad:
                        before = len(df)
                        df = df[~df['symbol'].isin(bad)]
                        logger.info(f"  增强ST过滤: 排除 {len(bad)} 只 (连续跌停/低价下跌)")
            except Exception:
                pass

        if exclude_prefixes:
            for pfx in exclude_prefixes:
                before = len(df)
                df = df[~df['symbol'].str.startswith(pfx)]
                dropped = before - len(df)
                if dropped:
                    logger.info(f"  排除 {pfx}*** 板块: {dropped} 只")
        bars = _bars(qconn, date_str, date_str)
        if not bars.empty:
            bars['symbol'] = bars['symbol'].astype(str)
            df = df.merge(bars[['symbol', 'close', 'pct_change', 'volume']], on='symbol', how='left', suffixes=('', '_b'))
            df = df[(df['pct_change'] < 9.95) & (df['volume'] > 0)]
    except Exception as e:
        logger.warning(f"过滤失败: {e}")
    return df
