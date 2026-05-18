"""mss_dynamic 实盘信号生成。

根据 MarketStateSelector 检测当前市场状态，按 V6a_3way 配置
组合多个子策略的选股信号，输出明日建仓信号。

用法：
    python3 -m live.signals.mss_dynamic --capital 50000
    python3 -m live.signals.mss_dynamic --capital 50000 --date 2026-05-19
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import subprocess
from datetime import date
from typing import Dict, List, Tuple

import duckdb
import numpy as np
import pandas as pd

logger = logging.getLogger('mss_live')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

SIGNAL_DIR = 'data_live/mss_dynamic'
os.makedirs(SIGNAL_DIR, exist_ok=True)

FACTOR_NAMES = list(set([
    'close', 'returns', 'rsi_14', 'volatility_20', 'macd', 'macd_signal',
    'momentum_5', 'momentum_20', 'volume_ratio', 'boll_position',
    'a27', 'a30', 'a31', 'a41', 'a42', 'a64', 'a69', 'a8', 'a80', 'a85',
    'a88', 'a91', 'a97', 'a98', 'a99', 'ff_mkt',
    'gtja103', 'gtja104', 'gtja105', 'gtja108', 'gtja113', 'gtja117',
    'gtja12', 'gtja120', 'gtja121', 'gtja123', 'gtja127', 'gtja13',
    'gtja139', 'gtja141', 'gtja142', 'gtja144', 'gtja148', 'gtja164',
    'gtja168', 'gtja171', 'gtja176', 'gtja185', 'gtja34', 'gtja49',
    'gtja62', 'gtja76', 'gtja83', 'gtja85', 'gtja90', 'gtja91', 'gtja99',
    'beta_20',
]))

V6A_ALLOCATION = {
    "bull": [("mf_d10_rp", 0.6), ("mf_vol_d10_rp", 0.2), ("chip_covrp", 0.2)],
    "bear": [("chip_covrp", 0.6), ("chip_equal_d3", 0.2), ("mf_vol_d10_rp", 0.2)],
    "oscillate": [("chip_covrp", 0.4), ("mf50_chip50", 0.3), ("c01_layered_d5", 0.3)],
    "recovery": [("chip_equal_d3", 0.4), ("mf60_chip40", 0.3), ("mf_vol_d10_rp", 0.3)],
}


def _db():
    src = os.path.abspath("data/quant_data.db")
    cache = os.path.abspath("data/cache/quant_copy.db")
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    if not os.path.exists(cache):
        subprocess.run(["cp", "-c", src, cache], check=True)
    return duckdb.connect(cache, read_only=True)


def _factors(conn, date_str: str, names: list) -> pd.DataFrame:
    cols = ", ".join([f'"{c}"' for c in names if c != 'close'])
    df = conn.execute(f"""
        SELECT f.date, f.symbol, b.close, {cols}
        FROM factors_wide f LEFT JOIN daily_bars b ON f.date=b.date AND f.symbol=b.symbol
        WHERE f.date='{date_str}'
    """).fetchdf()
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
    return df


def _bars(conn, start: str, end: str) -> pd.DataFrame:
    return conn.execute(
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


def mf_score(conn, date_str: str) -> pd.DataFrame:
    w = _weights()
    available = [c for c in w.keys() if c in FACTOR_NAMES]
    cols = list(set(available + ['close']))
    df = _factors(conn, date_str, cols)
    if df.empty:
        return df
    scores = np.zeros(len(df))
    for fn in [c for c in available if c in df.columns]:
        scores += df[fn].fillna(0).values.astype(float) * w[fn]
    df['score'] = _zscore(scores)
    return df


def chip_score(conn, date_str: str) -> pd.DataFrame:
    df = _factors(conn, date_str, ['volatility_20', 'momentum_20', 'close'])
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


def trend_factor(conn, date_str: str) -> float:
    df = _factors(conn, date_str, ['macd', 'macd_signal', 'momentum_5', 'momentum_20', 'rsi_14'])
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


def vol_factor(conn, date_str: str) -> float:
    df = _factors(conn, date_str, ['volatility_20'])
    if df.empty or 'volatility_20' not in df.columns:
        return 1.0
    return np.clip(1.0 - np.mean(df['volatility_20'].fillna(0).values > 0.05), 0.2, 1.0)


def market_state(conn, date_str: str) -> Tuple[str, float]:
    bars = _bars(conn, '2018-01-01', date_str)
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


def filter_buyable(df: pd.DataFrame, conn, date_str: str) -> pd.DataFrame:
    try:
        st = set()
        try:
            st = set(conn.execute("SELECT symbol FROM symbols WHERE UPPER(name) LIKE '%ST%'").fetchdf()['symbol'])
        except:
            pass
        df = df[~df['symbol'].isin(st)]
        bars = _bars(conn, date_str, date_str)
        if not bars.empty:
            bars['symbol'] = bars['symbol'].astype(str)
            df = df.merge(bars[['symbol', 'close', 'pct_change', 'volume']], on='symbol', how='left', suffixes=('', '_b'))
            df = df[(df['pct_change'] < 9.95) & (df['volume'] > 0)]
    except Exception as e:
        logger.warning(f"过滤失败: {e}")
    return df


def generate_orders(df: pd.DataFrame, capital: float, top_n: int) -> Tuple[list, float]:
    df = df.sort_values('score', ascending=False).head(top_n * 2).dropna(subset=['close'])
    df = df[df['close'] > 0]
    selected = []
    n = top_n
    while n >= 5 and not selected:
        a = capital / n
        for _, r in df.iterrows():
            p = float(r['close'])
            if p > 0 and 100 * p <= a:
                selected.append(r)
                if len(selected) == n:
                    break
        if not selected:
            n -= 1
    if not selected:
        a = capital / min(len(df), 10)
        for _, r in df.iterrows():
            p = float(r['close'])
            if p > 0 and 100 * p <= a:
                selected.append(r)
                if len(selected) == 10:
                    break
    orders, total = [], 0.0
    ap = capital / len(selected) if selected else capital
    for r in selected:
        p = float(r['close'])
        s = int(ap // (p * 100)) * 100
        if s < 100:
            continue
        c = s * p
        total += c
        orders.append({'symbol': r['symbol'], 'direction': '买入',
                       'shares': s, 'price': round(p, 2), 'cost': round(c, 2),
                       'reason': 'mss_dynamic_V6a_3way'})
    return orders, round(total, 2)


def run(capital: float = 50000, signal_date: str = None) -> dict:
    """生成信号，返回结果字典。"""
    ds = signal_date or str(date.today())
    conn = _db()

    # 回退到最新可用数据
    try:
        lf = conn.execute("SELECT MAX(date) FROM factors_wide").fetchone()[0]
        lb = conn.execute("SELECT MAX(date) FROM daily_bars WHERE close>0").fetchone()[0]
        ld = min(lf, lb) if lf and lb else (lf or lb)
        if ld and str(ld) != ds:
            logger.info(f"{ds} 无数据，使用 {ld}")
            ds = str(ld)
    except:
        pass

    state, conf = market_state(conn, ds)
    alloc = V6A_ALLOCATION.get(state, V6A_ALLOCATION["oscillate"])

    mf = mf_score(conn, ds)
    chip = chip_score(conn, ds)
    vr = vol_factor(conn, ds)
    tr = trend_factor(conn, ds)

    if not mf.empty:
        mfv = mf.copy()
        mfv['score'] = mfv['score'] * vr
        c01 = mf.copy()
        c01['score'] = c01['score'] * tr

    merged = {}
    for name, w in alloc:
        src = None
        if name == "mf_d10_rp":
            src = mf
        elif name == "mf_vol_d10_rp":
            src = mfv if not mf.empty else mf
        elif name in ("chip_covrp", "chip_equal_d3"):
            src = chip
        elif name == "c01_layered_d5":
            src = c01 if not mf.empty else mf
        elif name == "mf50_chip50" and not mf.empty and not chip.empty:
            src = combo_score(mf, chip, 0.5)
        elif name == "mf60_chip40" and not mf.empty and not chip.empty:
            src = combo_score(mf, chip, 0.6)
        if src is not None and 'score' in src.columns:
            s = src[['symbol', 'score']].dropna()
            ms = max(abs(s['score'].max()), abs(s['score'].min()), 1e-10)
            for _, r in s.iterrows():
                merged[r['symbol']] = merged.get(r['symbol'], 0) + float(r['score']) / ms * w

    df = pd.DataFrame({'symbol': list(merged.keys()), 'score': list(merged.values())})
    df['symbol'] = df['symbol'].astype(str)
    bars = _bars(conn, ds, ds)
    bars['symbol'] = bars['symbol'].astype(str)
    df = df.merge(bars[['symbol', 'close']], on='symbol', how='left').dropna(subset=['close'])
    df = filter_buyable(df, conn, ds)

    orders, cost = generate_orders(df, capital, top_n=15)

    names = {}
    try:
        ndf = conn.execute("SELECT symbol, name FROM symbols").fetchdf()
        names = dict(zip(ndf['symbol'], ndf['name']))
    except:
        pass
    conn.close()

    result = {
        'date': ds, 'state': state, 'confidence': round(float(conf), 3),
        'capital': capital, 'total_cost': cost,
        'remain': round(capital - cost, 2), 'orders': orders,
        'allocations': {n: w for n, w in alloc},
        'names': names,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description='mss_dynamic 实盘信号')
    parser.add_argument('--capital', type=float, default=50000)
    parser.add_argument('--date', help='信号日期(yyyy-mm-dd)')
    args = parser.parse_args()

    r = run(capital=args.capital, signal_date=args.date)

    sf = f'{SIGNAL_DIR}/build_{r["date"].replace("-","")}_{r["state"]}.json'
    with open(sf, 'w') as f:
        json.dump({
            'meta': {'strategy': 'mss_dynamic', 'config': 'V6a_3way',
                     'signal_date': r['date'], 'market_state': r['state'],
                     'confidence': r['confidence'], 'capital': r['capital'],
                     'total_cost': r['total_cost'], 'remain': r['remain'],
                     'n_buy': len(r['orders'])},
            'allocation': r['allocations'],
            'buy_orders': r['orders'],
        }, f, indent=2, ensure_ascii=False)
    logger.info(f'信号已写入 {sf}')

    print(f'\n╔{"═"*58}╗')
    print(f'║  ZEquant 实盘信号 — mss_dynamic (V6a_3way)')
    print(f'╠{"═"*58}╣')
    print(f'║  日期: {r["date"]}')
    print(f'║  市场状态: {r["state"]} ║ 置信度: {r["confidence"]:.2f}')
    print(f'║  资金: {r["capital"]:>8,.0f}')
    print(f'╠{"═"*58}╣')
    print(f'║  策略分配:')
    for n, w in r['allocations'].items():
        print(f'║    {n:<20s} {w*100:.0f}%')
    print(f'╠{"═"*58}╣')
    print(f'║  🟢 买入 ({len(r["orders"])}只):')
    print(f'║    {"代码":<8} {"名称":<10} {"股数":<6} {"价格":<7} {"金额":<8}')
    for o in r['orders']:
        n = r['names'].get(o['symbol'], '')
        print(f'║    {o["symbol"]:<8} {n:<10} {o["shares"]:<6} {o["price"]:<7.2f} {o["cost"]:<8,.0f}')
    print(f'╠{"═"*58}╣')
    print(f'║  占用: {r["total_cost"]:>8,.0f}  剩余: {r["remain"]:>8,.0f}')
    print(f'╚{"═"*58}╝')


if __name__ == '__main__':
    main()
