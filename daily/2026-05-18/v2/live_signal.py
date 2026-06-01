"""mss_dynamic 动态策略切换 — 实盘信号生成（DuckDB直连版）。

根据 MarketStateSelector 检测当前市场状态，按 V6a_3way 配置
组合多个子策略的选股信号，输出明日建仓/调仓信号。

用法：
    首次建仓：
        python3 daily/2026-05-18/v2/live_signal.py --capital 50000 --build
    日常运行：
        python3 daily/2026-05-18/v2/live_signal.py --capital 50000
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
from datetime import date, datetime
from typing import Dict, List, Tuple

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from core.screening.universe import SymbolUniverse

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


def get_conn():
    src = os.path.abspath("./data/quant_data.db")
    return duckdb.connect(src, read_only=True)


def get_factors_df(conn, date_str: str, factor_names: list) -> pd.DataFrame:
    """获取指定日期各股票的因子数据。"""
    cols = ", ".join([f'"{c}"' for c in factor_names if c != 'close'])
    df = conn.execute(f"""
        SELECT f.date, f.symbol, b.close, {cols}
        FROM factors_wide f
        LEFT JOIN daily_bars b ON f.date = b.date AND f.symbol = b.symbol
        WHERE f.date = '{date_str}'
    """).fetchdf()
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
    return df


def get_daily_bars(conn, start: str, end: str) -> pd.DataFrame:
    return conn.execute(f"""
        SELECT * FROM daily_bars
        WHERE date >= '{start}' AND date <= '{end}'
        ORDER BY date, symbol
    """).fetchdf()


def load_ga_weights() -> dict:
    cfg_path = 'core/strategies/impl/v1_ga_rp/config.json'
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        return cfg.get("selector", {}).get("weights", {})
    return {}


def zscore(vals: np.ndarray) -> np.ndarray:
    """截面zscore"""
    nz = vals[~np.isnan(vals)]
    if len(nz) < 2:
        return np.zeros_like(vals)
    lo, hi = np.percentile(nz, [1, 99])
    c = np.clip(vals, lo, hi)
    mu, sd = np.mean(c), np.std(c)
    return (c - mu) / sd if sd > 1e-10 else np.zeros_like(vals)


def compute_mf(conn, date_str: str) -> pd.DataFrame:
    """多因子选股得分。"""
    w = load_ga_weights()
    available = [c for c in w.keys() if c in FACTOR_NAMES]
    cols = list(set(available + ['close']))
    df = get_factors_df(conn, date_str, cols)
    if df.empty:
        return df
    scores = np.zeros(len(df))
    usable = [c for c in available if c in df.columns]
    for fn in usable:
        vals = df[fn].fillna(0).values.astype(float)
        scores += vals * w[fn]
    df['score'] = zscore(scores)
    return df


def compute_chip(conn, date_str: str) -> pd.DataFrame:
    """筹码集中选股得分。"""
    df = get_factors_df(conn, date_str, ['volatility_20', 'momentum_20', 'close'])
    if df.empty:
        return df
    scores = np.zeros(len(df))
    if 'volatility_20' in df.columns:
        v = df['volatility_20'].fillna(0).values.astype(float)
        scores += np.where(v < 0.3, 1.0, 0.0) * 0.5
    if 'momentum_20' in df.columns:
        v = df['momentum_20'].fillna(0).values.astype(float)
        v_z = zscore(v)
        scores += np.where(np.abs(v_z) < 0.3, 1.0, 0.0) * 0.3
    df['score'] = scores
    return df


def detrend(conn, date_str: str) -> float:
    """趋势择时 0~1"""
    df = get_factors_df(conn, date_str, ['macd', 'macd_signal', 'momentum_5', 'momentum_20', 'rsi_14'])
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


def devol(conn, date_str: str) -> float:
    """波动率择时 0~1"""
    df = get_factors_df(conn, date_str, ['volatility_20'])
    if df.empty or 'volatility_20' not in df.columns:
        return 1.0
    v = df['volatility_20'].fillna(0).values.astype(float)
    return np.clip(1.0 - np.mean(v > 0.05), 0.2, 1.0)


def detect_market_state(conn, date_str: str) -> Tuple[str, float]:
    """检测市场状态。"""
    bars = get_daily_bars(conn, '2018-01-01', date_str)
    if bars.empty:
        return "oscillate", 0.3
    bars = bars.sort_values('date').dropna(subset=['close'])
    daily = bars.groupby('date')['pct_change'].mean().fillna(0)
    price = (1 + daily).cumsum()
    p = price.values
    if len(p) < 200:
        return "oscillate", 0.3
    ma5 = pd.Series(p).rolling(5).mean().values
    ma20 = pd.Series(p).rolling(20).mean().values
    ma60 = pd.Series(p).rolling(60).mean().values
    ma200 = pd.Series(p).rolling(200).mean().values
    close = p[-1]
    a200 = (close - ma200[-1]) / ma200[-1] if ma200[-1] > 0 else 0
    def slope(arr, lb):
        if lb < 2: return 0.0
        return (arr[-1] - arr[-lb]) / arr[-lb] if arr[-lb] != 0 else 0.0
    s5 = slope(ma5, min(5, len(p)-1))
    s20 = slope(ma20, min(20, len(p)-1))
    s60 = slope(ma60, min(60, len(p)-1))
    if a200 > 0 and s20 > -0.001:
        return "bull", min(1.0, a200 * 2 + s20 * 20)
    if a200 < 0 and s20 < 0 and s60 < 0:
        return "bear", min(1.0, abs(a200)*2 + abs(s20)*10 + abs(s60)*10)
    if a200 < 0 and s5 > 0.005:
        return "recovery", s5 * 50
    sp = abs(ma5[-1]-ma20[-1])/max(abs(ma20[-1]),1e-10) + abs(ma20[-1]-ma60[-1])/max(abs(ma60[-1]),1e-10)
    if sp < 0.03:
        return "oscillate", max(0.3, 1.0 - sp*15)
    if a200 < 0 and s5 > 0:
        return "recovery", max(0.3, s5*30)
    return "oscillate", 0.3


def combo(df_mf: pd.DataFrame, df_chip: pd.DataFrame, w_mf: float) -> pd.DataFrame:
    """MF+Chip 组合得分。"""
    m = df_mf[['symbol','close','score']].rename(columns={'score':'ms'})
    c = df_chip[['symbol','score']].rename(columns={'score':'cs'})
    r = m.merge(c, on='symbol', how='left')
    r['cs'] = r['cs'].fillna(0)
    r['score'] = r['ms'] * w_mf + r['cs'] * (1-w_mf)
    r['score'] = zscore(r['score'].values)
    return r


def filter_buyable(df: pd.DataFrame, conn, date_str: str) -> pd.DataFrame:
    """过滤不可买入的股票（ST/涨停/停牌）。使用duckdb直连避免锁冲突。"""
    try:
        # 从symbols表获取ST名单
        st_syms = set()
        try:
            sym_df = conn.execute(
                "SELECT symbol FROM symbols WHERE UPPER(name) LIKE '%ST%'"
            ).fetchdf()
            st_syms = set(sym_df['symbol'].tolist())
        except:
            pass
        # 排除ST
        df = df[~df['symbol'].isin(st_syms)]
        # 排除涨停/停牌
        bars = get_daily_bars(conn, date_str, date_str)
        if not bars.empty:
            bars['symbol'] = bars['symbol'].astype(str)
            # 涨停过滤
            df = df.merge(bars[['symbol', 'close', 'pct_change', 'volume']],
                          on='symbol', how='left', suffixes=('', '_bar'))
            df = df[df['pct_change'] < 9.95]  # 排除涨停板
            df = df[df['volume'] > 0]  # 排除停牌
    except Exception as e:
        logger.warning(f"过滤失败，降低标准: {e}")
    return df


def generate_orders(df: pd.DataFrame, capital: float, top_n: int) -> list:
    """从最终得分生成买入订单。"""
    df = df.sort_values('score', ascending=False).head(top_n * 2)
    df = df.dropna(subset=['close'])
    df = df[df['close'] > 0]
    selected = []
    n = top_n
    while n >= 5 and not selected:
        alloc = capital / n
        for _, r in df.iterrows():
            p = float(r['close'])
            if p > 0 and 100 * p <= alloc:
                selected.append(r)
                if len(selected) == n:
                    break
        if not selected:
            n -= 1
    if not selected:
        alloc = capital / min(len(df), 10)
        for _, r in df.iterrows():
            p = float(r['close'])
            if p > 0 and 100 * p <= alloc:
                selected.append(r)
                if len(selected) == 10:
                    break
    orders, total = [], 0.0
    n_act = len(selected)
    ap = capital / n_act if n_act else capital
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--capital', type=float, default=50000)
    parser.add_argument('--date', help='信号日期(yyyy-mm-dd)')
    parser.add_argument('--build', action='store_true', default=True)
    args = parser.parse_args()
    signal_date = date.fromisoformat(args.date) if args.date else date.today()
    ds = str(signal_date)

    conn = get_conn()

    # 如果请求的日期没有数据，回退到最新可用日期
    try:
        latest_factor = conn.execute("SELECT MAX(date) FROM factors_wide").fetchone()[0]
        latest_bar = conn.execute("SELECT MAX(date) FROM daily_bars WHERE close > 0").fetchone()[0]
        latest_date = min(latest_factor, latest_bar) if latest_factor and latest_bar else (latest_factor or latest_bar)
        if latest_date and str(latest_date) != ds:
            logger.info(f"请求日期 {ds} 无完整数据，使用最新数据日期 {latest_date}")
            ds = str(latest_date)
            signal_date = date.fromisoformat(ds)
    except Exception as e:
        logger.warning(f"获取最新日期失败: {e}")

    # 1. 检测市场状态
    state, conf = detect_market_state(conn, ds)
    logger.info(f"市场状态: {state} (置信度={conf:.2f})")

    # 2. 获取分配
    alloc = V6A_ALLOCATION.get(state, V6A_ALLOCATION["oscillate"])

    # 3. 预计算各子策略信号
    logger.info("计算子策略信号...")
    mf_df = compute_mf(conn, ds)
    chip_df = compute_chip(conn, ds)
    vol_r = devol(conn, ds)
    trend_r = detrend(conn, ds)

    if not mf_df.empty and 'score' in mf_df.columns:
        mfv = mf_df.copy(); mfv['score'] = mfv['score'] * vol_r
        c01 = mf_df.copy(); c01['score'] = c01['score'] * trend_r

    # 4. 按分配合并得分
    logger.info(f"策略分配:")
    merged_scores = {}
    for name, w in alloc:
        logger.info(f"  {name}: {w*100:.0f}%")
        src_df = None
        if name == "mf_d10_rp":
            src_df = mf_df
        elif name == "mf_vol_d10_rp":
            src_df = mfv if not mf_df.empty else mf_df
        elif name in ("chip_covrp", "chip_equal_d3"):
            src_df = chip_df
        elif name == "c01_layered_d5":
            src_df = c01 if not mf_df.empty else mf_df
        elif name == "mf50_chip50" and not mf_df.empty and not chip_df.empty:
            src_df = combo(mf_df, chip_df, 0.5)
        elif name == "mf60_chip40" and not mf_df.empty and not chip_df.empty:
            src_df = combo(mf_df, chip_df, 0.6)

        if src_df is not None and 'score' in src_df.columns:
            scores = src_df[['symbol', 'score']].dropna()
            max_s = max(abs(scores['score'].max()), abs(scores['score'].min()), 1e-10)
            for _, r in scores.iterrows():
                sym = r['symbol']
                norm = float(r['score']) / max_s * w
                merged_scores[sym] = merged_scores.get(sym, 0) + norm

    # 5. 生成订单
    df_final = pd.DataFrame({'symbol': list(merged_scores.keys()),
                              'score': list(merged_scores.values())})
    df_final['symbol'] = df_final['symbol'].astype(str)
    # 合并收盘价
    bars = get_daily_bars(conn, ds, ds)
    bars['symbol'] = bars['symbol'].astype(str)
    df_final = df_final.merge(bars[['symbol', 'close']], on='symbol', how='left')
    df_final = df_final.dropna(subset=['close'])
    df_final = filter_buyable(df_final, conn, ds)

    orders, total_cost = generate_orders(df_final, args.capital, top_n=15)

    # 6. 输出
    sym_df = conn.execute("SELECT symbol, name FROM symbols").fetchdf()
    name_map = dict(zip(sym_df['symbol'], sym_df['name'])) if not sym_df.empty else {}

    sf = f'{SIGNAL_DIR}/build_{signal_date.strftime("%Y%m%d")}_{state}.json'
    with open(sf, 'w') as f:
        json.dump({
            'meta': {'strategy': 'mss_dynamic', 'config': 'V6a_3way',
                     'signal_date': ds, 'market_state': state,
                     'confidence': round(float(conf), 3),
                     'capital': args.capital, 'total_cost': total_cost,
                     'remain': round(args.capital - total_cost, 2),
                     'n_buy': len(orders)},
            'allocation': {n: w for n, w in alloc},
            'buy_orders': orders,
        }, f, indent=2, ensure_ascii=False)
    logger.info(f'信号已写入 {sf}')

    print(f'\n╔{"═"*58}╗')
    print(f'║  ZEquant 实盘信号 — mss_dynamic (V6a_3way)')
    print(f'╠{"═"*58}╣')
    print(f'║  日期: {signal_date}')
    print(f'║  市场状态: {state} ║ 置信度: {conf:.2f}')
    print(f'║  资金: {args.capital:>8,.0f}')
    print(f'╠{"═"*58}╣')
    print(f'║  策略分配:')
    for n, w in alloc:
        print(f'║    {n:<20s} {w*100:.0f}%')
    print(f'╠{"═"*58}╣')
    print(f'║  🟢 买入 ({len(orders)}只):')
    print(f'║    {"代码":<8} {"名称":<10} {"股数":<6} {"价格":<7} {"金额":<8}')
    for o in orders:
        n = name_map.get(o['symbol'], '')
        print(f'║    {o["symbol"]:<8} {n:<10} {o["shares"]:<6} {o["price"]:<7.2f} {o["cost"]:<8,.0f}')
    print(f'╠{"═"*58}╣')
    print(f'║  占用: {total_cost:>8,.0f}  剩余: {args.capital - total_cost:>8,.0f}')
    print(f'╚{"═"*58}╝')
    conn.close()


if __name__ == '__main__':
    main()
