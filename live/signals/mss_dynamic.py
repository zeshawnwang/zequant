"""mss_dynamic 实盘信号生成。

根据 MarketStateSelector 检测当前市场状态，按 V6a_3way 配置
组合多个子策略的选股信号，输出建仓/调仓信号。
持仓状态持久化在 data_live/live_data.db 中。

用法：
    python3 -m live.signals.mss_dynamic --capital 50000
    python3 -m live.signals.mss_dynamic --capital 50000 --date 2026-05-19
    python3 -m live.signals.mss_dynamic --capital 50000 --force
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.database import Database

logger = logging.getLogger('mss_live')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

SIGNAL_DIR = 'data_live/mss_dynamic'
LIVE_DB_PATH = "./data_live/live_data.db"
os.makedirs(SIGNAL_DIR, exist_ok=True)


def _signal_dir(date_key: str) -> str:
    """返回该日期的信号目录，如 data_live/mss_dynamic/20260519/"""
    d = os.path.join(SIGNAL_DIR, date_key)
    os.makedirs(d, exist_ok=True)
    return d

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

# V4 最优分配: recovery修复 + 与原V6a_3way一致 (已验证无可超越)
V6A_ALLOCATION = {
    "bull": [("mf_d10_rp", 0.6), ("mf_vol_d10_rp", 0.2), ("chip_covrp", 0.2)],
    "bear": [("chip_covrp", 0.6), ("chip_equal_d3", 0.2), ("mf_vol_d10_rp", 0.2)],
    "oscillate": [("chip_covrp", 0.4), ("mf50_chip50", 0.3), ("c01_layered_d5", 0.3)],
    # V4 recovery修复: osr_d10替代mf60_chip40 (超跌反弹更适合熊市反弹)
    "recovery": [("chip_covrp", 0.4), ("osr_d10", 0.3), ("mf_vol_d10_rp", 0.3)],
}

# V4 子策略参数: mf系列rf=5 (高频调仓验证更优)
SUB_STRATEGY_META = {
    "mf_d10_rp":       {"rebal_freq": 5,  "top_n": 10, "signal": "mf"},
    "mf_vol_d10_rp":   {"rebal_freq": 5,  "top_n": 8,  "signal": "mf_vol"},
    "chip_covrp":      {"rebal_freq": 3,  "top_n": 6,  "signal": "chip"},
    "chip_equal_d3":   {"rebal_freq": 3,  "top_n": 6,  "signal": "chip"},
    "c01_layered_d5":  {"rebal_freq": 5,  "top_n": 6,  "signal": "mf_trend"},
    "mf50_chip50":     {"rebal_freq": 5,  "top_n": 8,  "signal": "combo_50"},
    "mf60_chip40":     {"rebal_freq": 5,  "top_n": 8,  "signal": "combo_60"},
    "osr_d10":         {"rebal_freq": 10, "top_n": 6,  "signal": "osr"},
}

BOARD_PREFIXES = {
    "主板":   ["000", "001", "002", "003", "600", "601", "603", "605"],
    "创业板": ["300", "301"],
    "科创板": ["688", "689"],
    "北交所": ["430", "830", "831", "832", "833", "834", "835", "836", "837", "838", "839", "870", "871", "872", "873"],
}
DEFAULT_EXCLUDE = ["300", "301"]

# V4 紧止损配置 (6/8) — V4实验验证: 比原V2b止损(8/10) Calmar+1.0+
STOP_LOSS_CONFIG = {
    "mf_d10_rp": 0.06,
    "mf_vol_d10_rp": 0.06,
    "chip_covrp": 0.08,
    "chip_equal_d3": 0.08,
    "c01_layered_d5": 0.06,
    "mf50_chip50": 0.06,
    "mf60_chip40": 0.06,
    "osr_d10": 0.06,
}

# V4 移动止盈 3% — V6 walk-forward 三窗口验证最优 (可选5%稳健)
TRAILING_STOP_PCT = 0.03


# ══════════════════════════════════════
# DB 持久化（data_live/live_data.db）
# ══════════════════════════════════════

def _init_live_db():
    """初始化实盘数据库中的 mss 子策略状态表。"""
    db = _live_db()
    db.conn.execute("""
        CREATE TABLE IF NOT EXISTS sub_strategy_state (
            name            TEXT PRIMARY KEY,
            last_date       TEXT,
            last_state      TEXT,
            last_allocation TEXT,  -- JSON: 该策略权重，用于检测分配变化
            holdings        TEXT,  -- JSON: {symbol: {shares, price}}
            used_capital    REAL DEFAULT 0
        )
    """)
    db.conn.execute("""
        CREATE TABLE IF NOT EXISTS mss_meta (
            key   TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    # 清理旧 JSON 状态文件
    import glob
    for f in glob.glob(os.path.join(SIGNAL_DIR, "sub_state_*.json")):
        try:
            os.remove(f)
        except:
            pass
    db.conn.commit()
    db.close()


def _live_db() -> Database:
    return Database(LIVE_DB_PATH)


def _load_all_sub_states(live_db) -> dict:
    """读取所有子策略的持仓状态。返回 {name: {...}}。"""
    rows = live_db.conn.execute("SELECT name, last_date, last_state, last_allocation, holdings, used_capital FROM sub_strategy_state").fetchall()
    result = {}
    for r in rows:
        result[r[0]] = {
            "name": r[0], "last_date": r[1], "last_state": r[2],
            "last_allocation": json.loads(r[3]) if r[3] else {},
            "holdings": json.loads(r[4]) if r[4] else {},
            "used_capital": float(r[5]) if r[5] else 0.0,
        }
    return result


def _load_sub_state(live_db, name: str) -> dict:
    """读取单个子策略状态。"""
    r = live_db.conn.execute("SELECT last_date, last_state, last_allocation, holdings, used_capital FROM sub_strategy_state WHERE name=?", [name]).fetchone()
    if r:
        return {
            "last_date": r[0], "last_state": r[1],
            "last_allocation": json.loads(r[2]) if r[2] else {},
            "holdings": json.loads(r[3]) if r[3] else {},
            "used_capital": float(r[4]) if r[4] else 0.0,
        }
    return {"last_date": None, "last_state": None, "last_allocation": {}, "holdings": {}, "used_capital": 0.0}


def _save_sub_state(live_db, name: str, state: dict, current_state: str, current_weight: float):
    live_db.conn.execute("""
        INSERT INTO sub_strategy_state (name, last_date, last_state, last_allocation, holdings, used_capital)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT (name) DO UPDATE SET
            last_date=EXCLUDED.last_date, last_state=EXCLUDED.last_state,
            last_allocation=EXCLUDED.last_allocation, holdings=EXCLUDED.holdings,
            used_capital=EXCLUDED.used_capital
    """, [
        name, state.get("last_date"),
        current_state,
        json.dumps({"weight": current_weight}),
        json.dumps(state.get("holdings", {})),
        state.get("used_capital", 0.0),
    ])


def _delete_sub_state(live_db, name: str):
    live_db.conn.execute("DELETE FROM sub_strategy_state WHERE name=?", [name])


def _get_last_known_state(live_db) -> str:
    r = live_db.conn.execute("SELECT value FROM mss_meta WHERE key='last_known_state'").fetchone()
    return r[0] if r else None


def _set_last_known_state(live_db, state: str):
    live_db.conn.execute("""
        INSERT INTO mss_meta (key, value) VALUES ('last_known_state', ?)
        ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value
    """, [state])


# ══════════════════════════════════════
# 数据 / 信号计算（DuckDB 只读）
# ══════════════════════════════════════

def _trading_days_between(qconn, d1: str, d2: str) -> int:
    start, end = (d1, d2) if d1 <= d2 else (d2, d1)
    rows = qconn.execute(
        "SELECT COUNT(DISTINCT date) FROM daily_bars WHERE date > ? AND date <= ?",
        [start, end]
    ).fetchone()
    return rows[0] if rows else 0


def _qdb():
    return duckdb.connect("data/quant_data.db", read_only=True)


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
        except:
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


def generate_orders(df: pd.DataFrame, capital: float, top_n: int,
                     old_holdings: dict = None) -> Tuple[list, list, float, dict]:
    df = df.sort_values('score', ascending=False).dropna(subset=['close'])
    df = df[df['close'] > 0]

    kept = {}; kept_cost = 0.0
    if old_holdings:
        old_syms = set(old_holdings.keys())
        for _, r in df.iterrows():
            sym = r['symbol']
            if sym in old_syms and sym not in kept:
                kept[sym] = old_holdings[sym]
                kept_cost += kept[sym]['shares'] * kept[sym]['price']
    available = max(capital - kept_cost, capital * 0.3)
    new_slots = max(1, top_n - len(kept))

    # 从非保留股票中取 new_slots*3 候选，不限制 head 数量
    candidates = []
    for _, r in df.iterrows():
        if r['symbol'] not in kept:
            candidates.append(r)
            if len(candidates) >= new_slots * 3:
                break

    # 贪心填充：直到预算用完或填满 new_slots
    new_budget = available
    buy_orders, new_cost = [], 0.0
    new_syms = set()
    equal_share = new_budget / new_slots if new_slots > 0 else new_budget

    for r in candidates:
        p = float(r['close'])
        # 每只最多花 equal_share，最少1手
        budget_per = min(equal_share * 1.2, new_budget)
        s = int(budget_per // (p * 100)) * 100
        if s < 100:
            s = 100
        c = s * p
        if c > new_budget * 1.1:  # 允许轻微超支
            continue
        buy_orders.append({'symbol': r['symbol'], 'direction': '买入',
                           'shares': s, 'price': round(p, 2), 'cost': round(c, 2)})
        new_cost += c; new_budget -= c; new_syms.add(r['symbol'])
        if len(buy_orders) >= new_slots or new_budget < 100:
            break

    sell_orders = []
    if old_holdings:
        for sym, h in old_holdings.items():
            if sym not in new_syms and sym not in kept:
                sell_orders.append({'symbol': sym, 'direction': '卖出',
                                    'shares': h['shares'], 'price': h['price'],
                                    'reason': '子策略调仓'})
    return buy_orders, sell_orders, round(new_cost + kept_cost, 2), kept


# ══════════════════════════════════════
# 信号生成主逻辑
# ══════════════════════════════════════

def run(capital: float = 50000, signal_date: str = None,
        exclude_boards: list = None, force: bool = False,
        mode: str = "v2b") -> dict:
    """生成信号。signal_date 为信号请求日期（用于目录），ds 为实际数据日期。

    Args:
        mode: "baseline"=原始版, "v2b"=增强ST+止损(默认), "v2c"=增强ST+止损+置信度联动
    """
    exclude_boards = exclude_boards or DEFAULT_EXCLUDE
    signal_date = signal_date or str(date.today())

    enhanced_st = mode in ("v2b", "v2c")
    stop_loss_enabled = mode in ("v2b", "v2c")
    use_confidence_weights = (mode == "v2c")

    _init_live_db()

    qconn = _qdb()
    ds = signal_date
    latest_data = None
    try:
        lf = qconn.execute("SELECT MAX(date) FROM factors_wide").fetchone()[0]
        lb = qconn.execute("SELECT MAX(date) FROM daily_bars WHERE close>0").fetchone()[0]
        latest_data = min(lf, lb) if lf and lb else (lf or lb)
        if latest_data and str(latest_data) != ds:
            logger.info(f"{ds} 无数据，使用 {latest_data}")
            ds = str(latest_data)
    except:
        pass

    state, conf = market_state(qconn, ds)

    # V6 市场广度二次确认: breadth < 0.35 降级为 oscillate
    br = _market_breadth(qconn, ds)
    if br is not None and br < 0.35 and state != "oscillate":
        logger.info(f"  市场广度={br:.3f}<0.35, {state}→oscillate")
        state = "oscillate"

    current_alloc = V6A_ALLOCATION.get(state, V6A_ALLOCATION["oscillate"])

    # V2c: 置信度联动权重调整
    if use_confidence_weights and conf < 0.5:
        adjusted = []
        for name, weight in current_alloc:
            if name == "mf_d10_rp":
                weight = weight * 0.6
            elif name == "c01_layered_d5":
                weight = weight * 0.7
            adjusted.append((name, max(weight, 0.0)))
        total_w = sum(w for _, w in adjusted)
        if total_w > 0:
            adjusted = [(n, w / total_w) for n, w in adjusted]
        current_alloc = adjusted
        logger.info(f"  V2c: 置信度={conf:.2f}<0.5, 已调整分配权重")

    live_db = _live_db()

    # 读取上次状态
    last_state = _get_last_known_state(live_db)
    all_old_states = _load_all_sub_states(live_db)
    state_changed = (last_state is not None and last_state != state)

    # 当前分配的子策略名列表
    current_names = {name for name, w in current_alloc}
    # 旧持仓中的所有子策略名
    old_names = set(all_old_states.keys())

    # ===== 收集所有卖出 =====
    all_buy = []
    all_sell = []
    sub_details = []
    total_new_cost = 0.0
    total_capital_used = 0.0
    sold_off_symbols = set()  # 已卖出的，避免重复

    # 1. 处理状态切换 → 退出分配的子策略，全部清仓卖出
    orphan_names = old_names - current_names
    if orphan_names:
        logger.info(f"状态切换 {last_state}→{state}，以下子策略退出分配:")
    for name in orphan_names:
        old = all_old_states[name]
        holdings = old.get("holdings", {})
        if holdings:
            logger.info(f"  ❌ {name}: 清仓卖出 {len(holdings)} 只")
            for sym, h in holdings.items():
                all_sell.append({'symbol': sym, 'direction': '卖出',
                                 'shares': h['shares'], 'price': h.get('price', 0),
                                 'reason': f'{name}退出分配'})
                sold_off_symbols.add(sym)
        _delete_sub_state(live_db, name)
        # 清仓的资金会释放，不计入占用
        sub_details.append({"name": name, "status": "liquidated",
                            "n_sell": len(holdings)})

    # 2. 处理当前分配中的每个子策略
    for name, weight in current_alloc:
        sub_capital = capital * weight
        meta = SUB_STRATEGY_META.get(name, {"rebal_freq": 10, "top_n": 8})
        old_state = all_old_states.get(name, {})
        last_date_s = old_state.get("last_date")

        due = False
        is_new = (last_date_s is None)

        if is_new:
            due = True
            logger.info(f"  {name}: 首次激活 (权重{weight:.0%})")
        elif force:
            due = True
            logger.info(f"  {name}: 强制调仓")
        elif state_changed:
            due = True
            logger.info(f"  {name}: 状态切换→调仓")
        else:
            ndays = _trading_days_between(qconn, last_date_s, ds)
            due = ndays >= meta["rebal_freq"]

        if due:
            logger.info(f"  {name}: 到期调仓 (距上次 {last_date_s or '首次'} "
                        f"{_trading_days_between(qconn, last_date_s, ds) if last_date_s else 0}天 >= {meta['rebal_freq']}天)")
            scores = _get_scores(qconn, name, ds)
            if scores.empty or 'score' not in scores.columns:
                logger.warning(f"  {name}: 信号为空，跳过")
                sub_details.append({"name": name, "status": "skip_empty"})
                continue

            scores = scores[['symbol', 'close', 'score']].dropna()
            scores['symbol'] = scores['symbol'].astype(str)
            scores = filter_buyable(scores, qconn, ds, exclude_prefixes=exclude_boards, enhanced_st=enhanced_st)

            old_h = old_state.get("holdings", {})
            buys, sells, cost, kept = generate_orders(scores, sub_capital, meta["top_n"], old_holdings=old_h)

            # 更新状态：新买入 + 保留持仓
            new_holdings = {}
            for sym, h in kept.items():
                new_holdings[sym] = h
            for o in buys:
                new_holdings[o['symbol']] = {"shares": o['shares'], "price": o['price'], "peak": o['price']}
            _save_sub_state(live_db, name, {
                "last_date": ds, "holdings": new_holdings, "used_capital": cost,
            }, state, weight)

            all_buy.extend(buys)
            # 过滤掉已被清仓的卖出
            for o in sells:
                if o['symbol'] not in sold_off_symbols:
                    all_sell.append(o)
            total_new_cost += cost
            total_capital_used += cost
            sub_details.append({
                "name": name, "status": "rebalanced",
                "weight": weight, "capital": round(sub_capital, 2),
                "n_buy": len(buys), "n_sell": len(sells), "cost": cost,
            })
        else:
            old_h = old_state.get("holdings", {})
            kept_cap = old_state.get("used_capital", 0)

            # V2b/V2c: 个股止损 + V6 trailing stop 检查
            stop_sold = set()
            if old_h:
                try:
                    bars_today = _bars(qconn, ds, ds)
                    if not bars_today.empty:
                        pm = dict(zip(bars_today['symbol'].astype(str), bars_today['close'].astype(float)))
                        sl_pct = STOP_LOSS_CONFIG.get(name, 0.08)
                        trail_pct = TRAILING_STOP_PCT
                        for sym, h in list(old_h.items()):
                            cp = pm.get(sym)
                            if cp and h.get('price', 0) > 0:
                                entry = h['price']
                                loss = (cp - entry) / entry
                                # 固定止损
                                if loss < -sl_pct:
                                    all_sell.append({'symbol': sym, 'direction': '卖出',
                                                     'shares': h['shares'], 'price': cp,
                                                     'reason': f'{name}: 止损{sl_pct*100:.0f}%'})
                                    logger.info(f"  ⛔ {name}: {sym} 止损触发 (亏损{loss*100:.1f}%)")
                                    stop_sold.add(sym)
                                    continue
                                # V6 移动止盈: 从峰值回撤 > trail_pct 时卖出
                                peak = h.get('peak', entry)
                                if cp > peak:
                                    peak = cp
                                    old_h[sym]['peak'] = peak
                                if peak > 0 and cp < peak * (1.0 - trail_pct):
                                    drawdown = (cp - peak) / peak
                                    all_sell.append({'symbol': sym, 'direction': '卖出',
                                                     'shares': h['shares'], 'price': cp,
                                                     'reason': f'{name}: 移动止盈{trail_pct*100:.0f}%'})
                                    logger.info(f"  🎯 {name}: {sym} 移动止盈触发 (从峰值回撤{drawdown*100:.1f}%)")
                                    stop_sold.add(sym)
                                    continue
                                # 更新 peak
                                if cp > peak:
                                    old_h[sym]['peak'] = cp
                except Exception as e:
                    logger.warning(f"  {name}: 止损检查失败: {e}")

            total_capital_used += kept_cap
            next_rebal = _next_rebal_date(qconn, last_date_s, ds, meta["rebal_freq"])
            logger.info(f"  {name}: 未到期，维持 {len(old_h)} 只持仓"
                        + (f" (止损{len(stop_sold)}只)" if stop_sold else "")
                        + (f", 下次调仓≈{next_rebal}" if next_rebal else ""))
            sub_details.append({
                "name": name, "status": "hold",
                "weight": weight, "n_hold": len(old_h),
                "last_date": last_date_s,
                "rebal_freq": meta["rebal_freq"],
                "next_rebalance": next_rebal,
            })

    # 写入 last_known_state
    _set_last_known_state(live_db, state)
    live_db.conn.commit()

    # 股票名称
    names = {}
    try:
        ndf = qconn.execute("SELECT symbol, name FROM symbols").fetchdf()
        names = dict(zip(ndf['symbol'], ndf['name']))
    except:
        pass
    qconn.close()
    live_db.close()

    # 计算真实占用 = 既有持仓 + 新增买入 - 已卖出持仓中用到的部分
    # total_capital_used 已包含: 退出分配的old capital + 调仓的new cost + hold的kept capital
    occupied = round(total_capital_used, 2)
    if occupied == 0:
        occupied = round(total_new_cost, 2)  # 极端情况：只有新买没有旧持仓

    result = {
        'is_rebalance': len(all_buy) > 0,
        'state_changed': state_changed,
        'last_state': last_state,
        'date': signal_date, 'data_date': ds,
        'state': state, 'confidence': round(float(conf), 3),
        'breadth': round(float(br), 3) if br is not None else None,
        'capital': capital, 'total_cost': occupied,
        'total_capital_used': occupied,
        'remain': round(capital - occupied, 2),
        'orders': all_buy,
        'sell_orders': all_sell,
        'allocations': {n: w for n, w in current_alloc},
        'names': names,
        'sub_details': sub_details,
        'mode': mode,
    }
    return result


def _next_rebal_date(qconn, last_date: Optional[str], ds: str, rebal_freq: int) -> Optional[str]:
    """计算下次调仓日。"""
    if last_date is None:
        return None
    ndays = _trading_days_between(qconn, last_date, ds)
    remaining = max(0, rebal_freq - ndays)
    if remaining == 0:
        return None  # 今天就该调仓了
    # 获取 future trading days
    rows = qconn.execute(
        "SELECT DISTINCT date FROM daily_bars WHERE date > ? AND close>0 ORDER BY date",
        [ds]
    ).fetchall()
    if len(rows) < remaining:
        return str(rows[-1][0]) + "+" if rows else None  # 数据不够，用最后一天+表示至少
    return str(rows[remaining - 1][0])


def _get_scores(qconn, name: str, date_str: str) -> pd.DataFrame:
    meta = SUB_STRATEGY_META.get(name, {})
    sig = meta.get("signal", "mf")
    mf = mf_score(qconn, date_str)
    chip = chip_score(qconn, date_str)
    vr = vol_factor(qconn, date_str)
    tr = trend_factor(qconn, date_str)
    # V6 composite择时: trend×0.6 + vol×0.4
    cp = composite_factor(qconn, date_str)
    mfv = c01 = None
    if not mf.empty:
        mfv = mf.copy()
        mfv['score'] = mfv['score'] * cp
        c01 = mf.copy()
        c01['score'] = c01['score'] * tr

    if sig == "mf":
        return mf
    elif sig == "mf_vol":
        return mfv if mfv is not None else mf
    elif sig == "chip":
        return chip
    elif sig == "mf_trend":
        return c01 if c01 is not None else mf
    elif sig == "combo_50" and not mf.empty and not chip.empty:
        return combo_score(mf, chip, 0.5)
    elif sig == "combo_60" and not mf.empty and not chip.empty:
        return combo_score(mf, chip, 0.6)
    return mf


# ══════════════════════════════════════
# CLI
# ══════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='mss_dynamic 实盘信号')
    parser.add_argument('--capital', type=float, default=50000)
    parser.add_argument('--date', help='信号日期(yyyy-mm-dd)')
    parser.add_argument('--exclude-boards', nargs='*',
                        default=DEFAULT_EXCLUDE,
                        help='排除板块代码前缀，默认排除创业板 300 301')
    parser.add_argument('--force', action='store_true', help='强制所有子策略调仓')
    parser.add_argument('--email', action='store_true', help='信号生成后发送邮件通知')
    parser.add_argument('--mode', choices=['baseline', 'v2b', 'v2c'], default='v2b',
                        help='基线版 / V2b增强ST+止损(默认) / V2c+置信度联动')
    args = parser.parse_args()

    r = run(capital=args.capital, signal_date=args.date,
            exclude_boards=args.exclude_boards, force=args.force,
            mode=args.mode)

    date_key = r["date"].replace("-", "")
    sig_dir = _signal_dir(date_key)

    # 总是写入信号文件（即使是持有不动）
    meta = {
        'strategy': 'mss_dynamic', 'config': r.get('mode', 'v2b'),
        'signal_date': r['date'], 'data_date': r.get('data_date', r['date']), 'market_state': r['state'],
        'state_changed': r.get('state_changed', False),
        'last_state': r.get('last_state'),
        'confidence': r['confidence'],
        'capital': r['capital'],
        'total_cost': r['total_cost'],
        'total_capital_used': r['total_capital_used'],
        'remain': r['remain'],
        'breadth': r.get('breadth'),
        'n_buy': len(r['orders']),
        'n_sell': len(r.get('sell_orders', [])),
        'is_hold': len(r['orders']) == 0 and len(r.get('sell_orders', [])) == 0,
    }
    sf = os.path.join(sig_dir, 'build.json')
    with open(sf, 'w') as f:
        json.dump({'meta': meta, 'allocation': r['allocations'],
                    'sell_orders': r.get('sell_orders', []),
                    'buy_orders': r['orders'],
                    'sub_details': r.get('sub_details', []),
                    'names': r.get('names', {})},
                  f, indent=2, ensure_ascii=False)
    logger.info(f'信号已写入 {sf}')

    hf = os.path.join(sig_dir, 'build.html')
    _write_html(hf, r)
    logger.info(f'HTML已写入 {hf}')

    # 自动同步持仓快照
    if meta['is_hold']:
        _sync_hold_snapshot(r['date'], meta['total_capital_used'], meta['remain'])
    else:
        logger.info(f'调仓日，等待 record.py 录入成交')

    if args.email:
        _send_email(sf)


def _write_html(path: str, r: dict):
    rows = ""
    for i, o in enumerate(r['orders'], 1):
        n = r['names'].get(o['symbol'], '')
        rows += f"""<tr>
          <td>{i}</td>
          <td>{o['symbol']}</td>
          <td>{n}</td>
          <td class="num">{o['shares']}</td>
          <td class="num">{o['price']:.2f}</td>
          <td class="num">{o['cost']:,.0f}</td>
        </tr>"""

    sell_rows = ""
    for o in r.get('sell_orders', []):
        n = r['names'].get(o['symbol'], '')
        sell_rows += f"""<tr>
          <td>{o['symbol']}</td>
          <td>{n}</td>
          <td class="num">{o['shares']}</td>
          <td class="num">{o['price']:.2f}</td>
        </tr>"""

    sub_rows = ""
    status_icon_map = {"rebalanced": "🔄", "hold": "✅", "liquidated": "❌", "skip_empty": "⚠️"}
    for sd in r.get('sub_details', []):
        icon = status_icon_map.get(sd['status'], '❓')
        detail = ""
        if sd['status'] == 'rebalanced':
            detail = f"买入{sd.get('n_buy',0)}只 卖出{sd.get('n_sell',0)}只 占用{sd.get('cost',0):,.0f}/{sd.get('capital',0):,.0f}"
        elif sd['status'] == 'hold':
            detail = f"持仓{sd.get('n_hold',0)}只"
            nr = sd.get('next_rebalance')
            if nr: detail += f" (下次调仓≈{nr})"
        elif sd['status'] == 'liquidated':
            detail = f"清仓{sd.get('n_sell',0)}只"
        sub_rows += f"<li>{icon} {sd['name']} — {detail}</li>"

    alloc_rows = "".join(f"<li>{name} <strong>{w*100:.0f}%</strong></li>" for name, w in r['allocations'].items())

    state_icon = {"bull": "🐂", "bear": "🐻", "oscillate": "〰️", "recovery": "📈"}
    icon = state_icon.get(r['state'], "〰️")
    n_sell = len(r.get('sell_orders', []))

    sell_section = ""
    if n_sell > 0:
        sell_section = f"""<div class="card">
    <h3 style="margin-bottom:12px;">🔴 卖出清单 ({n_sell}只)</h3>
    <table>
      <thead><tr><th>代码</th><th>名称</th><th class="num">股数</th><th class="num">参考价</th></tr></thead>
      <tbody>{sell_rows}</tbody>
    </table>
  </div>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ZEquant 实盘信号 — mss_dynamic</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; background:#f5f7fa; color:#1a1a2e; padding:24px; }}
  .container {{ max-width:800px; margin:0 auto; }}
  .card {{ background:#fff; border-radius:12px; box-shadow:0 2px 12px rgba(0,0,0,.08); padding:24px; margin-bottom:20px; }}
  h1 {{ font-size:22px; margin-bottom:4px; }}
  .badge {{ display:inline-block; padding:4px 12px; border-radius:6px; font-size:14px; font-weight:600; }}
  .badge.bull {{ background:#e8f5e9; color:#2e7d32; }}
  .badge.bear {{ background:#ffebee; color:#c62828; }}
  .badge.oscillate {{ background:#fff3e0; color:#e65100; }}
  .badge.recovery {{ background:#e3f2fd; color:#1565c0; }}
  .meta {{ display:flex; gap:20px; flex-wrap:wrap; margin:12px 0; font-size:14px; color:#666; }}
  .meta span {{ background:#f5f5f5; padding:4px 10px; border-radius:4px; }}
  .alloc {{ display:flex; gap:12px; flex-wrap:wrap; list-style:none; }}
  .alloc li {{ background:#f0f4ff; padding:6px 14px; border-radius:6px; font-size:14px; }}
  .sub-list {{ list-style:none; }}
  .sub-list li {{ padding:6px 0; font-size:14px; }}
  table {{ width:100%; border-collapse:collapse; font-size:14px; }}
  th {{ text-align:left; padding:10px 8px; border-bottom:2px solid #e0e0e0; color:#666; font-weight:600; }}
  td {{ padding:8px; border-bottom:1px solid #f0f0f0; }}
  tr:last-child td {{ border-bottom:none; }}
  .num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  .total {{ font-weight:600; }}
  .remain {{ margin-top:12px; text-align:right; font-size:15px; color:#666; }}
  .footer {{ text-align:center; font-size:12px; color:#999; margin-top:20px; }}
</style>
</head>
<body>
<div class="container">
  <div class="card">
    <h1>ZEquant 实盘信号</h1>
    <div style="font-size:13px;color:#999;margin-bottom:8px;">mss_dynamic — V6a_3way 动态策略切换</div>
    <div class="badge {r['state']}">{icon} {r['state'].upper()}</div>
    <div class="meta">
      <span>📅 {r['date']}</span>
      <span>🎯 置信度 {(r['confidence']*100):.0f}%</span>
      <span>💰 资金 {r['capital']:,.0f}</span>
      <span>🔄 状态变化 {r.get('state_changed', False)}</span>
    </div>
  </div>
  <div class="card">
    <h3 style="margin-bottom:10px;">📊 子策略状态</h3>
    <ul class="sub-list">{sub_rows}</ul>
  </div>
  <div class="card">
    <h3 style="margin-bottom:10px;">📊 策略分配</h3>
    <ul class="alloc">{alloc_rows}</ul>
  </div>
  {sell_section}
  <div class="card">
    <h3 style="margin-bottom:12px;">🟢 买入清单 ({len(r['orders'])}只)</h3>
    <table>
      <thead><tr><th>#</th><th>代码</th><th>名称</th><th class="num">股数</th><th class="num">价格</th><th class="num">金额</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
    <div class="remain">
      新占用: <span class="total">{r['total_cost']:,.0f}</span> &nbsp;|&nbsp; 总占用: <span class="total">{r['total_capital_used']:,.0f}</span> &nbsp;|&nbsp; 剩余: <span class="total">{r['remain']:,.0f}</span>
    </div>
  </div>
  <div class="footer">ZEquant — Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
</div>
</body>
</html>"""
    with open(path, 'w') as f:
        f.write(html)

    print(f'\n╔{"═"*58}╗')
    print(f'║  ZEquant 实盘信号 — mss_dynamic (V6a_3way)')
    print(f'╠{"═"*58}╣')
    print(f'║  日期: {r["date"]}')
    print(f'║  市场状态: {r["state"]} ║ 置信度: {r["confidence"]:.2f}')
    if r.get('state_changed'):
        print(f'║  🔄 状态变化: {r.get("last_state")} → {r["state"]}')
    print(f'║  资金: {r["capital"]:>8,.0f}')
    print(f'╠{"═"*58}╣')
    print(f'║  子策略状态:')
    for sd in r.get('sub_details', []):
        st = sd['status']
        if st == 'rebalanced':
            print(f'║    🔄 {sd["name"]}: 买入{sd["n_buy"]}只 卖出{sd["n_sell"]}只')
        elif st == 'hold':
            print(f'║    ✅ {sd["name"]}: 持仓{sd["n_hold"]}只 (未到期)')
        elif st == 'liquidated':
            print(f'║    ❌ {sd["name"]}: 清仓{sd.get("n_sell",0)}只 (退出分配)')
        else:
            print(f'║    ⚠️ {sd["name"]}: 跳过')
    print(f'╠{"═"*58}╣')
    if r.get('sell_orders'):
        print(f'║  🔴 卖出 ({len(r["sell_orders"])}只):')
        for o in r['sell_orders']:
            n = r['names'].get(o['symbol'], '')
            print(f'║    {o["symbol"]} {n:10s} {o["shares"]}股')
    if r['orders']:
        print(f'║  🟢 买入 ({len(r["orders"])}只):')
        print(f'║    {"代码":<8} {"名称":<10} {"股数":<6} {"价格":<7} {"金额":<8}')
        for o in r['orders']:
            n = r['names'].get(o['symbol'], '')
            print(f'║    {o["symbol"]:<8} {n:<10} {o["shares"]:<6} {o["price"]:<7.2f} {o["cost"]:<8,.0f}')
    print(f'╠{"═"*58}╣')
    print(f'║  新占用: {r["total_cost"]:>8,.0f}  总占用: {r["total_capital_used"]:>8,.0f}  剩余: {r["remain"]:>8,.0f}')
    print(f'╚{"═"*58}╝')


def _send_email(signal_path: str):
    """发送信号邮件。"""
    try:
        from live.notification import Mailer
        mailer = Mailer()
        mailer.send_signal_from_file(signal_path)
    except Exception as e:
        logger.error("邮件发送失败: %s", e)


def _sync_hold_snapshot(today: str, used_capital: float, remain: float):
    """非调仓日：将上日的持仓快照同步到今日，用最新市价更新总资产。"""
    try:
        from core.database import Database
        live_db = Database(LIVE_DB_PATH)
        quant_db = Database("./data/quant_data.db")

        # 查上次快照
        last = live_db.conn.execute("""
            SELECT date, total_value, cash, positions, orders
            FROM daily_snapshots ORDER BY date DESC LIMIT 1
        """).fetchone()
        if not last:
            logger.warning("无上次快照，跳过同步")
            live_db.close()
            quant_db.close()
            return

        last_date = last[0]
        positions = json.loads(last[3]) if last[3] else {}
        cash = float(last[2]) if last[2] else 0

        # 用最新收盘价算持仓市值
        position_value = 0.0
        if positions:
            symbols = list(positions.keys())
            ph = ",".join("?" for _ in symbols)
            rows = quant_db.conn.execute(
                f"SELECT symbol, close FROM daily_bars WHERE date=? AND symbol IN ({ph})",
                [today] + symbols
            ).fetchall()
            prices = {r[0]: float(r[1]) for r in rows}
            # 补漏：取最近价
            for sym in symbols:
                if sym not in prices:
                    r = quant_db.conn.execute(
                        "SELECT close FROM daily_bars WHERE symbol=? AND date<=? ORDER BY date DESC LIMIT 1",
                        [sym, today]
                    ).fetchone()
                    if r:
                        prices[sym] = float(r[0])
            for sym, shares in positions.items():
                position_value += prices.get(sym, 0) * shares

        total_value = position_value + cash

        # 写入新快照
        live_db.conn.execute("""
            INSERT INTO daily_snapshots (date, strategy, total_value, cash, positions, orders)
            VALUES (?, 'mss_dynamic', ?, ?, ?, '[]')
            ON CONFLICT (date, strategy) DO UPDATE SET
                total_value=EXCLUDED.total_value, cash=EXCLUDED.cash,
                positions=EXCLUDED.positions, orders=EXCLUDED.orders
        """, [today, round(total_value, 2), round(cash, 2),
              json.dumps(positions)])

        # 更新 daily_performance
        prev_perf = live_db.conn.execute(
            "SELECT cumulative FROM daily_performance ORDER BY date DESC LIMIT 1"
        ).fetchone()
        prev_cum = float(prev_perf[0]) if prev_perf else 1.0
        prev_total = float(last[1]) if last[1] else total_value
        daily_ret = (total_value - prev_total) / prev_total if prev_total > 0 else 0.0
        cumulative = prev_cum * (1 + daily_ret)
        # 算回撤
        all_cum = [r[0] for r in live_db.conn.execute(
            "SELECT cumulative FROM daily_performance ORDER BY date").fetchall()]
        all_cum.append(cumulative)
        peak = max(all_cum) if all_cum else cumulative
        max_dd = (cumulative - peak) / peak if peak > 0 else 0.0

        # 基准收益
        bench_ret = 0.0
        br = quant_db.conn.execute(
            "SELECT pct_change FROM daily_bars WHERE symbol='000300' AND date=?",
            [today]
        ).fetchone()
        if br:
            bench_ret = float(br[0]) / 100.0

        live_db.conn.execute("""
            INSERT INTO daily_performance (date, total_value, daily_return, cumulative,
                max_drawdown, positions_count, turnover, benchmark_ret)
            VALUES (?, ?, ?, ?, ?, ?, 0, ?)
            ON CONFLICT (date) DO UPDATE SET
                total_value=EXCLUDED.total_value, daily_return=EXCLUDED.daily_return,
                cumulative=EXCLUDED.cumulative, max_drawdown=EXCLUDED.max_drawdown,
                positions_count=EXCLUDED.positions_count, benchmark_ret=EXCLUDED.benchmark_ret
        """, [today, round(total_value, 2), round(daily_ret, 6),
              round(cumulative, 6), round(max_dd, 6),
              len(positions), round(bench_ret, 6)])

        live_db.conn.commit()
        logger.info(f"快照同步: {today} 持仓{len(positions)}只 总资产{total_value:,.0f} "
                    f"日收益{daily_ret:+.4f} 累计{cumulative:.4f}")
        live_db.close()
        quant_db.close()
    except Exception as e:
        logger.warning("快照同步失败(非关键): %s", e)


if __name__ == '__main__':
    main()
