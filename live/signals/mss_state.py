"""mss_state — split from mss_dynamic.py."""
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

SIGNAL_DIR = 'data_live/mss_dynamic'
LIVE_DB_PATH = "./data_live/live_data.db"
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

# V7 实盘配置 (2026-06-02 上线)
#   bull: mf_d10_rp 60% + mf_vol_d10_rp 20% + mf50_chip50 15% + osr_d10 5%
#   bear: c01_layered_d5 50% + chip_equal_d3 25% + mf_vol_d10_rp 25%
#   oscillate: mf_d10_rp 40% + mf50_chip50 30% + c01_layered_d5 30%
#   recovery: c01_layered_d5 40% + osr_d10 30% + mf_vol_d10_rp 30%
V6A_ALLOCATION = {
    "bull": [("mf_d10_rp", 0.6), ("mf_vol_d10_rp", 0.2), ("mf50_chip50", 0.15), ("osr_d10", 0.05)],
    "bear": [("c01_layered_d5", 0.5), ("chip_equal_d3", 0.25), ("mf_vol_d10_rp", 0.25)],
    "oscillate": [("mf_d10_rp", 0.4), ("mf50_chip50", 0.3), ("c01_layered_d5", 0.3)],
    "recovery": [("c01_layered_d5", 0.4), ("osr_d10", 0.3), ("mf_vol_d10_rp", 0.3)],
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

# V7.1 状态切换冷却期: 新状态需连续出现 N 个交易日才切换
COOLDOWN_DAYS = 5



from core.database import Database



def _signal_dir(date_key: str) -> str:
    """返回该日期的信号目录，如 data_live/mss_dynamic/20260519/"""
    d = os.path.join(SIGNAL_DIR, date_key)
    os.makedirs(d, exist_ok=True)
    return d


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
        except OSError:
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


def _get_pending_state(live_db) -> tuple:
    """读取 pending 状态。返回 (pending_state, consecutive_days) 或 (None, 0)。"""
    state_r = live_db.conn.execute("SELECT value FROM mss_meta WHERE key='pending_state'").fetchone()
    days_r = live_db.conn.execute("SELECT value FROM mss_meta WHERE key='pending_days'").fetchone()
    if state_r and days_r:
        return state_r[0], int(days_r[0])
    return None, 0


def _set_pending_state(live_db, state: str, days: int):
    live_db.conn.execute("""
        INSERT INTO mss_meta (key, value) VALUES ('pending_state', ?)
        ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value
    """, [state])
    live_db.conn.execute("""
        INSERT INTO mss_meta (key, value) VALUES ('pending_days', ?)
        ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value
    """, [str(days)])


def _clear_pending_state(live_db):
    live_db.conn.execute("DELETE FROM mss_meta WHERE key IN ('pending_state', 'pending_days')")


def _trading_days_between(qconn, d1: str, d2: str) -> int:
    start, end = (d1, d2) if d1 <= d2 else (d2, d1)
    rows = qconn.execute(
        "SELECT COUNT(DISTINCT date) FROM daily_bars WHERE date > ? AND date <= ?",
        [start, end]
    ).fetchone()
    return rows[0] if rows else 0



def _qdb():
    return duckdb.connect("./data/quant_data.db", read_only=True)
