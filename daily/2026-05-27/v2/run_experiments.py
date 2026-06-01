"""mss_dynamic V2 全面优化实验框架

基于 V2b (增强ST+差异化止损) 和 V2c (V2b+置信度联动) 两个 baseline，
进行 10+ 项改进的单点消融、器级参数调优、组合优化。
借鉴 autoresearch 方法论：固定评估指标(Calmar)、keep/discard 机制、简洁性偏好。

实验阶段:
  Phase 0: V2b/V2c baseline 建立
  Phase 1: 10 项改进 × 2 baseline 的单点消融
  Phase 2: 器级参数调优 (分配权重/止损/选股数/调仓频率/置信度系数)
  Phase 3: 贪婪组合优化 (Calmar-driven keep/discard)
  Phase 4: 报告生成 + OOS 验证

用法:
    cd /Users/wangzeshang1/MyProjects/zequant
    python3 daily/2026-05-27/v2/run_experiments.py [--mode all|single|param|combo]
"""
from __future__ import annotations
import argparse
import copy
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import duckdb
from core.positioners import RPPortfolioWeights

# ════════════════════════════════════════════════════════════════════════════
# 全局配置
# ════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
LOG_PATH = os.path.join(SCRIPT_DIR, "experiment.log")
os.makedirs(RESULTS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("mss_v2_exp")

TX = 0.0012
DB_PATH = os.path.abspath("./data/quant_data.db")
GA_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..",
    "core", "strategies", "impl", "v1_ga_rp", "config.json",
)

FULL_RANGE = ("2019-01-02", "2026-05-22")
WINDOWS = [
    ("全区间",     "2019-01-02", "2026-05-22"),
    ("2022熊市",   "2022-01-04", "2022-12-30"),
    ("OOS修复牛",  "2024-07-01", "2026-05-22"),
]

FACTORS = list(set([
    'a27', 'a30', 'a31', 'a41', 'a42', 'a64', 'a69', 'a8', 'a80', 'a85',
    'a88', 'a91', 'a97', 'a98', 'a99', 'ff_mkt', 'gtja103', 'gtja104', 'gtja105',
    'gtja108', 'gtja113', 'gtja117', 'gtja12', 'gtja120', 'gtja121', 'gtja123',
    'gtja127', 'gtja13', 'gtja139', 'gtja141', 'gtja142', 'gtja144', 'gtja148',
    'gtja164', 'gtja168', 'gtja171', 'gtja176', 'gtja185', 'gtja34', 'gtja49',
    'gtja62', 'gtja76', 'gtja83', 'gtja85', 'gtja90', 'gtja91', 'gtja99',
    'returns', 'rsi_14', 'volatility_20', 'macd', 'macd_signal', 'momentum_5',
    'momentum_20', 'volume_ratio', 'boll_position', 'beta_20',
]))
NEW_FACTORS = [
    'ma5', 'ma10', 'ma20', 'ma21', 'ma60', 'ma120', 'ma_alignment_score',
    'ma60_trend', 'ma120_trend', 'macd_above_zero', 'macd_golden_cross',
    'volume_breakout_ratio', 'volume_contraction', 'chip_concentration', 'ma_angle_20',
]
ALL_FACTORS = list(set(FACTORS + NEW_FACTORS))

DEFAULT_STATE_STRATEGIES = {
    "bull": [
        {"strategy": "mf_d10_rp",     "weight": 0.6},
        {"strategy": "mf_vol_d10_rp", "weight": 0.2},
        {"strategy": "chip_covrp",    "weight": 0.2},
    ],
    "bear": [
        {"strategy": "chip_covrp",    "weight": 0.6},
        {"strategy": "mf_vol_d10_rp", "weight": 0.2},
        {"strategy": "chip_rp",       "weight": 0.2},
    ],
    "oscillate": [
        {"strategy": "chip_covrp",    "weight": 0.4},
        {"strategy": "mf50_chip50",   "weight": 0.3},
        {"strategy": "c01_layered_d5","weight": 0.3},
    ],
    "recovery": [
        {"strategy": "mf60_chip40",   "weight": 0.4},
        {"strategy": "chip_rp",       "weight": 0.3},
        {"strategy": "mf_vol_d10_rp", "weight": 0.3},
    ],
}

STOP_LOSS_V2B = {
    "mf_d10_rp": 0.08,
    "mf_vol_d10_rp": 0.08,
    "chip_rp": 0.10,
    "chip_covrp": 0.10,
    "c01_layered_d5": 0.08,
    "osr_d10": 0.08,
    "mf_base": 0.08,
}

# 子策略参数
DEFAULT_SUB_PARAMS = {
    "mf_d10_rp":       {"signal": "mf",     "rf": 10, "tn": 50, "mhd": 10, "timing": None},
    "mf_vol_d10_rp":   {"signal": "mf",     "rf": 10, "tn": 50, "mhd": 10, "timing": "vol"},
    "chip_rp":         {"signal": "chip",   "rf": 3,  "tn": 40, "mhd": 5,  "timing": None},
    "chip_covrp":      {"signal": "chip",   "rf": 3,  "tn": 40, "mhd": 5,  "timing": None},
    "osr_d10":         {"signal": "osr",    "rf": 10, "tn": 40, "mhd": 5,  "timing": None},
    "c01_layered_d5":  {"signal": "mf",     "rf": 5,  "tn": 40, "mhd": 5,  "timing": "trend"},
    "mf_base":         {"signal": "mf",     "rf": 3,  "tn": 40, "mhd": 5,  "timing": None},
}

# ════════════════════════════════════════════════════════════════════════════
# 数据加载 (与 mss_ab_comparison.py 保持一致)
# ════════════════════════════════════════════════════════════════════════════

def _get_conn():
    return duckdb.connect(DB_PATH, read_only=True)


def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                          List[str], List[str], int, int, List[pd.Timestamp],
                          Dict[str, int], Dict[str, np.ndarray]]:
    logger.info("=" * 60)
    logger.info("Phase 0: 加载数据")
    logger.info("=" * 60)
    t0 = time.time()
    conn = _get_conn()
    all_cols = [r[0] for r in conn.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name='factors_wide'"
    ).fetchall()]
    available = [c for c in ALL_FACTORS if c in all_cols]

    factor_cols = ", ".join([f'f."{c}"' for c in available])
    df = conn.execute(f"""
        SELECT f.date, f.symbol, b.close, b.pct_change, b.volume, {factor_cols}
        FROM factors_wide f
        LEFT JOIN daily_bars b ON f.date = b.date AND f.symbol = b.symbol
        WHERE f.date >= '2018-01-01' AND f.date <= '2026-05-22'
        ORDER BY f.date, f.symbol
    """).fetchdf()
    df['date'] = pd.to_datetime(df['date'])
    ds = sorted(df['date'].unique())
    tks = [r[0] for r in conn.execute("SELECT symbol FROM symbols ORDER BY symbol").fetchall()]
    nd, ns, nf = len(ds), len(tks), len(available)
    t2i = {t: i for i, t in enumerate(tks)}
    d2i = {d: i for i, d in enumerate(ds)}

    v3 = np.full((nd, ns, nf), np.nan, dtype=np.float32)
    dm = np.zeros((nd, ns), dtype=bool)
    cl = np.zeros((nd, ns), dtype=np.float32)
    pct = np.zeros((nd, ns), dtype=np.float32)
    vol = np.zeros((nd, ns), dtype=np.float32)

    di = np.array([d2i[d] for d in df['date']], dtype=np.int32)
    si = np.array([t2i.get(s, -1) for s in df['symbol']], dtype=np.int32)
    v = si >= 0
    di, si = di[v], si[v]

    for fi, fc in enumerate(available):
        if fc in df.columns:
            v3[di, si, fi] = df[fc].values[v].astype(np.float32)
    cl[di, si] = df['close'].values[v].astype(np.float32)
    if 'pct_change' in df.columns:
        pct[di, si] = df['pct_change'].values[v].astype(np.float32)
    if 'volume' in df.columns:
        vol[di, si] = df['volume'].values[v].astype(np.float32)
    dm[di, si] = True
    np.nan_to_num(v3, nan=0.0, copy=False)
    np.nan_to_num(cl, nan=0.0, copy=False)
    np.nan_to_num(pct, nan=0.0, copy=False)
    np.nan_to_num(vol, nan=0.0, copy=False)

    fwd = np.zeros((nd, ns), dtype=np.float32)
    for d in range(nd - 1):
        b = (cl[d] > 1e-10) & (cl[d + 1] > 1e-10)
        fwd[d, b] = (cl[d + 1, b] - cl[d, b]) / cl[d, b]

    z3 = np.zeros_like(v3)
    for fi in range(nf):
        a = v3[:, :, fi]
        for d in range(nd):
            r = a[d, :]
            nz = r[r != 0]
            if len(nz) > 1:
                lo, hi = np.quantile(nz, [0.01, 0.99])
                c = np.clip(r, lo, hi)
                mu, sd = np.mean(c), np.std(c)
                z3[d, :, fi] = (c - mu) / sd if sd > 1e-10 else 0.0

    per_symbol_info = {"pct": pct, "cl": cl, "vol": vol}
    logger.info(f"数据加载完成: {nd}天 × {ns}只 × {nf}因子 ({time.time()-t0:.1f}s)")
    conn.close()
    return z3, fwd, dm, cl, tks, available, nd, ns, ds, t2i, per_symbol_info


def load_ga_weights() -> Dict[str, float]:
    if os.path.exists(GA_CONFIG_PATH):
        with open(GA_CONFIG_PATH) as f:
            cfg = json.load(f)
        return cfg.get("selector", {}).get("weights", {})
    return {}


# ════════════════════════════════════════════════════════════════════════════
# 信号构建
# ════════════════════════════════════════════════════════════════════════════

def build_signals(z3, fwd, dm, cl, fnames, nd, ns, ds):
    fi = {fn: i for i, fn in enumerate(fnames)}

    mf_weights = load_ga_weights()
    if mf_weights:
        wv = np.zeros(len(fnames), dtype=np.float32)
        for fi_i, fc in enumerate(fnames):
            if fc in mf_weights:
                wv[fi_i] = float(mf_weights[fc])
        s = np.sum(np.abs(wv))
        if s > 0:
            wv /= s
        mf = np.nan_to_num(np.tensordot(z3, wv, axes=(2, 0)), nan=-1e10, neginf=-1e10)
    else:
        mf = np.nan_to_num(np.mean(z3, axis=2), nan=-1e10, neginf=-1e10)

    vol20_idx = fi.get('volatility_20')
    m20_idx = fi.get('momentum_20')
    chip_sig = np.full((nd, ns), -np.inf, dtype=np.float32)
    for d in range(nd):
        s_chip = np.zeros(ns)
        if vol20_idx is not None:
            s_chip += np.where(z3[d, :, vol20_idx] < -0.3, 1.0, 0.0) * 0.5
        if m20_idx is not None:
            s_chip += np.where(np.abs(z3[d, :, m20_idx]) < 0.3, 1.0, 0.0) * 0.3
        chip_sig[d] = np.nan_to_num(s_chip, nan=-1e10)

    rsi_idx = fi.get('rsi_14')
    m5_idx = fi.get('momentum_5')
    ret_idx = fi.get('returns')
    osr_sig = np.full((nd, ns), -np.inf, dtype=np.float32)
    for d in range(nd):
        s_osr = np.zeros(ns)
        if rsi_idx is not None:
            s_osr += np.where(z3[d, :, rsi_idx] < -0.5, 1.0, 0.0) * -0.5
        if m5_idx is not None:
            s_osr += np.where(z3[d, :, m5_idx] > 0.3, 1.0, 0.0) * 0.5
        if ret_idx is not None:
            s_osr += np.where(z3[d, :, ret_idx] < -0.5, 1.0, 0.0) * 0.3
        osr_sig[d] = np.nan_to_num(s_osr, nan=-1e10)

    vol_p = np.ones(nd, dtype=np.float32)
    if vol20_idx is not None:
        vol_p = np.clip(1.0 - np.mean(z3[:, :, vol20_idx] > 0.05, axis=1), 0.2, 1.0)

    im = fi.get('macd')
    ims = fi.get('macd_signal')
    im5 = fi.get('momentum_5')
    im20 = fi.get('momentum_20')
    ir = fi.get('rsi_14')
    trend_p = np.full(nd, 0.5, dtype=np.float32)
    for d in range(nd):
        sl = []
        if im is not None and ims is not None:
            sl.append(np.where(z3[d, :, im] > z3[d, :, ims], 1.0, 0.0))
        if im5 is not None and im20 is not None:
            m5v = z3[d, :, im5]
            m20v = z3[d, :, im20]
            sl.append(np.where((m5v > 0) & (m5v > m20v), 1.0,
                               np.where(m5v < 0, 0.0, 0.5)))
        if ir is not None:
            rv = z3[d, :, ir]
            sl.append(np.where(rv > 70, 0.0,
                               np.where(rv >= 50, 1.0,
                                        np.where(rv >= 30, 0.5, 0.0))))
        if sl:
            trend_p[d] = np.clip(np.mean(np.mean(sl, axis=0) >= 0.6) * 2.0, 0.1, 1.0)

    mkt_idx = np.zeros(nd, dtype=np.float64)
    for d in range(1, nd):
        active = dm[d] & (cl[d] > 1e-10)
        if np.any(active):
            mkt_idx[d] = np.mean(fwd[d - 1, active])
            mkt_idx[d] = 0.0 if np.isnan(mkt_idx[d]) or np.isinf(mkt_idx[d]) else mkt_idx[d]

    return {
        "mf": mf, "chip": chip_sig, "osr": osr_sig,
        "vol_p": vol_p, "trend_p": trend_p,
        "fi": fi, "market_index": mkt_idx, "close": cl,
    }


# ════════════════════════════════════════════════════════════════════════════
# 增强ST过滤 (V2b/V2c 自带)
# ════════════════════════════════════════════════════════════════════════════

def build_enhanced_st_mask(per_symbol_info, t2i, nd, ns):
    risk_map = {}
    pct = per_symbol_info["pct"]
    cl = per_symbol_info["cl"]

    flagged = set()
    for sym, idx in t2i.items():
        found = False
        for d in range(5, nd):
            if pct[d, idx] < -9.5 and pct[d - 1, idx] < -9.5:
                flagged.add(idx)
                found = True
                break
        if found:
            continue
        for d in range(25, nd):
            recent_p = pct[d - 4:d + 1, idx]
            recent_c = cl[d - 4:d + 1, idx]
            if np.all(recent_c > 0) and np.mean(recent_c) < 3.0 and np.mean(recent_p) < -2.0:
                flagged.add(idx)
                break

    for idx in flagged:
        risk_map[idx] = 'high'
    logger.info(f"增强ST: {len(flagged)} 只标记为高风险")
    return risk_map


# ════════════════════════════════════════════════════════════════════════════
# 市场状态检测
# ════════════════════════════════════════════════════════════════════════════

def detect_market_state(mkt_returns, nd):
    idx_price = np.zeros(nd, dtype=np.float64)
    idx_price[0] = 1000.0
    for i in range(1, nd):
        idx_price[i] = idx_price[i - 1] * (1.0 + mkt_returns[i])

    ma5 = pd.Series(idx_price).rolling(5).mean().values
    ma20 = pd.Series(idx_price).rolling(20).mean().values
    ma60 = pd.Series(idx_price).rolling(60).mean().values
    ma200 = pd.Series(idx_price).rolling(200).mean().values

    states = ["oscillate"] * nd
    confidence = np.zeros(nd, dtype=np.float32)

    for i in range(nd):
        if pd.isna(ma200[i]) or ma200[i] == 0:
            states[i] = "oscillate"
            continue

        close = idx_price[i]
        above_ma200 = (close - ma200[i]) / ma200[i]

        lb5 = min(5, i)
        ma5_slope = (ma5[i] - ma5[i - lb5]) / ma5[i - lb5] if (lb5 >= 2 and ma5[i - lb5] != 0) else 0.0
        lb20 = min(20, i)
        ma20_slope = (ma20[i] - ma20[i - lb20]) / ma20[i - lb20] if (lb20 >= 2 and ma20[i - lb20] != 0) else 0.0
        lb60 = min(60, i)
        ma60_slope = (ma60[i] - ma60[i - lb60]) / ma60[i - lb60] if (lb60 >= 2 and ma60[i - lb60] != 0) else 0.0

        bull = above_ma200 > 0 and ma20_slope > 0
        bear = above_ma200 < 0 and ma20_slope < 0 and ma60_slope < 0
        recovery = above_ma200 < 0 and ma5_slope > 0.005

        sp = 0.1
        if pd.notna(ma5[i]) and pd.notna(ma20[i]) and pd.notna(ma60[i]):
            sp = abs(ma5[i] - ma20[i]) / max(abs(ma20[i]), 1e-10) + \
                 abs(ma20[i] - ma60[i]) / max(abs(ma60[i]), 1e-10)
        oscillate = sp < 0.03

        if bull:
            states[i] = "bull"
            confidence[i] = min(1.0, above_ma200 * 2 + ma20_slope * 20)
        elif bear:
            states[i] = "bear"
            confidence[i] = min(1.0, abs(above_ma200) * 2 + abs(ma20_slope) * 10 + abs(ma60_slope) * 10)
        elif recovery:
            states[i] = "recovery"
            confidence[i] = min(1.0, ma5_slope * 50)
        elif oscillate:
            states[i] = "oscillate"
            confidence[i] = max(0.3, 1.0 - sp * 15)
        elif above_ma200 < 0 and ma5_slope > 0:
            states[i] = "recovery"
            confidence[i] = max(0.3, ma5_slope * 30)
        else:
            states[i] = "oscillate"
            confidence[i] = 0.3

    return states, confidence


def detect_market_state_hysteresis(mkt_returns, nd, hysteresis_days=3, cooldown_days=5):
    idx_price = np.zeros(nd, dtype=np.float64)
    idx_price[0] = 1000.0
    for i in range(1, nd):
        idx_price[i] = idx_price[i - 1] * (1.0 + mkt_returns[i])

    ma5 = pd.Series(idx_price).rolling(5).mean().values
    ma20 = pd.Series(idx_price).rolling(20).mean().values
    ma60 = pd.Series(idx_price).rolling(60).mean().values
    ma200 = pd.Series(idx_price).rolling(200).mean().values

    raw_states = ["oscillate"] * nd
    raw_conf = np.zeros(nd, dtype=np.float32)

    for i in range(nd):
        if pd.isna(ma200[i]) or ma200[i] == 0:
            continue
        close = idx_price[i]
        above_ma200 = (close - ma200[i]) / ma200[i]
        lb5 = min(5, i)
        ma5_slope = (ma5[i] - ma5[i - lb5]) / ma5[i - lb5] if (lb5 >= 2 and ma5[i - lb5] != 0) else 0.0
        lb20 = min(20, i)
        ma20_slope = (ma20[i] - ma20[i - lb20]) / ma20[i - lb20] if (lb20 >= 2 and ma20[i - lb20] != 0) else 0.0
        lb60 = min(60, i)
        ma60_slope = (ma60[i] - ma60[i - lb60]) / ma60[i - lb60] if (lb60 >= 2 and ma60[i - lb60] != 0) else 0.0

        bull = above_ma200 > 0 and ma20_slope > 0
        bear = above_ma200 < 0 and ma20_slope < 0 and ma60_slope < 0
        recovery = above_ma200 < 0 and ma5_slope > 0.005
        sp = 0.1
        if pd.notna(ma5[i]) and pd.notna(ma20[i]) and pd.notna(ma60[i]):
            sp = abs(ma5[i] - ma20[i]) / max(abs(ma20[i]), 1e-10) + \
                 abs(ma20[i] - ma60[i]) / max(abs(ma60[i]), 1e-10)
        oscillate = sp < 0.03

        if bull:
            raw_states[i] = "bull"
            raw_conf[i] = min(1.0, above_ma200 * 2 + ma20_slope * 20)
        elif bear:
            raw_states[i] = "bear"
            raw_conf[i] = min(1.0, abs(above_ma200) * 2 + abs(ma20_slope) * 10 + abs(ma60_slope) * 10)
        elif recovery:
            raw_states[i] = "recovery"
            raw_conf[i] = min(1.0, ma5_slope * 50)
        elif oscillate:
            raw_states[i] = "oscillate"
            raw_conf[i] = max(0.3, 1.0 - sp * 15)
        elif above_ma200 < 0 and ma5_slope > 0:
            raw_states[i] = "recovery"
            raw_conf[i] = max(0.3, ma5_slope * 30)

    filtered = ["oscillate"] * nd
    filtered_conf = np.zeros(nd, dtype=np.float32)
    filtered[0] = raw_states[0]
    filtered_conf[0] = raw_conf[0]
    last_switch = 0
    current_streak = 1

    for i in range(1, nd):
        if raw_states[i] == filtered[i - 1]:
            current_streak += 1
            filtered[i] = filtered[i - 1]
            filtered_conf[i] = raw_conf[i]
        else:
            current_streak = 1
            if i - last_switch < cooldown_days:
                filtered[i] = filtered[i - 1]
                filtered_conf[i] = filtered_conf[i - 1]
            else:
                confirm = True
                for k in range(1, min(hysteresis_days, nd - i)):
                    if i + k < nd and raw_states[i + k] != raw_states[i]:
                        confirm = False
                        break
                if confirm:
                    filtered[i] = raw_states[i]
                    filtered_conf[i] = raw_conf[i]
                    last_switch = i
                else:
                    filtered[i] = filtered[i - 1]
                    filtered_conf[i] = filtered_conf[i - 1]

    return filtered, filtered_conf


def detect_panic_state(mkt_returns, nd, single_day_thresh=-0.05, consecutive_thresh=-0.02, consecutive_days=3):
    panic = np.zeros(nd, dtype=bool)
    for i in range(nd):
        if mkt_returns[i] <= single_day_thresh:
            panic[i] = True
    for i in range(consecutive_days - 1, nd):
        if all(mkt_returns[i - k] <= consecutive_thresh for k in range(consecutive_days)):
            panic[i] = True
    return panic


def compute_market_breadth(pct, dm, nd):
    breadth = np.zeros(nd, dtype=np.float32)
    for i in range(nd):
        valid = dm[i] & (np.abs(pct[i]) < 100.0) & (pct[i] != 0)
        if np.any(valid):
            breadth[i] = np.mean(pct[i, valid] > 0)
    return breadth


def compute_weekly_slope(mkt_returns, nd):
    weekly = pd.Series(mkt_returns).rolling(5).sum().values
    slope = np.zeros(nd, dtype=np.float32)
    for i in range(4, nd):
        if i >= 23:
            x = np.arange(4)
            y = weekly[i - 3:i + 1]
            valid = ~np.isnan(y)
            if np.sum(valid) >= 3:
                slope[i] = np.polyfit(x[valid], y[valid], 1)[0]
    return slope


def build_industry_map(tks):
    industry = {}
    for s in tks:
        code = s.split('.')[0] if '.' in s else s
        if code.startswith('600') or code.startswith('601') or code.startswith('603') or code.startswith('605'):
            industry[s] = 'sh_main'
        elif code.startswith('000') or code.startswith('001') or code.startswith('002'):
            industry[s] = 'sz_main'
        elif code.startswith('300'):
            industry[s] = 'chinext'
        elif code.startswith('688'):
            industry[s] = 'star'
        else:
            industry[s] = 'other'
    return industry


def apply_sector_diversity(pw, t2i, tks, max_per_sector=3):
    industry = build_industry_map(tks)
    idx_to_ind = {}
    for tk, idx in t2i.items():
        idx_to_ind[idx] = industry.get(tk, 'other')
    order = np.argsort(-pw)
    sector_counts = {}
    result = np.zeros_like(pw)
    for idx in order:
        if pw[idx] <= 0:
            continue
        sec = idx_to_ind.get(idx, 'other')
        cnt = sector_counts.get(sec, 0)
        if cnt < max_per_sector:
            result[idx] = pw[idx]
            sector_counts[sec] = cnt + 1
    s = np.sum(result)
    if s > 0:
        result /= s
    return result


def compute_strategy_track_record(sub_drs, nd, lookback=60):
    eq = {}
    for name, dr in sub_drs.items():
        eq[name] = np.ones(nd, dtype=np.float64)
        for i in range(1, nd):
            eq[name][i] = eq[name][i - 1] * (1.0 + dr[i])
    return eq


def hmm_detect_states(mkt_returns, nd, n_states=2, max_iter=20):
    if nd < 100:
        return np.zeros(nd, dtype=np.int32)

    rets = np.array(mkt_returns[200:], dtype=np.float64)
    T = len(rets)

    pos = rets > 0
    neg = ~pos
    if np.sum(pos) < 10 or np.sum(neg) < 10:
        return np.zeros(nd, dtype=np.int32), np.zeros(nd, dtype=np.float32)

    mus = np.array([np.mean(rets[pos]), np.mean(rets[neg])])
    sigs = np.array([np.std(rets[pos]) + 1e-6, np.std(rets[neg]) + 1e-6])
    pi = np.array([0.5, 0.5])
    A = np.array([[0.95, 0.05], [0.10, 0.90]])

    for _ in range(max_iter):
        alpha = np.zeros((T, n_states))
        scale = np.zeros(T)
        for s in range(n_states):
            alpha[0, s] = pi[s] * (1.0 / (np.sqrt(2 * np.pi) * sigs[s]) *
                                    np.exp(-0.5 * ((rets[0] - mus[s]) / sigs[s]) ** 2))
        scale[0] = np.sum(alpha[0, :])
        if scale[0] > 0:
            alpha[0, :] /= scale[0]

        for t in range(1, T):
            for s in range(n_states):
                prob = 1.0 / (np.sqrt(2 * np.pi) * sigs[s]) * \
                       np.exp(-0.5 * ((rets[t] - mus[s]) / sigs[s]) ** 2)
                alpha[t, s] = prob * np.sum(alpha[t - 1, :] * A[:, s])
            scale[t] = np.sum(alpha[t, :])
            if scale[t] > 0:
                alpha[t, :] /= scale[t]

        beta = np.zeros((T, n_states))
        beta[T - 1, :] = 1.0
        for t in range(T - 2, -1, -1):
            for s in range(n_states):
                for sp in range(n_states):
                    prob = 1.0 / (np.sqrt(2 * np.pi) * sigs[sp]) * \
                           np.exp(-0.5 * ((rets[t + 1] - mus[sp]) / sigs[sp]) ** 2)
                    beta[t, s] += A[s, sp] * prob * beta[t + 1, sp]
            if np.sum(beta[t, :]) > 0:
                beta[t, :] /= np.sum(beta[t, :])

        gamma = alpha * beta
        gamma /= np.sum(gamma, axis=1, keepdims=True) + 1e-10

        xi = np.zeros((T - 1, n_states, n_states))
        for t in range(T - 1):
            for s in range(n_states):
                for sp in range(n_states):
                    prob = 1.0 / (np.sqrt(2 * np.pi) * sigs[sp]) * \
                           np.exp(-0.5 * ((rets[t + 1] - mus[sp]) / sigs[sp]) ** 2)
                    xi[t, s, sp] = alpha[t, s] * A[s, sp] * prob * beta[t + 1, sp]
            s_xi = np.sum(xi[t, :, :])
            if s_xi > 0:
                xi[t, :, :] /= s_xi

        pi_new = gamma[0, :]
        A_new = np.zeros_like(A)
        for s in range(n_states):
            denom = np.sum(gamma[:-1, s])
            if denom > 0:
                for sp in range(n_states):
                    A_new[s, sp] = np.sum(xi[:, s, sp]) / denom
        mus_new = np.zeros(n_states)
        sigs_new = np.zeros(n_states)
        for s in range(n_states):
            w = gamma[:, s]
            sw = np.sum(w)
            if sw > 0:
                mus_new[s] = np.sum(w * rets) / sw
                sigs_new[s] = np.sqrt(np.sum(w * (rets - mus_new[s]) ** 2) / sw) + 1e-6

        if (np.max(np.abs(mus - mus_new)) < 1e-4 and
            np.max(np.abs(sigs - sigs_new)) < 1e-4):
            break
        mus, sigs, pi, A = mus_new, sigs_new, pi_new, A_new

    viterbi = np.zeros((T, n_states))
    back = np.zeros((T, n_states), dtype=np.int32)
    for s in range(n_states):
        viterbi[0, s] = np.log(pi[s] + 1e-10) - 0.5 * np.log(2 * np.pi * sigs[s] ** 2) - \
                        0.5 * ((rets[0] - mus[s]) / sigs[s]) ** 2
    for t in range(1, T):
        for s in range(n_states):
            prob = -0.5 * np.log(2 * np.pi * sigs[s] ** 2) - \
                   0.5 * ((rets[t] - mus[s]) / sigs[s]) ** 2
            trans = viterbi[t - 1, :] + np.log(A[:, s] + 1e-10)
            back[t, s] = np.argmax(trans)
            viterbi[t, s] = prob + trans[back[t, s]]

    hmm_full = np.zeros(T, dtype=np.int32)
    hmm_full[T - 1] = np.argmax(viterbi[T - 1, :])
    for t in range(T - 2, -1, -1):
        hmm_full[t] = back[t + 1, hmm_full[t + 1]]

    hmm_states = np.zeros(nd, dtype=np.int32)
    hmm_conf = np.zeros(nd, dtype=np.float32)
    conf_values = np.max(gamma, axis=1)
    hmm_states[200:200 + T] = hmm_full
    hmm_conf[200:200 + T] = conf_values.astype(np.float32)

    return hmm_states, hmm_conf


# ════════════════════════════════════════════════════════════════════════════
# 回测引擎
# ════════════════════════════════════════════════════════════════════════════

def bt_sub_strategy_base(sig, fwd, dm, rebal_freq=10, top_n=50, min_hold_days=10,
                          pos_ratio=None):
    nd, ns = sig.shape
    alloc = RPPortfolioWeights(top_n=top_n, min_hold_days=min_hold_days)
    pw = np.zeros(ns, dtype=np.float32)
    hs = np.full(ns, -1, dtype=np.int32)
    rh = 0
    dr = np.zeros(nd, dtype=np.float64)

    for i in range(1, nd):
        rebal = (i % rebal_freq == 0)
        if rebal:
            nw = alloc.allocate(sig[i], fwd, i, pw, hs, rh)
            pw = nw
            for j in range(ns):
                if nw[j] > 0 and hs[j] < 0:
                    hs[j] = rh + 1
        else:
            mk = dm[i] & (pw > 0)
            if np.any(mk):
                p2 = pw[mk].copy() / float(np.sum(pw[mk]))
                pw = np.zeros(ns, dtype=np.float32)
                pw[mk] = p2
        pr = pos_ratio[i] if pos_ratio is not None else 1.0
        rt = pr * float(np.dot(pw, fwd[i]))
        dr[i] = 0.0 if (np.isnan(rt) or np.isinf(rt)) else rt
        rh += 1
    return dr


def bt_sub_strategy_v2(sig, fwd, dm, rebal_freq=10, top_n=50, min_hold_days=10,
                        pos_ratio=None, stop_loss_pct=0.08, symbol_risk_map=None,
                        use_trailing_stop=False, trailing_profit_pct=0.15):
    nd, ns = sig.shape
    alloc = RPPortfolioWeights(top_n=top_n, min_hold_days=min_hold_days)
    pw = np.zeros(ns, dtype=np.float32)
    hs = np.full(ns, -1, dtype=np.int32)
    rh = 0
    dr = np.zeros(nd, dtype=np.float64)
    entry_px = np.zeros(ns, dtype=np.float32)
    peak_px = np.zeros(ns, dtype=np.float32)
    nt_stop = 0
    nt_trail = 0

    for i in range(1, nd):
        # 固定止损
        if stop_loss_pct > 0 and np.any(pw > 0):
            for j in range(ns):
                if pw[j] > 0 and hs[j] >= 0 and entry_px[j] > 0:
                    if fwd[i, j] < -stop_loss_pct and fwd[i, j] > -0.95:
                        pw[j] = 0.0
                        hs[j] = -1
                        entry_px[j] = 0.0
                        peak_px[j] = 0.0
                        nt_stop += 1

        # Trailing stop
        if use_trailing_stop:
            for j in range(ns):
                if pw[j] > 0 and entry_px[j] > 0:
                    cur = entry_px[j] * (1.0 + fwd[i, j])
                    if cur > peak_px[j] or peak_px[j] <= 0:
                        peak_px[j] = cur
                    if peak_px[j] > 0 and cur < peak_px[j] * (1.0 - trailing_profit_pct):
                        pw[j] = 0.0
                        hs[j] = -1
                        entry_px[j] = 0.0
                        peak_px[j] = 0.0
                        nt_trail += 1

        rebal = (i % rebal_freq == 0)
        if rebal:
            masked = sig[i].copy()
            if symbol_risk_map:
                for j, level in symbol_risk_map.items():
                    if level == 'high':
                        masked[j] = -1e10
            nw = alloc.allocate(masked, fwd, i, pw, hs, rh)
            for j in range(ns):
                if nw[j] > 0 and pw[j] <= 0:
                    entry_px[j] = max(1.0, 1.0 + fwd[i, j])
                    peak_px[j] = entry_px[j]
                elif nw[j] > 0 and pw[j] > 0 and entry_px[j] <= 0:
                    entry_px[j] = max(1.0, 1.0 + fwd[i, j])
                    peak_px[j] = entry_px[j]
            pw = nw
            for j in range(ns):
                if nw[j] > 0 and hs[j] < 0:
                    hs[j] = rh + 1
        else:
            mk = dm[i] & (pw > 0)
            if np.any(mk):
                p2 = pw[mk].copy() / float(np.sum(pw[mk]))
                pw = np.zeros(ns, dtype=np.float32)
                pw[mk] = p2

        for j in range(ns):
            if pw[j] > 0 and entry_px[j] <= 0:
                entry_px[j] = max(1.0, 1.0 + fwd[i, j])

        pr = pos_ratio[i] if pos_ratio is not None else 1.0
        rt = pr * float(np.dot(pw, fwd[i]))
        dr[i] = 0.0 if (np.isnan(rt) or np.isinf(rt)) else rt
        rh += 1

    if nt_stop > 0 or nt_trail > 0:
        logger.info(f"    止损={nt_stop}次 移动止盈={nt_trail}次")
    return dr


def compute_sub_drs_with_config(signals, fwd, dm, nd,
                                 use_v2=True, symbol_risk_map=None,
                                 stop_loss_by_strategy=None,
                                 use_trailing_stop=False, trailing_profit_pct=0.15,
                                 sub_params=None):
    if sub_params is None:
        sub_params = DEFAULT_SUB_PARAMS
    sub_drs = {}
    for name, params in sub_params.items():
        sig = signals[params["signal"]]
        pr = None
        if params.get("timing") == "vol":
            pr = signals["vol_p"]
        elif params.get("timing") == "trend":
            pr = signals["trend_p"]

        if use_v2:
            sl = 0.0
            if stop_loss_by_strategy:
                sl = stop_loss_by_strategy.get(name, 0.08)
            dr = bt_sub_strategy_v2(
                sig, fwd, dm,
                rebal_freq=params["rf"],
                top_n=params["tn"],
                min_hold_days=params["mhd"],
                pos_ratio=pr,
                stop_loss_pct=sl,
                symbol_risk_map=symbol_risk_map,
                use_trailing_stop=use_trailing_stop,
                trailing_profit_pct=trailing_profit_pct,
            )
        else:
            dr = bt_sub_strategy_base(
                sig, fwd, dm,
                rebal_freq=params["rf"],
                top_n=params["tn"],
                min_hold_days=params["mhd"],
                pos_ratio=pr,
            )
        sub_drs[name] = dr
    return sub_drs


def run_mss(state_strategies, sub_drs, states, confidence, nd,
             use_confidence_weights=False, confidence_threshold=0.5,
             conf_adjust_mf=0.6, conf_adjust_c01=0.7,
             panic_states=None, panic_cash_ratio=0.8,
             breadth=None, breadth_bear_thresh=0.35,
             weekly_slope=None,
             hmm_states=None, hmm_conf=None,
             strategy_eq=None, elimination_lookback=60, elimination_threshold=-0.05,
             macro_overlay=False, macro_cash_ratio=0.5,
             sector_diversity=False, max_per_sector=3,
             t2i=None, tks=None):
    strat_names = sorted(set(
        a["strategy"] for allocs in state_strategies.values()
        for a in allocs if a["strategy"] in sub_drs
    ))
    eq = {n: np.ones(nd, dtype=np.float64) for n in strat_names}
    dr = np.zeros(nd, dtype=np.float64)
    eliminated = set()

    for i in range(1, nd):
        st = states[i] if i < len(states) else "oscillate"

        # Panic覆盖
        if panic_states is not None and i < len(panic_states) and panic_states[i]:
            st = "panic"

        allocs = state_strategies.get(st, state_strategies.get("oscillate", []))

        alloc_map = {}
        for a in allocs:
            if a["strategy"] in sub_drs and a["strategy"] not in eliminated:
                w = a["weight"]
                if use_confidence_weights and confidence[i] < confidence_threshold:
                    if a["strategy"] in ("mf_d10_rp",):
                        w *= conf_adjust_mf
                    elif a["strategy"] in ("c01_layered_d5",):
                        w *= conf_adjust_c01
                alloc_map[a["strategy"]] = max(w, 0.0)

        # HMM置信度调整
        if hmm_states is not None and hmm_conf is not None and i < len(hmm_states):
            if hmm_states[i] == 1:
                for name in alloc_map:
                    alloc_map[name] *= 0.6

        # 市场广度降级
        if breadth is not None and i < len(breadth) and breadth[i] < breadth_bear_thresh:
            st = "oscillate"
            if st in state_strategies:
                allocs2 = state_strategies[st]
                alloc_map = {}
                for a in allocs2:
                    if a["strategy"] in sub_drs and a["strategy"] not in eliminated:
                        alloc_map[a["strategy"]] = max(a["weight"], 0.0)

        # 多时间框架降级
        if weekly_slope is not None and i < len(weekly_slope) and weekly_slope[i] < -0.02:
            if st == "bull":
                st = "oscillate"
                if st in state_strategies:
                    allocs2 = state_strategies[st]
                    alloc_map = {}
                    for a in allocs2:
                        if a["strategy"] in sub_drs and a["strategy"] not in eliminated:
                            alloc_map[a["strategy"]] = max(a["weight"], 0.0)

        # 宏观因子叠加
        if macro_overlay and st not in ("bull",):
            for name in alloc_map:
                alloc_map[name] *= (1.0 - macro_cash_ratio)

        total_w = sum(alloc_map.values()) or 1.0
        for name in alloc_map:
            alloc_map[name] /= total_w

        # 更新子策略净值
        for name in strat_names:
            if name not in eliminated:
                eq[name][i] = eq[name][i - 1] * (1.0 + sub_drs[name][i])

        # 子策略淘汰
        if elimination_lookback > 0 and len(eliminated) < len(strat_names) - 1:
            lb = min(elimination_lookback, i)
            if lb >= 20:
                for name in strat_names:
                    if name not in eliminated:
                        ret = eq[name][i] / eq[name][i - lb] - 1.0
                        if ret < elimination_threshold:
                            eliminated.add(name)
                            logger.info(f"    淘汰子策略 {name} @ day {i} (ret={ret*100:.1f}%)")

        combined_ret = sum(alloc_map.get(n, 0.0) * sub_drs[n][i] for n in strat_names if n in sub_drs)
        dr[i] = combined_ret

        # 行业多样性
        if sector_diversity:
            pass

    return dr


# ════════════════════════════════════════════════════════════════════════════
# 指标计算
# ════════════════════════════════════════════════════════════════════════════

def compute_metrics(dr, name=""):
    nd = len(dr)
    eq = np.ones(nd, dtype=np.float64)
    for i in range(1, nd):
        eq[i] = eq[i - 1] * (1.0 + dr[i])

    ny = nd / 252.0
    ar = (float(eq[-1] / eq[0])) ** (1.0 / max(ny, 0.5)) - 1.0

    lr = np.log(eq[1:] / eq[:-1])
    sd_lr = np.std(lr)
    sp = float(np.mean(lr) / max(sd_lr, 1e-10) * np.sqrt(252))

    cm = np.maximum.accumulate(eq)
    dd = (eq - cm) / cm
    mdd = float(np.min(dd))
    cal = ar / abs(mdd) if abs(mdd) > 0 else 0
    wr = int(np.sum(dr > 0)) / max(int(np.sum(dr > 0)) + int(np.sum(dr < 0)), 1)

    return {
        "name": name,
        "annual_return": round(float(ar), 4),
        "sharpe": round(float(sp), 4),
        "max_drawdown": round(float(mdd), 4),
        "calmar": round(float(cal), 4),
        "win_rate": round(float(wr), 4),
    }


def window_analysis(dr, ds, windows):
    results = []
    for wname, ws, we in windows:
        sdt, edt = pd.Timestamp(ws), pd.Timestamp(we)
        idx = [i for i, d in enumerate(ds) if sdt <= d <= edt]
        if not idx:
            results.append({"name": wname, "n_days": 0})
            continue
        sub = dr[idx[0]:idx[-1] + 1]
        m = compute_metrics(sub, name=wname)
        m["n_days"] = len(sub)
        results.append(m)
    return results


# ════════════════════════════════════════════════════════════════════════════
# ExperimentConfig
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class ExperimentConfig:
    name: str = "unnamed"
    baseline: str = "v2b"

    # --- 数据层 ---
    use_enhanced_st: bool = True
    use_stop_loss: bool = True
    stop_loss_by_strategy: Optional[Dict[str, float]] = None

    # --- V2c 置信度联动 ---
    use_confidence_weights: bool = False
    confidence_threshold: float = 0.5
    conf_adjust_mf: float = 0.6
    conf_adjust_c01: float = 0.7

    # --- 改进 #1: 滞后确认 ---
    use_hysteresis: bool = False
    hysteresis_days: int = 3
    cooldown_days: int = 5

    # --- 改进 #2: 移动止盈 ---
    use_trailing_stop: bool = False
    trailing_profit_pct: float = 0.15

    # --- 改进 #3: 置信度动态权重 ---
    use_dynamic_weights: bool = False

    # --- 改进 #4: Panic状态 ---
    use_panic_state: bool = False
    panic_single_day: float = -0.05
    panic_consecutive: float = -0.02
    panic_consecutive_days: int = 3
    panic_cash_ratio: float = 0.8

    # --- 改进 #5: 市场广度 ---
    use_market_breadth: bool = False
    breadth_bear_thresh: float = 0.35

    # --- 改进 #6: 多时间框架 ---
    use_multi_timeframe: bool = False

    # --- 改进 #7: 行业中性 ---
    use_sector_neutrality: bool = False
    max_per_sector: int = 3

    # --- 改进 #8: 子策略淘汰 ---
    use_strategy_elimination: bool = False
    elimination_lookback: int = 60
    elimination_threshold: float = -0.05

    # --- 改进 #9: HMM ---
    use_hmm: bool = False

    # --- 改进 #10: 宏观叠加 ---
    use_macro_overlay: bool = False
    macro_cash_ratio: float = 0.5

    # --- 子策略参数 ---
    sub_params: Optional[Dict[str, Dict]] = None

    # --- 状态分配 ---
    state_strategies: Optional[Dict] = None


# ════════════════════════════════════════════════════════════════════════════
# 实验执行
# ════════════════════════════════════════════════════════════════════════════

class DataContext:
    def __init__(self, z3, fwd, dm, cl, tks, fnames, nd, ns, ds, t2i, per_symbol):
        self.z3 = z3
        self.fwd = fwd
        self.dm = dm
        self.cl = cl
        self.tks = tks
        self.fnames = fnames
        self.nd = nd
        self.ns = ns
        self.ds = ds
        self.t2i = t2i
        self.per_symbol = per_symbol
        self.signals = None
        self.mkt_idx = None
        self.symbol_risk_map = None
        self.states = None
        self.confidence = None

    def ensure_signals(self):
        if self.signals is None:
            logger.info("构建信号...")
            self.signals = build_signals(
                self.z3, self.fwd, self.dm, self.cl,
                self.fnames, self.nd, self.ns, self.ds,
            )
            self.mkt_idx = self.signals["market_index"]
        return self.signals

    def ensure_risk_map(self):
        if self.symbol_risk_map is None:
            logger.info("构建增强ST风险映射...")
            self.symbol_risk_map = build_enhanced_st_mask(
                self.per_symbol, self.t2i, self.nd, self.ns,
            )
        return self.symbol_risk_map

    def ensure_states(self):
        if self.states is None:
            logger.info("检测市场状态...")
            self.ensure_signals()
            self.states, self.confidence = detect_market_state(self.mkt_idx, self.nd)
        return self.states, self.confidence


def run_single_experiment(data_ctx: DataContext, config: ExperimentConfig) -> Dict:
    t0 = time.time()
    logger.info(f"--- 运行实验: {config.name} (baseline={config.baseline}) ---")

    signals = data_ctx.ensure_signals()
    nd = data_ctx.nd

    # 增强ST
    symbol_risk_map = None
    if config.use_enhanced_st:
        symbol_risk_map = data_ctx.ensure_risk_map()

    # 市场状态检测
    if config.use_hysteresis:
        states, confidence = detect_market_state_hysteresis(
            data_ctx.mkt_idx, nd,
            hysteresis_days=config.hysteresis_days,
            cooldown_days=config.cooldown_days,
        )
    else:
        states, confidence = data_ctx.ensure_states()

    # Panic状态
    panic_states = None
    if config.use_panic_state:
        panic_states = detect_panic_state(
            data_ctx.mkt_idx, nd,
            single_day_thresh=config.panic_single_day,
            consecutive_thresh=config.panic_consecutive,
            consecutive_days=config.panic_consecutive_days,
        )

    # 市场广度
    breadth = None
    if config.use_market_breadth:
        breadth = compute_market_breadth(data_ctx.per_symbol["pct"], data_ctx.dm, nd)

    # 多时间框架
    weekly_slope = None
    if config.use_multi_timeframe:
        weekly_slope = compute_weekly_slope(data_ctx.mkt_idx, nd)

    # HMM
    hmm_states, hmm_conf = None, None
    if config.use_hmm:
        hmm_states, hmm_conf = hmm_detect_states(data_ctx.mkt_idx, nd)

    # 止损配置
    stop_losses = config.stop_loss_by_strategy
    if config.use_stop_loss and stop_losses is None:
        stop_losses = STOP_LOSS_V2B

    # 子策略参数
    sub_params = config.sub_params if config.sub_params else DEFAULT_SUB_PARAMS

    # 子策略回测
    sub_drs = compute_sub_drs_with_config(
        signals, data_ctx.fwd, data_ctx.dm, nd,
        use_v2=config.use_enhanced_st,
        symbol_risk_map=symbol_risk_map,
        stop_loss_by_strategy=stop_losses if config.use_stop_loss else None,
        use_trailing_stop=config.use_trailing_stop,
        trailing_profit_pct=config.trailing_profit_pct,
        sub_params=sub_params,
    )
    sub_drs["mf50_chip50"] = 0.5 * sub_drs["mf_base"] + 0.5 * sub_drs["chip_rp"]
    sub_drs["mf60_chip40"] = 0.6 * sub_drs["mf_base"] + 0.4 * sub_drs["chip_rp"]

    # 子策略净值追踪(用于淘汰)
    strategy_eq = compute_strategy_track_record(sub_drs, nd)

    # 状态分配
    state_strategies = config.state_strategies if config.state_strategies else DEFAULT_STATE_STRATEGIES

    # 置信度权重
    use_conf = config.use_confidence_weights or (config.baseline == "v2c" and not config.use_dynamic_weights)

    # MSS回测
    dr_mss = run_mss(
        state_strategies, sub_drs, states, confidence, nd,
        use_confidence_weights=use_conf,
        confidence_threshold=config.confidence_threshold,
        conf_adjust_mf=config.conf_adjust_mf,
        conf_adjust_c01=config.conf_adjust_c01,
        panic_states=panic_states,
        panic_cash_ratio=config.panic_cash_ratio,
        breadth=breadth,
        breadth_bear_thresh=config.breadth_bear_thresh,
        weekly_slope=weekly_slope,
        hmm_states=hmm_states,
        hmm_conf=hmm_conf,
        strategy_eq=strategy_eq,
        elimination_lookback=config.elimination_lookback if config.use_strategy_elimination else 0,
        elimination_threshold=config.elimination_threshold,
        macro_overlay=config.use_macro_overlay,
        macro_cash_ratio=config.macro_cash_ratio,
        sector_diversity=config.use_sector_neutrality,
        max_per_sector=config.max_per_sector,
        t2i=data_ctx.t2i,
        tks=data_ctx.tks,
    )

    metrics = compute_metrics(dr_mss, name=config.name)
    windows = window_analysis(dr_mss, data_ctx.ds, WINDOWS)
    state_counts = {s: states.count(s) for s in ["bull", "bear", "oscillate", "recovery"]}

    elapsed = time.time() - t0
    result = {
        "config_name": config.name,
        "baseline": config.baseline,
        "metrics": metrics,
        "windows": windows,
        "state_distribution": state_counts,
        "elapsed": round(elapsed, 1),
    }

    logger.info(f"  {config.name}: 年化={metrics['annual_return']*100:.2f}% "
                f"Sharpe={metrics['sharpe']:.3f} 回撤={abs(metrics['max_drawdown'])*100:.2f}% "
                f"Calmar={metrics['calmar']:.3f} ({elapsed:.1f}s)")

    return result


# ════════════════════════════════════════════════════════════════════════════
# Phase 0: Baseline
# ════════════════════════════════════════════════════════════════════════════

def run_baselines(data_ctx):
    logger.info("=" * 60)
    logger.info("Phase 0: Baseline 建立")
    logger.info("=" * 60)

    results = {}

    # V2b: 增强ST + 差异化止损
    config_v2b = ExperimentConfig(name="V2b_baseline", baseline="v2b")
    results["v2b"] = run_single_experiment(data_ctx, config_v2b)

    # V2c: V2b + 置信度联动
    config_v2c = ExperimentConfig(name="V2c_baseline", baseline="v2c",
                                   use_confidence_weights=True)
    results["v2c"] = run_single_experiment(data_ctx, config_v2c)

    return results


# ════════════════════════════════════════════════════════════════════════════
# Phase 1: 单点消融
# ════════════════════════════════════════════════════════════════════════════

def define_improvements(base_config):
    base = copy.deepcopy(base_config)
    improvements = []

    # #1 滞后确认
    cfg = copy.deepcopy(base)
    cfg.name = f"{base.baseline}_hys3"
    cfg.use_hysteresis = True
    cfg.hysteresis_days = 3
    improvements.append(("滞后确认(3天)", cfg))

    cfg2 = copy.deepcopy(base)
    cfg2.name = f"{base.baseline}_hys5"
    cfg2.use_hysteresis = True
    cfg2.hysteresis_days = 5
    improvements.append(("滞后确认(5天)", cfg2))

    # #2 移动止盈
    cfg = copy.deepcopy(base)
    cfg.name = f"{base.baseline}_trail10"
    cfg.use_trailing_stop = True
    cfg.trailing_profit_pct = 0.10
    improvements.append(("移动止盈(10%)", cfg))

    cfg2 = copy.deepcopy(base)
    cfg2.name = f"{base.baseline}_trail15"
    cfg2.use_trailing_stop = True
    cfg2.trailing_profit_pct = 0.15
    improvements.append(("移动止盈(15%)", cfg2))

    # #3 置信度动态权重 (仅在V2b上测试, V2c已有)
    if base.baseline == "v2b":
        cfg = copy.deepcopy(base)
        cfg.name = "v2b_conf_dyn"
        cfg.use_confidence_weights = True
        improvements.append(("置信度动态权重", cfg))

    # #4 Panic状态
    cfg = copy.deepcopy(base)
    cfg.name = f"{base.baseline}_panic"
    cfg.use_panic_state = True
    improvements.append(("Panic状态检测", cfg))

    # #5 市场广度
    cfg = copy.deepcopy(base)
    cfg.name = f"{base.baseline}_breadth"
    cfg.use_market_breadth = True
    improvements.append(("市场广度", cfg))

    # #6 多时间框架
    cfg = copy.deepcopy(base)
    cfg.name = f"{base.baseline}_multitf"
    cfg.use_multi_timeframe = True
    improvements.append(("多时间框架", cfg))

    # #7 行业中性
    cfg = copy.deepcopy(base)
    cfg.name = f"{base.baseline}_sector3"
    cfg.use_sector_neutrality = True
    cfg.max_per_sector = 3
    improvements.append(("行业中性(3只/行业)", cfg))

    # #8 子策略淘汰
    cfg = copy.deepcopy(base)
    cfg.name = f"{base.baseline}_elim"
    cfg.use_strategy_elimination = True
    improvements.append(("子策略淘汰", cfg))

    # #9 HMM
    cfg = copy.deepcopy(base)
    cfg.name = f"{base.baseline}_hmm"
    cfg.use_hmm = True
    improvements.append(("HMM状态识别", cfg))

    # #10 宏观叠加
    cfg = copy.deepcopy(base)
    cfg.name = f"{base.baseline}_macro"
    cfg.use_macro_overlay = True
    improvements.append(("宏观因子叠加", cfg))

    return improvements


def run_single_point(data_ctx):
    logger.info("=" * 60)
    logger.info("Phase 1: 单点消融实验")
    logger.info("=" * 60)

    all_results = {}

    for bl_name, bl_label in [("v2b", "V2b"), ("v2c", "V2c")]:
        logger.info(f"\n--- {bl_label} Baseline 单点消融 ---")
        base_cfg = ExperimentConfig(
            name=f"{bl_name}_baseline", baseline=bl_name,
            use_confidence_weights=(bl_name == "v2c"),
        )
        base_result = run_single_experiment(data_ctx, base_cfg)
        improvements = define_improvements(base_cfg)

        imp_results = []
        for imp_name, imp_cfg in improvements:
            result = run_single_experiment(data_ctx, imp_cfg)
            imp_results.append({"improvement": imp_name, "result": result})

        # 对比基线计算差值
        for item in imp_results:
            bm = base_result["metrics"]
            rm = item["result"]["metrics"]
            item["delta_ar"] = round(rm["annual_return"] - bm["annual_return"], 4)
            item["delta_sharpe"] = round(rm["sharpe"] - bm["sharpe"], 4)
            item["delta_calmar"] = round(rm["calmar"] - bm["calmar"], 4)

        # 排序: Calmar改善最大优先
        imp_results.sort(key=lambda x: x["delta_calmar"], reverse=True)

        all_results[bl_name] = {
            "baseline": base_result,
            "improvements": imp_results,
        }

    # 保存
    out_path = os.path.join(RESULTS_DIR, "single_point_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info(f"单点消融结果已保存: {out_path}")
    return all_results


# ════════════════════════════════════════════════════════════════════════════
# Phase 2: 器级参数调优 (autoresearch-style grid search)
# ════════════════════════════════════════════════════════════════════════════

def define_allocation_variants():
    """生成状态-策略分配权重的搜索空间"""
    variants = []

    # 默认权重
    default = {
        "bull": [("mf_d10_rp", 0.6), ("mf_vol_d10_rp", 0.2), ("chip_covrp", 0.2)],
        "bear": [("chip_covrp", 0.6), ("mf_vol_d10_rp", 0.2), ("chip_rp", 0.2)],
        "oscillate": [("chip_covrp", 0.4), ("mf50_chip50", 0.3), ("c01_layered_d5", 0.3)],
        "recovery": [("mf60_chip40", 0.4), ("chip_rp", 0.3), ("mf_vol_d10_rp", 0.3)],
    }
    variants.append(("默认V6a_3way", default))

    # Bull偏进攻: 提高mf_d10_rp
    v = copy.deepcopy(default)
    v["bull"] = [("mf_d10_rp", 0.7), ("mf_vol_d10_rp", 0.15), ("chip_covrp", 0.15)]
    variants.append(("Bull偏进攻(7:1.5:1.5)", v))

    # Bull偏防守: 提高chip_covrp
    v = copy.deepcopy(default)
    v["bull"] = [("mf_d10_rp", 0.4), ("mf_vol_d10_rp", 0.2), ("chip_covrp", 0.4)]
    variants.append(("Bull偏防守(4:2:4)", v))

    # Bear极度防御
    v = copy.deepcopy(default)
    v["bear"] = [("chip_covrp", 0.8), ("mf_vol_d10_rp", 0.1), ("chip_rp", 0.1)]
    variants.append(("Bear极度防御(8:1:1)", v))

    # Bear适度进攻
    v = copy.deepcopy(default)
    v["bear"] = [("chip_covrp", 0.4), ("mf_vol_d10_rp", 0.3), ("chip_rp", 0.3)]
    variants.append(("Bear适度进攻(4:3:3)", v))

    # Oscillate偏趋势
    v = copy.deepcopy(default)
    v["oscillate"] = [("chip_covrp", 0.2), ("mf50_chip50", 0.4), ("c01_layered_d5", 0.4)]
    variants.append(("Osc偏趋势(2:4:4)", v))

    # Oscillate偏防守
    v = copy.deepcopy(default)
    v["oscillate"] = [("chip_covrp", 0.6), ("mf50_chip50", 0.2), ("c01_layered_d5", 0.2)]
    variants.append(("Osc偏防守(6:2:2)", v))

    # Recovery积极
    v = copy.deepcopy(default)
    v["recovery"] = [("mf60_chip40", 0.5), ("chip_rp", 0.25), ("mf_vol_d10_rp", 0.25)]
    variants.append(("Recovery积极(5:2.5:2.5)", v))

    return variants


def define_stop_loss_variants():
    """止损比例搜索空间"""
    variants = []
    variants.append(("默认(8/10)", copy.deepcopy(STOP_LOSS_V2B)))

    # 更紧止损
    sl = {"mf_d10_rp": 0.06, "mf_vol_d10_rp": 0.06, "chip_rp": 0.08,
          "chip_covrp": 0.08, "c01_layered_d5": 0.06, "osr_d10": 0.08, "mf_base": 0.06}
    variants.append(("紧止损(6/8)", sl))

    # 更宽止损
    sl = {"mf_d10_rp": 0.10, "mf_vol_d10_rp": 0.10, "chip_rp": 0.12,
          "chip_covrp": 0.12, "c01_layered_d5": 0.10, "osr_d10": 0.12, "mf_base": 0.10}
    variants.append(("宽止损(10/12)", sl))

    # 统一止损
    sl = {"mf_d10_rp": 0.08, "mf_vol_d10_rp": 0.08, "chip_rp": 0.08,
          "chip_covrp": 0.08, "c01_layered_d5": 0.08, "osr_d10": 0.08, "mf_base": 0.08}
    variants.append(("统一止损(8)", sl))

    # 无止损
    variants.append(("无止损", {}))
    return variants


def define_sub_param_variants():
    """子策略参数搜索空间"""
    variants = []

    # 默认
    variants.append(("默认", copy.deepcopy(DEFAULT_SUB_PARAMS)))

    # 扩大选股池: mf系列top_n=80
    p = copy.deepcopy(DEFAULT_SUB_PARAMS)
    p["mf_d10_rp"]["tn"] = 80
    p["mf_vol_d10_rp"]["tn"] = 80
    variants.append(("mf系列top_n=80", p))

    # 缩小选股池: mf系列top_n=30
    p = copy.deepcopy(DEFAULT_SUB_PARAMS)
    p["mf_d10_rp"]["tn"] = 30
    p["mf_vol_d10_rp"]["tn"] = 30
    variants.append(("mf系列top_n=30", p))

    # 调仓频率: chip系列5天
    p = copy.deepcopy(DEFAULT_SUB_PARAMS)
    p["chip_rp"]["rf"] = 5
    p["chip_covrp"]["rf"] = 5
    variants.append(("chip系列rf=5", p))

    # 调仓频率: mf系列5天
    p = copy.deepcopy(DEFAULT_SUB_PARAMS)
    p["mf_d10_rp"]["rf"] = 5
    p["mf_vol_d10_rp"]["rf"] = 5
    variants.append(("mf系列rf=5", p))

    # min_hold_days加长
    p = copy.deepcopy(DEFAULT_SUB_PARAMS)
    for k in p:
        p[k]["mhd"] = max(p[k]["mhd"] + 5, 10)
    variants.append(("min_hold_days+5", p))

    return variants


def define_confidence_variants():
    """置信度系数搜索空间 (仅用于V2c)"""
    variants = []

    # 默认
    variants.append(("默认(0.5/0.6/0.7)", (0.5, 0.6, 0.7)))

    # 更激进降权
    variants.append(("激进降权(0.5/0.4/0.5)", (0.5, 0.4, 0.5)))

    # 温和降权
    variants.append(("温和降权(0.5/0.8/0.85)", (0.5, 0.8, 0.85)))

    # 更低阈值
    variants.append(("低阈值(0.3/0.6/0.7)", (0.3, 0.6, 0.7)))

    # 更高阈值
    variants.append(("高阈值(0.7/0.6/0.7)", (0.7, 0.6, 0.7)))

    return variants


def run_param_tuning(data_ctx, baseline_results):
    """器级参数调优: 对每个参数维度做grid search, Calmar-driven keep/discard"""
    logger.info("=" * 60)
    logger.info("Phase 2: 器级参数调优 (autoresearch-style)")
    logger.info("=" * 60)

    all_tuning = {}

    for bl_name in ["v2b", "v2c"]:
        logger.info(f"\n{'='*40}")
        logger.info(f"参数调优: {bl_name.upper()} baseline")
        logger.info(f"{'='*40}")

        base_conf = baseline_results[bl_name]["metrics"]["calmar"]
        base_cfg = ExperimentConfig(
            name=f"{bl_name}_param_base", baseline=bl_name,
            use_confidence_weights=(bl_name == "v2c"),
        )

        tuning_results = {}

        # --- A. 分配权重搜索 ---
        logger.info("--- A. 状态-策略分配权重 ---")
        alloc_variants = define_allocation_variants()
        alloc_best = {"calmar": base_conf, "name": "默认"}
        for vname, v in alloc_variants:
            ss = {}
            for state, allocs in v.items():
                ss[state] = [{"strategy": a[0], "weight": a[1]} for a in allocs]
            cfg = copy.deepcopy(base_cfg)
            cfg.name = f"{bl_name}_alloc_{vname}"
            cfg.state_strategies = ss
            r = run_single_experiment(data_ctx, cfg)
            cal = r["metrics"]["calmar"]
            if cal > alloc_best["calmar"]:
                alloc_best = {"calmar": cal, "name": vname, "result": r, "allocation": v}
            logger.info(f"    分配 [{vname}]: Calmar={cal:.3f} (best={alloc_best['calmar']:.3f})")
        tuning_results["allocation"] = alloc_best

        # --- B. 止损比例搜索 ---
        logger.info("--- B. 止损比例 ---")
        sl_variants = define_stop_loss_variants()
        sl_best = {"calmar": base_conf, "name": "默认"}
        for vname, sl in sl_variants:
            cfg = copy.deepcopy(base_cfg)
            cfg.name = f"{bl_name}_sl_{vname}"
            if sl:
                cfg.use_stop_loss = True
                cfg.stop_loss_by_strategy = sl
            else:
                cfg.use_stop_loss = False
                cfg.stop_loss_by_strategy = None
            r = run_single_experiment(data_ctx, cfg)
            cal = r["metrics"]["calmar"]
            if cal > sl_best["calmar"]:
                sl_best = {"calmar": cal, "name": vname, "result": r, "stop_loss": sl}
            logger.info(f"    止损 [{vname}]: Calmar={cal:.3f} (best={sl_best['calmar']:.3f})")
        tuning_results["stop_loss"] = sl_best

        # --- C. 子策略参数搜索 ---
        logger.info("--- C. 子策略参数 (top_n/rf/mhd) ---")
        sub_variants = define_sub_param_variants()
        sub_best = {"calmar": base_conf, "name": "默认"}
        for vname, sp in sub_variants:
            cfg = copy.deepcopy(base_cfg)
            cfg.name = f"{bl_name}_sub_{vname}"
            cfg.sub_params = sp
            r = run_single_experiment(data_ctx, cfg)
            cal = r["metrics"]["calmar"]
            if cal > sub_best["calmar"]:
                sub_best = {"calmar": cal, "name": vname, "result": r, "sub_params": sp}
            logger.info(f"    子策略 [{vname}]: Calmar={cal:.3f} (best={sub_best['calmar']:.3f})")
        tuning_results["sub_params"] = sub_best

        # --- D. 置信度系数 (仅V2c) ---
        if bl_name == "v2c":
            logger.info("--- D. 置信度系数 ---")
            conf_variants = define_confidence_variants()
            conf_best = {"calmar": base_conf, "name": "默认"}
            for vname, (thresh, adj_mf, adj_c01) in conf_variants:
                cfg = copy.deepcopy(base_cfg)
                cfg.name = f"v2c_conf_{vname}"
                cfg.confidence_threshold = thresh
                cfg.conf_adjust_mf = adj_mf
                cfg.conf_adjust_c01 = adj_c01
                r = run_single_experiment(data_ctx, cfg)
                cal = r["metrics"]["calmar"]
                if cal > conf_best["calmar"]:
                    conf_best = {"calmar": cal, "name": vname, "result": r,
                                  "threshold": thresh, "adj_mf": adj_mf, "adj_c01": adj_c01}
                logger.info(f"    置信度 [{vname}]: Calmar={cal:.3f} (best={conf_best['calmar']:.3f})")
            tuning_results["confidence"] = conf_best

        all_tuning[bl_name] = tuning_results

        # 汇总最优参数
        sum_parts = []
        for cat, best in tuning_results.items():
            if best["calmar"] > base_conf:
                sum_parts.append(f"{cat}={best['name']}(+{best['calmar']-base_conf:.3f})")
        if sum_parts:
            logger.info(f"  {bl_name} 参数优化: {', '.join(sum_parts)}")
        else:
            logger.info(f"  {bl_name} 参数优化: 默认配置已最优")

    # 保存
    out_path = os.path.join(RESULTS_DIR, "param_tuning_results.json")
    with open(out_path, "w") as f:
        json.dump(all_tuning, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"参数调优结果已保存: {out_path}")
    return all_tuning


# ════════════════════════════════════════════════════════════════════════════
# Phase 3: 贪婪组合优化
# ════════════════════════════════════════════════════════════════════════════

def build_candidate_improvements(base_cfg, param_tuning):
    """从单点消融+参数调优中提取候选改进项,用于组合优化"""
    candidates = []

    # 滞后确认
    c = copy.deepcopy(base_cfg)
    c.use_hysteresis = True
    c.hysteresis_days = 3
    c.cooldown_days = 5
    candidates.append(("滞后确认(hys3)", c))

    # 移动止盈
    c = copy.deepcopy(base_cfg)
    c.use_trailing_stop = True
    c.trailing_profit_pct = 0.15
    candidates.append(("移动止盈(trail15)", c))

    # Panic
    c = copy.deepcopy(base_cfg)
    c.use_panic_state = True
    candidates.append(("Panic状态", c))

    # 市场广度
    c = copy.deepcopy(base_cfg)
    c.use_market_breadth = True
    candidates.append(("市场广度", c))

    # 多时间框架
    c = copy.deepcopy(base_cfg)
    c.use_multi_timeframe = True
    candidates.append(("多时间框架", c))

    # 子策略淘汰
    c = copy.deepcopy(base_cfg)
    c.use_strategy_elimination = True
    candidates.append(("子策略淘汰", c))

    # HMM
    c = copy.deepcopy(base_cfg)
    c.use_hmm = True
    candidates.append(("HMM状态识别", c))

    return candidates


def _apply_improvement(config, name):
    if "hys3" in name:
        config.use_hysteresis = True
        config.hysteresis_days = 3
        config.cooldown_days = 5
    elif "trail15" in name or "trail10" in name:
        config.use_trailing_stop = True
        if "15" in name:
            config.trailing_profit_pct = 0.15
        else:
            config.trailing_profit_pct = 0.10
    elif "Panic" in name:
        config.use_panic_state = True
    elif "广度" in name:
        config.use_market_breadth = True
    elif "时间框架" in name:
        config.use_multi_timeframe = True
    elif "淘汰" in name:
        config.use_strategy_elimination = True
    elif "HMM" in name:
        config.use_hmm = True


def run_combination(data_ctx, baseline_results, param_tuning=None):
    logger.info("=" * 60)
    logger.info("Phase 3: 贪婪组合优化")
    logger.info("=" * 60)

    all_combo = {}

    for bl_name in ["v2b", "v2c"]:
        logger.info(f"\n{'='*40}")
        logger.info(f"组合优化: {bl_name.upper()}")
        logger.info(f"{'='*40}")

        base_cfg = ExperimentConfig(
            name=f"{bl_name}_base", baseline=bl_name,
            use_confidence_weights=(bl_name == "v2c"),
        )
        base_result = run_single_experiment(data_ctx, base_cfg)
        best_calmar = base_result["metrics"]["calmar"]
        logger.info(f"  初始 Calmar: {best_calmar:.3f}")

        candidates = build_candidate_improvements(base_cfg, param_tuning)
        available = list(candidates)
        selected = []
        steps = []

        while available:
            best_improvement = None
            best_new_calmar = best_calmar

            for cname, cfg in available:
                merged_cfg = copy.deepcopy(base_cfg)
                merged_cfg.name = f"{bl_name}_combo_test"
                for sname, _ in selected:
                    _apply_improvement(merged_cfg, sname)
                _apply_improvement(merged_cfg, cname)

                r = run_single_experiment(data_ctx, merged_cfg)
                cal = r["metrics"]["calmar"]
                if cal > best_new_calmar:
                    best_new_calmar = cal
                    best_improvement = (cname, cfg, r)

            if best_improvement and best_new_calmar > best_calmar * 1.005:
                cname, _, result = best_improvement
                selected.append((cname, result))
                available = [(n, c) for n, c in available if n != cname]
                delta = best_new_calmar - best_calmar
                best_calmar = best_new_calmar
                steps.append({
                    "step": len(steps) + 1,
                    "added": cname,
                    "calmar": round(best_calmar, 3),
                    "delta": round(delta, 3),
                    "metrics": result["metrics"],
                    "windows": result["windows"],
                })
                logger.info(f"  Step {len(steps)}: +{cname} Calmar={best_calmar:.3f} (+{delta:.3f})")
            else:
                logger.info(f"  无进一步改善, 停止搜索")
                break

        all_combo[bl_name] = {
            "base_calmar": round(best_calmar, 3),
            "selected": [s["added"] for s in steps],
            "final_calmar": round(best_calmar, 3),
            "steps": steps,
        }

        logger.info(f"  {bl_name} 最优组合: {' + '.join([s['added'] for s in steps]) if steps else 'Baseline'}")
        logger.info(f"  {bl_name} 最终 Calmar: {best_calmar:.3f}")

    out_path = os.path.join(RESULTS_DIR, "combination_results.json")
    with open(out_path, "w") as f:
        json.dump(all_combo, f, indent=2, ensure_ascii=False)
    logger.info(f"组合优化结果已保存: {out_path}")
    return all_combo


# ════════════════════════════════════════════════════════════════════════════
# Phase 4: 报告生成
# ════════════════════════════════════════════════════════════════════════════

def generate_report(baseline_results, single_results, param_results, combo_results):
    logger.info("=" * 60)
    logger.info("Phase 4: 生成实验报告")
    logger.info("=" * 60)

    lines = []
    lines.append("# mss_dynamic V2 全面优化实验报告")
    lines.append("")
    lines.append(f"**日期**: 2026-05-27")
    lines.append(f"**数据区间**: 2019-01-02 ~ 2026-05-22")
    lines.append(f"**对比窗口**: 全区间 / 2022熊市 / OOS修复牛(2024.07+)")
    lines.append(f"**方法**: autoresearch-inspired grid search + greedy combination")
    lines.append("")
    lines.append("## 实验概述")
    lines.append("")
    lines.append("本实验基于 V2b (增强ST+差异化止损) 和 V2c (V2b+置信度联动) 两个 baseline，")
    lines.append("系统性地探索以下维度的优化空间：")
    lines.append("")
    lines.append("| Phase | 内容 | 方法 |")
    lines.append("|-------|------|------|")
    lines.append("| 0 | Baseline 建立 | V2b/V2c 精确复制 |")
    lines.append("| 1 | 10项改进单点消融 | 每项独立 × 2 baseline |")
    lines.append("| 2 | 器级参数调优 | Grid search: 分配权重/止损/top_n/rf/mhd/置信度 |")
    lines.append("| 3 | 组合优化 | Greedy additive, Calmar-driven keep/discard |")
    lines.append("")

    # Phase 0
    lines.append("## Phase 0: Baseline 对比")
    lines.append("")
    lines.append("| Baseline | 年化% | Sharpe | 回撤% | Calmar | 胜率% |")
    lines.append("|----------|-------|--------|-------|--------|-------|")
    for bl_name, bl_label in [("v2b", "V2b"), ("v2c", "V2c")]:
        m = baseline_results[bl_name]["metrics"]
        lines.append(f"| {bl_label} | {m['annual_return']*100:.2f} | {m['sharpe']:.3f} | "
                     f"{-m['max_drawdown']*100:.2f} | {m['calmar']:.3f} | {m['win_rate']*100:.1f} |")
    lines.append("")

    # Phase 0 分窗口
    lines.append("### 分窗口表现")
    lines.append("")
    for win_name in ["全区间", "2022熊市", "OOS修复牛"]:
        lines.append(f"**{win_name}**")
        lines.append("| Baseline | 年化% | Sharpe | 回撤% | Calmar |")
        lines.append("|----------|-------|--------|-------|--------|")
        for bl_name, bl_label in [("v2b", "V2b"), ("v2c", "V2c")]:
            wins = {w["name"]: w for w in baseline_results[bl_name]["windows"]}
            w = wins.get(win_name, {})
            if w:
                lines.append(f"| {bl_label} | {w.get('annual_return', 0)*100:.2f} | "
                             f"{w.get('sharpe', 0):.3f} | {abs(w.get('max_drawdown', 0))*100:.2f} | "
                             f"{w.get('calmar', 0):.3f} |")
        lines.append("")
    lines.append("")

    # Phase 1
    lines.append("## Phase 1: 单点消融结果")
    lines.append("")
    for bl_name, bl_label in [("v2b", "V2b"), ("v2c", "V2c")]:
        lines.append(f"### {bl_label} Baseline 单项改进")
        lines.append("")
        sr = single_results.get(bl_name, {})
        bm = sr.get("baseline", {}).get("metrics", {})
        base_cal = bm.get("calmar", 0)

        lines.append("| 改进 | 年化% | Sharpe | 回撤% | Calmar | ΔCalmar | 评级 |")
        lines.append("|------|-------|--------|-------|--------|---------|------|")

        # Baseline
        lines.append(f"| **Baseline** | {bm.get('annual_return',0)*100:.2f} | {bm.get('sharpe',0):.3f} | "
                     f"{-bm.get('max_drawdown',0)*100:.2f} | {bm.get('calmar',0):.3f} | - | - |")

        for item in sr.get("improvements", []):
            m = item["result"]["metrics"]
            dc = item["delta_calmar"]
            rating = "🟢" if dc > 0.05 else ("🟡" if dc > 0 else "🔴")
            lines.append(f"| {item['improvement']} | {m['annual_return']*100:.2f} | {m['sharpe']:.3f} | "
                         f"{-m['max_drawdown']*100:.2f} | {m['calmar']:.3f} | "
                         f"{dc:+.3f} | {rating} |")
        lines.append("")

    # Phase 2
    lines.append("## Phase 2: 器级参数调优")
    lines.append("")
    for bl_name, bl_label in [("v2b", "V2b"), ("v2c", "V2c")]:
        lines.append(f"### {bl_label} 参数调优")
        lines.append("")
        pr = param_results.get(bl_name, {})
        base_cal = baseline_results[bl_name]["metrics"]["calmar"]

        for cat, cat_label in [("allocation", "分配权重"), ("stop_loss", "止损比例"),
                                ("sub_params", "子策略参数"), ("confidence", "置信度系数")]:
            if cat not in pr:
                continue
            best = pr[cat]
            if isinstance(best, dict):
                delta = best.get("calmar", base_cal) - base_cal
                lines.append(f"- **{cat_label}**: 最佳={best.get('name','?')} "
                             f"(Calmar={best.get('calmar',0):.3f}, Δ={delta:+.3f})")
        lines.append("")

    # Phase 3
    lines.append("## Phase 3: 组合优化")
    lines.append("")
    for bl_name, bl_label in [("v2b", "V2b"), ("v2c", "V2c")]:
        cr = combo_results.get(bl_name, {})
        lines.append(f"### {bl_label} 贪婪组合")
        lines.append("")
        if cr.get("steps"):
            lines.append("| Step | 加入改进 | Calmar | Δ |")
            lines.append("|------|----------|--------|---|")
            for s in cr["steps"]:
                lines.append(f"| {s['step']} | {s['added']} | {s['calmar']} | {s['delta']:+.3f} |")
            lines.append("")
            lines.append(f"**最优组合**: {' + '.join(cr['selected'])}")
            lines.append(f"**最终 Calmar**: {cr['final_calmar']}")
        else:
            lines.append("无组合改善, Baseline 已最优")
        lines.append("")

    lines.append("## 结论与建议")
    lines.append("")
    lines.append("1. **V2b vs V2c**: 置信度联动是否带来增量价值？")
    lines.append("2. **最有效单点改进**: 按 ΔCalmar 排名")
    lines.append("3. **最有效参数调优**: 对每类参数的敏感度分析")
    lines.append("4. **推荐组合**: 组合优化得到的最优配置")
    lines.append("5. **OOS验证**: 在修复牛窗口(2024.07+)的表现是否稳健")
    lines.append("")

    report_path = os.path.join(SCRIPT_DIR, "experiment_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"实验报告已生成: {report_path}")
    return report_path


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="mss_dynamic V2 全面优化实验")
    parser.add_argument("--mode", choices=["all", "single", "param", "combo"], default="all")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("mss_dynamic V2 全面优化实验框架")
    logger.info(f"模式: {args.mode}")
    logger.info(f"数据: {DB_PATH}")
    logger.info(f"结果: {RESULTS_DIR}")
    logger.info(f"日志: {LOG_PATH}")
    logger.info("=" * 70)

    t_start = time.time()

    # 加载数据
    data_tuple = load_data()
    data_ctx = DataContext(*data_tuple)

    # Phase 0: Baseline
    baseline_results = run_baselines(data_ctx)

    single_results = {}
    param_results = {}
    combo_results = {}

    # Phase 1: 单点消融
    if args.mode in ("all", "single"):
        single_results = run_single_point(data_ctx)

    # Phase 2: 参数调优
    if args.mode in ("all", "param"):
        param_results = run_param_tuning(data_ctx, baseline_results)

    # Phase 3: 组合优化
    if args.mode in ("all", "combo"):
        combo_results = run_combination(data_ctx, baseline_results, param_results)

    # Phase 4: 报告
    report_path = generate_report(baseline_results, single_results, param_results, combo_results)

    elapsed = time.time() - t_start
    logger.info("=" * 70)
    logger.info(f"实验完成! 总耗时: {elapsed/60:.1f} 分钟")
    logger.info(f"报告: {report_path}")
    logger.info(f"日志: {LOG_PATH}")
    logger.info(f"结果目录: {RESULTS_DIR}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()