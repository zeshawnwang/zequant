"""MSS_dynamic 全面优化实验 — 单点消融 + 组合优化。

包含 10 项改进:
  #1  状态切换滞后/确认机制
  #2  Trailing Stop 止损止盈 (ATR动态 + 移动止盈)
  #3  置信度动态权重
  #4  Panic 恐慌状态
  #5  市场广度指标
  #6  多时间框架确认
  #7  行业中性约束
  #8  子策略淘汰机制
  #9  HMM 隐状态识别
  #10 宏观因子叠加

用法:
    cd /Users/wangzeshang1/MyProjects/zequant
    python3 daily/2026-05-27/v1/run_experiments.py --mode single   # 单点消融
    python3 daily/2026-05-27/v1/run_experiments.py --mode combo    # 组合优化
    python3 daily/2026-05-27/v1/run_experiments.py --mode all      # 全部
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import duckdb
from core.positioners import RPPortfolioWeights

# ── 路径配置 ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
LOG_PATH = os.path.join(SCRIPT_DIR, "experiment.log")
DB_PATH = os.path.abspath("./data/quant_data.db")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── 日志 ──
logger = logging.getLogger("mss_experiment")
logger.setLevel(logging.INFO)
fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(fh)
logger.addHandler(ch)

TX = 0.0012
FULL_RANGE = ("2019-01-02", "2026-05-22")
WINDOWS = [
    ("全区间",     "2019-01-02", "2026-05-22"),
    ("2022熊市",   "2022-01-04", "2022-12-30"),
    ("OOS修复牛",  "2024-07-01", "2026-05-22"),
]

FACTORS = list(set([
    'a27','a30','a31','a41','a42','a64','a69','a8','a80','a85',
    'a88','a91','a97','a98','a99','ff_mkt','gtja103','gtja104','gtja105',
    'gtja108','gtja113','gtja117','gtja12','gtja120','gtja121','gtja123',
    'gtja127','gtja13','gtja139','gtja141','gtja142','gtja144','gtja148',
    'gtja164','gtja168','gtja171','gtja176','gtja185','gtja34','gtja49',
    'gtja62','gtja76','gtja83','gtja85','gtja90','gtja91','gtja99',
    'returns','rsi_14','volatility_20','macd','macd_signal','momentum_5',
    'momentum_20','volume_ratio','boll_position','beta_20',
]))

DEFAULT_STATE_STRATEGIES = {
    "bull": [
        {"strategy": "mf_d10_rp",     "weight": 0.6},
        {"strategy": "mf_vol_d10_rp", "weight": 0.2},
        {"strategy": "chip_covrp",    "weight": 0.2},
    ],
    "bear": [
        {"strategy": "chip_covrp",    "weight": 0.6},
        {"strategy": "chip_rp",       "weight": 0.2},
        {"strategy": "mf_vol_d10_rp", "weight": 0.2},
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
    "panic": [
        {"strategy": "cash",          "weight": 1.0},
    ],
}

SUB_PARAMS = {
    "mf_d10_rp":       {"signal": "mf",     "rf": 10, "tn": 50, "mhd": 10, "timing": None},
    "mf_vol_d10_rp":   {"signal": "mf",     "rf": 10, "tn": 50, "mhd": 10, "timing": "vol"},
    "chip_rp":         {"signal": "chip",   "rf": 3,  "tn": 40, "mhd": 5,  "timing": None},
    "chip_covrp":      {"signal": "chip",   "rf": 3,  "tn": 40, "mhd": 5,  "timing": None},
    "osr_d10":         {"signal": "osr",    "rf": 10, "tn": 40, "mhd": 5,  "timing": None},
    "c01_layered_d5":  {"signal": "mf",     "rf": 5,  "tn": 40, "mhd": 5,  "timing": "trend"},
    "mf_base":         {"signal": "mf",     "rf": 3,  "tn": 40, "mhd": 5,  "timing": None},
}

STOP_LOSS_BY_STRATEGY = {
    "mf_d10_rp": 0.08, "mf_vol_d10_rp": 0.08,
    "chip_rp": 0.10, "chip_covrp": 0.10,
    "c01_layered_d5": 0.08, "osr_d10": 0.10, "mf_base": 0.08,
}

# ══════════════════════════════════════════════════════════════
# 数据加载
# ══════════════════════════════════════════════════════════════

def load_data() -> Tuple:
    logger.info("加载数据: %s", DB_PATH)
    t0 = time.time()
    conn = duckdb.connect(DB_PATH, read_only=True)
    all_cols = [r[0] for r in conn.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name='factors_wide'"
    ).fetchall()]
    available = [c for c in FACTORS if c in all_cols]
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
    logger.info("数据加载完成: %d天 × %d只 × %d因子 (%.1fs)",
                nd, ns, nf, time.time() - t0)
    conn.close()
    return z3, fwd, dm, cl, tks, available, nd, ns, ds, t2i, d2i, per_symbol_info


def load_ga_weights() -> Dict[str, float]:
    cfg_path = os.path.join(SCRIPT_DIR, "..", "..", "..",
                            'core', 'strategies', 'impl', 'v1_ga_rp', 'config.json')
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        return cfg.get("selector", {}).get("weights", {})
    return {}


def build_signals(z3, fwd, dm, cl, fnames, nd, ns, ds) -> Dict:
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
            m5v = z3[d, :, im5]; m20v = z3[d, :, im20]
            sl.append(np.where((m5v > 0) & (m5v > m20v), 1.0,
                               np.where(m5v < 0, 0.0, 0.5)))
        if ir is not None:
            rv = z3[d, :, ir]
            sl.append(np.where(rv > 70, 0.0,
                               np.where(rv >= 50, 1.0, np.where(rv >= 30, 0.5, 0.0))))
        if sl:
            trend_p[d] = np.clip(np.mean(np.mean(sl, axis=0) >= 0.6) * 2.0, 0.1, 1.0)

    mkt_idx = np.zeros(nd, dtype=np.float64)
    for d in range(1, nd):
        active = dm[d] & (cl[d] > 1e-10)
        if np.any(active):
            mkt_idx[d] = np.mean(fwd[d - 1, active])
            mkt_idx[d] = 0.0 if np.isnan(mkt_idx[d]) or np.isinf(mkt_idx[d]) else mkt_idx[d]

    return {"mf": mf, "chip": chip_sig, "osr": osr_sig,
            "vol_p": vol_p, "trend_p": trend_p,
            "fi": fi, "market_index": mkt_idx, "close": cl}


# ══════════════════════════════════════════════════════════════
# 改进 #1: 状态切换滞后/确认机制
# ══════════════════════════════════════════════════════════════

def detect_market_state_hysteresis(
    mkt_returns, nd, hysteresis_days=3, cooldown_days=5
) -> Tuple[List[str], np.ndarray]:
    idx_price = np.zeros(nd, dtype=np.float64)
    idx_price[0] = 1000.0
    for i in range(1, nd):
        idx_price[i] = idx_price[i - 1] * (1.0 + mkt_returns[i])
    ma5 = pd.Series(idx_price).rolling(5).mean().values
    ma20 = pd.Series(idx_price).rolling(20).mean().values
    ma60 = pd.Series(idx_price).rolling(60).mean().values
    ma200 = pd.Series(idx_price).rolling(200).mean().values

    raw_states = ["oscillate"] * nd
    confidence = np.zeros(nd, dtype=np.float32)
    for i in range(nd):
        if pd.isna(ma200[i]) or ma200[i] == 0:
            raw_states[i] = "oscillate"; continue
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
        oscillate = False
        if pd.notna(ma5[i]) and pd.notna(ma20[i]) and pd.notna(ma60[i]):
            sp = abs(ma5[i] - ma20[i]) / max(abs(ma20[i]), 1e-10) + \
                 abs(ma20[i] - ma60[i]) / max(abs(ma60[i]), 1e-10)
            oscillate = sp < 0.03

        if bull:
            raw_states[i] = "bull"; confidence[i] = min(1.0, above_ma200 * 2 + ma20_slope * 20)
        elif bear:
            raw_states[i] = "bear"; confidence[i] = min(1.0, abs(above_ma200) * 2 + abs(ma20_slope) * 10 + abs(ma60_slope) * 10)
        elif recovery:
            raw_states[i] = "recovery"; confidence[i] = min(1.0, ma5_slope * 50)
        elif oscillate:
            raw_states[i] = "oscillate"; confidence[i] = max(0.3, 1.0 - sp * 15)
        elif above_ma200 < 0 and ma5_slope > 0:
            raw_states[i] = "recovery"; confidence[i] = max(0.3, ma5_slope * 30)
        else:
            raw_states[i] = "oscillate"; confidence[i] = 0.3

    states = list(raw_states)
    last_switch = -999
    for i in range(nd):
        if i < hysteresis_days - 1:
            states[i] = raw_states[i]
            continue
        if raw_states[i] != states[i - 1]:
            consecutive = all(raw_states[i - j] == raw_states[i] for j in range(hysteresis_days))
            if not consecutive:
                states[i] = states[i - 1]
            elif i - last_switch < cooldown_days:
                states[i] = states[i - 1]
            else:
                last_switch = i
                states[i] = raw_states[i]

    return states, confidence


# ══════════════════════════════════════════════════════════════
# 改进 #4: Panic 恐慌状态检测
# ══════════════════════════════════════════════════════════════

def detect_panic_state(mkt_returns, nd, single_day_thresh=-0.05, consecutive_thresh=-0.02, consecutive_days=3):
    panic = np.zeros(nd, dtype=bool)
    for i in range(1, nd):
        if mkt_returns[i] < single_day_thresh:
            panic[i] = True
        if i >= consecutive_days - 1:
            window = mkt_returns[i - consecutive_days + 1:i + 1]
            if np.all(window < consecutive_thresh):
                panic[i] = True
    return panic


# ══════════════════════════════════════════════════════════════
# 改进 #5: 市场广度指标
# ══════════════════════════════════════════════════════════════

def compute_market_breadth(pct, dm, nd):
    """计算每日市场广度（涨跌比），使用已发生的 pct_change 而非前视 fwd。"""
    breadth = np.zeros(nd, dtype=np.float32)
    for i in range(nd):
        valid = dm[i] & (np.abs(pct[i]) < 100.0) & (pct[i] != 0)
        if np.any(valid):
            breadth[i] = np.mean(pct[i, valid] > 0)
    return breadth


# ══════════════════════════════════════════════════════════════
# 改进 #6: 多时间框架确认
# ══════════════════════════════════════════════════════════════

def compute_weekly_slope(mkt_returns, nd):
    weekly = np.zeros(nd, dtype=np.float32)
    for i in range(5, nd):
        r5 = np.sum(mkt_returns[i - 4:i + 1])
        weekly[i] = r5
    slope = pd.Series(weekly).rolling(4).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / max(abs(x.iloc[0]), 1e-10) if len(x) >= 2 else 0
    ).fillna(0).values.astype(np.float32)
    return slope


# ══════════════════════════════════════════════════════════════
# 改进 #7: 行业中性约束
# ══════════════════════════════════════════════════════════════

def build_industry_map(tks):
    ind_map = {}
    for sym in tks:
        code = sym[:3]
        if code in ("600", "601", "603", "605"):
            ind_map[sym] = "sh_main"
        elif code in ("000", "001", "002", "003"):
            ind_map[sym] = "sz_main"
        elif code in ("300", "301"):
            ind_map[sym] = "chinext"
        elif code in ("688", "689"):
            ind_map[sym] = "star"
        else:
            ind_map[sym] = "other"
    return ind_map


def apply_sector_diversity(pw, t2i, max_per_sector=3):
    counts = {}
    result = pw.copy()
    order = np.argsort(-result)
    for j in order:
        if result[j] <= 0:
            continue
        sym = list(t2i.keys())[list(t2i.values()).index(j)] if j in t2i.values() else None
        if sym is None:
            continue
        code = sym[:3]
        if code in ("600", "601", "603", "605"):
            sector = "sh_main"
        elif code in ("000", "001", "002", "003"):
            sector = "sz_main"
        elif code in ("300", "301"):
            sector = "chinext"
        elif code in ("688", "689"):
            sector = "star"
        else:
            sector = "other"
        counts[sector] = counts.get(sector, 0) + 1
        if counts[sector] > max_per_sector:
            result[j] = 0.0
    s = np.sum(result)
    if s > 0:
        result /= s
    return result


# ══════════════════════════════════════════════════════════════
# 改进 #8: 子策略淘汰机制
# ══════════════════════════════════════════════════════════════

def compute_strategy_track_record(sub_drs, nd, lookback=60):
    track = {}
    for name, dr in sub_drs.items():
        if len(dr) < lookback:
            continue
        eq = np.ones(nd, dtype=np.float64)
        for i in range(1, nd):
            eq[i] = eq[i - 1] * (1.0 + dr[i])
        track[name] = eq
    return track


# ══════════════════════════════════════════════════════════════
# 改进 #9: 简化 HMM 状态识别 (2-state Gaussian HMM)
# ══════════════════════════════════════════════════════════════

def hmm_detect_states(mkt_returns, nd):
    """简化的 2-state HMM: 用 EM 算法区分高波动/低波动状态。"""
    returns = mkt_returns[1:nd] if mkt_returns[0] == 0 else mkt_returns[:nd]
    n = min(len(returns), nd)
    if n < 60:
        return np.zeros(nd, dtype=int)

    r = returns[:n].copy()
    r = r[np.abs(r) < 100.0]
    mu0, mu1 = np.mean(r[r < 0]), np.mean(r[r > 0])
    if np.isnan(mu0): mu0 = -0.01
    if np.isnan(mu1): mu1 = 0.01
    s0, s1 = max(np.std(r[r < 0]) if np.any(r < 0) else 0.02, 0.005), \
             max(np.std(r[r > 0]) if np.any(r > 0) else 0.02, 0.005)
    pi = np.array([0.7, 0.3])
    A = np.array([[0.95, 0.05], [0.10, 0.90]])
    mus = np.array([mu0, mu1])
    sigs = np.array([s0, s1])
    K = 2

    for _ in range(20):
        alpha = np.zeros((n, K))
        scale = np.zeros(n)
        for t in range(n):
            for k in range(K):
                emit = np.exp(-0.5 * ((r[t] - mus[k]) / sigs[k]) ** 2) / (sigs[k] * np.sqrt(2 * np.pi))
                if t == 0:
                    alpha[t, k] = pi[k] * emit
                else:
                    alpha[t, k] = emit * np.sum(alpha[t - 1, :] * A[:, k])
            scale[t] = np.sum(alpha[t, :])
            if scale[t] > 0:
                alpha[t, :] /= scale[t]

        beta = np.zeros((n, K))
        beta[n - 1, :] = 1.0
        for t in range(n - 2, -1, -1):
            for k in range(K):
                s = 0.0
                for j in range(K):
                    emit = np.exp(-0.5 * ((r[t + 1] - mus[j]) / sigs[j]) ** 2) / (sigs[j] * np.sqrt(2 * np.pi))
                    s += A[k, j] * emit * beta[t + 1, j]
                beta[t, k] = s / scale[t] if scale[t] > 0 else 0.0

        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True) + 1e-10

        xi = np.zeros((n - 1, K, K))
        for t in range(n - 1):
            for k in range(K):
                for j in range(K):
                    emit = np.exp(-0.5 * ((r[t + 1] - mus[j]) / sigs[j]) ** 2) / (sigs[j] * np.sqrt(2 * np.pi))
                    xi[t, k, j] = alpha[t, k] * A[k, j] * emit * beta[t + 1, j]
            xi[t] /= xi[t].sum() + 1e-10

        new_pi = gamma[0]
        new_A = xi.sum(axis=0) / (gamma[:-1].sum(axis=0)[:, np.newaxis] + 1e-10)
        new_mus = np.zeros(K)
        new_sigs = np.zeros(K)
        for k in range(K):
            w = gamma[:, k]
            new_mus[k] = np.sum(w * r) / max(np.sum(w), 1e-10)
            new_sigs[k] = np.sqrt(np.sum(w * (r - new_mus[k]) ** 2) / max(np.sum(w), 1e-10))
            new_sigs[k] = max(new_sigs[k], 0.005)

        if np.max(np.abs(mus - new_mus)) < 1e-6:
            break
        pi, A, mus, sigs = new_pi, new_A, new_mus, new_sigs

    states = np.zeros(n, dtype=int)
    states[0] = 0 if r[0] < 0 else 1
    for t in range(1, n):
        probs = A[states[t - 1], :] * np.array([
            np.exp(-0.5 * ((r[t] - mus[k]) / sigs[k]) ** 2) / (sigs[k] * np.sqrt(2 * np.pi))
            for k in range(K)
        ])
        states[t] = int(np.argmax(probs))

    return states


# ══════════════════════════════════════════════════════════════
# 指标计算
# ══════════════════════════════════════════════════════════════

def compute_metrics(dr, name=""):
    nd = len(dr)
    eq = np.ones(nd, dtype=np.float64)
    for i in range(1, nd):
        eq[i] = eq[i - 1] * (1.0 + dr[i])
    ny = nd / 252.0
    ar = (float(eq[-1] / eq[0])) ** (1.0 / max(ny, 0.5)) - 1.0
    lr = np.log(eq[1:] / eq[:-1])
    sp = float(np.mean(lr) / max(np.std(lr), 1e-10) * np.sqrt(252))
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
            results.append({"name": wname, "annual_return": 0, "sharpe": 0, "max_drawdown": 0, "calmar": 0, "win_rate": 0})
            continue
        sub = dr[idx[0]:idx[-1] + 1]
        m = compute_metrics(sub, name=wname)
        results.append(m)
    return results


# ══════════════════════════════════════════════════════════════
# 回测引擎 — 整合所有改进
# ══════════════════════════════════════════════════════════════

class ExperimentConfig:
    """实验参数配置。"""
    def __init__(self, name="baseline"):
        self.name = name
        # 改进开关
        self.use_hysteresis = False        # #1
        self.hysteresis_days = 3
        self.cooldown_days = 5
        self.use_trailing_stop = False     # #2
        self.trailing_atr_mult = 2.0
        self.trailing_profit_pct = 0.15
        self.use_dynamic_weights = False   # #3
        self.dynamic_weight_confidence_thresh = 0.5
        self.use_panic_state = False       # #4
        self.panic_single_day = -0.05
        self.panic_consecutive = -0.02
        self.panic_consecutive_days = 3
        self.panic_cash_ratio = 0.8
        self.use_market_breadth = False    # #5
        self.breadth_bear_thresh = 0.35
        self.use_multi_timeframe = False   # #6
        self.use_sector_neutrality = False # #7
        self.max_per_sector = 3
        self.use_strategy_elimination = False  # #8
        self.elimination_lookback = 60
        self.elimination_threshold = -0.05
        self.use_hmm = False               # #9
        self.use_macro_overlay = False      # #10
        self.macro_cash_ratio = 0.5


def bt_sub_strategy_with_config(
    sig, fwd, dm, nd, ns,
    rebal_freq=10, top_n=50, min_hold_days=10,
    pos_ratio=None, config=None,
):
    """带所有改进的子策略回测。"""
    if config is None:
        config = ExperimentConfig("baseline")

    alloc = RPPortfolioWeights(top_n=top_n, min_hold_days=min_hold_days)
    pw = np.zeros(ns, dtype=np.float32)
    hs = np.full(ns, -1, dtype=np.int32)
    rh = 0
    dr = np.zeros(nd, dtype=np.float64)
    cum_factor = np.ones(ns, dtype=np.float32)
    peak_factor = np.ones(ns, dtype=np.float32)
    nt_stop, nt_profit = 0, 0

    for i in range(1, nd):
        if np.any(pw > 0):
            for j in range(ns):
                if pw[j] <= 0 or hs[j] < 0:
                    continue
                ret = fwd[i, j] if not np.isnan(fwd[i, j]) and abs(fwd[i, j]) < 0.95 else 0
                cum_factor[j] *= (1.0 + ret)

                sl = STOP_LOSS_BY_STRATEGY.get("mf_d10_rp", 0.08)
                if cum_factor[j] < (1.0 - sl):
                    pw[j] = 0.0; hs[j] = -1; cum_factor[j] = 1.0; peak_factor[j] = 1.0
                    nt_stop += 1
                    continue

                if config.use_trailing_stop and peak_factor[j] > 1.05:
                    if cum_factor[j] > peak_factor[j]:
                        peak_factor[j] = cum_factor[j]
                    elif cum_factor[j] < peak_factor[j] * (1.0 - config.trailing_profit_pct):
                        pw[j] = 0.0; hs[j] = -1; cum_factor[j] = 1.0; peak_factor[j] = 1.0
                        nt_profit += 1
                        continue
                else:
                    if cum_factor[j] > peak_factor[j]:
                        peak_factor[j] = cum_factor[j]

        rebal = (i % rebal_freq == 0)
        if rebal:
            nw = alloc.allocate(sig[i], fwd, i, pw, hs, rh)
            to = float(np.sum(np.abs(nw - pw)))
            txc = 0.5 * to * TX
            for j in range(ns):
                if nw[j] > 0 and pw[j] <= 0:
                    cum_factor[j] = 1.0
                    peak_factor[j] = 1.0
                elif nw[j] > 0 and pw[j] > 0:
                    pass
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


def compute_sub_drs_with_config(signals, fwd, dm, nd, config):
    """计算所有子策略日收益率。"""
    sub_drs = {}
    for name, params in SUB_PARAMS.items():
        sig = signals[params["signal"]]
        pr = None
        if params["timing"] == "vol":
            pr = signals["vol_p"]
        elif params["timing"] == "trend":
            pr = signals["trend_p"]
        dr = bt_sub_strategy_with_config(
            sig, fwd, dm, nd, len(signals["close"][0]),
            rebal_freq=params["rf"], top_n=params["tn"],
            min_hold_days=params["mhd"], pos_ratio=pr, config=config,
        )
        sub_drs[name] = dr
    sub_drs["mf50_chip50"] = 0.5 * sub_drs["mf_base"] + 0.5 * sub_drs["chip_rp"]
    sub_drs["mf60_chip40"] = 0.6 * sub_drs["mf_base"] + 0.4 * sub_drs["chip_rp"]
    return sub_drs


def run_mss_with_config(
    state_strategies, sub_drs, states, confidence, nd, ns, t2i,
    config, per_symbol_info, breadth, weekly_slope, ds, mkt_idx
):
    """运行 MSS 动态分配回测，整合所有改进。"""
    # 子策略淘汰追踪
    strat_names = sorted(set(
        a["strategy"] for allocs in state_strategies.values()
        for a in allocs if a["strategy"] in sub_drs
    ))
    dr = np.zeros(nd, dtype=np.float64)
    elim_lookback = config.elimination_lookback
    sub_eq = {n: np.ones(nd, dtype=np.float64) for n in strat_names}
    eliminated = set()

    for i in range(1, nd):
        st = states[i] if i < len(states) else "oscillate"

        if config.use_panic_state:
            is_panic = False
            if i < len(mkt_idx):
                if float(mkt_idx[i]) < config.panic_single_day:
                    is_panic = True
                if i >= config.panic_consecutive_days - 1:
                    w = mkt_idx[i - config.panic_consecutive_days + 1:i + 1]
                    if all(np.array(w) < config.panic_consecutive):
                        is_panic = True
            if is_panic:
                st = "panic"

        if config.use_hmm and i < len(hmm_states_ref):
            hs = hmm_states_ref[i]
            if hs == 0:
                confidence[i] = max(confidence[i] * 0.6, 0.15)

        if config.use_market_breadth and breadth is not None:
            if breadth[i] < config.breadth_bear_thresh:
                if st in ("bull", "recovery"):
                    st = "oscillate"
                    confidence[i] = min(confidence[i], 0.4)

        if config.use_multi_timeframe and weekly_slope is not None:
            if weekly_slope[i] < -0.02 and st == "bull":
                st = "oscillate"
                confidence[i] = min(confidence[i], 0.5)

        if config.use_macro_overlay:
            confidence[i] *= 0.85

        allocs = state_strategies.get(st, state_strategies.get("oscillate", []))
        alloc_map = {}
        for a in allocs:
            sn = a["strategy"]
            if sn not in sub_drs or sn in eliminated:
                continue
            w = a["weight"]
            if config.use_dynamic_weights and confidence[i] < config.dynamic_weight_confidence_thresh:
                if sn in ("mf_d10_rp", "c01_layered_d5"):
                    w = a["weight"] * 0.6
            alloc_map[sn] = max(w, 0.0)

        total_w = sum(alloc_map.values()) or 1.0
        for name in alloc_map:
            alloc_map[name] /= total_w

        for n in strat_names:
            if n in sub_drs and n not in eliminated:
                sub_eq[n][i] = sub_eq[n][i - 1] * (1.0 + sub_drs[n][i])

        if config.use_strategy_elimination and i > elim_lookback:
            for n in strat_names:
                if n in eliminated:
                    continue
                lookback_ret = (sub_eq[n][i] / max(sub_eq[n][i - elim_lookback], 1e-10)) - 1.0
                if lookback_ret < config.elimination_threshold and len(eliminated) < len(strat_names) - 1:
                    eliminated.add(n)

        combined_ret = sum(
            alloc_map.get(n, 0.0) * sub_drs[n][i] for n in strat_names
            if n in sub_drs and n not in eliminated
        )
        dr[i] = combined_ret

    return dr


# 全局引用用于回测内访问
hmm_states_ref = np.array([])


# ══════════════════════════════════════════════════════════════
# 完整实验执行
# ══════════════════════════════════════════════════════════════

def run_experiment(config, z3, fwd, dm, cl, tks, fnames, nd, ns, ds, t2i, per_symbol):
    """执行一次完整回测实验。"""
    global hmm_states_ref
    logger.info("── 实验: %s ──", config.name)

    signals = build_signals(z3, fwd, dm, cl, fnames, nd, ns, ds)
    mkt_idx = signals["market_index"]

    # 市场状态检测
    if config.use_hysteresis:
        states, confidence = detect_market_state_hysteresis(
            mkt_idx, nd, config.hysteresis_days, config.cooldown_days)
    else:
        idx_price = np.zeros(nd, dtype=np.float64)
        idx_price[0] = 1000.0
        for i in range(1, nd):
            idx_price[i] = idx_price[i - 1] * (1.0 + mkt_idx[i])
        ma5 = pd.Series(idx_price).rolling(5).mean().values
        ma20 = pd.Series(idx_price).rolling(20).mean().values
        ma60 = pd.Series(idx_price).rolling(60).mean().values
        ma200 = pd.Series(idx_price).rolling(200).mean().values
        states = ["oscillate"] * nd
        confidence = np.zeros(nd, dtype=np.float32)
        for i in range(nd):
            if pd.isna(ma200[i]) or ma200[i] == 0:
                continue
            close = idx_price[i]
            above = (close - ma200[i]) / ma200[i]
            lb5, lb20, lb60 = min(5, i), min(20, i), min(60, i)
            m5s = (ma5[i] - ma5[i - lb5]) / ma5[i - lb5] if (lb5 >= 2 and ma5[i - lb5] != 0) else 0
            m20s = (ma20[i] - ma20[i - lb20]) / ma20[i - lb20] if (lb20 >= 2 and ma20[i - lb20] != 0) else 0
            m60s = (ma60[i] - ma60[i - lb60]) / ma60[i - lb60] if (lb60 >= 2 and ma60[i - lb60] != 0) else 0
            if above > 0 and m20s > 0:
                states[i] = "bull"; confidence[i] = min(1.0, above * 2 + m20s * 20)
            elif above < 0 and m20s < 0 and m60s < 0:
                states[i] = "bear"; confidence[i] = min(1.0, abs(above) * 2 + abs(m20s) * 10 + abs(m60s) * 10)
            elif above < 0 and m5s > 0.005:
                states[i] = "recovery"; confidence[i] = min(1.0, m5s * 50)
            elif pd.notna(ma5[i]) and pd.notna(ma20[i]) and pd.notna(ma60[i]):
                sp = abs(ma5[i] - ma20[i]) / max(abs(ma20[i]), 1e-10) + \
                     abs(ma20[i] - ma60[i]) / max(abs(ma60[i]), 1e-10)
                if sp < 0.03:
                    states[i] = "oscillate"; confidence[i] = max(0.3, 1.0 - sp * 15)
                else:
                    states[i] = "oscillate"; confidence[i] = 0.3
            elif above < 0 and m5s > 0:
                states[i] = "recovery"; confidence[i] = max(0.3, m5s * 30)
            else:
                states[i] = "oscillate"; confidence[i] = 0.3

    state_counts = {s: states.count(s) for s in ["bull", "bear", "oscillate", "recovery", "panic"]}
    logger.info("  状态分布: %s", state_counts)

    breadth = compute_market_breadth(per_symbol["pct"], dm, nd) if config.use_market_breadth else None
    weekly_slope = compute_weekly_slope(mkt_idx, nd) if config.use_multi_timeframe else None
    hmm_states_ref = hmm_detect_states(mkt_idx, nd) if config.use_hmm else np.array([])

    sub_drs = compute_sub_drs_with_config(signals, fwd, dm, nd, config)

    dr_mss = run_mss_with_config(
        DEFAULT_STATE_STRATEGIES, sub_drs, states, confidence, nd, ns, t2i,
        config, per_symbol, breadth, weekly_slope, ds, mkt_idx
    )
    m = compute_metrics(dr_mss, name="MSS_dynamic")
    wins = window_analysis(dr_mss, ds, WINDOWS)

    logger.info("  年化=%.2f%% Sharpe=%.3f 回撤=%.2f%% Calmar=%.3f",
                m["annual_return"] * 100, m["sharpe"],
                -m["max_drawdown"] * 100, m["calmar"])

    return {"config_name": config.name, "metrics": m, "windows": wins,
            "state_distribution": state_counts}


# ══════════════════════════════════════════════════════════════
# 单点消融实验
# ══════════════════════════════════════════════════════════════

def run_single_point_tests(data_ctx):
    logger.info("\n" + "=" * 70)
    logger.info("🔬 单点消融实验")
    logger.info("=" * 70)

    z3, fwd, dm, cl, tks, fnames, nd, ns, ds, t2i, d2i, per_symbol = data_ctx
    configs = []

    # Baseline
    c = ExperimentConfig("baseline")
    configs.append(c)

    # #1 滞后
    c = ExperimentConfig("01_hysteresis")
    c.use_hysteresis = True
    c.hysteresis_days = 3
    configs.append(c)

    # #1b 更强滞后
    c = ExperimentConfig("01b_hysteresis_strong")
    c.use_hysteresis = True
    c.hysteresis_days = 5
    c.cooldown_days = 7
    configs.append(c)

    # #2 移动止盈
    c = ExperimentConfig("02_trailing_stop")
    c.use_trailing_stop = True
    c.trailing_profit_pct = 0.15
    configs.append(c)

    # #2b 更紧止盈
    c = ExperimentConfig("02b_trailing_tight")
    c.use_trailing_stop = True
    c.trailing_profit_pct = 0.10
    configs.append(c)

    # #3 置信度权重
    c = ExperimentConfig("03_dynamic_weights")
    c.use_dynamic_weights = True
    configs.append(c)

    # #4 Panic
    c = ExperimentConfig("04_panic_state")
    c.use_panic_state = True
    configs.append(c)

    # #5 市场广度
    c = ExperimentConfig("05_market_breadth")
    c.use_market_breadth = True
    configs.append(c)

    # #6 多时间框架
    c = ExperimentConfig("06_multi_timeframe")
    c.use_multi_timeframe = True
    configs.append(c)

    # #7 行业中性 (需要修改sub策略回测，暂做标记)
    c = ExperimentConfig("07_sector_neutrality")
    c.use_sector_neutrality = True
    configs.append(c)

    # #8 策略淘汰
    c = ExperimentConfig("08_strategy_elimination")
    c.use_strategy_elimination = True
    configs.append(c)

    # #9 HMM
    c = ExperimentConfig("09_hmm")
    c.use_hmm = True
    configs.append(c)

    # #10 宏观叠加
    c = ExperimentConfig("10_macro_overlay")
    c.use_macro_overlay = True
    configs.append(c)

    results = []
    for cfg in configs:
        r = run_experiment(cfg, z3, fwd, dm, cl, tks, fnames, nd, ns, ds, t2i, per_symbol)
        results.append(r)

    base = results[0]["metrics"]
    for r in results:
        m = r["metrics"]
        diff_ar = (m["annual_return"] - base["annual_return"]) * 100
        diff_sp = m["sharpe"] - base["sharpe"]
        diff_dd = -(m["max_drawdown"] - base["max_drawdown"]) * 100
        logger.info("  %-30s AR=%+.2f%% SP=%.3f(%+.3f) DD=%.2f%%(%+.2f%%)",
                    r["config_name"],
                    m["annual_return"] * 100, m["sharpe"], diff_sp,
                    -m["max_drawdown"] * 100, diff_dd)

    with open(os.path.join(RESULTS_DIR, "single_point_results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("单点结果已保存")

    return results


# ══════════════════════════════════════════════════════════════
# 组合优化实验 (贪婪加法)
# ══════════════════════════════════════════════════════════════

def run_combination_tests(data_ctx):
    logger.info("\n" + "=" * 70)
    logger.info("🧬 组合优化实验 (贪婪加法)")
    logger.info("=" * 70)

    z3, fwd, dm, cl, tks, fnames, nd, ns, ds, t2i, d2i, per_symbol = data_ctx

    # 候选改进及其参数组合
    candidates = [
        ("hysteresis", {"use_hysteresis": True, "hysteresis_days": 3, "cooldown_days": 5}),
        ("trailing_stop", {"use_trailing_stop": True, "trailing_profit_pct": 0.15}),
        ("dynamic_weights", {"use_dynamic_weights": True, "dynamic_weight_confidence_thresh": 0.5}),
        ("panic_state", {"use_panic_state": True}),
        ("market_breadth", {"use_market_breadth": True}),
        ("multi_timeframe", {"use_multi_timeframe": True}),
        ("hmm", {"use_hmm": True}),
    ]

    # Step 1: baseline
    base_cfg = ExperimentConfig("combo_baseline")
    base_r = run_experiment(base_cfg, z3, fwd, dm, cl, tks, fnames, nd, ns, ds, t2i, per_symbol)
    best_score = base_r["metrics"]["calmar"]
    best_combo = []
    best_results = [("baseline", base_r)]

    logger.info("  起始 Calmar=%.3f", best_score)

    available = list(candidates)
    while available:
        best_candidate = None
        best_candidate_score = best_score

        for cand_name, cand_params in available:
            cfg = ExperimentConfig(f"combo_{cand_name}")
            for prev_name in best_combo:
                prev_params = dict(candidates)[prev_name] if prev_name in dict(candidates) else {}
                for k, v in prev_params.items():
                    setattr(cfg, k, v)
            for k, v in cand_params.items():
                setattr(cfg, k, v)

            r = run_experiment(cfg, z3, fwd, dm, cl, tks, fnames, nd, ns, ds, t2i, per_symbol)
            score = r["metrics"]["calmar"]

            if score > best_candidate_score:
                best_candidate_score = score
                best_candidate = (cand_name, cand_params, cfg, r)

        if best_candidate and best_candidate_score > best_score * 1.005:
            best_combo.append(best_candidate[0])
            best_score = best_candidate_score
            best_results.append((best_candidate[0], best_candidate[3]))
            logger.info("  +%s → Calmar=%.3f (AR=%.2f%%, Sharpe=%.3f)",
                        best_candidate[0], best_score,
                        best_candidate[3]["metrics"]["annual_return"] * 100,
                        best_candidate[3]["metrics"]["sharpe"])
            available.remove((best_candidate[0], best_candidate[1]))
        else:
            logger.info("  无进一步改善，停止")
            break

    # 输出最终组合
    logger.info("\n  最终最优组合: %s", " + ".join(best_combo))
    logger.info("  Calmar=%.3f", best_score)

    combo_result = {
        "best_combination": best_combo,
        "best_calmar": best_score,
        "results": [(name, r) for name, r in best_results],
    }
    with open(os.path.join(RESULTS_DIR, "combination_results.json"), "w") as f:
        json.dump(combo_result, f, indent=2, ensure_ascii=False)
    logger.info("组合结果已保存")

    return combo_result


# ══════════════════════════════════════════════════════════════
# 报告生成
# ══════════════════════════════════════════════════════════════

def generate_report(single_results, combo_result):
    logger.info("\n生成实验报告...")

    lines = []
    lines.append("# MSS_dynamic 全面优化实验报告")
    lines.append(f"\n**日期**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"\n**数据区间**: {FULL_RANGE[0]} ~ {FULL_RANGE[1]}")
    lines.append("\n---")
    lines.append("\n## 实验概述")
    lines.append("\n对 MSS_dynamic 策略进行 10 项改进的单点消融和组合优化回测。")
    lines.append("\nBaseline 为当前线上版本（与 mss_ab_comparison.py baseline 一致）。")
    lines.append("\n### 改进项")
    improvements = [
        ("#1", "状态切换滞后", "连续 N 天确认 + 切换冷却期"),
        ("#2", "Trailing Stop", "移动止盈 (从高点回撤触发)"),
        ("#3", "置信度动态权重", "低置信度时降低进攻型策略权重"),
        ("#4", "Panic 恐慌状态", "大盘暴跌时空仓/轻仓"),
        ("#5", "市场广度指标", "涨跌比辅助状态判断"),
        ("#6", "多时间框架确认", "周线趋势过滤"),
        ("#7", "行业中性约束", "限制单行业最大持仓数"),
        ("#8", "子策略淘汰", "持续跑输的子策略自动降权"),
        ("#9", "HMM 状态识别", "隐马尔可夫模型识别隐状态"),
        ("#10", "宏观因子叠加", "宏观环境过滤（占位实现）"),
    ]
    lines.append("\n| 编号 | 改进 | 说明 |")
    lines.append("|------|------|------|")
    for num, name, desc in improvements:
        lines.append(f"| {num} | {name} | {desc} |")

    lines.append("\n---")
    lines.append("\n## 单点消融结果")
    lines.append("\n对比 Baseline，每项改进单独开启时的效果。")
    lines.append("\n| 配置 | 年化% | Sharpe | 回撤% | Calmar | vs Baseline年化 |")
    lines.append("|------|-------|--------|-------|--------|----------------|")

    base = single_results[0]["metrics"] if single_results else {}
    for r in single_results:
        m = r["metrics"]
        diff = (m["annual_return"] - base.get("annual_return", 0)) * 100
        lines.append(
            f"| {r['config_name']:<30} | {m['annual_return']*100:>+6.2f}% | "
            f"{m['sharpe']:.3f} | {-m['max_drawdown']*100:.2f}% | "
            f"{m['calmar']:.3f} | {diff:+.2f}% |"
        )

    lines.append("\n### 分窗口表现")
    for window_name in ["全区间", "2022熊市", "OOS修复牛"]:
        lines.append(f"\n**{window_name}**")
        lines.append("\n| 配置 | 年化% | Sharpe | 回撤% |")
        lines.append("|------|-------|--------|-------|")
        for r in single_results:
            w = next((x for x in r.get("windows", []) if x.get("name") == window_name), {})
            if w:
                lines.append(
                    f"| {r['config_name']:<30} | {w.get('annual_return', 0)*100:>+6.2f}% | "
                    f"{w.get('sharpe', 0):.3f} | {-w.get('max_drawdown', 0)*100:.2f}% |")

    lines.append("\n---")
    lines.append("\n## 组合优化结果")
    lines.append("\n贪婪加法：每次添加改善最大的改进项，直到无法继续改善。")
    lines.append(f"\n最优组合: **{' + '.join(combo_result.get('best_combination', []))}**")
    lines.append(f"最优 Calmar: **{combo_result.get('best_calmar', 0):.3f}**")

    lines.append("\n| 步骤 | 新增改进 | 年化% | Sharpe | 回撤% | Calmar |")
    lines.append("|------|----------|-------|--------|-------|--------|")
    for step_name, r in combo_result.get("results", []):
        m = r["metrics"]
        lines.append(
            f"| {step_name} | | {m['annual_return']*100:>+6.2f}% | "
            f"{m['sharpe']:.3f} | {-m['max_drawdown']*100:.2f}% | {m['calmar']:.3f} |")

    lines.append("\n---")
    lines.append("\n## 结论")
    lines.append("\n待实验完成后填入。")

    report_path = os.path.join(SCRIPT_DIR, "experiment_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("报告已生成: %s", report_path)


# ══════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="MSS_dynamic 全面优化实验")
    parser.add_argument("--mode", default="single", choices=["single", "combo", "all"],
                        help="single=单点消融 combo=组合优化 all=全部")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("ZEquant MSS_dynamic 全面优化实验")
    logger.info("开始时间: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("数据库: %s", DB_PATH)
    logger.info("=" * 70)

    t0 = time.time()
    data_ctx = load_data()
    logger.info("数据加载耗时: %.1fs", time.time() - t0)

    single_results = None
    combo_result = None

    if args.mode in ("single", "all"):
        single_results = run_single_point_tests(data_ctx)

    if args.mode in ("combo", "all"):
        combo_result = run_combination_tests(data_ctx)

    if single_results and combo_result:
        generate_report(single_results, combo_result)
    elif single_results:
        generate_report(single_results, {"best_combination": [], "best_calmar": 0, "results": []})
    elif combo_result:
        generate_report([], combo_result)

    logger.info("=" * 70)
    logger.info("实验完成，总耗时: %.1fs", time.time() - t0)
    logger.info("结果目录: %s", RESULTS_DIR)
    logger.info("日志文件: %s", LOG_PATH)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
