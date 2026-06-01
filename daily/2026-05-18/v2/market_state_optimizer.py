"""
MarketStateSelector 动态策略切换 — 深度优化管道。

优化维度：
  Phase 1: 子策略替换 — 用更强的子策略替换弱策略（osr_d10 → chip_equal_d3 / c01_layered_d5）
  Phase 2: 多策略组合 — 尝试每状态2~3个策略的各种组合
  Phase 3: 置信度加权 — 根据信号强度动态调整权重
  Phase 4: 动量过滤 — 全局动量调整总仓位
  Phase 5: 综合最优 — 组合所有最佳技巧

用法：
    python3 daily/2026-05-18/v2/market_state_optimizer.py --phase 1
    python3 daily/2026-05-18/v2/market_state_optimizer.py --phase 2
    python3 daily/2026-05-18/v2/market_state_optimizer.py --phase 3
    python3 daily/2026-05-18/v2/market_state_optimizer.py --phase 4
    python3 daily/2026-05-18/v2/market_state_optimizer.py --phase 5
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from core.database import Database
from core.positioners import RPPortfolioWeights

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("mss_optimizer")

V2_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(V2_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

TX = 0.0012

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

WINDOWS = [
    ("全区间",     "2019-01-02", "2026-04-30"),
    ("2019修复牛", "2019-01-02", "2019-12-31"),
    ("2020疫情",   "2020-01-02", "2020-12-31"),
    ("2021结构牛", "2021-01-04", "2021-12-31"),
    ("2022熊市",   "2022-01-04", "2022-12-30"),
    ("2023震荡",   "2023-01-03", "2023-12-29"),
    ("2024反弹",   "2024-01-02", "2024-12-31"),
    ("2025至今",   "2025-01-02", "2026-04-30"),
    ("OOS修复牛",  "2024-07-01", "2026-04-30"),
]

# ═══════════════════════════════════════════════
# 数据加载 & 信号构建（复用 market_state_pipeline 的逻辑）
# ═══════════════════════════════════════════════

import duckdb


def _get_conn():
    src = os.path.abspath("./data/quant_data.db")
    return duckdb.connect(src, read_only=True)


def load_data():
    conn = _get_conn()
    all_cols = [
        r[0] for r in conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name='factors_wide'"
        ).fetchall()
    ]
    available = [c for c in ALL_FACTORS if c in all_cols]

    factor_cols = ", ".join([f'f."{c}"' for c in available])
    df = conn.execute(f"""
        SELECT f.date, f.symbol, b.close, {factor_cols}
        FROM factors_wide f
        LEFT JOIN daily_bars b ON f.date = b.date AND f.symbol = b.symbol
        WHERE f.date >= '2018-01-01' AND f.date <= '2026-04-30'
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

    di = np.array([d2i[d] for d in df['date']], dtype=np.int32)
    si = np.array([t2i.get(s, -1) for s in df['symbol']], dtype=np.int32)
    v = si >= 0
    di, si = di[v], si[v]

    for fi, fc in enumerate(available):
        if fc in df.columns:
            v3[di, si, fi] = df[fc].values[v].astype(np.float32)
    cl[di, si] = df['close'].values[v].astype(np.float32)
    dm[di, si] = True
    np.nan_to_num(v3, nan=0.0, copy=False)
    np.nan_to_num(cl, nan=0.0, copy=False)

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

    logger.info(f"数据: {nd}天 × {ns}只 × {nf}因子")
    conn.close()
    return z3, fwd, dm, cl, tks, available, nd, ns, ds


def load_ga_weights():
    cfg_path = os.path.join(
        os.path.dirname(__file__), '..', '..', '..',
        'core', 'strategies', 'impl', 'v1_ga_rp', 'config.json',
    )
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        return cfg.get("selector", {}).get("weights", {})
    return {}


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
        "mf": mf,
        "chip": chip_sig,
        "osr": osr_sig,
        "vol_p": vol_p,
        "trend_p": trend_p,
        "fi": fi,
        "market_index": mkt_idx,
        "close": cl,
    }


# ═══════════════════════════════════════════════
# 子策略回测
# ═══════════════════════════════════════════════

def bt_sub_strategy(sig, fwd, dm, rebal_freq=10, top_n=50, min_hold_days=10,
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
            pr = pos_ratio[i] if pos_ratio is not None else 1.0
            nw = alloc.allocate(sig[i], fwd, i, pw, hs, rh)
            to = float(np.sum(np.abs(nw - pw)))
            txc = 0.5 * to * TX
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


def run_mss_unified_backtest(state_strategies, sub_drs, states, nd, rebal_freq=10):
    """统一间隔回测：每 rebal_freq 天检测状态并分配权重，中间固定持有。"""
    dr = np.zeros(nd, dtype=np.float64)

    # 每期的起始权重
    block_weights = {}
    for i in range(1, nd):
        if i % rebal_freq == 0 or i == 1:
            state = states[i] if i < len(states) else "oscillate"
            allocs = state_strategies.get(state, state_strategies.get("oscillate", []))
            block_weights = {}
            for a in allocs:
                block_weights[a["strategy"]] = a["weight"]
            tw = sum(block_weights.values()) or 1.0
            block_weights = {k: v/tw for k, v in block_weights.items()}

        combined_ret = 0.0
        total_w = 0.0
        for name, w in block_weights.items():
            if name in sub_drs:
                combined_ret += w * sub_drs[name][i]
                total_w += w

        dr[i] = combined_ret / total_w if total_w > 0 else 0.0

    return dr


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

    mdd_idx = np.argmin(dd)
    rec = np.where(eq[mdd_idx:] >= cm[mdd_idx])[0]
    recovery_days = int(rec[0]) if len(rec) > 0 else nd - mdd_idx - 1

    peak_idx = np.argmax(cm[:mdd_idx + 1] == cm[mdd_idx])
    drawdown_duration = int(mdd_idx - peak_idx)

    cal = ar / abs(mdd) if abs(mdd) > 0 else 0
    wr = int(np.sum(dr > 0)) / max(int(np.sum(dr > 0)) + int(np.sum(dr < 0)), 1)

    # 综合评分（与INDEX.md一致）
    def s_r(v): return min(100, max(0, v / 0.50 * 100))
    def s_s(v): return min(100, max(0, v / 2.0 * 100))
    def s_dd(v): return min(100, max(0, (1.0 - v / 0.50) * 100))
    def s_rec(v): return min(100, max(0, (1.0 - v / 500) * 100))

    score = (s_r(ar) * 0.20 + s_s(sp) * 0.20 +
             s_dd(abs(mdd)) * 0.20 + s_rec(recovery_days) * 0.15)

    return {
        "name": name,
        "annual_return": round(float(ar), 4),
        "sharpe": round(float(sp), 4),
        "max_drawdown": round(float(mdd), 4),
        "calmar": round(float(cal), 4),
        "win_rate": round(float(wr), 4),
        "recovery_days": int(recovery_days),
        "drawdown_duration": int(drawdown_duration),
        "composite_score": round(float(score), 1),
    }


def window_analysis(dr, ds, windows):
    results = []
    for wname, wstart, wend in windows:
        start_dt = pd.Timestamp(wstart)
        end_dt = pd.Timestamp(wend)
        idx = [i for i, d in enumerate(ds) if start_dt <= d <= end_dt]
        if not idx:
            results.append({"name": wname, "n_days": 0})
            continue
        sub_dr = dr[idx[0]:idx[-1] + 1]
        m = compute_metrics(sub_dr, name=wname)
        m["n_days"] = len(sub_dr)
        results.append(m)
    return results


def print_results_table(results, title=""):
    if title:
        print(f"\n{'=' * 110}")
        print(f"  {title}")
        print(f"{'=' * 110}")
    print(f"{'策略':<30} {'年化%':<7} {'Sharpe':<7} {'回撤%':<7} {'Calmar':<7} {'修复(d)':<7} {'评分':<6} {'评级'}")
    print('-' * 110)
    for r in sorted(results, key=lambda x: x.get('composite_score', 0), reverse=True):
        ar = r.get('annual_return', 0) * 100
        sp = r.get('sharpe', 0)
        dd = abs(r.get('max_drawdown', 0)) * 100
        ca = r.get('calmar', 0)
        rd = r.get('recovery_days', 9999)
        sc = r.get('composite_score', 0)
        cls = "🏆" if sc >= 75 else ("✅" if sc >= 60 else ("⚠️" if sc >= 40 else "❌"))
        print(f"{cls} {r['name']:<28} {ar:>6.2f}% {sp:>6.3f} {dd:>6.2f}% {ca:>6.3f} {rd:>4}d {sc:>5.1f}")
    print('=' * 110)


# ═══════════════════════════════════════════════
# 市场状态检测
# ═══════════════════════════════════════════════

def detect_market_state(mkt_returns, nd, bull_easy=True):
    idx_price = np.zeros(nd, dtype=np.float64)
    idx_price[0] = 1000.0
    for i in range(1, nd):
        idx_price[i] = idx_price[i - 1] * (1.0 + mkt_returns[i])

    ma5 = pd.Series(idx_price).rolling(5).mean().values
    ma20 = pd.Series(idx_price).rolling(20).mean().values
    ma60 = pd.Series(idx_price).rolling(60).mean().values
    ma200 = pd.Series(idx_price).rolling(200).mean().values

    states = ["oscillate"] * nd
    conf = [0.0] * nd  # 置信度

    for i in range(nd):
        if pd.isna(ma200[i]) or ma200[i] == 0:
            states[i] = "oscillate"
            continue

        close = idx_price[i]
        above_ma200 = (close - ma200[i]) / ma200[i]

        lookback5 = min(5, i)
        ma5_slope = (ma5[i] - ma5[i - lookback5]) / ma5[i - lookback5] if (lookback5 >= 2 and ma5[i - lookback5] != 0) else 0.0

        lookback20 = min(20, i)
        ma20_slope = (ma20[i] - ma20[i - lookback20]) / ma20[i - lookback20] if (lookback20 >= 2 and ma20[i - lookback20] != 0) else 0.0

        lookback60 = min(60, i)
        ma60_slope = (ma60[i] - ma60[i - lookback60]) / ma60[i - lookback60] if (lookback60 >= 2 and ma60[i - lookback60] != 0) else 0.0

        bull_slope_thresh = -0.001 if bull_easy else 0.0

        if above_ma200 > 0 and ma20_slope > bull_slope_thresh:
            states[i] = "bull"
            conf[i] = min(1.0, above_ma200 * 2 + ma20_slope * 20)
        elif above_ma200 < 0 and ma20_slope < 0 and ma60_slope < 0:
            states[i] = "bear"
            conf[i] = min(1.0, abs(above_ma200) * 2 + abs(ma20_slope) * 10 + abs(ma60_slope) * 10)
        elif above_ma200 < 0 and ma5_slope > 0.005:
            states[i] = "recovery"
            conf[i] = ma5_slope * 50
        elif pd.notna(ma5[i]) and pd.notna(ma20[i]) and pd.notna(ma60[i]):
            spread = abs(ma5[i] - ma20[i]) / max(abs(ma20[i]), 1e-10) + abs(ma20[i] - ma60[i]) / max(abs(ma60[i]), 1e-10)
            if spread < 0.03:
                states[i] = "oscillate"
                conf[i] = 1.0 - spread * 15
            elif above_ma200 < 0 and ma5_slope > 0:
                states[i] = "recovery"
                conf[i] = ma5_slope * 30
            else:
                states[i] = "oscillate"
                conf[i] = 0.3
        else:
            states[i] = "oscillate"
            conf[i] = 0.3

    return states, conf


# ═══════════════════════════════════════════════
# MSS 回测引擎
# ═══════════════════════════════════════════════

def run_mss_backtest(state_strategies, sub_drs, states, nd, conf=None, rebal_cost=0.001):
    dr = np.zeros(nd, dtype=np.float64)
    for i in range(1, nd):
        state = states[i] if i < len(states) else "oscillate"
        allocs = state_strategies.get(state, state_strategies.get("oscillate", []))

        c = conf[i] if conf is not None else 1.0

        combined_ret = 0.0
        total_w = 0.0
        for a in allocs:
            name = a["strategy"]
            w = a["weight"]
            w_adj = w * (0.5 + 0.5 * c) if a.get("confidence_adjust", False) else w
            if name in sub_drs:
                combined_ret += w_adj * sub_drs[name][i]
                total_w += w_adj

        dr[i] = combined_ret / total_w if total_w > 0 else 0.0

    return dr


def run_mss_momentum_backtest(state_strategies, sub_drs, states, nd,
                               mkt_returns, momentum_lookback=20):
    """带动量过滤的 MSS：当市场动量弱时降低总仓位。"""
    idx_price = np.zeros(nd, dtype=np.float64)
    idx_price[0] = 1000.0
    for i in range(1, nd):
        idx_price[i] = idx_price[i - 1] * (1.0 + mkt_returns[i])

    m20 = pd.Series(idx_price).rolling(momentum_lookback).mean().values

    dr = np.zeros(nd, dtype=np.float64)
    for i in range(1, nd):
        state = states[i] if i < len(states) else "oscillate"
        allocs = state_strategies.get(state, state_strategies.get("oscillate", []))

        # 动量系数：当前价/20日均线，越低仓位越低
        mom_ratio = idx_price[i] / m20[i] if m20[i] > 0 else 1.0
        mom_ratio = np.clip(mom_ratio, 0.5, 1.0)

        combined_ret = 0.0
        total_w = 0.0
        for a in allocs:
            name = a["strategy"]
            w = a["weight"] * mom_ratio
            if name in sub_drs:
                combined_ret += w * sub_drs[name][i]
                total_w += w

        dr[i] = combined_ret / total_w if total_w > 0 else 0.0

    return dr


# ═══════════════════════════════════════════════
# 扩展子策略参数
# ═══════════════════════════════════════════════

def get_extended_strategy_params():
    """扩展的子策略参数，从INDEX.md选出更强的策略。"""
    return {
        # 核心策略（MF系列）
        "mf_d10_rp":       {"signal": "mf",     "rf": 10, "tn": 50, "mhd": 10, "timing": None},
        "mf_vol_d10_rp":   {"signal": "mf",     "rf": 10, "tn": 50, "mhd": 10, "timing": "vol"},

        # Chip系列（防御核心）
        "chip_rp":         {"signal": "chip",   "rf": 3,  "tn": 40, "mhd": 5,  "timing": None},
        "chip_covrp":      {"signal": "chip",   "rf": 3,  "tn": 40, "mhd": 5,  "timing": None},

        # 新增强策略
        "chip_equal_d3":   {"signal": "chip",   "rf": 3,  "tn": 40, "mhd": 3,  "timing": None},
        "chip_vol_rp":     {"signal": "chip",   "rf": 3,  "tn": 40, "mhd": 5,  "timing": "vol"},

        # 特殊策略
        "osr_d10":         {"signal": "osr",    "rf": 10, "tn": 40, "mhd": 5,  "timing": None},
        "c01_layered_d5":  {"signal": "mf",     "rf": 5,  "tn": 40, "mhd": 5,  "timing": "trend"},

        # 基础参考
        "mf_base":         {"signal": "mf",     "rf": 3,  "tn": 40, "mhd": 5,  "timing": None},
    }


def compute_all_sub_drs(signals, fwd, dm, nd):
    """预计算所有子策略日收益率。"""
    sub_params = get_extended_strategy_params()
    sub_drs = {}

    for name, params in sub_params.items():
        sig = signals[params["signal"]]
        pr = None
        if params["timing"] == "vol":
            pr = signals["vol_p"]
        elif params["timing"] == "trend":
            pr = signals["trend_p"]

        dr = bt_sub_strategy(
            sig, fwd, dm,
            rebal_freq=params["rf"],
            top_n=params["tn"],
            min_hold_days=params["mhd"],
            pos_ratio=pr,
        )
        sub_drs[name] = dr

    # Combo 策略
    sub_drs["mf50_chip50"] = 0.5 * sub_drs["mf_base"] + 0.5 * sub_drs["chip_rp"]
    sub_drs["mf60_chip40"] = 0.6 * sub_drs["mf_base"] + 0.4 * sub_drs["chip_rp"]

    return sub_drs


# ═══════════════════════════════════════════════
# 分配方案定义
# ═══════════════════════════════════════════════

ALLOCATION_VARIANTS = {
    # ═══ V5系列：替换弱子策略 ═══
    "V5_original": {
        "bull": [("mf_d10_rp", 0.7), ("chip_rp", 0.3)],
        "bear": [("chip_covrp", 0.7), ("mf_vol_d10_rp", 0.3)],
        "oscillate": [("mf50_chip50", 0.5), ("chip_covrp", 0.5)],
        "recovery": [("mf60_chip40", 0.6), ("osr_d10", 0.4)],
    },
    "V5a_recovery_eq": {
        "bull": [("mf_d10_rp", 0.7), ("chip_rp", 0.3)],
        "bear": [("chip_covrp", 0.7), ("mf_vol_d10_rp", 0.3)],
        "oscillate": [("mf50_chip50", 0.5), ("chip_covrp", 0.5)],
        "recovery": [("mf60_chip40", 0.6), ("chip_equal_d3", 0.4)],  # osr→chip_equal_d3
    },
    "V5b_recovery_c01": {
        "bull": [("mf_d10_rp", 0.7), ("chip_rp", 0.3)],
        "bear": [("chip_covrp", 0.7), ("mf_vol_d10_rp", 0.3)],
        "oscillate": [("mf50_chip50", 0.5), ("chip_covrp", 0.5)],
        "recovery": [("mf60_chip40", 0.6), ("c01_layered_d5", 0.4)],  # osr→c01
    },
    "V5c_bull_covrp": {
        "bull": [("mf_d10_rp", 0.7), ("chip_covrp", 0.3)],  # chip_rp→chip_covrp
        "bear": [("chip_covrp", 0.7), ("mf_vol_d10_rp", 0.3)],
        "oscillate": [("mf50_chip50", 0.5), ("chip_covrp", 0.5)],
        "recovery": [("mf60_chip40", 0.6), ("osr_d10", 0.4)],
    },
    "V5d_bear_eq": {
        "bull": [("mf_d10_rp", 0.7), ("chip_rp", 0.3)],
        "bear": [("chip_equal_d3", 0.7), ("mf_vol_d10_rp", 0.3)],  # covrp→equal_d3
        "oscillate": [("mf50_chip50", 0.5), ("chip_covrp", 0.5)],
        "recovery": [("mf60_chip40", 0.6), ("osr_d10", 0.4)],
    },
    "V5e_all_chip_equal": {
        "bull": [("mf_d10_rp", 0.7), ("chip_equal_d3", 0.3)],
        "bear": [("chip_equal_d3", 0.7), ("mf_vol_d10_rp", 0.3)],
        "oscillate": [("mf50_chip50", 0.5), ("chip_equal_d3", 0.5)],
        "recovery": [("mf60_chip40", 0.6), ("chip_equal_d3", 0.4)],
    },
    "V5f_c01_oscillate": {
        "bull": [("mf_d10_rp", 0.7), ("chip_rp", 0.3)],
        "bear": [("chip_covrp", 0.7), ("mf_vol_d10_rp", 0.3)],
        "oscillate": [("c01_layered_d5", 0.5), ("chip_covrp", 0.5)],  # mf50→c01
        "recovery": [("mf60_chip40", 0.6), ("osr_d10", 0.4)],
    },

    # ═══ V6系列：3策略/状态（丰富配置） ═══
    "V6a_3way_bull_heavy": {
        "bull": [("mf_d10_rp", 0.6), ("mf_vol_d10_rp", 0.2), ("chip_covrp", 0.2)],
        "bear": [("chip_covrp", 0.6), ("chip_equal_d3", 0.2), ("mf_vol_d10_rp", 0.2)],
        "oscillate": [("chip_covrp", 0.4), ("mf50_chip50", 0.3), ("c01_layered_d5", 0.3)],
        "recovery": [("chip_equal_d3", 0.4), ("mf60_chip40", 0.3), ("mf_vol_d10_rp", 0.3)],
    },
    "V6b_3way_defense": {
        "bull": [("mf_d10_rp", 0.5), ("chip_covrp", 0.3), ("mf_vol_d10_rp", 0.2)],
        "bear": [("chip_covrp", 0.5), ("chip_equal_d3", 0.3), ("mf_vol_d10_rp", 0.2)],
        "oscillate": [("chip_covrp", 0.4), ("chip_equal_d3", 0.3), ("c01_layered_d5", 0.3)],
        "recovery": [("chip_equal_d3", 0.4), ("mf_vol_d10_rp", 0.3), ("c01_layered_d5", 0.3)],
    },
    "V6c_3way_c01_core": {
        "bull": [("mf_d10_rp", 0.5), ("chip_covrp", 0.3), ("c01_layered_d5", 0.2)],
        "bear": [("chip_covrp", 0.5), ("c01_layered_d5", 0.3), ("mf_vol_d10_rp", 0.2)],
        "oscillate": [("c01_layered_d5", 0.4), ("chip_covrp", 0.3), ("mf50_chip50", 0.3)],
        "recovery": [("mf60_chip40", 0.4), ("c01_layered_d5", 0.3), ("osr_d10", 0.3)],
    },

    # ═══ V7系列：极简（精简版） ═══
    "V7a_minimal": {
        "bull": [("mf_d10_rp", 0.8), ("chip_covrp", 0.2)],
        "bear": [("chip_covrp", 0.8), ("mf_vol_d10_rp", 0.2)],
        "oscillate": [("chip_covrp", 0.6), ("mf50_chip50", 0.4)],
        "recovery": [("mf_d10_rp", 0.6), ("chip_equal_d3", 0.4)],
    },
}


# ═══════════════════════════════════════════════
# 实验入口
# ═══════════════════════════════════════════════

def run_phase_base():
    """基础：加载数据 + 预计算子策略 + 打印子策略表现。"""
    logger.info("=" * 60)
    logger.info("Phase 0: 加载数据 & 子策略")
    logger.info("=" * 60)
    z3, fwd, dm, cl, tks, fnames, nd, ns, ds = load_data()
    signals = build_signals(z3, fwd, dm, cl, fnames, nd, ns, ds)
    sub_drs = compute_all_sub_drs(signals, fwd, dm, nd)

    sub_metrics = []
    for name, dr in sorted(sub_drs.items()):
        m = compute_metrics(dr, name=name)
        sub_metrics.append(m)
        logger.info(f"  {name}: 年化={m['annual_return']*100:.2f}% "
                    f"Sharpe={m['sharpe']:.3f} 回撤={m['max_drawdown']*100:.2f}% "
                    f"评分={m['composite_score']}")

    print_results_table(sub_metrics, "子策略表现一览")

    return z3, fwd, dm, cl, tks, fnames, nd, ns, ds, signals, sub_drs


def run_phase1():
    """Phase 1: 替换弱子策略 — 系统对比V5系列变种。"""
    logger.info("=" * 60)
    logger.info("Phase 1: 替换弱子策略")
    logger.info("=" * 60)

    z3, fwd, dm, cl, tks, fnames, nd, ns, ds = load_data()
    signals = build_signals(z3, fwd, dm, cl, fnames, nd, ns, ds)
    sub_drs = compute_all_sub_drs(signals, fwd, dm, nd)

    mkt_idx = signals["market_index"]
    states, _ = detect_market_state(mkt_idx, nd, bull_easy=True)

    results = []
    variant_keys = [k for k in ALLOCATION_VARIANTS.keys() if k.startswith("V5")]

    for vk in variant_keys:
        v = ALLOCATION_VARIANTS[vk]
        ssm = {s: [{"strategy": n, "weight": w} for n, w in v[s]]
               for s in ["bull", "bear", "oscillate", "recovery"]}

        dr = run_mss_backtest(ssm, sub_drs, states, nd)
        m = compute_metrics(dr, name=vk)
        results.append(m)
        logger.info(f"  {vk}: 年化={m['annual_return']*100:.2f}% "
                    f"Sharpe={m['sharpe']:.3f} 回撤={m['max_drawdown']*100:.2f}% "
                    f"评分={m['composite_score']}")

    print_results_table(results, "Phase 1: 子策略替换 (V5变种)")

    out_path = os.path.join(RESULTS_DIR, "phase1_sub_swap.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("已保存到 %s", out_path)

    return results


def run_phase2():
    """Phase 2: 多策略组合 — 尝试各种状态映射方案。"""
    logger.info("=" * 60)
    logger.info("Phase 2: 多策略组合探索")
    logger.info("=" * 60)

    z3, fwd, dm, cl, tks, fnames, nd, ns, ds = load_data()
    signals = build_signals(z3, fwd, dm, cl, fnames, nd, ns, ds)
    sub_drs = compute_all_sub_drs(signals, fwd, dm, nd)

    mkt_idx = signals["market_index"]
    states, _ = detect_market_state(mkt_idx, nd, bull_easy=True)

    results = []
    for vk, v in ALLOCATION_VARIANTS.items():
        ssm = {s: [{"strategy": n, "weight": w} for n, w in v[s]]
               for s in ["bull", "bear", "oscillate", "recovery"]}

        dr = run_mss_backtest(ssm, sub_drs, states, nd)
        m = compute_metrics(dr, name=vk)
        results.append(m)
        logger.info(f"  {vk}: 年化={m['annual_return']*100:.2f}% "
                    f"Sharpe={m['sharpe']:.3f} 回撤={m['max_drawdown']*100:.2f}% "
                    f"评分={m['composite_score']}")

    print_results_table(results, "Phase 2: 所有分配方案对比")

    out_path = os.path.join(RESULTS_DIR, "phase2_allocation.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("已保存到 %s", out_path)

    return results


def run_phase3():
    """Phase 3: 置信度加权 — 根据信号强度动态调整分配权重。"""
    logger.info("=" * 60)
    logger.info("Phase 3: 置信度加权分配")
    logger.info("=" * 60)

    z3, fwd, dm, cl, tks, fnames, nd, ns, ds = load_data()
    signals = build_signals(z3, fwd, dm, cl, fnames, nd, ns, ds)
    sub_drs = compute_all_sub_drs(signals, fwd, dm, nd)

    mkt_idx = signals["market_index"]
    states, conf = detect_market_state(mkt_idx, nd, bull_easy=True)

    results = []
    for vk, v in ALLOCATION_VARIANTS.items():
        if not vk.startswith("V5"):
            continue

        ssm = {s: [{"strategy": n, "weight": w, "confidence_adjust": True}
                   for n, w in v[s]]
               for s in ["bull", "bear", "oscillate", "recovery"]}

        dr = run_mss_backtest(ssm, sub_drs, states, nd, conf=conf)
        m = compute_metrics(dr, name=f"{vk}_conf")
        results.append(m)
        logger.info(f"  {vk}_conf: 年化={m['annual_return']*100:.2f}% "
                    f"Sharpe={m['sharpe']:.3f} 回撤={m['max_drawdown']*100:.2f}%"
                    f" 评分={m['composite_score']}")

    # 对比：不加置信度的版本
    for vk, v in ALLOCATION_VARIANTS.items():
        if not vk.startswith("V5"):
            continue

        ssm = {s: [{"strategy": n, "weight": w} for n, w in v[s]]
               for s in ["bull", "bear", "oscillate", "recovery"]}

        dr = run_mss_backtest(ssm, sub_drs, states, nd)
        m = compute_metrics(dr, name=f"{vk}_plain")
        results.append(m)

    print_results_table(results, "Phase 3: 置信度加权对比")

    # 置信度统计
    conf_by_state = {"bull": [], "bear": [], "oscillate": [], "recovery": []}
    for i in range(nd):
        s = states[i]
        if s in conf_by_state:
            conf_by_state[s].append(conf[i])

    logger.info("置信度统计:")
    for s, vals in conf_by_state.items():
        if vals:
            logger.info(f"  {s}: 均值={np.mean(vals):.3f} 中位={np.median(vals):.3f} "
                        f"范围=[{min(vals):.3f}, {max(vals):.3f}]")

    out_path = os.path.join(RESULTS_DIR, "phase3_confidence.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("已保存到 %s", out_path)

    return results


def run_phase4():
    """Phase 4: 动量过滤器 — 在动态切换上层加全局动量风控。"""
    logger.info("=" * 60)
    logger.info("Phase 4: 动量过滤器")
    logger.info("=" * 60)

    z3, fwd, dm, cl, tks, fnames, nd, ns, ds = load_data()
    signals = build_signals(z3, fwd, dm, cl, fnames, nd, ns, ds)
    sub_drs = compute_all_sub_drs(signals, fwd, dm, nd)

    mkt_idx = signals["market_index"]
    states, conf = detect_market_state(mkt_idx, nd, bull_easy=True)

    results = []

    # 测试V5_original + 不同动量lookback
    best_vk = "V5_original"

    for lookback in [10, 20, 30, 60]:
        v = ALLOCATION_VARIANTS[best_vk]
        ssm = {s: [{"strategy": n, "weight": w} for n, w in v[s]]
               for s in ["bull", "bear", "oscillate", "recovery"]}

        dr = run_mss_momentum_backtest(ssm, sub_drs, states, nd, mkt_idx, momentum_lookback=lookback)
        m = compute_metrics(dr, name=f"V5_mom{lookback}")
        results.append(m)
        logger.info(f"  V5_mom{lookback}: 年化={m['annual_return']*100:.2f}% "
                    f"Sharpe={m['sharpe']:.3f} 回撤={m['max_drawdown']*100:.2f}%"
                    f" 评分={m['composite_score']}")

    # 对最佳变种加动量过滤
    variants_to_test = ["V5a_recovery_eq", "V5b_recovery_c01", "V5c_bull_covrp",
                         "V5e_all_chip_equal", "V6a_3way_bull_heavy"]
    for vk in variants_to_test:
        if vk not in ALLOCATION_VARIANTS:
            continue
        v = ALLOCATION_VARIANTS[vk]
        ssm = {s: [{"strategy": n, "weight": w} for n, w in v[s]]
               for s in ["bull", "bear", "oscillate", "recovery"]}

        dr = run_mss_momentum_backtest(ssm, sub_drs, states, nd, mkt_idx, momentum_lookback=20)
        m = compute_metrics(dr, name=f"{vk}_mom20")
        results.append(m)
        logger.info(f"  {vk}_mom20: 年化={m['annual_return']*100:.2f}% "
                    f"Sharpe={m['sharpe']:.3f} 回撤={m['max_drawdown']*100:.2f}%"
                    f" 评分={m['composite_score']}")

    print_results_table(results, "Phase 4: 动量过滤")

    out_path = os.path.join(RESULTS_DIR, "phase4_momentum.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("已保存到 %s", out_path)

    return results


def run_phase_compare():
    """对比实验：统一10天 vs 子策略各自周期。"""
    logger.info("=" * 60)
    logger.info("Phase Compare: 统一10天 vs 子策略各自周期")
    logger.info("=" * 60)

    z3, fwd, dm, cl, tks, fnames, nd, ns, ds = load_data()
    signals = build_signals(z3, fwd, dm, cl, fnames, nd, ns, ds)
    sub_drs = compute_all_sub_drs(signals, fwd, dm, nd)

    mkt_idx = signals["market_index"]
    states, conf = detect_market_state(mkt_idx, nd, bull_easy=True)

    results = []

    for cname, alloc in {
        "V5_original": {
            "bull": [("mf_d10_rp", 0.7), ("chip_rp", 0.3)],
            "bear": [("chip_covrp", 0.7), ("mf_vol_d10_rp", 0.3)],
            "oscillate": [("mf50_chip50", 0.5), ("chip_covrp", 0.5)],
            "recovery": [("mf60_chip40", 0.6), ("osr_d10", 0.4)],
        },
        "V6a_3way": {
            "bull": [("mf_d10_rp", 0.6), ("mf_vol_d10_rp", 0.2), ("chip_covrp", 0.2)],
            "bear": [("chip_covrp", 0.6), ("chip_equal_d3", 0.2), ("mf_vol_d10_rp", 0.2)],
            "oscillate": [("chip_covrp", 0.4), ("mf50_chip50", 0.3), ("c01_layered_d5", 0.3)],
            "recovery": [("chip_equal_d3", 0.4), ("mf60_chip40", 0.3), ("mf_vol_d10_rp", 0.3)],
        },
    }.items():
        ssm = {s: [{"strategy": n, "weight": w} for n, w in alloc[s]]
               for s in ["bull", "bear", "oscillate", "recovery"]}

        # A: 子策略各自周期（每日状态检测）
        dr_daily = run_mss_backtest(ssm, sub_drs, states, nd)
        m_daily = compute_metrics(dr_daily, name=f"{cname}_每日状态")

        # B: 统一10天（每10天检测一次状态）
        dr_uni = run_mss_unified_backtest(ssm, sub_drs, states, nd, rebal_freq=10)
        m_uni = compute_metrics(dr_uni, name=f"{cname}_统一10天")

        # C: 统一5天
        dr_uni5 = run_mss_unified_backtest(ssm, sub_drs, states, nd, rebal_freq=5)
        m_uni5 = compute_metrics(dr_uni5, name=f"{cname}_统一5天")

        # D: 统一3天
        dr_uni3 = run_mss_unified_backtest(ssm, sub_drs, states, nd, rebal_freq=3)
        m_uni3 = compute_metrics(dr_uni3, name=f"{cname}_统一3天")

        results += [m_daily, m_uni, m_uni5, m_uni3]

        for m in [m_daily, m_uni, m_uni5, m_uni3]:
            w = window_analysis(_get_dr_by_name(m['name'], {
                f"{cname}_每日状态": dr_daily,
                f"{cname}_统一10天": dr_uni,
                f"{cname}_统一5天": dr_uni5,
                f"{cname}_统一3天": dr_uni3,
            }), ds, WINDOWS)
            pos = sum(1 for ww in w if ww.get("annual_return", 0) > 0 and ww.get("n_days", 0) > 0)
            total = sum(1 for ww in w if ww.get("n_days", 0) > 0)
            m["windows_positive"] = f"{pos}/{total}"

        logger.info(f"  {cname}:")
        logger.info(f"    每日状态: 年化={m_daily['annual_return']*100:.2f}% Sharpe={m_daily['sharpe']:.3f} 回撤={m_daily['max_drawdown']*100:.2f}% 窗口={m_daily['windows_positive']}")
        logger.info(f"    统一10天: 年化={m_uni['annual_return']*100:.2f}% Sharpe={m_uni['sharpe']:.3f} 回撤={m_uni['max_drawdown']*100:.2f}% 窗口={m_uni['windows_positive']}")
        logger.info(f"    统一5天:  年化={m_uni5['annual_return']*100:.2f}% Sharpe={m_uni5['sharpe']:.3f} 回撤={m_uni5['max_drawdown']*100:.2f}% 窗口={m_uni5['windows_positive']}")
        logger.info(f"    统一3天:  年化={m_uni3['annual_return']*100:.2f}% Sharpe={m_uni3['sharpe']:.3f} 回撤={m_uni3['max_drawdown']*100:.2f}% 窗口={m_uni3['windows_positive']}")

    print_results_table(results, "对比: 每日状态 vs 统一间隔")

    out_path = os.path.join(RESULTS_DIR, "phase_compare.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("已保存到 %s", out_path)
    return results


def _get_dr_by_name(name, dr_map):
    for k, v in dr_map.items():
        if k == name:
            return v
    return None


def run_phase5():
    """Phase 5: 综合最优 — 组合所有最佳配置做完整窗口分析。"""
    logger.info("=" * 60)
    logger.info("Phase 5: 综合最优验证")
    logger.info("=" * 60)

    z3, fwd, dm, cl, tks, fnames, nd, ns, ds = load_data()
    signals = build_signals(z3, fwd, dm, cl, fnames, nd, ns, ds)
    sub_drs = compute_all_sub_drs(signals, fwd, dm, nd)

    mkt_idx = signals["market_index"]
    states, conf = detect_market_state(mkt_idx, nd, bull_easy=True)

    all_results = []

    # 候选方案：基于前4个phases的最佳结果
    candidates = {
        "MSS_v5_original": {  # 原始最好
            "bull": [("mf_d10_rp", 0.7), ("chip_rp", 0.3)],
            "bear": [("chip_covrp", 0.7), ("mf_vol_d10_rp", 0.3)],
            "oscillate": [("mf50_chip50", 0.5), ("chip_covrp", 0.5)],
            "recovery": [("mf60_chip40", 0.6), ("osr_d10", 0.4)],
        },
        "MSS_v5a_rec_eq": {  # recovery使用chip_equal_d3
            "bull": [("mf_d10_rp", 0.7), ("chip_rp", 0.3)],
            "bear": [("chip_covrp", 0.7), ("mf_vol_d10_rp", 0.3)],
            "oscillate": [("mf50_chip50", 0.5), ("chip_covrp", 0.5)],
            "recovery": [("mf60_chip40", 0.6), ("chip_equal_d3", 0.4)],
        },
        "MSS_v5c_bull_covrp": {  # bull使用chip_covrp
            "bull": [("mf_d10_rp", 0.7), ("chip_covrp", 0.3)],
            "bear": [("chip_covrp", 0.7), ("mf_vol_d10_rp", 0.3)],
            "oscillate": [("mf50_chip50", 0.5), ("chip_covrp", 0.5)],
            "recovery": [("mf60_chip40", 0.6), ("osr_d10", 0.4)],
        },
        "MSS_v6a_3way": {  # 3策略/状态
            "bull": [("mf_d10_rp", 0.6), ("mf_vol_d10_rp", 0.2), ("chip_covrp", 0.2)],
            "bear": [("chip_covrp", 0.6), ("chip_equal_d3", 0.2), ("mf_vol_d10_rp", 0.2)],
            "oscillate": [("chip_covrp", 0.4), ("mf50_chip50", 0.3), ("c01_layered_d5", 0.3)],
            "recovery": [("chip_equal_d3", 0.4), ("mf60_chip40", 0.3), ("mf_vol_d10_rp", 0.3)],
        },
        "MSS_v5e_all_eq": {  # 全系chip_equal_d3
            "bull": [("mf_d10_rp", 0.7), ("chip_equal_d3", 0.3)],
            "bear": [("chip_equal_d3", 0.7), ("mf_vol_d10_rp", 0.3)],
            "oscillate": [("mf50_chip50", 0.5), ("chip_equal_d3", 0.5)],
            "recovery": [("mf60_chip40", 0.6), ("chip_equal_d3", 0.4)],
        },
        "MSS_v6b_3way_def": {  # 3策略防御型
            "bull": [("mf_d10_rp", 0.5), ("chip_covrp", 0.3), ("mf_vol_d10_rp", 0.2)],
            "bear": [("chip_covrp", 0.5), ("chip_equal_d3", 0.3), ("mf_vol_d10_rp", 0.2)],
            "oscillate": [("chip_covrp", 0.4), ("chip_equal_d3", 0.3), ("c01_layered_d5", 0.3)],
            "recovery": [("chip_equal_d3", 0.4), ("mf_vol_d10_rp", 0.3), ("c01_layered_d5", 0.3)],
        },
    }

    best_windows = {}

    for cname, alloc in candidates.items():
        ssm = {}
        for s in ["bull", "bear", "oscillate", "recovery"]:
            ssm[s] = [{"strategy": n, "weight": w} for n, w in alloc[s]]

        # 普通版
        dr = run_mss_backtest(ssm, sub_drs, states, nd)
        m = compute_metrics(dr, name=cname)
        all_results.append(m)

        # 窗口分析
        windows = window_analysis(dr, ds, WINDOWS)
        positive = sum(1 for w in windows if w.get("annual_return", 0) > 0 and w.get("n_days", 0) > 0)
        total = sum(1 for w in windows if w.get("n_days", 0) > 0)
        m["windows_positive"] = f"{positive}/{total}"

        # 动量过滤版
        dr_mom = run_mss_momentum_backtest(ssm, sub_drs, states, nd, mkt_idx, momentum_lookback=20)
        m_mom = compute_metrics(dr_mom, name=f"{cname}_mom20")
        all_results.append(m_mom)
        windows_mom = window_analysis(dr_mom, ds, WINDOWS)
        positive_mom = sum(1 for w in windows_mom if w.get("annual_return", 0) > 0 and w.get("n_days", 0) > 0)
        m_mom["windows_positive"] = f"{positive_mom}/{total}"

        # 置信度版
        ssm_conf = {}
        for s in ["bull", "bear", "oscillate", "recovery"]:
            ssm_conf[s] = [{"strategy": n, "weight": w, "confidence_adjust": True} for n, w in alloc[s]]
        dr_conf = run_mss_backtest(ssm_conf, sub_drs, states, nd, conf=conf)
        m_conf = compute_metrics(dr_conf, name=f"{cname}_conf")
        all_results.append(m_conf)
        windows_conf = window_analysis(dr_conf, ds, WINDOWS)
        positive_conf = sum(1 for w in windows_conf if w.get("annual_return", 0) > 0 and w.get("n_days", 0) > 0)
        m_conf["windows_positive"] = f"{positive_conf}/{total}"

        # 保存最佳版本的窗口数据
        best_windows[cname] = windows
        best_windows[f"{cname}_mom20"] = windows_mom

        logger.info(f"  {cname}: 年化={m['annual_return']*100:.2f}% "
                    f"Sharpe={m['sharpe']:.3f} 回撤={m['max_drawdown']*100:.2f}% "
                    f"评分={m['composite_score']} 窗口={m['windows_positive']}")

    print_results_table(all_results, "Phase 5: 综合最优对比")

    # 对最高评分的版本做详细窗口分析
    best = max(all_results, key=lambda x: x.get('composite_score', 0))
    logger.info(f"🏆 最佳: {best['name']}: 年化={best['annual_return']*100:.2f}% "
                f"Sharpe={best['sharpe']:.3f} 回撤={best['max_drawdown']*100:.2f}% "
                f"评分={best['composite_score']}")

    # 输出最佳方案的窗口
    base_name = best['name'].replace('_mom20', '').replace('_conf', '')
    wkey = base_name
    if '_mom20' in best['name']:
        wkey = f"{base_name}_mom20"

    if wkey in best_windows:
        print(f"\n{'=' * 80}")
        print(f"  🏆 最佳方案窗口分析: {best['name']}")
        print(f"{'=' * 80}")
        print(f"{'窗口':<20} {'年化%':<8} {'Sharpe':<8} {'回撤%':<8} {'Calmar':<8}")
        print('-' * 60)
        for w in best_windows[wkey]:
            if w.get("n_days", 0) == 0:
                continue
            ar = w.get("annual_return", 0) * 100
            sp = w.get("sharpe", 0)
            dd = abs(w.get("max_drawdown", 0)) * 100
            ca = w.get("calmar", 0)
            print(f"  {w['name']:<18} {ar:>6.2f}% {sp:>7.3f} {dd:>6.2f}% {ca:>7.3f}")

    out_path = os.path.join(RESULTS_DIR, "phase5_final.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info("已保存到 %s", out_path)

    return best


def main():
    parser = argparse.ArgumentParser(description="MarketStateSelector 深度优化")
    parser.add_argument("--phase", choices=["base", "1", "2", "3", "4", "5", "compare", "all"],
                        default="all")
    args = parser.parse_args()

    if args.phase == "base":
        run_phase_base()
    elif args.phase == "1":
        run_phase1()
    elif args.phase == "2":
        run_phase2()
    elif args.phase == "3":
        run_phase3()
    elif args.phase == "4":
        run_phase4()
    elif args.phase == "5":
        run_phase5()
    elif args.phase == "compare":
        run_phase_compare()
    elif args.phase == "all":
        logger.info("⚠️ 运行全部phase，每个phase独立加载数据")
        run_phase1()
        run_phase2()
        run_phase3()
        run_phase4()
        best = run_phase5()
        logger.info(f"🏆 最终最佳: {best['name']}")


if __name__ == "__main__":
    main()
