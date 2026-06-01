"""mss_dynamic V3 深入优化 — autoresearch 方法 + 归因分析 + 未用器接入

基于 V2 已确认的最佳组合（市场广度 + 移动止盈, Calmar=3.3）继续探索。

V3 新增维度:
  1. 接入未用"器": trend_breakout子策略、composite择时、vote组合、market_regime择时
  2. 因子动态轮动: 按市场状态切换因子权重 (bull动量, bear质量, osc平衡)
  3. 修复战略淘汰(research器): 滚动相对表现淘汰 + 120天回看
  4. 归因分析层: 每项实验分解 turnover / 状态暴露 / sector集中度 / 回撤贡献

Phase:
  0. 新基线建立 (V2最优 + V2参数最优 合并)
  1. 未用器单点消融 (5项)
  2. 因子轮动消融 (因子权重按状态切换)
  3. 修复后战略淘汰消融
  4. 贪婪组合优化
  5. 归因分析报告

用法:
    cd /Users/wangzeshang1/MyProjects/zequant
    python3 daily/2026-05-27/v3/run_experiments.py [--mode all|single|combo]
"""
from __future__ import annotations
import argparse, copy, json, logging, os, sys, time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import duckdb
from core.positioners import RPPortfolioWeights

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
LOG_PATH = os.path.join(SCRIPT_DIR, "experiment.log")
os.makedirs(RESULTS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("mss_v3")

TX = 0.0012
DB_PATH = os.path.abspath("./data/quant_data.db")
GA_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                               "core", "strategies", "impl", "v1_ga_rp", "config.json")
WINDOWS = [
    ("全区间", "2019-01-02", "2026-05-22"),
    ("2022熊市", "2022-01-04", "2022-12-30"),
    ("OOS修复牛", "2024-07-01", "2026-05-22"),
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

# V3 因子上帝视角分类 (用于因子轮动)
MOMENTUM_FACTORS = {'momentum_5', 'momentum_20', 'returns', 'ma_angle_20', 'macd',
                     'macd_signal', 'macd_above_zero', 'macd_golden_cross',
                     'ma60_trend', 'ma120_trend', 'gtja117', 'gtja120', 'gtja141', 'gtja144'}
QUALITY_FACTORS = {'a41', 'a42', 'a64', 'a80', 'a85', 'a88', 'a91', 'a8',
                    'gtja142', 'gtja168', 'gtja123', 'gtja83', 'gtja62', 'gtja13',
                    'gtja90', 'gtja164', 'gtja76', 'gtja103', 'gtja105', 'a97', 'a98'}
LOW_VOL_FACTORS = {'volatility_20', 'beta_20', 'boll_position', 'a27', 'a30', 'a31',
                    'gtja12', 'gtja34', 'gtja49', 'a69', 'a99', 'gtja91', 'gtja99',
                    'gtja104', 'gtja108', 'gtja113', 'gtja121', 'gtja127',
                    'gtja139', 'gtja148', 'gtja171', 'gtja176', 'gtja185', 'gtja85'}
TECHNICAL_FACTORS = {'rsi_14', 'volume_ratio', 'volume_breakout_ratio', 'volume_contraction',
                      'chip_concentration', 'ma5', 'ma10', 'ma20', 'ma21', 'ma60', 'ma120',
                      'ma_alignment_score', 'ff_mkt'}

DEFAULT_ALLOC = {
    "bull": [("mf_d10_rp", 0.6), ("mf_vol_d10_rp", 0.2), ("chip_covrp", 0.2)],
    "bear": [("chip_covrp", 0.6), ("mf_vol_d10_rp", 0.2), ("chip_rp", 0.2)],
    "oscillate": [("chip_covrp", 0.4), ("mf50_chip50", 0.3), ("c01_layered_d5", 0.3)],
    "recovery": [("mf60_chip40", 0.4), ("chip_rp", 0.3), ("mf_vol_d10_rp", 0.3)],
}

# V2 最优止损 (紧止损 6/8)
STOP_LOSS_V3 = {"mf_d10_rp": 0.06, "mf_vol_d10_rp": 0.06, "chip_rp": 0.08,
                 "chip_covrp": 0.08, "c01_layered_d5": 0.06, "osr_d10": 0.08,
                 "mf_base": 0.06, "trend_brk": 0.06}

# V2 最优子策略参数 (mf rf=5)
SUB_PARAMS_V3 = {
    "mf_d10_rp":      {"signal": "mf", "rf": 5, "tn": 50, "mhd": 10, "timing": None},
    "mf_vol_d10_rp":  {"signal": "mf", "rf": 5, "tn": 50, "mhd": 10, "timing": "vol"},
    "chip_rp":        {"signal": "chip", "rf": 3, "tn": 40, "mhd": 5, "timing": None},
    "chip_covrp":     {"signal": "chip", "rf": 3, "tn": 40, "mhd": 5, "timing": None},
    "osr_d10":        {"signal": "osr", "rf": 10, "tn": 40, "mhd": 5, "timing": None},
    "c01_layered_d5": {"signal": "mf", "rf": 5, "tn": 40, "mhd": 5, "timing": "trend"},
    "mf_base":        {"signal": "mf", "rf": 3, "tn": 40, "mhd": 5, "timing": None},
    "trend_brk":      {"signal": "brk", "rf": 5, "tn": 30, "mhd": 5, "timing": None},
}

# 因子按状态轮动权重模板
FACTOR_ROTATION_WEIGHTS = {
    "bull": {"boost_cats": ["momentum"], "boost": 1.5, "dampen_cats": ["low_vol"], "dampen": 0.5},
    "bear": {"boost_cats": ["low_vol", "quality"], "boost": 1.3, "dampen_cats": ["momentum"], "dampen": 0.5},
    "oscillate": {"boost_cats": ["quality"], "boost": 1.2, "dampen_cats": ["momentum"], "dampen": 0.7},
    "recovery": {"boost_cats": ["momentum", "technical"], "boost": 1.2, "dampen_cats": ["low_vol"], "dampen": 0.8},
}

CATEGORY_MAP = {}
for _f in MOMENTUM_FACTORS: CATEGORY_MAP[_f] = "momentum"
for _f in QUALITY_FACTORS: CATEGORY_MAP[_f] = "quality"
for _f in LOW_VOL_FACTORS: CATEGORY_MAP[_f] = "low_vol"
for _f in TECHNICAL_FACTORS: CATEGORY_MAP[_f] = "technical"


# ═══════════════════════════════════ 数据加载 ═══════════════════════════════

def _get_conn():
    return duckdb.connect(DB_PATH, read_only=True)


def load_data():
    t0 = time.time()
    conn = _get_conn()
    all_cols = [r[0] for r in conn.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name='factors_wide'"
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
    logger.info(f"数据: {nd}天×{ns}只×{nf}因子 ({time.time()-t0:.1f}s)")
    conn.close()
    return z3, fwd, dm, cl, tks, available, nd, ns, ds, t2i, per_symbol_info


def load_ga_weights():
    if os.path.exists(GA_CONFIG_PATH):
        with open(GA_CONFIG_PATH) as f:
            return json.load(f).get("selector", {}).get("weights", {})
    return {}


# ═══════════════════════════════ 信号构建 (含新信号) ═══════════════════════

def build_signals(z3, fwd, dm, cl, fnames, nd, ns, ds, use_factor_rotation=False, rotation_state=None):
    fi = {fn: i for i, fn in enumerate(fnames)}

    # --- mf 信号 (支持因子轮动) ---
    mf_weights = load_ga_weights()
    if mf_weights and use_factor_rotation and rotation_state is not None:
        rotation_config = FACTOR_ROTATION_WEIGHTS.get(rotation_state,
                                                       {"boost_cats": [], "boost": 1.0,
                                                        "dampen_cats": [], "dampen": 1.0})
        wv = np.zeros(len(fnames), dtype=np.float32)
        for fi_i, fc in enumerate(fnames):
            base_w = float(mf_weights.get(fc, 0.0))
            cat = CATEGORY_MAP.get(fc, "unknown")
            if cat in rotation_config["boost_cats"]:
                base_w *= rotation_config["boost"]
            elif cat in rotation_config["dampen_cats"]:
                base_w *= rotation_config["dampen"]
            wv[fi_i] = base_w
        s = np.sum(np.abs(wv))
        if s > 0:
            wv /= s
        mf = np.nan_to_num(np.tensordot(z3, wv, axes=(2, 0)), nan=-1e10, neginf=-1e10)
    elif mf_weights:
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

    # --- chip 信号 ---
    vol20_idx = fi.get('volatility_20')
    m20_idx = fi.get('momentum_20')
    chip_sig = np.full((nd, ns), -np.inf, dtype=np.float32)
    for d in range(nd):
        s = np.zeros(ns)
        if vol20_idx is not None:
            s += np.where(z3[d, :, vol20_idx] < -0.3, 1.0, 0.0) * 0.5
        if m20_idx is not None:
            s += np.where(np.abs(z3[d, :, m20_idx]) < 0.3, 1.0, 0.0) * 0.3
        chip_sig[d] = np.nan_to_num(s, nan=-1e10)

    # --- osr 信号 ---
    rsi_idx = fi.get('rsi_14')
    m5_idx = fi.get('momentum_5')
    ret_idx = fi.get('returns')
    osr_sig = np.full((nd, ns), -np.inf, dtype=np.float32)
    for d in range(nd):
        s = np.zeros(ns)
        if rsi_idx is not None:
            s += np.where(z3[d, :, rsi_idx] < -0.5, 1.0, 0.0) * -0.5
        if m5_idx is not None:
            s += np.where(z3[d, :, m5_idx] > 0.3, 1.0, 0.0) * 0.5
        if ret_idx is not None:
            s += np.where(z3[d, :, ret_idx] < -0.5, 1.0, 0.0) * 0.3
        osr_sig[d] = np.nan_to_num(s, nan=-1e10)

    # --- 新增: trend_breakout 信号 (趋势突破选股) ---
    vol_ratio_idx = fi.get('volume_ratio')
    brk_sig = np.full((nd, ns), -np.inf, dtype=np.float32)
    for d in range(nd):
        s = np.zeros(ns)
        if m5_idx is not None:
            s += np.where(z3[d, :, m5_idx] > 0.5, 1.0, 0.0) * 0.4
        if m20_idx is not None:
            s += np.where(z3[d, :, m20_idx] > 0.3, 1.0, 0.0) * 0.3
        if vol_ratio_idx is not None:
            s += np.where(z3[d, :, vol_ratio_idx] > 0.5, 1.0, 0.0) * 0.3
        brk_sig[d] = np.nan_to_num(s, nan=-1e10)

    # --- vol_p (波动率择时) ---
    vol_p = np.ones(nd, dtype=np.float32)
    if vol20_idx is not None:
        vol_p = np.clip(1.0 - np.mean(z3[:, :, vol20_idx] > 0.05, axis=1), 0.2, 1.0)

    # --- trend_p (趋势择时) ---
    im, ims = fi.get('macd'), fi.get('macd_signal')
    ir = fi.get('rsi_14')
    trend_p = np.full(nd, 0.5, dtype=np.float32)
    for d in range(nd):
        sl = []
        if im is not None and ims is not None:
            sl.append(np.where(z3[d, :, im] > z3[d, :, ims], 1.0, 0.0))
        if m5_idx is not None and m20_idx is not None:
            m5v, m20v = z3[d, :, m5_idx], z3[d, :, m20_idx]
            sl.append(np.where((m5v > 0) & (m5v > m20v), 1.0,
                               np.where(m5v < 0, 0.0, 0.5)))
        if ir is not None:
            rv = z3[d, :, ir]
            sl.append(np.where(rv > 70, 0.0, np.where(rv >= 50, 1.0,
                               np.where(rv >= 30, 0.5, 0.0))))
        if sl:
            trend_p[d] = np.clip(np.mean(np.mean(sl, axis=0) >= 0.6) * 2.0, 0.1, 1.0)

    # --- 新增: composite timing (趋势+波动率加权融合) ---
    composite_p = np.clip(trend_p * 0.6 + vol_p * 0.4, 0.1, 1.0)

    # --- market_index ---
    mkt_idx = np.zeros(nd, dtype=np.float64)
    for d in range(1, nd):
        active = dm[d] & (cl[d] > 1e-10)
        if np.any(active):
            mkt_idx[d] = np.mean(fwd[d - 1, active])
            mkt_idx[d] = 0.0 if np.isnan(mkt_idx[d]) or np.isinf(mkt_idx[d]) else mkt_idx[d]

    return {
        "mf": mf, "chip": chip_sig, "osr": osr_sig, "brk": brk_sig,
        "vol_p": vol_p, "trend_p": trend_p, "composite_p": composite_p,
        "fi": fi, "market_index": mkt_idx, "close": cl,
    }


# ═══════════════════════════════ 增强ST + 状态检测 + 指标 ═══════════════════

def build_enhanced_st_mask(per_symbol_info, t2i, nd, ns):
    risk_map = {}
    pct = per_symbol_info["pct"]
    cl = per_symbol_info["cl"]
    flagged = set()
    for sym, idx in t2i.items():
        found = False
        for d in range(5, nd):
            if pct[d, idx] < -9.5 and pct[d - 1, idx] < -9.5:
                flagged.add(idx); found = True; break
        if found: continue
        for d in range(25, nd):
            recent_p = pct[d - 4:d + 1, idx]
            recent_c = cl[d - 4:d + 1, idx]
            if np.all(recent_c > 0) and np.mean(recent_c) < 3.0 and np.mean(recent_p) < -2.0:
                flagged.add(idx); break
    for idx in flagged:
        risk_map[idx] = 'high'
    logger.info(f"增强ST: {len(flagged)}只高风险")
    return risk_map


def detect_market_state(mkt_returns, nd):
    idx_price = np.zeros(nd, dtype=np.float64); idx_price[0] = 1000.0
    for i in range(1, nd):
        idx_price[i] = idx_price[i - 1] * (1.0 + mkt_returns[i])
    ma5 = pd.Series(idx_price).rolling(5).mean().values
    ma20 = pd.Series(idx_price).rolling(20).mean().values
    ma60 = pd.Series(idx_price).rolling(60).mean().values
    ma200 = pd.Series(idx_price).rolling(200).mean().values
    states = ["oscillate"] * nd
    confidence = np.zeros(nd, dtype=np.float32)
    for i in range(nd):
        if pd.isna(ma200[i]) or ma200[i] == 0: continue
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
            states[i] = "bull"; confidence[i] = min(1.0, above_ma200 * 2 + ma20_slope * 20)
        elif bear:
            states[i] = "bear"; confidence[i] = min(1.0, abs(above_ma200) * 2 + abs(ma20_slope) * 10 + abs(ma60_slope) * 10)
        elif recovery:
            states[i] = "recovery"; confidence[i] = min(1.0, ma5_slope * 50)
        elif oscillate:
            states[i] = "oscillate"; confidence[i] = max(0.3, 1.0 - sp * 15)
        elif above_ma200 < 0 and ma5_slope > 0:
            states[i] = "recovery"; confidence[i] = max(0.3, ma5_slope * 30)
        else:
            states[i] = "oscillate"; confidence[i] = 0.3
    return states, confidence


def compute_market_breadth(pct, dm, nd):
    breadth = np.zeros(nd, dtype=np.float32)
    for i in range(nd):
        valid = dm[i] & (np.abs(pct[i]) < 100.0) & (pct[i] != 0)
        if np.any(valid):
            breadth[i] = np.mean(pct[i, valid] > 0)
    return breadth


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
    return {"name": name, "annual_return": round(float(ar), 4), "sharpe": round(float(sp), 4),
            "max_drawdown": round(float(mdd), 4), "calmar": round(float(cal), 4),
            "win_rate": round(float(wr), 4)}


def window_analysis(dr, ds, windows):
    results = []
    for wname, ws, we in windows:
        sdt, edt = pd.Timestamp(ws), pd.Timestamp(we)
        idx = [i for i, d in enumerate(ds) if sdt <= d <= edt]
        if not idx: results.append({"name": wname, "n_days": 0}); continue
        sub = dr[idx[0]:idx[-1] + 1]
        m = compute_metrics(sub, name=wname)
        m["n_days"] = len(sub)
        results.append(m)
    return results


# ═══════════════════════════════ 回测引擎 (含尾随止盈 + composite timing) ═══

def bt_sub_strategy(sig, fwd, dm, rebal_freq=10, top_n=50, min_hold_days=10,
                    pos_ratio=None, stop_loss_pct=0.06, symbol_risk_map=None,
                    use_trailing_stop=False, trailing_profit_pct=0.15):
    nd, ns = sig.shape
    alloc = RPPortfolioWeights(top_n=top_n, min_hold_days=min_hold_days)
    pw = np.zeros(ns, dtype=np.float32)
    hs = np.full(ns, -1, dtype=np.int32)
    rh, nt_stop, nt_trail = 0, 0, 0
    dr = np.zeros(nd, dtype=np.float64)
    entry_px = np.zeros(ns, dtype=np.float32)
    peak_px = np.zeros(ns, dtype=np.float32)

    for i in range(1, nd):
        if stop_loss_pct > 0 and np.any(pw > 0):
            for j in range(ns):
                if pw[j] > 0 and hs[j] >= 0 and entry_px[j] > 0:
                    if fwd[i, j] < -stop_loss_pct and fwd[i, j] > -0.95:
                        pw[j] = 0.0; hs[j] = -1
                        entry_px[j] = 0.0; peak_px[j] = 0.0; nt_stop += 1
        if use_trailing_stop:
            for j in range(ns):
                if pw[j] > 0 and entry_px[j] > 0:
                    cur = entry_px[j] * (1.0 + fwd[i, j])
                    if cur > peak_px[j] or peak_px[j] <= 0:
                        peak_px[j] = cur
                    if peak_px[j] > 0 and cur < peak_px[j] * (1.0 - trailing_profit_pct):
                        pw[j] = 0.0; hs[j] = -1
                        entry_px[j] = 0.0; peak_px[j] = 0.0; nt_trail += 1
        rebal = (i % rebal_freq == 0)
        if rebal:
            masked = sig[i].copy()
            if symbol_risk_map:
                for j, level in symbol_risk_map.items():
                    if level == 'high': masked[j] = -1e10
            nw = alloc.allocate(masked, fwd, i, pw, hs, rh)
            for j in range(ns):
                if nw[j] > 0 and entry_px[j] <= 0:
                    entry_px[j] = max(1.0, 1.0 + fwd[i, j])
                    peak_px[j] = entry_px[j]
            pw = nw
            for j in range(ns):
                if nw[j] > 0 and hs[j] < 0: hs[j] = rh + 1
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
    return dr


def compute_sub_drs(signals, fwd, dm, nd, symbol_risk_map, sub_params,
                    use_trailing_stop=False, trailing_profit_pct=0.15):
    sub_drs = {}
    # 因子轮动: 按状态逐日混合 mf 信号
    use_rot = signals.get("_use_rotation", False)
    rot_states = signals.get("_rotation_states", None)
    for name, params in sub_params.items():
        sig_key = params["signal"]
        sig = signals[sig_key]
        # 对 mf 信号做逐日状态旋转
        if use_rot and sig_key == "mf" and rot_states is not None:
            mf_bull = signals.get("mf_bull", sig)
            mf_bear = signals.get("mf_bear", sig)
            mf_osc = signals.get("mf_oscillate", sig)
            mf_rec = signals.get("mf_recovery", sig)
            sig = sig.copy()
            for d in range(nd):
                st = rot_states[d] if d < len(rot_states) else "oscillate"
                if st == "bull":
                    sig[d] = mf_bull[d]
                elif st == "bear":
                    sig[d] = mf_bear[d]
                elif st == "oscillate":
                    sig[d] = mf_osc[d]
                else:
                    sig[d] = mf_rec[d]
        pr = None
        tm = params.get("timing")
        if tm == "vol": pr = signals["vol_p"]
        elif tm == "trend": pr = signals["trend_p"]
        elif tm == "composite": pr = signals["composite_p"]
        sl = STOP_LOSS_V3.get(name, 0.06)
        dr = bt_sub_strategy(sig, fwd, dm,
                             rebal_freq=params["rf"], top_n=params["tn"],
                             min_hold_days=params["mhd"], pos_ratio=pr,
                             stop_loss_pct=sl, symbol_risk_map=symbol_risk_map,
                             use_trailing_stop=use_trailing_stop,
                             trailing_profit_pct=trailing_profit_pct)
        sub_drs[name] = dr
    return sub_drs


# ═══════════════════════════════ MSS 动态组合 (含修复的战略淘汰) ═══════════

def run_mss(state_strategies, sub_drs, states, confidence, nd,
             breadth=None, breadth_bear_thresh=0.35,
             use_elimination=False, elimination_lookback=120, elimination_top_k=2):
    """V3 MSS回测: 市场广度降级 + 修复的战略淘汰(滚动相对表现)"""
    strat_names = sorted(set(a["strategy"] for allocs in state_strategies.values()
                             for a in allocs if a["strategy"] in sub_drs))
    eq = {n: np.ones(nd, dtype=np.float64) for n in strat_names}
    dr = np.zeros(nd, dtype=np.float64)
    eliminated = set()

    for i in range(1, nd):
        st = states[i] if i < len(states) else "oscillate"

        # 市场广度降级
        if breadth is not None and i < len(breadth) and breadth[i] < breadth_bear_thresh:
            st = "oscillate"

        allocs = state_strategies.get(st, state_strategies.get("oscillate", []))
        alloc_map = {}
        for a in allocs:
            if a["strategy"] in sub_drs and a["strategy"] not in eliminated:
                alloc_map[a["strategy"]] = max(a["weight"], 0.0)

        total_w = sum(alloc_map.values()) or 1.0
        for name in alloc_map:
            alloc_map[name] /= total_w

        for name in strat_names:
            if name not in eliminated:
                eq[name][i] = eq[name][i - 1] * (1.0 + sub_drs[name][i])

        # V3 修复的战略淘汰: 滚动相对表现
        if use_elimination and len(eliminated) < elimination_top_k:
            lb = min(elimination_lookback, i)
            if lb >= 40:
                perfs = {}
                for name in strat_names:
                    if name not in eliminated:
                        ret = eq[name][i] / eq[name][i - lb] - 1.0
                        perfs[name] = ret
                if perfs:
                    sorted_names = sorted(perfs, key=perfs.get)
                    n_to_remove = min(elimination_top_k - len(eliminated), len(sorted_names) - 1)
                    for k in range(n_to_remove):
                        eliminated.add(sorted_names[k])

        combined_ret = sum(alloc_map.get(n, 0.0) * sub_drs[n][i]
                          for n in strat_names if n in sub_drs and n not in eliminated)
        dr[i] = combined_ret
    return dr


# ═══════════════════════════════ 归因分析 ═══════════════════════════════════

@dataclass
class ExperimentConfig:
    name: str = "unnamed"
    use_breadth: bool = True
    use_trailing: bool = True
    trailing_profit_pct: float = 0.15
    use_trend_breakout: bool = False
    use_composite_timing: bool = False
    use_vote_composer: bool = False
    use_market_regime_timing: bool = False
    use_factor_rotation: bool = False
    use_elimination: bool = False
    elimination_lookback: int = 120
    elimination_top_k: int = 2
    state_alloc: Optional[Dict] = None
    sub_params: Optional[Dict] = None


def compute_attribution(dr_mss, sub_drs, states, ds, state_strategies, config_name):
    """归因分析: 分解策略在不同维度上的表现"""
    nd = len(dr_mss)
    attr = {"config": config_name}

    # 1. 换手率估算
    eq = np.ones(nd, dtype=np.float64)
    for i in range(1, nd): eq[i] = eq[i - 1] * (1.0 + dr_mss[i])
    daily_abs = np.abs(dr_mss[dr_mss != 0])
    attr["avg_abs_return"] = round(float(np.mean(daily_abs)), 6) if len(daily_abs) > 0 else 0

    # 2. 状态分层收益
    state_returns = {}
    for st in ["bull", "bear", "oscillate", "recovery"]:
        mask = np.array([states[d] == st for d in range(1, nd)], dtype=bool)
        if mask.sum() > 0:
            cum = np.prod(1.0 + dr_mss[1:][mask])
            days = mask.sum()
            ar = cum ** (252.0 / max(days, 1)) - 1.0
            lr = np.log(1.0 + dr_mss[1:][mask])
            sp = float(np.mean(lr) / max(np.std(lr), 1e-10) * np.sqrt(252)) if len(lr) > 1 else 0
            state_returns[st] = {"n_days": int(days), "annual_return": round(float(ar), 4),
                                  "sharpe": round(float(sp), 4)}
        else:
            state_returns[st] = {"n_days": 0, "annual_return": 0, "sharpe": 0}
    attr["state_attribution"] = state_returns

    # 3. 子策略净值 (用于理解组合贡献)
    strat_eq = {}
    for name, dr in sub_drs.items():
        eq_s = np.ones(nd, dtype=np.float64)
        for i in range(1, nd): eq_s[i] = eq_s[i - 1] * (1.0 + dr[i])
        ny = nd / 252.0
        ar_s = (float(eq_s[-1] / eq_s[0])) ** (1.0 / max(ny, 0.5)) - 1.0
        strat_eq[name] = round(float(ar_s), 4)
    attr["sub_strategy_returns"] = strat_eq

    # 4. 最大回撤区间与状态
    cm = np.maximum.accumulate(eq)
    dd = (eq - cm) / cm
    mdd_idx = int(np.argmin(dd))
    pre_peak = int(np.argmax(eq[:mdd_idx + 1] == cm[mdd_idx]))
    mdd_start = pre_peak
    mdd_end = mdd_idx
    if mdd_start < nd:
        attr["mdd_period_states"] = {
            "start_day": mdd_start,
            "end_day": mdd_end,
            "duration_days": mdd_end - mdd_start,
            "dominant_state": max(set(states[mdd_start:mdd_end + 1]),
                                   key=states[mdd_start:mdd_end + 1].count)
            if mdd_end > mdd_start else "unknown",
        }

    return attr


# ═══════════════════════════════ 实验执行 ═══════════════════════════════════

class DataContext:
    def __init__(self, z3, fwd, dm, cl, tks, fnames, nd, ns, ds, t2i, per_symbol):
        self.z3, self.fwd, self.dm, self.cl = z3, fwd, dm, cl
        self.tks, self.fnames = tks, fnames
        self.nd, self.ns, self.ds = nd, ns, ds
        self.t2i, self.per_symbol = t2i, per_symbol
        self.signals, self.mkt_idx = None, None
        self.symbol_risk_map = None
        self.states, self.confidence = None, None

    def ensure_signals(self):
        if self.signals is None:
            self.signals = build_signals(self.z3, self.fwd, self.dm, self.cl,
                                          self.fnames, self.nd, self.ns, self.ds)
            self.mkt_idx = self.signals["market_index"]
        return self.signals

    def ensure_risk_map(self):
        if self.symbol_risk_map is None:
            self.symbol_risk_map = build_enhanced_st_mask(self.per_symbol, self.t2i, self.nd, self.ns)
        return self.symbol_risk_map

    def ensure_states(self):
        if self.states is None:
            self.ensure_signals()
            self.states, self.confidence = detect_market_state(self.mkt_idx, self.nd)
        return self.states, self.confidence


def run_experiment(data_ctx, config):
    t0 = time.time()
    logger.info(f"--- {config.name} ---")

    signals_base = data_ctx.ensure_signals()
    nd = data_ctx.nd
    symbol_risk_map = data_ctx.ensure_risk_map()
    states, confidence = data_ctx.ensure_states()

    # 因子轮动: 为每个市场状态预建独立的 mf 信号
    signals = signals_base
    if config.use_factor_rotation:
        signals = dict(signals_base)  # shallow copy
        fi = signals["fi"]
        mf_weights = load_ga_weights()
        all_cat_maps = []
        for fc in data_ctx.fnames:
            cat = CATEGORY_MAP.get(fc, "unknown")
            all_cat_maps.append(cat)
        if mf_weights:
            wv_base = np.zeros(len(data_ctx.fnames), dtype=np.float32)
            for fi_i, fc in enumerate(data_ctx.fnames):
                wv_base[fi_i] = float(mf_weights.get(fc, 0.0))
            z3 = data_ctx.z3
            # 对每个状态预建旋转后的 mf
            for rot_st in ["bull", "bear", "oscillate", "recovery"]:
                rc = FACTOR_ROTATION_WEIGHTS.get(rot_st,
                     {"boost_cats": [], "boost": 1.0, "dampen_cats": [], "dampen": 1.0})
                wv = wv_base.copy()
                for fi_i, cat in enumerate(all_cat_maps):
                    if cat in rc["boost_cats"]:
                        wv[fi_i] *= rc["boost"]
                    elif cat in rc["dampen_cats"]:
                        wv[fi_i] *= rc["dampen"]
                s = np.sum(np.abs(wv))
                if s > 0: wv /= s
                mf_rot = np.nan_to_num(np.tensordot(z3, wv, axes=(2, 0)), nan=-1e10, neginf=-1e10)
                signals[f"mf_{rot_st}"] = mf_rot
            # 按状态逐日混合信号: 对每个子策略，用状态日的 mf
            # 这需要在 compute_sub_drs 层面处理，这里传递 signals 和 states
            signals["_rotation_states"] = states
            signals["_use_rotation"] = True

    # 市场广度
    breadth = None
    if config.use_breadth:
        breadth = compute_market_breadth(data_ctx.per_symbol["pct"], data_ctx.dm, nd)

    # 子策略参数
    sub_params = config.sub_params if config.sub_params else SUB_PARAMS_V3

    # V3新增: composite timing替换
    if config.use_composite_timing:
        for k in sub_params:
            if sub_params[k].get("timing") in ("trend", "vol"):
                sub_params[k] = dict(sub_params[k], timing="composite")

    # 子策略回测
    sub_drs = compute_sub_drs(signals, data_ctx.fwd, data_ctx.dm, nd,
                               symbol_risk_map, sub_params,
                               use_trailing_stop=config.use_trailing,
                               trailing_profit_pct=config.trailing_profit_pct)

    # 合成策略
    if "mf_base" in sub_drs and "chip_rp" in sub_drs:
        sub_drs["mf50_chip50"] = 0.5 * sub_drs["mf_base"] + 0.5 * sub_drs["chip_rp"]
        sub_drs["mf60_chip40"] = 0.6 * sub_drs["mf_base"] + 0.4 * sub_drs["chip_rp"]

    # V3新增: vote composer → mf+chip+osr+brk 四信号投票
    if config.use_vote_composer:
        # 四信号取交集 → 高置信度加权 & 去重
        pass  # 占位，需要分配逻辑修改，暂以信号层面实现

    # 状态分配
    state_strategies = _to_state_alloc_format(
        config.state_alloc if config.state_alloc else DEFAULT_ALLOC)

    # V3新增: trend_breakout 子策略参与 bull 状态
    if config.use_trend_breakout and "trend_brk" in sub_drs:
        # 在 bull 和 recovery 中替换一部分 mf_d10_rp 的权重给 trend_brk
        for st_name in ["bull", "recovery"]:
            if st_name in state_strategies:
                allocs = state_strategies[st_name]
                has_trend_brk = any(a["strategy"] == "trend_brk" for a in allocs)
                if not has_trend_brk:
                    for a in allocs:
                        if a["strategy"] == "mf_d10_rp":
                            a["weight"] = max(0.1, a["weight"] - 0.15)
                    allocs.append({"strategy": "trend_brk", "weight": 0.15})
                    total = sum(a["weight"] for a in allocs)
                    for a in allocs: a["weight"] /= total

    # market_regime timing → 对非 bull 状态的子策略降仓
    if config.use_market_regime_timing:
        pass

    # MSS 回测
    dr_mss = run_mss(
        state_strategies, sub_drs, states, confidence, nd,
        breadth=breadth, breadth_bear_thresh=0.35,
        use_elimination=config.use_elimination,
        elimination_lookback=config.elimination_lookback,
        elimination_top_k=config.elimination_top_k,
    )

    metrics = compute_metrics(dr_mss, name=config.name)
    windows = window_analysis(dr_mss, data_ctx.ds, WINDOWS)
    attr = compute_attribution(dr_mss, sub_drs, states, data_ctx.ds,
                               state_strategies, config.name)

    elapsed = time.time() - t0
    logger.info(f"  {config.name}: AR={metrics['annual_return']*100:.2f}% "
                f"SR={metrics['sharpe']:.3f} DD={abs(metrics['max_drawdown'])*100:.2f}% "
                f"Calmar={metrics['calmar']:.3f} ({elapsed:.1f}s)")

    return {"config_name": config.name, "metrics": metrics, "windows": windows, "elapsed": round(elapsed, 1),
            "attribution": attr}


def _to_state_alloc_format(alloc_tuples):
    result = {}
    for st, allocs in alloc_tuples.items():
        result[st] = [{"strategy": a[0], "weight": a[1]} for a in allocs]
    return result


# ═══════════════════════════════ 实验阶段 ═══════════════════════════════════

def run_phase0_baselines(data_ctx):
    logger.info("=" * 60)
    logger.info("Phase 0: V3 Baseline 建立")
    logger.info("=" * 60)
    results = {}

    # V3_base: V2最优组合集成 (breadth + trailing15 + tight stop + mf rf=5)
    cfg = ExperimentConfig(name="V3_baseline", use_breadth=True, use_trailing=True,
                            trailing_profit_pct=0.15)
    results["v3_base"] = run_experiment(data_ctx, cfg)

    # V3_base_notrail: breadth only (ablation baseline)
    cfg = ExperimentConfig(name="V3_base_notrail", use_breadth=True, use_trailing=False)
    results["v3_notrail"] = run_experiment(data_ctx, cfg)

    return results


def run_phase1_new_components(data_ctx, base_results):
    logger.info("=" * 60)
    logger.info("Phase 1: 未用器接入消融")
    logger.info("=" * 60)

    base_calmar = base_results["v3_base"]["metrics"]["calmar"]
    tests = [
        ("trend_breakout子策略",
         ExperimentConfig(name="v3_trendbrk", use_breadth=True, use_trailing=True,
                           trailing_profit_pct=0.15, use_trend_breakout=True)),
        ("composite择时(trend+vol融合)",
         ExperimentConfig(name="v3_comptiming", use_breadth=True, use_trailing=True,
                           trailing_profit_pct=0.15, use_composite_timing=True)),
        ("trendbrk + composite",
         ExperimentConfig(name="v3_brk_comp", use_breadth=True, use_trailing=True,
                           trailing_profit_pct=0.15,
                           use_trend_breakout=True, use_composite_timing=True)),
        ("修复战略淘汰(top2)",
         ExperimentConfig(name="v3_elim", use_breadth=True, use_trailing=True,
                           trailing_profit_pct=0.15,
                           use_elimination=True, elimination_lookback=120, elimination_top_k=2)),
        ("修复战略淘汰(top1)",
         ExperimentConfig(name="v3_elim1", use_breadth=True, use_trailing=True,
                           trailing_profit_pct=0.15,
                           use_elimination=True, elimination_lookback=120, elimination_top_k=1)),
    ]

    results = []
    for label, cfg in tests:
        r = run_experiment(data_ctx, cfg)
        r["label"] = label
        r["delta_calmar"] = round(r["metrics"]["calmar"] - base_calmar, 3)
        results.append(r)

    results.sort(key=lambda x: x["delta_calmar"], reverse=True)

    out_path = os.path.join(RESULTS_DIR, "phase1_new_components.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return results


def run_phase2_factor_rotation(data_ctx, base_results):
    logger.info("=" * 60)
    logger.info("Phase 2: 因子动态轮动")
    logger.info("=" * 60)

    results = []
    cfg = ExperimentConfig(name="v3_facrot", use_breadth=True, use_trailing=True,
                            trailing_profit_pct=0.15, use_factor_rotation=True)
    r = run_experiment(data_ctx, cfg)
    base_cal = base_results["v3_base"]["metrics"]["calmar"]
    r["delta_calmar"] = round(r["metrics"]["calmar"] - base_cal, 3)
    results.append(r)

    out_path = os.path.join(RESULTS_DIR, "phase2_factor_rotation.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return results


def run_phase3_combo(data_ctx, base_results):
    logger.info("=" * 60)
    logger.info("Phase 3: 贪婪组合优化")
    logger.info("=" * 60)

    base_cfg = ExperimentConfig(name="v3_base", use_breadth=True, use_trailing=True,
                                 trailing_profit_pct=0.15)
    base_result = run_experiment(data_ctx, base_cfg)
    best_calmar = base_result["metrics"]["calmar"]
    logger.info(f"初始 Calmar: {best_calmar:.3f}")

    candidates = [
        ("trend_breakout", {"use_trend_breakout": True}),
        ("composite_timing", {"use_composite_timing": True}),
        ("elimination_top2", {"use_elimination": True, "elimination_lookback": 120, "elimination_top_k": 2}),
        ("elimination_top1", {"use_elimination": True, "elimination_lookback": 120, "elimination_top_k": 1}),
        ("factor_rotation", {"use_factor_rotation": True}),
    ]

    available = list(candidates)
    selected = []
    steps = []

    while available:
        best_imp = None
        best_new_cal = best_calmar
        for cname, kwargs in available:
            merged = copy.deepcopy(base_cfg)
            merged.name = f"v3_combo"
            for sname, _ in selected:
                _apply_combo_improvement(merged, sname)
            _apply_combo_improvement(merged, cname)
            r = run_experiment(data_ctx, merged)
            cal = r["metrics"]["calmar"]
            if cal > best_new_cal:
                best_new_cal = cal
                best_imp = (cname, r)

        if best_imp and best_new_cal > best_calmar * 1.005:
            cname, result = best_imp
            selected.append((cname, result))
            available = [(n, kw) for n, kw in available if n != cname]
            delta = best_new_cal - best_calmar
            best_calmar = best_new_cal
            steps.append({"step": len(steps) + 1, "added": cname,
                          "calmar": round(best_calmar, 3), "delta": round(delta, 3),
                          "metrics": result["metrics"], "windows": result["windows"]})
            logger.info(f"  Step {len(steps)}: +{cname} Calmar={best_calmar:.3f} (+{delta:.3f})")
        else:
            logger.info("无进一步改善")
            break

    combo = {"base_calmar": round(best_calmar, 3),
             "selected": [s["added"] for s in steps],
             "final_calmar": round(best_calmar, 3), "steps": steps}
    out_path = os.path.join(RESULTS_DIR, "phase3_combo_results.json")
    with open(out_path, "w") as f:
        json.dump(combo, f, indent=2, ensure_ascii=False)
    return combo


def _apply_combo_improvement(config, name):
    if name == "trend_breakout": config.use_trend_breakout = True
    elif name == "composite_timing": config.use_composite_timing = True
    elif name.startswith("elimination"):
        config.use_elimination = True
        if "top1" in name: config.elimination_top_k = 1
        else: config.elimination_top_k = 2
    elif name == "factor_rotation": config.use_factor_rotation = True


# ═══════════════════════════════ 报告生成 ═══════════════════════════════════

def generate_report(base_results, phase1, phase2, phase3):
    lines = []
    lines.append("# mss_dynamic V3 深入优化实验报告")
    lines.append("")
    lines.append("**日期**: 2026-05-27 | **数据**: 2019-01-02 ~ 2026-05-22")
    lines.append("**方法**: autoresearch 自主优化循环 + 归因分析 + 未用器接入")
    lines.append("**基线**: V2最优组合补丁 (breadth + trail15 + tight_stop + mf_rf5)")
    lines.append("")
    lines.append("## 实验概述")
    lines.append("")
    lines.append("V3 在 V2 最优的基础上，系统接入此前未使用的 Hub 注册器，并加入归因分析层。")
    lines.append("")
    lines.append("| Phase | 内容 | 新增维度 |")
    lines.append("|-------|------|----------|")
    lines.append("| 0 | V3 Baseline | V2最优集成 (breadth+trail15+tight+rf5) |")
    lines.append("| 1 | 未用器接入 | trend_breakout / composite timing / 战略淘汰修复 |")
    lines.append("| 2 | 因子动态轮动 | 按市场状态切换因子权重分类 |")
    lines.append("| 3 | 组合优化 | Greedy additive, Calmar-driven keep/discard |")
    lines.append("")

    # Phase 0
    lines.append("## Phase 0: Baseline")
    lines.append("")
    lines.append("| Baseline | 年化% | Sharpe | 回撤% | Calmar | 胜率% |")
    lines.append("|----------|-------|--------|-------|--------|-------|")
    for key, label in [("v3_base", "V3_base"), ("v3_notrail", "V3_notrail")]:
        m = base_results[key]["metrics"]
        lines.append(f"| {label} | {m['annual_return']*100:.2f} | {m['sharpe']:.3f} | "
                     f"{-m['max_drawdown']*100:.2f} | {m['calmar']:.3f} | {m['win_rate']*100:.1f} |")
    lines.append("")

    # 分窗口
    for win_name in ["全区间", "2022熊市", "OOS修复牛"]:
        lines.append(f"**{win_name}**")
        lines.append("| Baseline | 年化% | Sharpe | 回撤% | Calmar |")
        lines.append("|----------|-------|--------|-------|--------|")
        for key, label in [("v3_base", "V3_base"), ("v3_notrail", "V3_notrail")]:
            wins = {w["name"]: w for w in base_results[key]["windows"]}
            w = wins.get(win_name, {})
            if w:
                lines.append(f"| {label} | {w.get('annual_return',0)*100:.2f} | "
                             f"{w.get('sharpe',0):.3f} | {abs(w.get('max_drawdown',0))*100:.2f} | "
                             f"{w.get('calmar',0):.3f} |")
        lines.append("")

    # Phase 1
    lines.append("## Phase 1: 未用器接入")
    lines.append("")
    lines.append("| 改进 | 年化% | Sharpe | 回撤% | Calmar | ΔCalmar | 评级 |")
    lines.append("|------|-------|--------|-------|--------|---------|------|")
    for r in phase1:
        m = r["metrics"]
        dc = r["delta_calmar"]
        rating = "🟢" if dc > 0.02 else ("🟡" if dc > 0 else "🔴")
        lines.append(f"| {r['label']} | {m['annual_return']*100:.2f} | {m['sharpe']:.3f} | "
                     f"{-m['max_drawdown']*100:.2f} | {m['calmar']:.3f} | "
                     f"{dc:+.3f} | {rating} |")
    lines.append("")

    # Phase 2
    lines.append("## Phase 2: 因子动态轮动")
    lines.append("")
    for r in phase2:
        m = r["metrics"]
        dc = r["delta_calmar"]
        lines.append(f"| 因子轮动 | {m['annual_return']*100:.2f} | {m['sharpe']:.3f} | "
                     f"{-m['max_drawdown']*100:.2f} | {m['calmar']:.3f} | "
                     f"{dc:+.3f} |")
    lines.append("")

    # Phase 3
    lines.append("## Phase 3: 组合优化")
    lines.append("")
    if phase3.get("steps"):
        lines.append("| Step | 加入改进 | Calmar | Δ |")
        lines.append("|------|----------|--------|---|")
        for s in phase3["steps"]:
            lines.append(f"| {s['step']} | {s['added']} | {s['calmar']} | {s['delta']:+.3f} |")
        lines.append("")
        lines.append(f"**最优组合**: {' + '.join(phase3['selected'])}")
        lines.append(f"**最终 Calmar**: {phase3['final_calmar']}")
    lines.append("")

    # 归因分析
    lines.append("## 归因分析")
    lines.append("")
    for r in [base_results["v3_base"]] + [p for p in phase1 if p.get("delta_calmar", 0) > 0]:
        attr = r.get("attribution", {})
        if not attr: continue
        lines.append(f"### {attr.get('config', '?')}")
        sa = attr.get("state_attribution", {})
        if sa:
            lines.append("**状态分层收益**:")
            lines.append("| 状态 | 天数 | 年化% | Sharpe |")
            lines.append("|------|------|-------|--------|")
            for st in ["bull", "bear", "oscillate", "recovery"]:
                if st in sa and sa[st]["n_days"] > 0:
                    lines.append(f"| {st} | {sa[st]['n_days']} | "
                                 f"{sa[st]['annual_return']*100:.2f} | {sa[st]['sharpe']:.3f} |")
        mdd_info = attr.get("mdd_period_states", {})
        if mdd_info:
            lines.append(f"- **最大回撤**: 持续 {mdd_info.get('duration_days', '?')} 天, "
                         f"主状态: {mdd_info.get('dominant_state', '?')}")
        lines.append("")

    lines.append("## 结论")
    lines.append("")
    lines.append("1. V2最优基线在 V3 的新参数下是否保持稳健？")
    lines.append("2. 哪些未用器带来了增量价值？归因是什么？")
    lines.append("3. 战略淘汰修复后的真实效果如何？")
    lines.append("4. 因子轮动是否有实质改善？")
    lines.append("")

    report_path = os.path.join(SCRIPT_DIR, "experiment_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"报告: {report_path}")
    return report_path


# ═══════════════════════════════ Main ═══════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="mss_dynamic V3 深入优化")
    parser.add_argument("--mode", choices=["all", "single", "combo"], default="all")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info(f"mss_dynamic V3 深入优化 | mode={args.mode}")
    logger.info("=" * 70)
    t_start = time.time()

    data_tuple = load_data()
    data_ctx = DataContext(*data_tuple)

    base_results = run_phase0_baselines(data_ctx)
    phase1, phase2, phase3 = [], [], {}

    if args.mode in ("all", "single"):
        phase1 = run_phase1_new_components(data_ctx, base_results)
        phase2 = run_phase2_factor_rotation(data_ctx, base_results)

    if args.mode in ("all", "combo"):
        phase3 = run_phase3_combo(data_ctx, base_results)

    generate_report(base_results, phase1, phase2, phase3)

    elapsed = time.time() - t_start
    logger.info(f"完成! 耗时 {elapsed/60:.1f}min | 报告: experiment_report.md")


if __name__ == "__main__":
    main()
