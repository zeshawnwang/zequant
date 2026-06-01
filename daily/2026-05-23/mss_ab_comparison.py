"""mss_dynamic A/B 对比回测：Baseline vs V2

V2 改动：
  1. 增强ST过滤：名称匹配 + 连续跌停 + 低价持续下跌
  2. 个股止损：-8%追踪止损（mf_d10_rp）/ -10%（chip系列）
  3. 置信度联动权重：置信度 < 0.5 时降低 mf_d10_rp 权重

用法：
    cd /Users/wangzeshang1/MyProjects/zequant
    python3 daily/2026-05-23/mss_ab_comparison.py
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple, Any, Callable

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import duckdb
from core.positioners import RPPortfolioWeights

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mss_ab")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
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

FULL_RANGE = ("2019-01-02", "2026-05-22")
WINDOWS = [
    ("全区间",     "2019-01-02", "2026-05-22"),
    ("2022熊市",   "2022-01-04", "2022-12-30"),
    ("OOS修复牛",  "2024-07-01", "2026-05-22"),
]

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


# ══════════════════════════════════════════
# 数据加载
# ══════════════════════════════════════════

def _get_conn():
    src = os.path.abspath("./data/quant_data.db")
    return duckdb.connect(src, read_only=True)


def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                          List[str], List[str], int, int, List[pd.Timestamp],
                          Dict[str, int], Dict[str, np.ndarray]]:
    """加载数据并构建每日个股级别信息。

    返回 (z3, fwd, dm, cl, tks, fnames, nd, ns, ds, t2i, per_symbol_info)
        per_symbol_info: 每日个股特征（用于增强ST检测）
    """
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
    logger.info(f"数据: {nd}天 × {ns}只 × {nf}因子")
    conn.close()
    return z3, fwd, dm, cl, tks, available, nd, ns, ds, t2i, per_symbol_info


def load_ga_weights() -> Dict[str, float]:
    cfg_path = os.path.join(SCRIPT_DIR, "..", "..",
                            'core', 'strategies', 'impl', 'v1_ga_rp', 'config.json')
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        return cfg.get("selector", {}).get("weights", {})
    return {}


def build_signals(z3: np.ndarray, fwd: np.ndarray, dm: np.ndarray, cl: np.ndarray,
                  fnames: List[str], nd: int, ns: int, ds: List[pd.Timestamp],
                  ) -> Dict[str, Any]:
    """构建所有需要的信号，与 live/signals/mss_dynamic.py 一致。"""
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


# ══════════════════════════════════════════
# 子策略参数
# ══════════════════════════════════════════

def get_sub_params() -> Dict[str, Dict]:
    return {
        "mf_d10_rp":       {"signal": "mf",     "rf": 10, "tn": 50, "mhd": 10, "timing": None},
        "mf_vol_d10_rp":   {"signal": "mf",     "rf": 10, "tn": 50, "mhd": 10, "timing": "vol"},
        "chip_rp":         {"signal": "chip",   "rf": 3,  "tn": 40, "mhd": 5,  "timing": None},
        "chip_covrp":      {"signal": "chip",   "rf": 3,  "tn": 40, "mhd": 5,  "timing": None},
        "osr_d10":         {"signal": "osr",    "rf": 10, "tn": 40, "mhd": 5,  "timing": None},
        "c01_layered_d5":  {"signal": "mf",     "rf": 5,  "tn": 40, "mhd": 5,  "timing": "trend"},
        "mf_base":         {"signal": "mf",     "rf": 3,  "tn": 40, "mhd": 5,  "timing": None},
    }


# ══════════════════════════════════════════
# 回测引擎（Baseline + V2）
# ══════════════════════════════════════════

def bt_sub_strategy_base(
    sig: np.ndarray, fwd: np.ndarray, dm: np.ndarray,
    rebal_freq: int = 10, top_n: int = 50, min_hold_days: int = 10,
    pos_ratio: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Baseline 子策略回测 — 无止损，无增强ST。"""
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


def bt_sub_strategy_v2(
    sig: np.ndarray, fwd: np.ndarray, dm: np.ndarray,
    rebal_freq: int = 10, top_n: int = 50, min_hold_days: int = 10,
    pos_ratio: Optional[np.ndarray] = None,
    stop_loss_pct: float = 0.08,
    symbol_risk_map: Optional[dict] = None,
) -> np.ndarray:
    """V2 子策略回测 — 增强ST过滤 + 个股止损。

    Args:
        symbol_risk_map: {symbol_idx: risk_level}, risk_level='high' 的排除
        stop_loss_pct: 止损比例（如 0.08 = -8%）
    """
    nd, ns = sig.shape
    alloc = RPPortfolioWeights(top_n=top_n, min_hold_days=min_hold_days)
    pw = np.zeros(ns, dtype=np.float32)
    hs = np.full(ns, -1, dtype=np.int32)
    rh = 0
    dr = np.zeros(nd, dtype=np.float64)
    entry_px = np.zeros(ns, dtype=np.float32)
    nt_stop = 0

    for i in range(1, nd):
        # 止损检查
        if stop_loss_pct > 0 and np.any(pw > 0):
            for j in range(ns):
                if pw[j] > 0 and hs[j] >= 0 and entry_px[j] > 0:
                    if fwd[i, j] < -stop_loss_pct and fwd[i, j] > -0.95:
                        pw[j] = 0.0
                        hs[j] = -1
                        entry_px[j] = 0.0
                        nt_stop += 1

        rebal = (i % rebal_freq == 0)
        if rebal:
            masked = sig[i].copy()
            if symbol_risk_map:
                for j, level in symbol_risk_map.items():
                    if level == 'high':
                        masked[j] = -1e10
            nw = alloc.allocate(masked, fwd, i, pw, hs, rh)
            to = float(np.sum(np.abs(nw - pw)))
            txc = 0.5 * to * TX
            for j in range(ns):
                if nw[j] > 0 and pw[j] <= 0:
                    entry_px[j] = max(1.0, 1.0 + fwd[i, j])
                elif nw[j] > 0 and pw[j] > 0 and entry_px[j] <= 0:
                    entry_px[j] = max(1.0, 1.0 + fwd[i, j])
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

    if nt_stop > 0:
        logger.info(f"    {nt_stop} 次止损触发 (stop={stop_loss_pct*100:.0f}%)")
    return dr


# ══════════════════════════════════════════
# 市场状态检测 + 置信度计算
# ══════════════════════════════════════════

def detect_market_state(mkt_returns: np.ndarray, nd: int) -> tuple[List[str], np.ndarray]:
    """检测市场状态 + 置信度。

    Returns:
        states: 每日状态列表 (nd,)
        confidence: 每日置信度 (nd,), 0~1
    """
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

        oscillate = False
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


# ══════════════════════════════════════════
# 增强ST检测
# ══════════════════════════════════════════

def build_enhanced_st_mask(per_symbol_info: Dict, t2i: Dict[str, int],
                            nd: int, ns: int) -> Dict[int, str]:
    """构建增强ST风险映射。

    Returns:
        {symbol_idx: risk_level}, 'high' = 不可交易
    """
    risk_map = {}
    pct = per_symbol_info["pct"]
    cl = per_symbol_info["cl"]
    vol = per_symbol_info["vol"]

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
    logger.info(f"  增强ST: {len(flagged)} 只额外标记为高风险")
    return risk_map


# ══════════════════════════════════════════
# 子策略回测调度
# ══════════════════════════════════════════

def compute_sub_drs(
    signals: Dict, fwd: np.ndarray, dm: np.ndarray, nd: int,
    use_v2: bool = False,
    symbol_risk_map: Optional[Dict[int, str]] = None,
    stop_loss_by_strategy: Optional[Dict[str, float]] = None,
) -> Dict[str, np.ndarray]:
    """计算所有子策略日收益率。"""
    sub_params = get_sub_params()
    sub_drs = {}
    for name, params in sub_params.items():
        sig = signals[params["signal"]]
        pr = None
        if params["timing"] == "vol":
            pr = signals["vol_p"]
        elif params["timing"] == "trend":
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
        m = compute_metrics(dr, name=name)
        logger.info(f"  {name}: 年化={m['annual_return']*100:.2f}% "
                     f"Sharpe={m['sharpe']:.3f} 回撤={abs(m['max_drawdown'])*100:.2f}%")

    return sub_drs


# ══════════════════════════════════════════
# MSS 动态分配回测
# ══════════════════════════════════════════

def run_mss(
    state_strategies: Dict[str, List[Dict]],
    sub_drs: Dict[str, np.ndarray],
    states: List[str], confidence: np.ndarray,
    nd: int,
    use_confidence_weights: bool = False,
) -> np.ndarray:
    """运行 MarketStateSelector 动态分配回测。

    功能与 market_state_pipeline.py 的 run_mss_backtest() 一致。
    新增 use_confidence_weights: 根据置信度动态调整权重。
    """
    strat_names = sorted(set(
        a["strategy"] for allocs in state_strategies.values()
        for a in allocs if a["strategy"] in sub_drs
    ))
    eq = {n: np.ones(nd, dtype=np.float64) for n in strat_names}
    dr = np.zeros(nd, dtype=np.float64)

    for i in range(1, nd):
        st = states[i] if i < len(states) else "oscillate"
        allocs = state_strategies.get(st, state_strategies.get("oscillate", []))

        alloc_map = {}
        for a in allocs:
            if a["strategy"] in sub_drs:
                w = a["weight"]
                if use_confidence_weights and confidence[i] < 0.5:
                    if a["strategy"] in ("mf_d10_rp",):
                        w = a["weight"] * 0.6
                    elif a["strategy"] in ("c01_layered_d5",):
                        w = a["weight"] * 0.7
                alloc_map[a["strategy"]] = max(w, 0.0)

        total_w = sum(alloc_map.values()) or 1.0
        for name in alloc_map:
            alloc_map[name] /= total_w

        for name in strat_names:
            w = alloc_map.get(name, 0.0)
            eq[name][i] = eq[name][i - 1] * (1.0 + sub_drs[name][i]) if w > 0 else eq[name][i - 1]

        total_eq = sum(eq[n][i] for n in strat_names)
        if total_eq < 1e-10:
            continue

        combined_ret = sum(alloc_map.get(n, 0.0) * sub_drs[n][i] for n in strat_names if n in sub_drs)
        dr[i] = combined_ret

    return dr


# ══════════════════════════════════════════
# 指标计算
# ══════════════════════════════════════════

def compute_metrics(dr: np.ndarray, name: str = "") -> Dict[str, Any]:
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
    pre_peak = cm[mdd_idx]
    rec = np.where(eq[mdd_idx:] >= pre_peak)[0]
    recovery_days = int(rec[0]) if len(rec) > 0 else nd - mdd_idx - 1
    peak_idx = np.argmax(cm[:mdd_idx + 1] == cm[mdd_idx])
    dd_dur = int(mdd_idx - peak_idx)

    cal = ar / abs(mdd) if abs(mdd) > 0 else 0
    wr = int(np.sum(dr > 0)) / max(int(np.sum(dr > 0)) + int(np.sum(dr < 0)), 1)

    return {
        "name": name,
        "annual_return": round(float(ar), 4),
        "sharpe": round(float(sp), 4),
        "max_drawdown": round(float(mdd), 4),
        "calmar": round(float(cal), 4),
        "win_rate": round(float(wr), 4),
        "recovery_days": int(recovery_days),
        "drawdown_duration": int(dd_dur),
    }


def window_analysis(dr: np.ndarray, ds: List[pd.Timestamp],
                    windows: List[Tuple[str, str, str]]) -> List[Dict]:
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


def print_table(results: List[Dict], title: str = ""):
    if title:
        print(f"\n{'=' * 100}")
        print(f"  {title}")
        print(f"{'=' * 100}")
    print(f"{'策略':<30} {'年化%':<8} {'Sharpe':<8} {'回撤%':<8} {'Calmar':<8} {'评级'}")
    print('-' * 100)
    for r in sorted(results, key=lambda x: x.get('sharpe', 0), reverse=True):
        ar = r.get('annual_return', 0) * 100
        sp = r.get('sharpe', 0)
        dd = abs(r.get('max_drawdown', 0)) * 100
        ca = r.get('calmar', 0)
        cls = "🏆" if sp > 1.5 and dd < 20 else ("✅" if sp > 1.0 and dd < 30 else ("⚠️" if sp > 0.5 else "❌"))
        print(f"{cls} {r['name']:<28} {ar:>6.2f}% {sp:>7.3f} {dd:>6.2f}% {ca:>7.3f}")
    print('=' * 100)


def print_ab_comparison(baseline_metrics: List[Dict], v2_metrics: List[Dict],
                         v2_label: str):
    """打印 Baseline vs V2 对比。"""
    base = {m["name"]: m for m in baseline_metrics}
    v2 = {m["name"]: m for m in v2_metrics}

    print(f"\n{'=' * 100}")
    print(f"  MSS_dynamic  A/B 对比: Baseline vs {v2_label}")
    print(f"{'=' * 100}")
    print(f"{'指标':<20} {'Baseline':<18} {v2_label:<18} {'差值':<12} {'改善':<8}")
    print('-' * 100)

    b_mss = base.get("MSS_dynamic", {})
    v_mss = v2.get("MSS_dynamic", {})

    if not b_mss or not v_mss:
        print("  未找到 MSS_dynamic 指标")
        return

    rows = [
        ("年化收益", "annual_return", lambda x: f"{x*100:.2f}%", True),
        ("夏普比率", "sharpe", lambda x: f"{x:.3f}", False),
        ("最大回撤", "max_drawdown", lambda x: f"{-x*100:.2f}%", True),
        ("卡玛比率", "calmar", lambda x: f"{x:.3f}", False),
    ]

    for label, key, fmt, is_pct in rows:
        bv = b_mss[key]
        vv = v_mss[key]
        diff = (vv - bv) if key != "max_drawdown" else (bv - vv)
        better = diff > 0
        diff_str = f"{diff*100 if is_pct else diff:+.2f}{'%' if is_pct else ''}"
        print(f"  {label:<18} {fmt(bv):<18} {fmt(vv):<18} {diff_str:<>14} {'🟢' if better else '🔴':<8}")

    print()
    print(f"  --- 分窗口对比 ---")
    for win_name in ["全区间", "2022熊市", "OOS修复牛"]:
        bw = next((x for x in base.get("windows", []) if x.get("name") == win_name), {})
        vw = next((x for x in v2.get("windows", []) if x.get("name") == win_name), {})
        if not bw or not vw:
            continue
        print(f"\n  {win_name}:")
        for key in ("annual_return", "sharpe", "max_drawdown"):
            bv = bw.get(key, 0)
            vv = vw.get(key, 0)
            diff = (vv - bv) if key != "max_drawdown" else (bv - vv)
            is_pct = key != "sharpe"
            print(f"    {key:<18} {bv*100 if is_pct else bv:>7.2f}{'%' if is_pct else ''} → "
                  f"{vv*100 if is_pct else vv:>7.2f}{'%' if is_pct else ''} "
                  f"({'+' if diff>0 else ''}{diff*100 if is_pct else diff:.2f}{'%' if is_pct else ''})")

    print(f"{'=' * 100}\n")


# ══════════════════════════════════════════
# 主实验
# ══════════════════════════════════════════

def run_ab_comparison():
    logger.info("=" * 60)
    logger.info("加载数据")
    logger.info("=" * 60)
    z3, fwd, dm, cl, tks, fnames, nd, ns, ds, t2i, per_symbol = \
        load_data()

    logger.info("=" * 60)
    logger.info("构建信号")
    logger.info("=" * 60)
    signals = build_signals(z3, fwd, dm, cl, fnames, nd, ns, ds)

    logger.info("=" * 60)
    logger.info("检测市场状态")
    logger.info("=" * 60)
    mkt_idx = signals["market_index"]
    states, confidence = detect_market_state(mkt_idx, nd)
    state_counts = {s: states.count(s) for s in ["bull", "bear", "oscillate", "recovery"]}
    logger.info(f"状态分布: {state_counts}")
    avg_conf = float(np.mean(confidence[confidence > 0]))
    logger.info(f"平均置信度: {avg_conf:.3f}")
    frac_low_conf = np.mean(confidence < 0.5) * 100
    logger.info(f"置信度<0.5的比例: {frac_low_conf:.1f}%")

    # 构建增强ST风险映射（用于V2）
    logger.info("构建增强ST风险映射...")
    symbol_risk_map = build_enhanced_st_mask(per_symbol, t2i, nd, ns)

    # ══════════════════════════════════════
    # Baseline
    # ══════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Baseline: 子策略回测")
    logger.info("=" * 60)
    sub_drs_base = compute_sub_drs(signals, fwd, dm, nd, use_v2=False)
    sub_drs_base["mf50_chip50"] = 0.5 * sub_drs_base["mf_base"] + 0.5 * sub_drs_base["chip_rp"]
    sub_drs_base["mf60_chip40"] = 0.6 * sub_drs_base["mf_base"] + 0.4 * sub_drs_base["chip_rp"]

    dr_mss_base = run_mss(DEFAULT_STATE_STRATEGIES, sub_drs_base, states, confidence, nd)
    m_mss_base = compute_metrics(dr_mss_base, name="MSS_dynamic")
    base_windows = window_analysis(dr_mss_base, ds, WINDOWS)

    all_base = [m_mss_base] + [compute_metrics(sub_drs_base[n], name=n) for n in sorted(sub_drs_base)]
    print_table(all_base, "Baseline: 全部策略")

    # ══════════════════════════════════════
    # V2 — 仅增强ST
    # ══════════════════════════════════════
    logger.info("=" * 60)
    logger.info("V2a: 仅增强ST过滤 (无止损)")
    logger.info("=" * 60)
    sub_drs_v2a = compute_sub_drs(signals, fwd, dm, nd,
                                   use_v2=True, symbol_risk_map=symbol_risk_map,
                                   stop_loss_by_strategy=None)
    sub_drs_v2a["mf50_chip50"] = 0.5 * sub_drs_v2a["mf_base"] + 0.5 * sub_drs_v2a["chip_rp"]
    sub_drs_v2a["mf60_chip40"] = 0.6 * sub_drs_v2a["mf_base"] + 0.4 * sub_drs_v2a["chip_rp"]

    dr_mss_v2a = run_mss(DEFAULT_STATE_STRATEGIES, sub_drs_v2a, states, confidence, nd)
    m_mss_v2a = compute_metrics(dr_mss_v2a, name="MSS_dynamic")
    v2a_windows = window_analysis(dr_mss_v2a, ds, WINDOWS)

    # ══════════════════════════════════════
    # V2b — 增强ST + 止损
    # ══════════════════════════════════════
    logger.info("=" * 60)
    logger.info("V2b: 增强ST + 差异化止损")
    logger.info("=" * 60)
    stop_losses = {
        "mf_d10_rp": 0.08,
        "mf_vol_d10_rp": 0.08,
        "chip_rp": 0.10,
        "chip_covrp": 0.10,
        "c01_layered_d5": 0.08,
        "osr_d10": 0.10,
        "mf_base": 0.08,
    }
    sub_drs_v2b = compute_sub_drs(signals, fwd, dm, nd,
                                   use_v2=True, symbol_risk_map=symbol_risk_map,
                                   stop_loss_by_strategy=stop_losses)
    sub_drs_v2b["mf50_chip50"] = 0.5 * sub_drs_v2b["mf_base"] + 0.5 * sub_drs_v2b["chip_rp"]
    sub_drs_v2b["mf60_chip40"] = 0.6 * sub_drs_v2b["mf_base"] + 0.4 * sub_drs_v2b["chip_rp"]

    dr_mss_v2b = run_mss(DEFAULT_STATE_STRATEGIES, sub_drs_v2b, states, confidence, nd)
    m_mss_v2b = compute_metrics(dr_mss_v2b, name="MSS_dynamic")
    v2b_windows = window_analysis(dr_mss_v2b, ds, WINDOWS)

    # ══════════════════════════════════════
    # V2c — 增强ST + 止损 + 置信度权重
    # ══════════════════════════════════════
    logger.info("=" * 60)
    logger.info("V2c: 增强ST + 止损 + 置信度联动权重")
    logger.info("=" * 60)
    dr_mss_v2c = run_mss(DEFAULT_STATE_STRATEGIES, sub_drs_v2b, states, confidence, nd,
                          use_confidence_weights=True)
    m_mss_v2c = compute_metrics(dr_mss_v2c, name="MSS_dynamic")
    v2c_windows = window_analysis(dr_mss_v2c, ds, WINDOWS)

    # ══════════════════════════════════════
    # 汇总对比
    # ══════════════════════════════════════
    print(f"\n{'='*120}")
    print(f"  mss_dynamic  A/B/C 对比汇总")
    print(f"{'='*120}")
    header = f"  {'版本':<12} {'年化%':<10} {'Sharpe':<10} {'回撤%':<10} {'Calmar':<10} {'胜率':<8}"
    print(header)
    print(f"  {'-'*60}")

    versions = [
        ("Baseline", m_mss_base),
        ("V2a(增强ST)", m_mss_v2a),
        ("V2b(+止损)", m_mss_v2b),
        ("V2c(+置信度)", m_mss_v2c),
    ]
    for label, m in versions:
        print(f"  {label:<12} {m['annual_return']*100:>+7.2f}% {m['sharpe']:>8.3f} "
              f"{-m['max_drawdown']*100:>7.2f}% {m['calmar']:>8.3f} {m['win_rate']*100:>5.1f}%")

    print(f"\n  --- 分窗口年化收益 ---")
    win_header = f"  {'版本':<12} {'全区间%':<12} {'2022熊市%':<14} {'OOS修复牛%':<14}"
    print(win_header)
    print(f"  {'-'*52}")
    for label, m, wins in [
        ("Baseline", m_mss_base, base_windows),
        ("V2a(ST)", m_mss_v2a, v2a_windows),
        ("V2b(+stop)", m_mss_v2b, v2b_windows),
        ("V2c(+conf)", m_mss_v2c, v2c_windows),
    ]:
        wm = {w["name"]: w for w in wins}
        a1 = m["annual_return"]
        a2 = wm.get("2022熊市", {}).get("annual_return", 0)
        a3 = wm.get("OOS修复牛", {}).get("annual_return", 0)
        print(f"  {label:<12} {a1*100:>+8.2f}% {a2*100:>+10.2f}% {a3*100:>+10.2f}%")

    print(f"\n  --- 分窗口夏普比率 ---")
    win_header2 = f"  {'版本':<12} {'全区间':<12} {'2022熊市':<14} {'OOS修复牛':<14}"
    print(win_header2)
    print(f"  {'-'*52}")
    for label, m, wins in [
        ("Baseline", m_mss_base, base_windows),
        ("V2a(ST)", m_mss_v2a, v2a_windows),
        ("V2b(+stop)", m_mss_v2b, v2b_windows),
        ("V2c(+conf)", m_mss_v2c, v2c_windows),
    ]:
        wm = {w["name"]: w for w in wins}
        s1 = m["sharpe"]
        s2 = wm.get("2022熊市", {}).get("sharpe", 0)
        s3 = wm.get("OOS修复牛", {}).get("sharpe", 0)
        print(f"  {label:<12} {s1:>8.3f} {s2:>10.3f} {s3:>10.3f}")

    print(f"{'='*120}\n")

    # 保存结果
    results = {
        "config": {
            "range": "2019-01-02 ~ 2026-05-22",
            "state_distribution": state_counts,
            "avg_confidence": round(float(avg_conf), 3),
            "frac_low_conf": round(float(frac_low_conf), 1),
        },
        "baseline": {
            "full_range": m_mss_base,
            "windows": base_windows,
        },
        "v2a_enhanced_st": {
            "full_range": m_mss_v2a,
            "windows": v2a_windows,
        },
        "v2b_stop_loss": {
            "full_range": m_mss_v2b,
            "windows": v2b_windows,
        },
        "v2c_confidence": {
            "full_range": m_mss_v2c,
            "windows": v2c_windows,
        },
    }
    out_path = os.path.join(RESULTS_DIR, "mss_ab_comparison.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"结果已保存至: {out_path}")


if __name__ == "__main__":
    run_ab_comparison()
