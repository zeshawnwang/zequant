"""
MarketStateSelector 动态策略切换实验管道。

核心思路：根据市场状态（牛/熊/震荡/反弹）自动切换子策略组合，
让每个市场环境下都用最合适的策略。

实验流程：
  Phase 1: 加载数据 + 构建所有子策略信号
  Phase 2: 预计算各子策略日收益率序列
  Phase 3: 实现 MarketStateSelector 动态分配回测
  Phase 4: 对比静态组合和各子策略
  Phase 5: 窗口分析 + 参数调优

用法：
    python3 daily/2026-05-18/v2/market_state_pipeline.py --experiment v1_baseline
    python3 daily/2026-05-18/v2/market_state_pipeline.py --experiment tune_detection
    python3 daily/2026-05-18/v2/market_state_pipeline.py --experiment tune_allocation
    python3 daily/2026-05-18/v2/market_state_pipeline.py --experiment final
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

import duckdb
from core.positioners import RPPortfolioWeights

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("mss_experiment")

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
# Phase 1: 数据加载 & 信号构建
# ═══════════════════════════════════════════════

import shutil


def _get_conn():
    src = os.path.abspath("./data/quant_data.db")
    return duckdb.connect(src, read_only=True)


def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                          List[str], List[str], int, int, List[pd.Timestamp]]:
    """加载因子数据并构建3D张量。

    返回 (z3, fwd, dm, cl, tks, fnames, nd, ns, ds)
    """
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


def build_signals(z3: np.ndarray, fwd: np.ndarray, dm: np.ndarray, cl: np.ndarray,
                  fnames: List[str], nd: int, ns: int, ds: List[pd.Timestamp],
                  mf_weights: Optional[Dict[str, float]] = None,
                  ) -> Dict[str, Any]:
    """构建所有需要的信号。

    Args:
        mf_weights: 因子权重字典 {因子名: 权重}，None=自动加载V1 GA权重

    返回:
        signals = {
            "mf": MF信号矩阵 (nd×ns),
            "chip": Chip信号矩阵 (nd×ns),
            "osr": OSR信号矩阵 (nd×ns),
            "vol_p": 波动率择时向量 (nd,),
            "trend_p": 趋势择时向量 (nd,),
            "fi": 因子名→索引的字典,
            "market_index": 等权市场指数 (nd,),
        }
    """
    fi = {fn: i for i, fn in enumerate(fnames)}

    # ── MF信号：GA优化加权或等权因子合成 ──
    if mf_weights is None:
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

    # ── Chip信号：波动率低 + 动量弱 ──
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

    # ── OSR信号：超跌反弹 ──
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

    # ── 波动率择时信号 ──
    vol_p = np.ones(nd, dtype=np.float32)
    if vol20_idx is not None:
        vol_p = np.clip(1.0 - np.mean(z3[:, :, vol20_idx] > 0.05, axis=1), 0.2, 1.0)

    # ── 趋势择时信号 ──
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

    # ── 市场指数（等权） ──
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
# 子策略回测引擎
# ═══════════════════════════════════════════════

def bt_sub_strategy(sig: np.ndarray, fwd: np.ndarray, dm: np.ndarray,
                    rebal_freq: int = 10, top_n: int = 50, min_hold_days: int = 10,
                    pos_ratio: Optional[np.ndarray] = None) -> np.ndarray:
    """对单一信号矩阵运行标准回测，返回日收益率序列 (nd,)。

    Args:
        sig: 信号矩阵 (nd×ns)
        fwd: 前向收益 (nd×ns)
        dm: 数据掩码 (nd×ns)
        rebal_freq: 调仓频率（天数）
        top_n: 选股数量
        min_hold_days: 最低持仓天数
        pos_ratio: 仓位系数 (nd,)，None=满仓
    """
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


# ═══════════════════════════════════════════════
# 市场状态检测
# ═══════════════════════════════════════════════

def detect_market_state_from_idx(mkt_returns: np.ndarray, nd: int) -> List[str]:
    """从市场指数收益率序列，每日检测市场状态。

    模拟 MarketStateSelector.detect_market_state 的逻辑，
    使用等权市场指数收盘价计算均线和斜率。

    Args:
        mkt_returns: 市场指数日收益率 (nd,)
        nd: 总天数

    Returns:
        states: 每日市场状态列表 (nd,) ∈ {"bull","bear","oscillate","recovery"}
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

        bull_cond = above_ma200 > 0 and ma20_slope > 0
        bear_cond = above_ma200 < 0 and ma20_slope < 0 and ma60_slope < 0
        recovery_cond = above_ma200 < 0 and ma5_slope > 0.005

        oscillate_cond = False
        if pd.notna(ma5[i]) and pd.notna(ma20[i]) and pd.notna(ma60[i]):
            spread = abs(ma5[i] - ma20[i]) / max(abs(ma20[i]), 1e-10) + abs(ma20[i] - ma60[i]) / max(abs(ma60[i]), 1e-10)
            oscillate_cond = spread < 0.03

        if bull_cond:
            states[i] = "bull"
        elif bear_cond:
            states[i] = "bear"
        elif recovery_cond:
            states[i] = "recovery"
        elif oscillate_cond:
            states[i] = "oscillate"
        elif above_ma200 < 0 and ma5_slope > 0:
            states[i] = "recovery"
        else:
            states[i] = "oscillate"

    return states


def detect_market_state_tuned(mkt_returns: np.ndarray, nd: int,
                              bull_ma200_thresh: float = 0.0,
                              bull_ma20_slope_thresh: float = 0.0,
                              bear_ma20_slope_thresh: float = 0.0,
                              bear_ma60_slope_thresh: float = 0.0,
                              recovery_ma5_slope_thresh: float = 0.005,
                              oscillate_spread_thresh: float = 0.03,
                              ) -> List[str]:
    """带参数的市场状态检测。"""
    idx_price = np.zeros(nd, dtype=np.float64)
    idx_price[0] = 1000.0
    for i in range(1, nd):
        idx_price[i] = idx_price[i - 1] * (1.0 + mkt_returns[i])

    ma5 = pd.Series(idx_price).rolling(5).mean().values
    ma20 = pd.Series(idx_price).rolling(20).mean().values
    ma60 = pd.Series(idx_price).rolling(60).mean().values
    ma200 = pd.Series(idx_price).rolling(200).mean().values

    states = ["oscillate"] * nd
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

        if above_ma200 > bull_ma200_thresh and ma20_slope > bull_ma20_slope_thresh:
            states[i] = "bull"
        elif above_ma200 < -bull_ma200_thresh and ma20_slope < bear_ma20_slope_thresh and ma60_slope < bear_ma60_slope_thresh:
            states[i] = "bear"
        elif above_ma200 < 0 and ma5_slope > recovery_ma5_slope_thresh:
            states[i] = "recovery"
        elif pd.notna(ma5[i]) and pd.notna(ma20[i]) and pd.notna(ma60[i]):
            spread = abs(ma5[i] - ma20[i]) / max(abs(ma20[i]), 1e-10) + abs(ma20[i] - ma60[i]) / max(abs(ma60[i]), 1e-10)
            if spread < oscillate_spread_thresh:
                states[i] = "oscillate"
            elif above_ma200 < 0 and ma5_slope > 0:
                states[i] = "recovery"
            else:
                states[i] = "oscillate"
        else:
            states[i] = "oscillate"

    return states


# ═══════════════════════════════════════════════
# MarketStateSelector 动态分配回测
# ═══════════════════════════════════════════════

def run_mss_backtest(
    state_strategies: Dict[str, List[Dict]],
    sub_drs: Dict[str, np.ndarray],
    states: List[str],
    nd: int,
    rebal_cost: float = 0.001,
) -> np.ndarray:
    """运行 MarketStateSelector 动态分配回测。

    每个子策略有独立的notional equity曲线，每天根据市场状态重新分配资金。
    分配变化时产生再平衡成本。

    Args:
        state_strategies: 状态→策略分配，格式同 MarketStateSelector.state_strategies
        sub_drs: 子策略名→日收益率序列 (nd,)
        states: 每日市场状态 (nd,)
        nd: 总天数
        rebal_cost: 分配再平衡费率

    Returns:
        dr: 合并后的日收益率序列 (nd,)
    """
    n_strats = {}
    for state, allocs in state_strategies.items():
        for a in allocs:
            if a["strategy"] in sub_drs:
                n_strats[a["strategy"]] = True

    strat_names = sorted(n_strats.keys())

    eq = {name: np.ones(nd, dtype=np.float64) for name in strat_names}
    prev_alloc = {name: 0.0 for name in strat_names}
    dr = np.zeros(nd, dtype=np.float64)

    for i in range(1, nd):
        state = states[i] if i < len(states) else "oscillate"
        allocs = state_strategies.get(state, state_strategies.get("oscillate", []))

        alloc_map = {}
        for a in allocs:
            if a["strategy"] in sub_drs:
                alloc_map[a["strategy"]] = a["weight"]

        total_w = sum(alloc_map.values()) if alloc_map else 1.0
        if total_w > 0:
            alloc_map = {k: v / total_w for k, v in alloc_map.items()}

        for name in strat_names:
            w = alloc_map.get(name, 0.0)
            if w > 0:
                eq[name][i] = eq[name][i - 1] * (1.0 + sub_drs[name][i])
            else:
                eq[name][i] = eq[name][i - 1]

        total_eq = sum(eq[name][i] for name in strat_names)
        if total_eq < 1e-10:
            dr[i] = 0.0
            prev_alloc = alloc_map.copy()
            continue

        combined_ret = 0.0
        rebal_txc = 0.0
        for name in strat_names:
            w = alloc_map.get(name, 0.0)
            weight_in_total = eq[name][i] / total_eq if total_eq > 0 else 0.0
            combined_ret += w * sub_drs[name][i]

            alloc_change = abs(w - prev_alloc.get(name, 0.0))
            rebal_txc += alloc_change * rebal_cost

        dr[i] = combined_ret - rebal_txc
        prev_alloc = alloc_map.copy()

    return dr


def run_mss_simple_backtest(
    state_strategies: Dict[str, List[Dict]],
    sub_drs: Dict[str, np.ndarray],
    states: List[str],
    nd: int,
) -> np.ndarray:
    """简化的 MarketStateSelector 回测——直接按状态权重加权各子策略日收益。

    不追踪各子策略独立净值，而是每天直接按状态权重混合各子策略收益率。
    这种更简单，但忽略了子策略间的业绩分化。
    """
    dr = np.zeros(nd, dtype=np.float64)
    for i in range(1, nd):
        state = states[i] if i < len(states) else "oscillate"
        allocs = state_strategies.get(state, state_strategies.get("oscillate", []))

        combined_ret = 0.0
        total_w = 0.0
        for a in allocs:
            name = a["strategy"]
            w = a["weight"]
            if name in sub_drs:
                combined_ret += w * sub_drs[name][i]
                total_w += w

        dr[i] = combined_ret / total_w if total_w > 0 else 0.0

    return dr


# ═══════════════════════════════════════════════
# 指标计算
# ═══════════════════════════════════════════════

def compute_metrics(dr: np.ndarray, name: str = "") -> Dict[str, Any]:
    """从日收益率序列计算完整回测指标。"""
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
    pre_peak_val = cm[mdd_idx]
    rec = np.where(eq[mdd_idx:] >= pre_peak_val)[0]
    recovery_days = int(rec[0]) if len(rec) > 0 else nd - mdd_idx - 1

    peak_idx = np.argmax(cm[:mdd_idx + 1] == cm[mdd_idx])
    drawdown_duration = int(mdd_idx - peak_idx)

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
        "drawdown_duration": int(drawdown_duration),
    }


def window_analysis(dr: np.ndarray, ds: List[pd.Timestamp],
                    windows: List[Tuple[str, str, str]]) -> List[Dict]:
    """分窗口分析。"""
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


def print_results_table(results: List[Dict], title: str = ""):
    """打印结果表格。"""
    if title:
        print(f"\n{'=' * 100}")
        print(f"  {title}")
        print(f"{'=' * 100}")
    print(f"{'策略':<30} {'年化%':<8} {'Sharpe':<8} {'回撤%':<8} {'Calmar':<8} {'修复(d)':<8} {'评级'}")
    print('-' * 100)
    for r in sorted(results, key=lambda x: x.get('sharpe', 0), reverse=True):
        ar = r.get('annual_return', 0) * 100
        sp = r.get('sharpe', 0)
        dd = abs(r.get('max_drawdown', 0)) * 100
        ca = r.get('calmar', 0)
        rd = r.get('recovery_days', 9999)
        cls = "🏆" if dd < 20 and sp > 1.0 else ("✅" if dd < 30 and sp > 0.5 else ("⚠️" if dd < 40 else "❌"))
        print(f"{cls} {r['name']:<28} {ar:>6.2f}% {sp:>7.3f} {dd:>6.2f}% {ca:>7.3f} {rd:>4}d")
    print('=' * 100)


# ═══════════════════════════════════════════════
# MarketStateSelector 默认配置
# ═══════════════════════════════════════════════

DEFAULT_STATE_STRATEGIES = {
    "bull": [
        {"strategy": "mf_d10_rp",     "weight": 0.6, "note": "牛市无择时满仓"},
        {"strategy": "mf_vol_d10_rp", "weight": 0.2, "note": "波动保护"},
        {"strategy": "chip_rp",       "weight": 0.2, "note": "低波动防御"},
    ],
    "bear": [
        {"strategy": "chip_vol_rp",   "weight": 0.5, "note": "熊市主打防守"},
        {"strategy": "chip_covrp",    "weight": 0.3, "note": "低风险底仓"},
        {"strategy": "mf_vol_d10_rp", "weight": 0.2, "note": "少量做多"},
    ],
    "oscillate": [
        {"strategy": "mf50_chip50",   "weight": 0.4, "note": "均衡配置"},
        {"strategy": "chip_covrp",    "weight": 0.3, "note": "低回撤底仓"},
        {"strategy": "c01_layered_d5","weight": 0.3, "note": "趋势择时"},
    ],
    "recovery": [
        {"strategy": "mf60_chip40",   "weight": 0.5, "note": "温和进攻"},
        {"strategy": "osr_d10",       "weight": 0.3, "note": "超跌反弹"},
        {"strategy": "mf_vol_d10_rp", "weight": 0.2, "note": "波动保护"},
    ],
}


# ═══════════════════════════════════════════════
# 子策略定义
# ═══════════════════════════════════════════════

def get_sub_strategy_params() -> Dict[str, Dict]:
    """所有子策略的参数定义。"""
    return {
        "mf_d10_rp":       {"signal": "mf",     "rf": 10, "tn": 50, "mhd": 10, "timing": None},
        "mf_vol_d10_rp":   {"signal": "mf",     "rf": 10, "tn": 50, "mhd": 10, "timing": "vol"},
        "chip_rp":         {"signal": "chip",   "rf": 3,  "tn": 40, "mhd": 5,  "timing": None},
        "chip_vol_rp":     {"signal": "chip",   "rf": 3,  "tn": 40, "mhd": 5,  "timing": "vol"},
        "chip_covrp":      {"signal": "chip",   "rf": 3,  "tn": 40, "mhd": 5,  "timing": None},
        "osr_d10":         {"signal": "osr",    "rf": 10, "tn": 40, "mhd": 5,  "timing": None},
        "c01_layered_d5":  {"signal": "mf",     "rf": 5,  "tn": 40, "mhd": 5,  "timing": "trend"},
        "mf_base":         {"signal": "mf",     "rf": 3,  "tn": 40, "mhd": 5,  "timing": None},
    }


# ═══════════════════════════════════════════════
# Phase 2: 预计算各子策略日收益率
# ═══════════════════════════════════════════════

def load_ga_weights() -> Dict[str, float]:
    """加载V1 GA优化因子权重。"""
    cfg_path = os.path.join(
        os.path.dirname(__file__), '..', '..', '..',
        'core', 'strategies', 'impl', 'v1_ga_rp', 'config.json',
    )
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        return cfg.get("selector", {}).get("weights", {})
    return {}


def compute_all_sub_strategy_drs(signals: Dict, fwd: np.ndarray, dm: np.ndarray,
                                  nd: int) -> Dict[str, np.ndarray]:
    """预计算所有子策略的日收益率序列。"""
    sub_params = get_sub_strategy_params()
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
        m = compute_metrics(dr, name=name)
        ar = m["annual_return"] * 100
        sp = m["sharpe"]
        dd = m["max_drawdown"] * 100
        logger.info(f"  {name}: 年化={ar:.2f}% Sharpe={sp:.3f} 回撤={dd:.2f}%")

    return sub_drs


# ═══════════════════════════════════════════════
# 实验入口
# ═══════════════════════════════════════════════

def run_experiment_v1_baseline():
    """实验V1: 运行MarketStateSelector原始配置，对比静态组合和独立子策略。"""
    logger.info("=" * 60)
    logger.info("Phase 1: 加载数据")
    logger.info("=" * 60)
    z3, fwd, dm, cl, tks, fnames, nd, ns, ds = load_data()

    logger.info("=" * 60)
    logger.info("Phase 2: 构建信号")
    logger.info("=" * 60)
    signals = build_signals(z3, fwd, dm, cl, fnames, nd, ns, ds)

    logger.info("=" * 60)
    logger.info("Phase 3: 预计算子策略日收益率")
    logger.info("=" * 60)
    sub_drs = compute_all_sub_strategy_drs(signals, fwd, dm, nd)

    # 预计算combo策略
    sub_drs["mf50_chip50"] = 0.5 * sub_drs["mf_base"] + 0.5 * sub_drs["chip_rp"]
    sub_drs["mf60_chip40"] = 0.6 * sub_drs["mf_base"] + 0.4 * sub_drs["chip_rp"]

    logger.info("=" * 60)
    logger.info("Phase 4: 市场状态检测")
    logger.info("=" * 60)
    mkt_idx = signals["market_index"]
    close = signals["close"]

    states = detect_market_state_from_idx(mkt_idx, nd)
    state_counts = {s: states.count(s) for s in ["bull", "bear", "oscillate", "recovery"]}
    logger.info(f"状态分布: {state_counts}")

    total_days = nd
    for s, c in state_counts.items():
        pct = c / max(total_days, 1) * 100
        logger.info(f"  {s}: {c}天 ({pct:.1f}%)")

    logger.info("=" * 60)
    logger.info("Phase 5: 运行回测")
    logger.info("=" * 60)

    all_results = []

    # 5.1 各独立子策略
    for name in sorted(sub_drs.keys()):
        m = compute_metrics(sub_drs[name], name=name)
        all_results.append(m)

    # 5.2 MarketStateSelector 动态分配（简化版）
    dr_mss_simple = run_mss_simple_backtest(
        DEFAULT_STATE_STRATEGIES, sub_drs, states, nd,
    )
    m = compute_metrics(dr_mss_simple, name="MSS_dynamic_simple")
    all_results.append(m)

    # 5.3 MarketStateSelector 动态分配（净值追踪版）
    dr_mss_full = run_mss_backtest(
        DEFAULT_STATE_STRATEGIES, sub_drs, states, nd,
    )
    m = compute_metrics(dr_mss_full, name="MSS_dynamic_full")
    all_results.append(m)

    # 5.4 静态组合（固定权重，作为对比基线）
    static_alloc = {"mf_d10_rp": 0.4, "chip_rp": 0.3, "mf_vol_d10_rp": 0.2, "osr_d10": 0.1}
    dr_static = np.zeros(nd, dtype=np.float64)
    for name, w in static_alloc.items():
        if name in sub_drs:
            dr_static += w * sub_drs[name]
    m = compute_metrics(dr_static, name="static_40_30_20_10")
    all_results.append(m)

    # 5.5 全部等权组合
    eq_names = [n for n in sub_drs.keys() if n not in ("mf50_chip50", "mf60_chip40", "mf_base")]
    dr_eq = np.zeros(nd, dtype=np.float64)
    for name in eq_names:
        dr_eq += sub_drs[name]
    dr_eq /= len(eq_names)
    m = compute_metrics(dr_eq, name="equal_weight_all")
    all_results.append(m)

    # 5.6 买入持有市场指数
    idx_eq = np.ones(nd, dtype=np.float64)
    for i in range(1, nd):
        idx_eq[i] = idx_eq[i - 1] * (1.0 + mkt_idx[i])
    mkt_dr = np.zeros(nd, dtype=np.float64)
    for i in range(1, nd):
        mkt_dr[i] = mkt_idx[i]
    m = compute_metrics(mkt_dr, name="market_index_ew")
    all_results.append(m)

    # 打印结果
    print_results_table(all_results, "V1 Baseline: MarketStateSelector 对比实验")

    # 窗口分析
    logger.info("=" * 60)
    logger.info("MSS 动态分配窗口分析 (simple)")
    logger.info("=" * 60)
    w_results = window_analysis(dr_mss_simple, ds, WINDOWS)
    for w in w_results:
        if w.get("n_days", 0) == 0:
            continue
        ar = w.get("annual_return", 0) * 100
        sp = w.get("sharpe", 0)
        dd = abs(w.get("max_drawdown", 0)) * 100
        logger.info(f"  {w['name']}: 年化={ar:.2f}% Sharpe={sp:.3f} 回撤={dd:.2f}%")

    # 状态转移分析
    transitions = 0
    for i in range(1, nd):
        if states[i] != states[i - 1]:
            transitions += 1
    logger.info(f"状态切换次数: {transitions}")

    # 存储结果
    out = {
        "experiment": "v1_baseline",
        "state_distribution": state_counts,
        "state_transitions": transitions,
        "all_results": all_results,
        "windows_mss_simple": [
            w for w in w_results if w.get("n_days", 0) > 0
        ],
    }
    out_path = os.path.join(RESULTS_DIR, "v1_baseline.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    logger.info("结果已保存到 %s", out_path)


def run_experiment_tune_detection():
    """实验: 调优市场状态检测参数。

    扫描关键阈值，寻找最佳状态划分。
    """
    logger.info("=" * 60)
    logger.info("加载数据")
    logger.info("=" * 60)
    z3, fwd, dm, cl, tks, fnames, nd, ns, ds = load_data()
    signals = build_signals(z3, fwd, dm, cl, fnames, nd, ns, ds)
    sub_drs = compute_all_sub_strategy_drs(signals, fwd, dm, nd)
    sub_drs["mf50_chip50"] = 0.5 * sub_drs["mf_base"] + 0.5 * sub_drs["chip_rp"]
    sub_drs["mf60_chip40"] = 0.6 * sub_drs["mf_base"] + 0.4 * sub_drs["chip_rp"]

    mkt_idx = signals["market_index"]

    scan_results = []

    scan_params = [
        # oscillate_spread_thresh
        {"oscillate_spread_thresh": 0.02, "label": "spread_0.02"},
        {"oscillate_spread_thresh": 0.03, "label": "spread_0.03"},
        {"oscillate_spread_thresh": 0.05, "label": "spread_0.05"},
        {"oscillate_spread_thresh": 0.08, "label": "spread_0.08"},

        # recovery slope
        {"recovery_ma5_slope_thresh": 0.003, "label": "rec_slope_0.003"},
        {"recovery_ma5_slope_thresh": 0.005, "label": "rec_slope_0.005"},
        {"recovery_ma5_slope_thresh": 0.008, "label": "rec_slope_0.008"},
        {"recovery_ma5_slope_thresh": 0.01,  "label": "rec_slope_0.01"},

        # bear more sensitive
        {"bear_ma20_slope_thresh": -0.001, "bear_ma60_slope_thresh": -0.001, "label": "bear_easy"},
        {"bear_ma20_slope_thresh": 0.0,    "bear_ma60_slope_thresh": 0.0,    "label": "bear_normal"},
        {"bear_ma20_slope_thresh": 0.002,  "bear_ma60_slope_thresh": 0.002,  "label": "bear_hard"},

        # bull more/less sensitive
        {"bull_ma20_slope_thresh": -0.001, "label": "bull_easy"},
        {"bull_ma20_slope_thresh": 0.0,    "label": "bull_normal"},
        {"bull_ma20_slope_thresh": 0.002,  "label": "bull_hard"},
    ]

    for params in scan_params:
        label = params.get("label", "unknown")
        states = detect_market_state_tuned(
            mkt_idx, nd,
            bull_ma20_slope_thresh=params.get("bull_ma20_slope_thresh", 0.0),
            bull_ma200_thresh=params.get("bull_ma200_thresh", 0.0),
            bear_ma20_slope_thresh=params.get("bear_ma20_slope_thresh", 0.0),
            bear_ma60_slope_thresh=params.get("bear_ma60_slope_thresh", 0.0),
            recovery_ma5_slope_thresh=params.get("recovery_ma5_slope_thresh", 0.005),
            oscillate_spread_thresh=params.get("oscillate_spread_thresh", 0.03),
        )
        state_counts = {s: states.count(s) for s in ["bull", "bear", "oscillate", "recovery"]}

        dr = run_mss_simple_backtest(DEFAULT_STATE_STRATEGIES, sub_drs, states, nd)
        m = compute_metrics(dr, name=f"MSS_detection_{label}")
        m["state_distribution"] = state_counts
        scan_results.append(m)

        logger.info(f"  {label}: 年化={m['annual_return']*100:.2f}% "
                    f"Sharpe={m['sharpe']:.3f} 回撤={m['max_drawdown']*100:.2f}% "
                    f"状态={state_counts}")

    print_results_table(scan_results, "Detection Threshold Tuning")

    out_path = os.path.join(RESULTS_DIR, "tune_detection.json")
    with open(out_path, "w") as f:
        json.dump(scan_results, f, indent=2, ensure_ascii=False)
    logger.info("结果已保存到 %s", out_path)

    return scan_results


def run_experiment_tune_allocation():
    """实验: 调优状态→策略映射和权重。

    基于最佳检测参数，尝试不同的状态→策略分配方案。
    """
    logger.info("=" * 60)
    logger.info("加载数据")
    logger.info("=" * 60)
    z3, fwd, dm, cl, tks, fnames, nd, ns, ds = load_data()
    signals = build_signals(z3, fwd, dm, cl, fnames, nd, ns, ds)
    sub_drs = compute_all_sub_strategy_drs(signals, fwd, dm, nd)
    sub_drs["mf50_chip50"] = 0.5 * sub_drs["mf_base"] + 0.5 * sub_drs["chip_rp"]
    sub_drs["mf60_chip40"] = 0.6 * sub_drs["mf_base"] + 0.4 * sub_drs["chip_rp"]

    mkt_idx = signals["market_index"]
    states = detect_market_state_from_idx(mkt_idx, nd)

    alloc_variants = [
        {
            "name": "MSS_v1_original",
            "bull": [("mf_d10_rp", 0.6), ("mf_vol_d10_rp", 0.2), ("chip_rp", 0.2)],
            "bear": [("chip_vol_rp", 0.5), ("chip_covrp", 0.3), ("mf_vol_d10_rp", 0.2)],
            "oscillate": [("mf50_chip50", 0.4), ("chip_covrp", 0.3), ("c01_layered_d5", 0.3)],
            "recovery": [("mf60_chip40", 0.5), ("osr_d10", 0.3), ("mf_vol_d10_rp", 0.2)],
        },
        {
            "name": "MSS_v2_mf_heavy_bear_defense",
            "bull": [("mf_d10_rp", 0.7), ("chip_rp", 0.2), ("mf_vol_d10_rp", 0.1)],
            "bear": [("chip_vol_rp", 0.4), ("chip_covrp", 0.4), ("mf_vol_d10_rp", 0.2)],
            "oscillate": [("mf50_chip50", 0.5), ("chip_covrp", 0.3), ("c01_layered_d5", 0.2)],
            "recovery": [("mf60_chip40", 0.4), ("osr_d10", 0.3), ("mf_vol_d10_rp", 0.3)],
        },
        {
            "name": "MSS_v3_bear_covrp_heavy",
            "bull": [("mf_d10_rp", 0.6), ("chip_rp", 0.2), ("mf_vol_d10_rp", 0.2)],
            "bear": [("chip_covrp", 0.6), ("chip_vol_rp", 0.2), ("mf_vol_d10_rp", 0.2)],
            "oscillate": [("chip_covrp", 0.4), ("mf50_chip50", 0.3), ("c01_layered_d5", 0.3)],
            "recovery": [("osr_d10", 0.4), ("mf60_chip40", 0.3), ("mf_vol_d10_rp", 0.3)],
        },
        {
            "name": "MSS_v4_trend_oscillate",
            "bull": [("mf_d10_rp", 0.5), ("chip_rp", 0.3), ("mf_vol_d10_rp", 0.2)],
            "bear": [("chip_covrp", 0.5), ("chip_vol_rp", 0.3), ("mf_vol_d10_rp", 0.2)],
            "oscillate": [("c01_layered_d5", 0.5), ("chip_covrp", 0.3), ("mf50_chip50", 0.2)],
            "recovery": [("mf60_chip40", 0.4), ("osr_d10", 0.3), ("c01_layered_d5", 0.3)],
        },
        {
            "name": "MSS_v5_simple",
            "bull": [("mf_d10_rp", 0.7), ("chip_rp", 0.3)],
            "bear": [("chip_covrp", 0.7), ("mf_vol_d10_rp", 0.3)],
            "oscillate": [("mf50_chip50", 0.5), ("chip_covrp", 0.5)],
            "recovery": [("mf60_chip40", 0.6), ("osr_d10", 0.4)],
        },
        {
            "name": "MSS_v6_mf_dominant",
            "bull": [("mf_d10_rp", 0.8), ("chip_rp", 0.2)],
            "bear": [("chip_covrp", 0.5), ("mf_vol_d10_rp", 0.3), ("chip_vol_rp", 0.2)],
            "oscillate": [("mf_d10_rp", 0.5), ("chip_covrp", 0.3), ("c01_layered_d5", 0.2)],
            "recovery": [("mf_d10_rp", 0.5), ("osr_d10", 0.3), ("mf_vol_d10_rp", 0.2)],
        },
        {
            "name": "MSS_v7_recovery_osr",
            "bull": [("mf_d10_rp", 0.6), ("chip_rp", 0.2), ("mf_vol_d10_rp", 0.2)],
            "bear": [("chip_covrp", 0.4), ("chip_vol_rp", 0.3), ("mf_vol_d10_rp", 0.3)],
            "oscillate": [("mf50_chip50", 0.4), ("chip_covrp", 0.3), ("c01_layered_d5", 0.3)],
            "recovery": [("osr_d10", 0.5), ("c01_layered_d5", 0.3), ("mf_vol_d10_rp", 0.2)],
        },
        {
            "name": "MSS_v8_conservative",
            "bull": [("mf_vol_d10_rp", 0.5), ("chip_rp", 0.3), ("mf_d10_rp", 0.2)],
            "bear": [("chip_covrp", 0.5), ("chip_vol_rp", 0.3), ("mf_vol_d10_rp", 0.2)],
            "oscillate": [("chip_covrp", 0.4), ("c01_layered_d5", 0.3), ("mf50_chip50", 0.3)],
            "recovery": [("chip_covrp", 0.4), ("osr_d10", 0.3), ("mf_vol_d10_rp", 0.3)],
        },
    ]

    alloc_results = []
    for variant in alloc_variants:
        ssm = {}
        for state in ["bull", "bear", "oscillate", "recovery"]:
            ssm[state] = [
                {"strategy": name, "weight": w, "note": ""}
                for name, w in variant[state]
            ]

        dr = run_mss_simple_backtest(ssm, sub_drs, states, nd)
        m = compute_metrics(dr, name=variant["name"])
        alloc_results.append(m)
        logger.info(f"  {variant['name']}: 年化={m['annual_return']*100:.2f}% "
                    f"Sharpe={m['sharpe']:.3f} 回撤={m['max_drawdown']*100:.2f}%")

    print_results_table(alloc_results, "Allocation Variant Tuning")

    out_path = os.path.join(RESULTS_DIR, "tune_allocation.json")
    with open(out_path, "w") as f:
        json.dump(alloc_results, f, indent=2, ensure_ascii=False)
    logger.info("结果已保存到 %s", out_path)

    return alloc_results


def run_experiment_final():
    """最终实验: 最佳配置 + 完整窗口分析 + 对比基线。

    基于前序实验的最佳参数，做完整的验证。
    """
    logger.info("=" * 60)
    logger.info("最终实验: 加载数据 + 完整分析")
    logger.info("=" * 60)

    z3, fwd, dm, cl, tks, fnames, nd, ns, ds = load_data()
    signals = build_signals(z3, fwd, dm, cl, fnames, nd, ns, ds)
    sub_drs = compute_all_sub_strategy_drs(signals, fwd, dm, nd)
    sub_drs["mf50_chip50"] = 0.5 * sub_drs["mf_base"] + 0.5 * sub_drs["chip_rp"]
    sub_drs["mf60_chip40"] = 0.6 * sub_drs["mf_base"] + 0.4 * sub_drs["chip_rp"]

    mkt_idx = signals["market_index"]
    states = detect_market_state_from_idx(mkt_idx, nd)

    state_counts = {s: states.count(s) for s in ["bull", "bear", "oscillate", "recovery"]}
    logger.info(f"状态分布: {state_counts}")

    # V5_simple: 调优最佳 (26.31%/1.472/-13.16%)
    best_config_v5 = {
        "bull": [
            {"strategy": "mf_d10_rp", "weight": 0.7, "note": ""},
            {"strategy": "chip_rp", "weight": 0.3, "note": ""},
        ],
        "bear": [
            {"strategy": "chip_covrp", "weight": 0.7, "note": ""},
            {"strategy": "mf_vol_d10_rp", "weight": 0.3, "note": ""},
        ],
        "oscillate": [
            {"strategy": "mf50_chip50", "weight": 0.5, "note": ""},
            {"strategy": "chip_covrp", "weight": 0.5, "note": ""},
        ],
        "recovery": [
            {"strategy": "mf60_chip40", "weight": 0.6, "note": ""},
            {"strategy": "osr_d10", "weight": 0.4, "note": ""},
        ],
    }
    dr_best_v5 = run_mss_simple_backtest(best_config_v5, sub_drs, states, nd)
    m_best_v5 = compute_metrics(dr_best_v5, name="MSS_v5_simple")

    # 使用 V6 MF主导版
    best_config = {
        "bull": [
            {"strategy": "mf_d10_rp", "weight": 0.8, "note": ""},
            {"strategy": "chip_rp", "weight": 0.2, "note": ""},
        ],
        "bear": [
            {"strategy": "chip_covrp", "weight": 0.5, "note": ""},
            {"strategy": "mf_vol_d10_rp", "weight": 0.3, "note": ""},
            {"strategy": "chip_vol_rp", "weight": 0.2, "note": ""},
        ],
        "oscillate": [
            {"strategy": "mf_d10_rp", "weight": 0.5, "note": ""},
            {"strategy": "chip_covrp", "weight": 0.3, "note": ""},
            {"strategy": "c01_layered_d5", "weight": 0.2, "note": ""},
        ],
        "recovery": [
            {"strategy": "mf_d10_rp", "weight": 0.5, "note": ""},
            {"strategy": "osr_d10", "weight": 0.3, "note": ""},
            {"strategy": "mf_vol_d10_rp", "weight": 0.2, "note": ""},
        ],
    }

    dr_best = run_mss_simple_backtest(best_config, sub_drs, states, nd)
    m_best = compute_metrics(dr_best, name="MSS_v6_mf_dominant")

    # 也尝试 V3 熊市CovRP主导版
    best_config_v3 = {
        "bull": [
            {"strategy": "mf_d10_rp", "weight": 0.6, "note": ""},
            {"strategy": "chip_rp", "weight": 0.2, "note": ""},
            {"strategy": "mf_vol_d10_rp", "weight": 0.2, "note": ""},
        ],
        "bear": [
            {"strategy": "chip_covrp", "weight": 0.6, "note": ""},
            {"strategy": "chip_vol_rp", "weight": 0.2, "note": ""},
            {"strategy": "mf_vol_d10_rp", "weight": 0.2, "note": ""},
        ],
        "oscillate": [
            {"strategy": "chip_covrp", "weight": 0.4, "note": ""},
            {"strategy": "mf50_chip50", "weight": 0.3, "note": ""},
            {"strategy": "c01_layered_d5", "weight": 0.3, "note": ""},
        ],
        "recovery": [
            {"strategy": "osr_d10", "weight": 0.4, "note": ""},
            {"strategy": "mf60_chip40", "weight": 0.3, "note": ""},
            {"strategy": "mf_vol_d10_rp", "weight": 0.3, "note": ""},
        ],
    }
    dr_best_v3 = run_mss_simple_backtest(best_config_v3, sub_drs, states, nd)
    m_best_v3 = compute_metrics(dr_best_v3, name="MSS_v3_bear_covrp")

    # 窗口分析
    w_best = window_analysis(dr_best, ds, WINDOWS)
    w_best_v5 = window_analysis(dr_best_v5, ds, WINDOWS)
    w_best_v3 = window_analysis(dr_best_v3, ds, WINDOWS)

    all_results = [m_best_v5, m_best, m_best_v3]

    # 对比基线：各子策略 + 静态组合
    for name in ["mf_d10_rp", "chip_covrp", "osr_d10", "c01_layered_d5",
                  "mf_vol_d10_rp", "mf50_chip50", "mf60_chip40"]:
        if name in sub_drs:
            all_results.append(compute_metrics(sub_drs[name], name=name))

    # 静态均衡组合
    dr_eq4 = np.zeros(nd, dtype=np.float64)
    for name in ["mf_d10_rp", "chip_covrp", "c01_layered_d5", "osr_d10"]:
        dr_eq4 += sub_drs[name] * 0.25
    all_results.append(compute_metrics(dr_eq4, name="eq4_mf_covrp_c01_osr"))

    print_results_table(all_results, "Final: Best MSS vs Baselines")

    print(f"\n{'=' * 100}")
    print("  MSS V6 窗口分析")
    print(f"{'=' * 100}")
    print(f"{'窗口':<20} {'年化%':<8} {'Sharpe':<8} {'回撤%':<8} {'Calmar':<8}")
    print('-' * 60)
    for w in w_best:
        if w.get("n_days", 0) == 0:
            continue
        ar = w.get("annual_return", 0) * 100
        sp = w.get("sharpe", 0)
        dd = abs(w.get("max_drawdown", 0)) * 100
        ca = w.get("calmar", 0)
        print(f"  {w['name']:<18} {ar:>6.2f}% {sp:>7.3f} {dd:>6.2f}% {ca:>7.3f}")

    print(f"\n{'=' * 100}")
    print("  MSS V5_simple 窗口分析 🏆")
    print(f"{'=' * 100}")
    print(f"{'窗口':<20} {'年化%':<8} {'Sharpe':<8} {'回撤%':<8} {'Calmar':<8}")
    print('-' * 60)
    for w in w_best_v5:
        if w.get("n_days", 0) == 0:
            continue
        ar = w.get("annual_return", 0) * 100
        sp = w.get("sharpe", 0)
        dd = abs(w.get("max_drawdown", 0)) * 100
        ca = w.get("calmar", 0)
        print(f"  {w['name']:<18} {ar:>6.2f}% {sp:>7.3f} {dd:>6.2f}% {ca:>7.3f}")

    print(f"\n{'=' * 100}")
    print("  MSS V3 窗口分析")
    print(f"{'=' * 100}")
    print(f"{'窗口':<20} {'年化%':<8} {'Sharpe':<8} {'回撤%':<8} {'Calmar':<8}")
    print('-' * 60)
    for w in w_best_v3:
        if w.get("n_days", 0) == 0:
            continue
        ar = w.get("annual_return", 0) * 100
        sp = w.get("sharpe", 0)
        dd = abs(w.get("max_drawdown", 0)) * 100
        ca = w.get("calmar", 0)
        print(f"  {w['name']:<18} {ar:>6.2f}% {sp:>7.3f} {dd:>6.2f}% {ca:>7.3f}")

    # 存储最终结果
    out = {
        "experiment": "final",
        "state_distribution": state_counts,
        "best_config_v5": {
            "config": best_config_v5,
            "metrics": m_best_v5,
            "windows": [w for w in w_best_v5 if w.get("n_days", 0) > 0],
        },
        "best_config_v6": {
            "config": best_config,
            "metrics": m_best,
            "windows": [w for w in w_best if w.get("n_days", 0) > 0],
        },
        "best_config_v3": {
            "config": best_config_v3,
            "metrics": m_best_v3,
            "windows": [w for w in w_best_v3 if w.get("n_days", 0) > 0],
        },
        "all_results": all_results,
    }
    out_path = os.path.join(RESULTS_DIR, "final_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    return out


def main():
    parser = argparse.ArgumentParser(description="MarketStateSelector 实验管道")
    parser.add_argument("--experiment", choices=[
        "v1_baseline", "tune_detection", "tune_allocation", "final",
    ], default="v1_baseline")
    args = parser.parse_args()

    if args.experiment == "v1_baseline":
        run_experiment_v1_baseline()
    elif args.experiment == "tune_detection":
        run_experiment_tune_detection()
    elif args.experiment == "tune_allocation":
        run_experiment_tune_allocation()
    elif args.experiment == "final":
        run_experiment_final()


if __name__ == "__main__":
    main()
