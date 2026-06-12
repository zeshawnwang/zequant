"""mss_dynamic 状态切换优化回测 — 三种方式对比（复用原始回测引擎）

研究三种解决"频繁状态切换"问题的方式：
  方式一: 状态切换冷却期 (state_cooldown_days)
  方式二: bull 状态广度门槛 (breadth_bull_thresh)
  方式三: 子策略退出最低持有期 (strategy_min_hold_days)

用法:
    python3 daily/2026-06-02/cooldown_backtest.py
"""
from __future__ import annotations
import argparse, copy, json, os, sys, time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import duckdb
from core.positioners.impl.rp_weights import RPPortfolioWeights

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

TX = 0.0012
DB_PATH = os.path.abspath("./data/quant_data.db")

# ═══════════════ 数据加载（复用原始回测逻辑）═══════════════════════════════════

def load_data(start_date="2018-01-01", end_date="2026-06-02"):
    """与 comprehensive_backtest.py 完全一致的数据加载"""
    t0 = time.time()
    conn = duckdb.connect(DB_PATH, read_only=True)

    # 因子列表
    FACTORS = list(set([
        'a27','a30','a31','a41','a42','a64','a69','a8','a80','a85','a88','a91',
        'a97','a98','a99','ff_mkt','gtja103','gtja104','gtja105','gtja108',
        'gtja113','gtja117','gtja12','gtja120','gtja121','gtja123','gtja127',
        'gtja13','gtja139','gtja141','gtja142','gtja144','gtja148','gtja164',
        'gtja168','gtja171','gtja176','gtja185','gtja34','gtja49','gtja62',
        'gtja76','gtja83','gtja85','gtja90','gtja91','gtja99',
        'returns','rsi_14','volatility_20','macd','macd_signal',
        'momentum_5','momentum_20','volume_ratio','boll_position','beta_20'
    ]))
    all_cols = [r[0] for r in conn.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name='factors_wide'"
    ).fetchall()]
    available = [c for c in FACTORS if c in all_cols]
    factor_cols = ", ".join([f'f."{c}"' for c in available])
    df = conn.execute(
        f'SELECT f.date,f.symbol,b.close,b.pct_change,b.volume,{factor_cols} '
        f'FROM factors_wide f LEFT JOIN daily_bars b ON f.date=b.date AND f.symbol=b.symbol '
        f"WHERE f.date>='{start_date}' AND f.date<='{end_date}' ORDER BY f.date,f.symbol"
    ).fetchdf()
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
    di = np.array([d2i.get(d, -1) for d in df['date']], dtype=np.int32)
    si = np.array([t2i.get(s, -1) for s in df['symbol']], dtype=np.int32)
    v = (di >= 0) & (si >= 0)
    di, si = di[v], si[v]
    for fi, fc in enumerate(available):
        if fc in df.columns:
            v3[di, si, fi] = df[fc].values[v].astype(np.float32)
    cl[di, si] = df['close'].values[v].astype(np.float32)
    if 'pct_change' in df.columns:
        pct[di, si] = df['pct_change'].values[v].astype(np.float32)
    dm[di, si] = True

    # 标准化 & 前向收益（与原始回测一致）
    for a in [v3, cl, pct]: np.nan_to_num(a, nan=0.0, copy=False)
    fwd = np.zeros((nd, ns), dtype=np.float32)
    for d in range(nd - 1):
        b = (cl[d] > 1e-10) & (cl[d + 1] > 1e-10)
        fwd[d, b] = (cl[d + 1, b] - cl[d, b]) / cl[d, b]

    # Z-score with outlier clipping
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

    conn.close()
    print(f"[数据] {nd}天×{ns}只×{nf}因子 ({time.time()-t0:.1f}s)", flush=True)
    return z3, fwd, dm, cl, tks, available, nd, ns, ds, t2i


# ═══════════════ 信号构建（与原始回测一致）══════════════════════════════════════

def load_ga_weights():
    GA_CONFIG_PATH = os.path.abspath("./core/strategies/impl/v1_ga_rp/config.json")
    if os.path.exists(GA_CONFIG_PATH):
        with open(GA_CONFIG_PATH) as f:
            return json.load(f).get("selector", {}).get("weights", {})
    return {}

def build_signals(z3, fwd, dm, cl, fnames, nd, ns):
    """与 comprehensive_backtest.py build_signals() 完全一致"""
    fi = {fn: i for i, fn in enumerate(fnames)}
    mf_weights = load_ga_weights()
    if mf_weights:
        wv = np.zeros(len(fnames), dtype=np.float32)
        for fi_i, fc in enumerate(fnames):
            if fc in mf_weights:
                wv[fi_i] = float(mf_weights[fc])
        s = np.sum(np.abs(wv))
        if s > 0: wv /= s
        mf = np.nan_to_num(np.tensordot(z3, wv, axes=(2, 0)), nan=-1e10, neginf=-1e10)
    else:
        mf = np.nan_to_num(np.mean(z3, axis=2), nan=-1e10, neginf=-1e10)

    vol20_idx = fi.get('volatility_20')
    m20_idx = fi.get('momentum_20')
    m5_idx = fi.get('momentum_5')
    rsi_idx = fi.get('rsi_14')
    ret_idx = fi.get('returns')

    # chip 信号
    chip_sig = np.full((nd, ns), -np.inf, dtype=np.float32)
    for d in range(nd):
        s = np.zeros(ns)
        if vol20_idx is not None:
            s += np.where(z3[d, :, vol20_idx] < -0.3, 1.0, 0.0) * 0.5
        if m20_idx is not None:
            s += np.where(np.abs(z3[d, :, m20_idx]) < 0.3, 1.0, 0.0) * 0.3
        chip_sig[d] = np.nan_to_num(s, nan=-1e10)

    # osr 信号
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

    # timing 信号
    vol_p = np.clip(1.0 - np.mean(z3[:, :, vol20_idx] > 0.05, axis=1), 0.2, 1.0) if vol20_idx else np.ones(nd, dtype=np.float32)
    im, ims = fi.get('macd'), fi.get('macd_signal')
    ir = fi.get('rsi_14')
    trend_p = np.full(nd, 0.5, dtype=np.float32)
    for d in range(nd):
        sl = []
        if im and ims:
            sl.append(np.where(z3[d, :, im] > z3[d, :, ims], 1.0, 0.0))
        if m5_idx and m20_idx:
            m5v, m20v = z3[d, :, m5_idx], z3[d, :, m20_idx]
            sl.append(np.where((m5v > 0) & (m5v > m20v), 1.0, np.where(m5v < 0, 0.0, 0.5)))
        if ir:
            rv = z3[d, :, ir]
            sl.append(np.where(rv > 70, 0.0, np.where(rv >= 50, 1.0, np.where(rv >= 30, 0.5, 0.0))))
        if sl:
            trend_p[d] = np.clip(np.mean(np.mean(sl, axis=0) >= 0.6) * 2.0, 0.1, 1.0)
    composite_p = np.clip(trend_p * 0.6 + vol_p * 0.4, 0.1, 1.0)

    # 市场收益率
    mkt_idx = np.zeros(nd, dtype=np.float64)
    for d in range(1, nd):
        active = dm[d] & (cl[d] > 1e-10)
        if np.any(active):
            mkt_idx[d] = np.mean(fwd[d - 1, active])

    return {
        "mf": mf, "chip": chip_sig, "osr": osr_sig,
        "vol_p": vol_p, "trend_p": trend_p, "composite_p": composite_p,
        "market_index": mkt_idx, "close": cl
    }


# ═══════════════ 市场状态检测 ═══════════════════════════════════════════════════

def detect_market_state(mkt_returns, nd):
    """与 comprehensive_backtest.py 完全一致"""
    ip = np.zeros(nd, dtype=np.float64)
    ip[0] = 1000.0
    for i in range(1, nd):
        ip[i] = ip[i - 1] * (1.0 + mkt_returns[i])
    ma5 = pd.Series(ip).rolling(5).mean().values
    ma20 = pd.Series(ip).rolling(20).mean().values
    ma60 = pd.Series(ip).rolling(60).mean().values
    ma200 = pd.Series(ip).rolling(200).mean().values
    states = ["oscillate"] * nd
    conf = np.zeros(nd, dtype=np.float32)
    for i in range(nd):
        if pd.isna(ma200[i]) or ma200[i] == 0:
            continue
        abv = (ip[i] - ma200[i]) / ma200[i]
        lb5 = min(5, i)
        ms5 = (ma5[i] - ma5[i - lb5]) / ma5[i - lb5] if (lb5 >= 2 and ma5[i - lb5] != 0) else 0.0
        lb20 = min(20, i)
        ms20 = (ma20[i] - ma20[i - lb20]) / ma20[i - lb20] if (lb20 >= 2 and ma20[i - lb20] != 0) else 0.0
        lb60 = min(60, i)
        ms60 = (ma60[i] - ma60[i - lb60]) / ma60[i - lb60] if (lb60 >= 2 and ma60[i - lb60] != 0) else 0.0
        sp_vol = 0.1
        if pd.notna(ma5[i]) and pd.notna(ma20[i]) and pd.notna(ma60[i]):
            sp_vol = abs(ma5[i] - ma20[i]) / max(abs(ma20[i]), 1e-10) + \
                     abs(ma20[i] - ma60[i]) / max(abs(ma60[i]), 1e-10)
        osc = sp_vol < 0.03
        if abv > 0 and ms20 > -0.001:
            states[i] = "bull"; conf[i] = min(1.0, abv * 2 + ms20 * 20)
        elif abv < 0 and ms20 < 0 and ms60 < 0:
            states[i] = "bear"; conf[i] = min(1.0, abs(abv) * 2 + abs(ms20) * 10 + abs(ms60) * 10)
        elif abv < 0 and ms5 > 0.005:
            states[i] = "recovery"; conf[i] = min(1.0, ms5 * 50)
        elif osc:
            states[i] = "oscillate"; conf[i] = max(0.3, 1.0 - sp_vol * 15)
        elif abv < 0 and ms5 > 0:
            states[i] = "recovery"; conf[i] = max(0.3, ms5 * 30)
        else:
            states[i] = "oscillate"; conf[i] = 0.3
    return states, conf

def compute_market_breadth(pct, dm, nd):
    """市场广度"""
    b = np.zeros(nd, dtype=np.float32)
    for i in range(nd):
        v = dm[i] & (np.abs(pct[i]) < 100.0) & (pct[i] != 0)
        if np.any(v):
            b[i] = np.mean(pct[i, v] > 0)
    return b

def apply_state_cooldown(states, nd, cooldown_days):
    """方式一：状态切换冷却期"""
    if cooldown_days <= 0:
        return states
    adjusted = list(states)
    last_switch = -cooldown_days - 1
    last_state = None
    for i in range(nd):
        if adjusted[i] != last_state:
            if i - last_switch < cooldown_days:
                adjusted[i] = last_state  # 冷却期内强制保持上一状态
            else:
                last_switch = i
                last_state = adjusted[i]
    return adjusted

def apply_breadth_bull_adjustment(states, breadth, nd, bull_thresh=0.35):
    """方式二：广度高于阈值时强制 bull"""
    adjusted = list(states)
    for i in range(nd):
        if breadth[i] > bull_thresh and adjusted[i] == "oscillate":
            adjusted[i] = "bull"
    return adjusted

def count_state_switches(states):
    n = 0
    for i in range(1, len(states)):
        if states[i] != states[i - 1]:
            n += 1
    return n


# ═══════════════ 指标计算 ═══════════════════════════════════════════════════

def compute_metrics(dr, n_days_metric=None):
    if n_days_metric is None:
        n_days_metric = len(dr)
    eq = np.ones(n_days_metric, dtype=np.float64)
    for i in range(1, n_days_metric):
        eq[i] = eq[i - 1] * (1.0 + dr[i])
    total_ret = float(eq[-1] / eq[0] - 1.0)
    ny = n_days_metric / 252.0
    ar = (float(eq[-1] / eq[0])) ** (1.0 / max(ny, 0.5)) - 1.0
    lr = np.log(eq[1:] / eq[:-1])
    std_lr = np.std(lr)
    sp = float(np.mean(lr) / max(std_lr, 1e-10) * np.sqrt(252))
    cm = np.maximum.accumulate(eq)
    dd = (eq - cm) / cm
    mdd = float(np.min(dd))
    Calmar = ar / abs(mdd) if abs(mdd) > 1e-10 else 0.0
    return {
        "total_return": round(total_ret, 4),
        "annual_return": round(ar, 4),
        "sharpe": round(sp, 4),
        "max_drawdown": round(mdd, 4),
        "calmar": round(Calmar, 4),
    }


# ═══════════════ 子策略回测（复用原始逻辑）═════════════════════════════════════

SUB_PARAMS = {
    "mf_d10_rp":     {"signal": "mf",      "rf": 5,  "tn": 10, "mhd": 10, "timing": None},
    "mf_vol_d10_rp": {"signal": "mf",      "rf": 5,  "tn": 8,  "mhd": 10, "timing": "composite"},
    "chip_covrp":    {"signal": "chip",    "rf": 3,  "tn": 6,  "mhd": 5,  "timing": None},
    "chip_equal_d3": {"signal": "chip",    "rf": 5,  "tn": 10, "mhd": 3,  "timing": None},
    "c01_layered_d5":{"signal": "mf",      "rf": 5,  "tn": 6,  "mhd": 5,  "timing": "composite"},
    "mf50_chip50":   {"signal": "chip",   "rf": 5,  "tn": 8,  "mhd": 5,  "timing": None},
    "osr_d10":       {"signal": "osr",     "rf": 10, "tn": 6,  "mhd": 5,  "timing": None},
}

STOP_LOSS = {
    "mf_d10_rp": 0.05, "mf_vol_d10_rp": 0.05,
    "chip_covrp": 0.08, "chip_equal_d3": 0.08,
    "c01_layered_d5": 0.05, "mf50_chip50": 0.05, "osr_d10": 0.05,
}

# V7 最新分配
V7_ALLOCATION = {
    "bull":      [("mf_d10_rp", 0.6), ("mf_vol_d10_rp", 0.2), ("mf50_chip50", 0.15), ("osr_d10", 0.05)],
    "bear":      [("c01_layered_d5", 0.5), ("chip_equal_d3", 0.25), ("mf_vol_d10_rp", 0.25)],
    "oscillate": [("mf_d10_rp", 0.4), ("mf50_chip50", 0.3), ("c01_layered_d5", 0.3)],
    "recovery":  [("c01_layered_d5", 0.4), ("osr_d10", 0.3), ("mf_vol_d10_rp", 0.3)],
}


def bt_sub_strategy(sig, fwd, dm, rebal_freq=5, top_n=10, min_hold_days=10,
                    pos_ratio=None, stop_loss_pct=0.05,
                    symbol_risk_map=None,
                    use_trailing_stop=True, trailing_profit_pct=0.03,
                    state_trailing_map=None, states=None,
                    strategy_min_hold_days=0,
                    tx_cost=TX):
    """回测单个子策略 — 与 comprehensive_backtest.py 原始引擎一致
    新增: strategy_min_hold_days — 方式三: 止盈止损时检查最低持有期
    """
    n_days, ns = sig.shape
    alloc = RPPortfolioWeights(top_n=top_n, min_hold_days=min_hold_days)
    pw = np.zeros(ns, dtype=np.float32)
    hs = np.full(ns, -1, dtype=np.int32)
    entry_px = np.zeros(ns, dtype=np.float32)
    peak_px = np.zeros(ns, dtype=np.float32)
    rh = 0
    dr = np.zeros(n_days, dtype=np.float64)

    for i in range(1, n_days):
        eff_trail = trailing_profit_pct
        if state_trailing_map and states and i < len(states):
            eff_trail = state_trailing_map.get(states[i], trailing_profit_pct)

        # ── 止损（单日跌幅）─ 与原始引擎一致 ──
        if stop_loss_pct > 0 and np.any(pw > 0):
            for j in range(ns):
                if pw[j] > 0 and hs[j] >= 0 and entry_px[j] > 0:
                    days_held = rh - hs[j] + 1
                    if fwd[i, j] < -stop_loss_pct and fwd[i, j] > -0.95:
                        if days_held >= strategy_min_hold_days:
                            pw[j] = 0.0; hs[j] = -1; entry_px[j] = 0.0; peak_px[j] = 0.0

        # ── 移动止盈（基于入场价）─ 与原始引擎一致 ──
        if use_trailing_stop:
            for j in range(ns):
                if pw[j] > 0 and entry_px[j] > 0:
                    days_held = rh - hs[j] + 1
                    cur = entry_px[j] * (1.0 + fwd[i, j])
                    if cur > peak_px[j] or peak_px[j] <= 0:
                        peak_px[j] = cur
                    if peak_px[j] > 0 and cur < peak_px[j] * (1.0 - eff_trail):
                        if days_held >= strategy_min_hold_days:
                            pw[j] = 0.0; hs[j] = -1; entry_px[j] = 0.0; peak_px[j] = 0.0

        # ── 再平衡 ──
        rebal = (i % rebal_freq == 0)
        if rebal:
            masked = sig[i].copy().astype(np.float32)
            if symbol_risk_map:
                for j, lv in symbol_risk_map.items():
                    if lv == 'high':
                        masked[j] = -1e10
            nw = alloc.allocate(masked, fwd, i, pw, hs, rh)
            for j in range(ns):
                if nw[j] > 0 and hs[j] < 0:
                    hs[j] = rh + 1
                if nw[j] > 0 and entry_px[j] <= 0:
                    entry_px[j] = max(1.0, 1.0 + fwd[i, j])
                    peak_px[j] = entry_px[j]
            to = float(np.sum(np.abs(nw - pw)))
            pw = nw
        else:
            mk = dm[i] & (pw > 0)
            if np.any(mk):
                p2 = pw[mk].copy() / float(np.sum(pw[mk]))
                pw = np.zeros(ns, dtype=np.float32)
                pw[mk] = p2
            to = 0.0

        pr = pos_ratio[i] if pos_ratio is not None else 1.0
        rt = pr * float(np.dot(pw, fwd[i])) - 0.5 * to * tx_cost
        dr[i] = 0.0 if (np.isnan(rt) or np.isinf(rt)) else rt

        if np.any(pw > 0):
            rh += 1

    return dr


def compute_sub_drs(signals, fwd, dm, nd, sub_params,
                   symbol_risk_map=None,
                   use_trailing_stop=True, trailing_profit_pct=0.03,
                   state_trailing_map=None, states=None,
                   strategy_min_hold_days=0, tx_cost=TX):
    sub_drs = {}
    for name, params in sub_params.items():
        sig = signals[params["signal"]]
        pr = None
        tm = params.get("timing")
        if tm == "vol":
            pr = signals.get("vol_p")
        elif tm == "trend":
            pr = signals.get("trend_p")
        elif tm == "composite":
            pr = signals.get("composite_p")
        sl = STOP_LOSS.get(name, 0.06)
        rf = int(params["rf"])
        dr = bt_sub_strategy(
            sig, fwd, dm, rebal_freq=rf, top_n=params["tn"],
            min_hold_days=params["mhd"], pos_ratio=pr,
            stop_loss_pct=sl, symbol_risk_map=symbol_risk_map,
            use_trailing_stop=use_trailing_stop,
            trailing_profit_pct=trailing_profit_pct,
            state_trailing_map=state_trailing_map, states=states,
            strategy_min_hold_days=strategy_min_hold_days, tx_cost=tx_cost
        )
        sub_drs[name] = dr
    return sub_drs


# ═══════════════ MSS 主策略 ═════════════════════════════════════════════════════

TRAILING_MAP = {"bull": 0.03, "oscillate": 0.03, "bear": 0.05, "recovery": 0.03}

def run_mss(signals, fwd, dm, states, n_days, sub_drs,
           breadth=None, bull_breadth_thresh=0.35,
           state_cooldown_days=0):
    """运行 MSS 主策略"""
    # 方式一: 冷却期
    if state_cooldown_days > 0:
        states = apply_state_cooldown(states, n_days, state_cooldown_days)

    # 方式二: 广度强制 bull
    if breadth is not None:
        states = apply_breadth_bull_adjustment(states, breadth, n_days, bull_thresh=bull_breadth_thresh)

    alloc_fmt = {st: [{"strategy": a[0], "weight": a[1]} for a in al]
                 for st, al in V7_ALLOCATION.items()}

    dr = np.zeros(n_days, dtype=np.float64)
    for i in range(1, n_days):
        st = states[i] if i < len(states) else "oscillate"
        allocs = alloc_fmt.get(st, alloc_fmt.get("oscillate", []))
        am = {}
        for a in allocs:
            if a["strategy"] in sub_drs:
                am[a["strategy"]] = max(a["weight"], 0.0)
        tw = sum(am.values()) or 1.0
        for n in am:
            am[n] /= tw
        dr[i] = sum(am.get(n, 0.0) * sub_drs[n][i] for n in am if n in sub_drs)

    return dr, count_state_switches(states)


# ═══════════════ 主回测 ═══════════════════════════════════════════════════════

@dataclass
class Result:
    cooldown: int; bull_thresh: float; min_hold: int
    annual: float; mdd: float; sharpe: float; calmar: float
    switches: int; total: float
    def __str__(self):
        return (f"cd={self.cooldown} bt={self.bull_thresh:.2f} mh={self.min_hold} | "
                f"annual={self.annual:.2%} mdd={self.mdd:.2%} "
                f"calmar={self.calmar:.4f} sharpe={self.sharpe:.4f} switches={self.switches}")


def run_backtest(cooldown=0, bull_thresh=0.35, min_hold=0,
                start="2019-01-02", end="2026-06-02"):
    """运行单次回测"""
    z3, fwd, dm, cl, tks, fnames, nd, ns, ds, t2i = load_data(start, end)

    # 子集切片（ds 包含字符串格式日期）
    ds_str = [str(d.date()) if hasattr(d, 'date') else str(d) for d in ds]
    si = next((i for i, d in enumerate(ds_str) if d >= start), 0)
    ei = len(ds_str) - 1 - next((i for i, d in enumerate(reversed(ds_str)) if d <= end), 0)
    nd_s = ei - si + 1
    ds_s = ds[si:ei+1]
    dm_s = dm[si:ei+1]
    cl_s = cl[si:ei+1]
    fwd_s = fwd[si:ei+1]

    # 信号
    sigs = build_signals(z3, fwd, dm, cl, fnames, nd, ns)
    sig_s = {}
    for k, v in sigs.items():
        if hasattr(v, 'shape'):
            if len(v.shape) >= 2 and v.shape[0] == nd:
                sig_s[k] = v[si:ei+1]
            elif len(v.shape) == 1 and v.shape[0] == nd:
                sig_s[k] = v[si:ei+1]
            else:
                sig_s[k] = v
        else:
            sig_s[k] = v

    # 市场状态
    mkt = np.zeros(nd_s, dtype=np.float64)
    for d in range(1, nd_s):
        active = dm_s[d] & (cl_s[d] > 1e-10)
        if np.any(active):
            mkt[d] = np.mean(fwd_s[d - 1, active])
    states, _ = detect_market_state(mkt, nd_s)

    # 广度
    breadth = compute_market_breadth(fwd_s, dm_s, nd_s)

    # 子策略
    sub_drs = compute_sub_drs(
        sig_s, fwd_s, dm_s, nd_s, SUB_PARAMS,
        symbol_risk_map=None,
        use_trailing_stop=True,
        trailing_profit_pct=0.03,
        state_trailing_map=TRAILING_MAP,
        states=states,
        strategy_min_hold_days=min_hold, tx_cost=TX
    )

    # 主策略
    n_days = ei - si + 1  # 实际回测天数（避免与 load_data 返回的 nd 混淆）
    dr_len = {k: len(v) for k, v in sub_drs.items()}
    print(f"    [DEBUG] n_days={n_days}, sub_drs keys={dr_len}", flush=True)
    dr, n_switches = run_mss(
        sig_s, fwd_s, dm_s, states, n_days, sub_drs,
        breadth=breadth, bull_breadth_thresh=bull_thresh,
        state_cooldown_days=cooldown
    )

    # DEBUG
    # dr_len = {k: len(v) for k, v in sub_drs.items()}  # 可用于调试
    metrics = compute_metrics(dr)
    return Result(
        cooldown=cooldown, bull_thresh=bull_thresh, min_hold=min_hold,
        annual=metrics["annual_return"], mdd=metrics["max_drawdown"],
        sharpe=metrics["sharpe"], calmar=metrics["calmar"],
        switches=n_switches, total=metrics["total_return"]
    )


def main():
    t0 = time.time()
    print("=" * 80, flush=True)
    print("mss_dynamic 状态切换优化回测 — 三种方式对比", flush=True)
    print("回测区间: 2019-01-02 ~ 2026-06-02", flush=True)
    print("=" * 80, flush=True)

    results = []

    # ── 基线 ──────────────────────────────────────────────────────────────────
    print("\n[0] 基线: 原始策略", flush=True)
    baseline = run_backtest(cooldown=0, bull_thresh=0.35, min_hold=0)
    results.append(("baseline", baseline))
    print(f"  {baseline}", flush=True)

    # ── 方式一: 冷却期单独 ──────────────────────────────────────────────────
    print("\n" + "─" * 60, flush=True)
    print("方式一: 状态切换冷却期（单独）", flush=True)
    print("─" * 60, flush=True)
    for cd in [3, 5]:
        r = run_backtest(cooldown=cd, bull_thresh=0.35, min_hold=0)
        results.append((f"cooldown={cd}", r))
        tag = " ✅" if r.calmar > baseline.calmar else ""
        print(f"  cd={cd}: annual={r.annual:.2%} mdd={r.mdd:.2%} calmar={r.calmar:.4f} sharpe={r.sharpe:.4f} switches={r.switches}{tag}", flush=True)

    # ── 方式二: 广度门槛单独 ──────────────────────────────────────────────
    print("\n" + "─" * 60, flush=True)
    print("方式二: bull广度门槛（单独）", flush=True)
    print("─" * 60, flush=True)
    for bt in [0.40, 0.45]:
        r = run_backtest(cooldown=0, bull_thresh=bt, min_hold=0)
        results.append((f"bull_thresh={bt}", r))
        tag = " ✅" if r.calmar > baseline.calmar else ""
        print(f"  bt={bt:.2f}: annual={r.annual:.2%} mdd={r.mdd:.2%} calmar={r.calmar:.4f} sharpe={r.sharpe:.4f} switches={r.switches}{tag}", flush=True)

    # ── 方式三: 最低持有期单独 ──────────────────────────────────────────────
    print("\n" + "─" * 60, flush=True)
    print("方式三: 子策略退出最低持有期（单独）", flush=True)
    print("─" * 60, flush=True)
    for mh in [3, 5]:
        r = run_backtest(cooldown=0, bull_thresh=0.35, min_hold=mh)
        results.append((f"min_hold={mh}", r))
        tag = " ✅" if r.calmar > baseline.calmar else ""
        print(f"  mh={mh}d: annual={r.annual:.2%} mdd={r.mdd:.2%} calmar={r.calmar:.4f} sharpe={r.sharpe:.4f}{tag}", flush=True)

    # ── 组合实验 ──────────────────────────────────────────────────────────
    print("\n" + "─" * 60, flush=True)
    print("组合实验（方式一 + 方式二 + 方式三）", flush=True)
    print("─" * 60, flush=True)
    combos = [
        (3, 0.40, 0),    # cd=3 + bt=0.40（方式一+二，无三）
        (3, 0.45, 0),    # cd=3 + bt=0.45（方式一+二，无三）
        (5, 0.40, 0),    # cd=5 + bt=0.40（方式一+二，无三）
        (3, 0.35, 3),    # cd=3 + mh=3（方式一+三，无二）
        (3, 0.40, 3),    # 三者叠加
    ]
    for cd, bt, mh in combos:
        r = run_backtest(cooldown=cd, bull_thresh=bt, min_hold=mh)
        results.append((f"cd={cd}_bt={bt}_mh={mh}", r))
        tag = " ✅" if r.calmar > baseline.calmar else ""
        print(f"  cd={cd} bt={bt:.2f} mh={mh}d: annual={r.annual:.2%} mdd={r.mdd:.2%} calmar={r.calmar:.4f} sharpe={r.sharpe:.4f} switches={r.switches}{tag}", flush=True)

    # ── 汇总 ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80, flush=True)
    print("结果汇总（按 Calmar 降序）", flush=True)
    print("=" * 80, flush=True)
    sorted_r = sorted(results, key=lambda x: x[1].calmar, reverse=True)
    print(f"\n{'配置':>30}  {'年化':>8}  {'最大回撤':>10}  {'Calmar':>8}  {'Sharpe':>8}  {'切换':>6}  {'vs基线':>8}", flush=True)
    print("-" * 85, flush=True)
    for label, r in sorted_r:
        vs = f"{r.calmar/baseline.calmar-1:+.1%}" if baseline.calmar > 0 else "N/A"
        marker = " ★基线" if label == "baseline" else (" ✅最优" if r.calmar == sorted_r[0][1].calmar else "")
        print(f"  {label:30}  {r.annual:>7.2%}  {r.mdd:>9.2%}  {r.calmar:>7.4f}  {r.sharpe:>7.3f}  {r.switches:>5}  {vs:>7}{marker}", flush=True)

    best_label, best = sorted_r[0]
    print(f"\n🏆 最优: {best_label}", flush=True)
    print(f"   年化收益: {best.annual:.2%}", flush=True)
    print(f"   最大回撤: {best.mdd:.2%}", flush=True)
    print(f"   Calmar: {best.calmar:.4f} (vs 基线 {baseline.calmar:.4f}, {best.calmar/baseline.calmar-1:+.1%})", flush=True)
    print(f"   Sharpe: {best.sharpe:.4f}", flush=True)
    print(f"   状态切换: {best.switches} 次 (基线 {baseline.switches} 次)", flush=True)
    print(f"\n耗时: {time.time()-t0:.1f}s", flush=True)

    # 最后一行供解析
    print(f"\nRESULT: cd={best.cooldown} bt={best.bull_thresh:.2f} mh={best.min_hold} "
          f"annual={best.annual:.4f} mdd={best.mdd:.4f} calmar={best.calmar:.4f} "
          f"sharpe={best.sharpe:.4f} switches={best.switches}", flush=True)


if __name__ == "__main__":
    main()
