"""V7 vs V8 全面回测 — 彻底验证因子翻转改善效果

基于 comprehensive_backtest.py 框架，对比 V7(原始权重) vs V8(翻转reversal+dead降0)：
  1. 主策略全区间 + 分年窗口 + 2022熊市专项
  2. 各子策略独立回测 V7 vs V8
  3. Walk-forward OOS 验证(3窗口)
  4. 各市场状态下子策略贡献归因
  5. 子策略收益相关性分析
  6. 月度/逐年收益对比
  7. V8信号与V7信号相关性(确认非随机翻转)

用法:
    python3 daily/2026-06-15/v8_full_backtest.py
    python3 daily/2026-06-15/v8_full_backtest.py --end 2026-06-11
"""
from __future__ import annotations
import argparse, copy, json, logging, os, sys, time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np, pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import duckdb
from core.positioners import RPPortfolioWeights

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

for h in logging.root.handlers[:]: logging.root.removeHandler(h)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(os.path.join(SCRIPT_DIR, "v8_backtest.log"), mode="w", encoding="utf-8"),
              logging.StreamHandler(sys.stdout)], force=True)
logger = logging.getLogger("v8_bt")

TX = 0.0012
DB_PATH = os.path.abspath("./data/quant_data.db")

# V7权重(原始)
V7_CONFIG = os.path.abspath("./core/strategies/impl/v1_ga_rp/config.json")
# V8权重(翻转)
V8_CONFIG = os.path.join(SCRIPT_DIR, "weights", "v8_flipped.json")

FACTORS = list(set(['a27','a30','a31','a41','a42','a64','a69','a8','a80','a85','a88','a91','a97','a98','a99',
    'ff_mkt','gtja103','gtja104','gtja105','gtja108','gtja113','gtja117','gtja12','gtja120','gtja121','gtja123','gtja127',
    'gtja13','gtja139','gtja141','gtja142','gtja144','gtja148','gtja164','gtja168','gtja171','gtja176','gtja185','gtja34',
    'gtja49','gtja62','gtja76','gtja83','gtja85','gtja90','gtja91','gtja99','returns','rsi_14','volatility_20',
    'macd','macd_signal','momentum_5','momentum_20','volume_ratio','boll_position','beta_20']))

WINDOWS = [
    ("2019修复牛", "2019-01-02", "2019-12-31"),
    ("2020疫情",   "2020-01-02", "2020-12-31"),
    ("2021结构牛", "2021-01-04", "2021-12-31"),
    ("2022熊市",   "2022-01-04", "2022-12-30"),
    ("2023震荡",   "2023-01-03", "2023-12-29"),
    ("2024反弹",   "2024-01-02", "2024-12-31"),
    ("2025至今",   "2025-01-02", "2026-06-11"),
]

# V7 实盘分配
V7_ALLOC = {
    "bull": [("mf_d10_rp", 0.6), ("mf_vol_d10_rp", 0.2), ("mf50_chip50", 0.15), ("osr_d10", 0.05)],
    "bear": [("c01_layered_d5", 0.5), ("chip_equal_d3", 0.25), ("mf_vol_d10_rp", 0.25)],
    "oscillate": [("mf_d10_rp", 0.4), ("mf50_chip50", 0.3), ("c01_layered_d5", 0.3)],
    "recovery": [("c01_layered_d5", 0.4), ("osr_d10", 0.3), ("mf_vol_d10_rp", 0.3)],
}

SUB_PARAMS = {
    "mf_d10_rp": {"signal": "mf", "rf": 5, "tn": 10, "mhd": 10, "timing": None},
    "mf_vol_d10_rp": {"signal": "mf", "rf": 5, "tn": 8, "mhd": 10, "timing": "composite"},
    "chip_covrp": {"signal": "chip", "rf": 3, "tn": 6, "mhd": 5, "timing": None},
    "chip_equal_d3": {"signal": "chip", "rf": 3, "tn": 6, "mhd": 5, "timing": None},
    "chip_rp": {"signal": "chip", "rf": 3, "tn": 6, "mhd": 5, "timing": None},
    "osr_d10": {"signal": "osr", "rf": 10, "tn": 6, "mhd": 5, "timing": None},
    "c01_layered_d5": {"signal": "mf", "rf": 5, "tn": 6, "mhd": 5, "timing": "composite"},
    "mf_base": {"signal": "mf", "rf": 3, "tn": 40, "mhd": 5, "timing": None},
}

STOP_LOSS = {"mf_d10_rp": 0.06, "mf_vol_d10_rp": 0.06, "chip_covrp": 0.08, "chip_equal_d3": 0.08,
    "c01_layered_d5": 0.06, "osr_d10": 0.06, "chip_rp": 0.08, "mf_base": 0.06}

LIVE_TOP_N = {"mf_d10_rp": 10, "mf_vol_d10_rp": 8, "chip_covrp": 6, "chip_equal_d3": 6,
    "chip_rp": 6, "osr_d10": 6, "c01_layered_d5": 6, "mf_base": 6}

# ═══════════════ 数据加载 ═══════════════

def _get_conn(): return duckdb.connect(DB_PATH, read_only=True)

def load_data(start_date="2018-01-01", end_date="2026-06-11"):
    t0 = time.time(); conn = _get_conn()
    all_cols = [r[0] for r in conn.execute("SELECT column_name FROM information_schema.columns WHERE table_name='factors_wide'").fetchall()]
    available = [c for c in FACTORS if c in all_cols]
    factor_cols = ", ".join([f'f."{c}"' for c in available])
    df = conn.execute(f"SELECT f.date,f.symbol,b.close,b.pct_change,b.volume,{factor_cols} FROM factors_wide f LEFT JOIN daily_bars b ON f.date=b.date AND f.symbol=b.symbol WHERE f.date>='{start_date}' AND f.date<='{end_date}' ORDER BY f.date,f.symbol").fetchdf()
    df['date'] = pd.to_datetime(df['date']); ds = sorted(df['date'].unique())
    tks = [r[0] for r in conn.execute("SELECT symbol FROM symbols ORDER BY symbol").fetchall()]
    nd, ns, nf = len(ds), len(tks), len(available)
    t2i = {t: i for i, t in enumerate(tks)}; d2i = {d: i for i, d in enumerate(ds)}
    v3 = np.full((nd, ns, nf), np.nan, dtype=np.float32); dm = np.zeros((nd, ns), dtype=bool)
    cl = np.zeros((nd, ns), dtype=np.float32); pct = np.zeros((nd, ns), dtype=np.float32)
    di = np.array([d2i[d] for d in df['date']], dtype=np.int32); si = np.array([t2i.get(s, -1) for s in df['symbol']], dtype=np.int32)
    v = si >= 0; di, si = di[v], si[v]
    for fi, fc in enumerate(available):
        if fc in df.columns: v3[di, si, fi] = df[fc].values[v].astype(np.float32)
    cl[di, si] = df['close'].values[v].astype(np.float32)
    if 'pct_change' in df.columns: pct[di, si] = df['pct_change'].values[v].astype(np.float32)
    dm[di, si] = True
    for a in [v3, cl, pct]: np.nan_to_num(a, nan=0.0, copy=False)
    fwd = np.zeros((nd, ns), dtype=np.float32)
    for d in range(nd - 1):
        b = (cl[d] > 1e-10) & (cl[d + 1] > 1e-10)
        fwd[d, b] = (cl[d + 1, b] - cl[d, b]) / cl[d, b]
    z3 = np.zeros_like(v3)
    for fi in range(nf):
        a = v3[:, :, fi]
        for d in range(nd):
            r = a[d, :]; nz = r[r != 0]
            if len(nz) > 1:
                lo, hi = np.quantile(nz, [0.01, 0.99]); c = np.clip(r, lo, hi)
                mu, sd = np.mean(c), np.std(c)
                z3[d, :, fi] = (c - mu) / sd if sd > 1e-10 else 0.0
    per = {"pct": pct, "cl": cl}
    logger.info(f"数据: {nd}天×{ns}只×{nf}因子 ({time.time()-t0:.1f}s), {start_date}~{end_date}")
    conn.close()
    return z3, fwd, dm, cl, tks, available, nd, ns, ds, t2i, per

def load_weights(path):
    if os.path.exists(path):
        with open(path) as f: return json.load(f).get("selector", {}).get("weights", {})
    return {}

# ═══════════════ 信号构建 ═══════════════

def build_mf_signal(z3, fwd, dm, cl, tks, available, nd, ns, weights, top_n=10, rebal_freq=5):
    """构建MF因子信号并回测，返回日频收益序列"""
    wv = np.zeros(len(available), dtype=np.float32)
    for i, f in enumerate(available):
        wv[i] = weights.get(f, 0.0)
    scores = z3 @ wv  # (nd, ns)
    # 反转：分数越高越好(做多)
    daily_ret = np.zeros(nd, dtype=np.float64)
    for d in range(nd):
        if d % rebal_freq != 0:
            daily_ret[d] = daily_ret[d - 1] if d > 0 else 0.0
            continue
        mask = dm[d] & (cl[d] > 1e-10) & (scores[d] != 0)
        if mask.sum() < 3:
            daily_ret[d] = 0.0
            continue
        ranked = np.argsort(-scores[d, mask])
        n = min(top_n, len(ranked))
        sel_idx = np.where(mask)[0][ranked[:n]]
        rets = fwd[d, sel_idx]
        valid = np.abs(rets) < 0.2
        if valid.sum() > 0:
            daily_ret[d] = np.mean(rets[valid]) - TX
        else:
            daily_ret[d] = 0.0
    return daily_ret

def build_chip_signal(z3, fwd, dm, cl, tks, available, nd, ns, top_n=6, rebal_freq=3):
    """构建chip信号(筹码因子组合)"""
    chip_factors = ['gtja164', 'gtja168', 'gtja171', 'gtja176', 'gtja185']
    wv = np.zeros(len(available), dtype=np.float32)
    for i, f in enumerate(available):
        if f in chip_factors:
            wv[i] = 1.0 / len(chip_factors)
    scores = z3 @ wv
    daily_ret = np.zeros(nd, dtype=np.float64)
    for d in range(nd):
        if d % rebal_freq != 0:
            daily_ret[d] = daily_ret[d - 1] if d > 0 else 0.0
            continue
        mask = dm[d] & (cl[d] > 1e-10) & (scores[d] != 0)
        if mask.sum() < 3:
            daily_ret[d] = 0.0
            continue
        ranked = np.argsort(-scores[d, mask])
        n = min(top_n, len(ranked))
        sel_idx = np.where(mask)[0][ranked[:n]]
        rets = fwd[d, sel_idx]
        valid = np.abs(rets) < 0.2
        if valid.sum() > 0:
            daily_ret[d] = np.mean(rets[valid]) - TX
        else:
            daily_ret[d] = 0.0
    return daily_ret

def build_osr_signal(z3, fwd, dm, cl, tks, available, nd, ns, top_n=6, rebal_freq=10):
    """构建OSR信号(超卖反弹)"""
    osr_factors = ['rsi_14', 'gtja103', 'gtja104', 'gtja105']
    wv = np.zeros(len(available), dtype=np.float32)
    for i, f in enumerate(available):
        if f in osr_factors:
            wv[i] = -1.0 / len(osr_factors)  # 反转：超卖=低RSI=做多
    scores = z3 @ wv
    daily_ret = np.zeros(nd, dtype=np.float64)
    for d in range(nd):
        if d % rebal_freq != 0:
            daily_ret[d] = daily_ret[d - 1] if d > 0 else 0.0
            continue
        mask = dm[d] & (cl[d] > 1e-10) & (scores[d] != 0)
        if mask.sum() < 3:
            daily_ret[d] = 0.0
            continue
        ranked = np.argsort(-scores[d, mask])
        n = min(top_n, len(ranked))
        sel_idx = np.where(mask)[0][ranked[:n]]
        rets = fwd[d, sel_idx]
        valid = np.abs(rets) < 0.2
        if valid.sum() > 0:
            daily_ret[d] = np.mean(rets[valid]) - TX
        else:
            daily_ret[d] = 0.0
    return daily_ret

def build_composite_timing(z3, fwd, dm, cl, pct, available, nd, ns):
    """构建composite timing信号(市场广度+动量)"""
    vol_factors = ['volatility_20', 'boll_position', 'beta_20']
    trend_factors = ['momentum_20', 'macd', 'macd_signal']
    wv = np.zeros(len(available), dtype=np.float32)
    for i, f in enumerate(available):
        if f in vol_factors: wv[i] = -0.5 / len(vol_factors)
        if f in trend_factors: wv[i] = 0.5 / len(trend_factors)
    scores = z3 @ wv
    # 市场广度
    breadth = np.zeros(nd, dtype=np.float32)
    for d in range(nd):
        mask = dm[d] & (pct[d] != 0)
        if mask.sum() > 0: breadth[d] = (pct[d, mask] > 0).sum() / mask.sum()
    timing = np.zeros(nd, dtype=np.float32)
    for d in range(nd):
        timing[d] = 0.5 * scores[d].mean() + 0.5 * (breadth[d] - 0.5)
    return timing

# ═══════════════ 子策略回测引擎 ═══════════════

def bt_sub_strategy(name, z3, fwd, dm, cl, pct, tks, available, nd, ns, ds, weights, params=None):
    """完整子策略回测：含止损、移动止盈、timing过滤、增强ST"""
    if params is None: params = SUB_PARAMS.get(name, SUB_PARAMS["mf_d10_rp"])
    rf = params["rf"]; tn = params["tn"]; mhd = params["mhd"]
    sig_type = params["signal"]; timing_type = params.get("timing")
    top_n = LIVE_TOP_N.get(name, 10); sl_pct = STOP_LOSS.get(name, 0.06)

    # 构建信号
    if sig_type == "mf":
        raw_ret = build_mf_signal(z3, fwd, dm, cl, tks, available, nd, ns, weights, top_n, rf)
    elif sig_type == "chip":
        raw_ret = build_chip_signal(z3, fwd, dm, cl, tks, available, nd, ns, top_n, rf)
    elif sig_type == "osr":
        raw_ret = build_osr_signal(z3, fwd, dm, cl, tks, available, nd, ns, top_n, rf)
    else:
        raw_ret = build_mf_signal(z3, fwd, dm, cl, tks, available, nd, ns, weights, top_n, rf)

    # Timing过滤
    if timing_type == "composite":
        timing = build_composite_timing(z3, fwd, dm, cl, pct, available, nd, ns)
        for d in range(nd):
            if timing[d] < -0.3: raw_ret[d] *= 0.3
            elif timing[d] < 0: raw_ret[d] *= 0.6

    # 止损 + 移动止盈
    cum = np.cumprod(1 + raw_ret)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    for d in range(1, nd):
        if dd[d] < -sl_pct: raw_ret[d] = 0.0  # 止损后空仓
        trailing = (cum[d] - peak[d]) / peak[d]
        if trailing < -0.03 and dd[d] < -0.02: raw_ret[d] *= 0.5  # 移动止盈减仓

    # 最小持仓天数
    hold_count = 0
    for d in range(nd):
        if raw_ret[d] != 0: hold_count += 1
        else:
            if hold_count > 0 and hold_count < mhd:
                for dd2 in range(d - hold_count, d): raw_ret[dd2] = 0.0
            hold_count = 0

    return raw_ret

# ═══════════════ 指标计算 ═══════════════

def calc_metrics(ret, label=""):
    """计算回测指标"""
    r = ret[ret != 0]
    if len(r) < 2: return {"total_ret": 0, "ann_ret": 0, "ann_vol": 0, "sharpe": 0, "max_dd": 0, "calmar": 0, "win_rate": 0, "n_days": len(ret)}
    cum = np.cumprod(1 + ret)
    total = cum[-1] / cum[0] - 1
    n = len(ret); ann_r = (1 + total) ** (252 / max(n, 1)) - 1
    ann_v = np.std(ret) * np.sqrt(252)
    sharpe = ann_r / ann_v if ann_v > 1e-10 else 0
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    max_dd = dd.min()
    calmar = ann_r / abs(max_dd) if abs(max_dd) > 1e-10 else 0
    wr = (ret > 0).sum() / len(r) if len(r) > 0 else 0
    return {"total_ret": round(total, 4), "ann_ret": round(ann_r, 4), "ann_vol": round(ann_v, 4),
            "sharpe": round(sharpe, 2), "max_dd": round(max_dd, 4), "calmar": round(calmar, 2),
            "win_rate": round(wr, 4), "n_days": n}

# ═══════════════ 市场状态检测 ═══════════════

def detect_market_state(pct, dm, nd, ns, lookback=40):
    """简化市场状态检测"""
    states = ["oscillate"] * nd
    for d in range(lookback, nd):
        mask = dm[d] & (pct[d] != 0)
        if mask.sum() < 10: continue
        mkt_ret = pct[d - lookback:d, mask].mean()
        pos_ratio = (pct[d - lookback:d, mask] > 0).mean()
        vol = pct[d - lookback:d, mask].std()
        if mkt_ret > 0.001 and pos_ratio > 0.55: states[d] = "bull"
        elif mkt_ret < -0.001 and pos_ratio < 0.45: states[d] = "bear"
        elif vol > 0.03: states[d] = "bear"
        elif mkt_ret > 0 and pos_ratio > 0.5: states[d] = "recovery"
        else: states[d] = "oscillate"
    return states

# ═══════════════ MSS混合策略 ═══════════════

def run_mss(sub_rets, states, nd, alloc):
    """运行MSS动态策略"""
    mss_ret = np.zeros(nd, dtype=np.float64)
    for d in range(nd):
        st = states[d]
        for sname, wt in alloc.get(st, []):
            if sname in sub_rets:
                mss_ret[d] += wt * sub_rets[sname][d]
    return mss_ret

# ═══════════════ 窗口分析 ═══════════════

def window_analysis(ret, ds, nd, windows):
    """分年窗口分析"""
    results = {}
    for wname, wstart, wend in windows:
        sd = pd.Timestamp(wstart); ed = pd.Timestamp(wend)
        mask = np.array([(ds[i] >= sd and ds[i] <= ed) for i in range(nd)])
        if mask.sum() < 10: continue
        wr = ret[mask]
        results[wname] = calc_metrics(wr, wname)
    return results

def walk_forward(z3, fwd, dm, cl, pct, tks, available, nd, ns, ds, weights_v7, weights_v8, wf_windows):
    """Walk-forward OOS验证"""
    results = {}
    for wname, is_start, is_end, oos_start, oos_end in wf_windows:
        is_sd, is_ed = pd.Timestamp(is_start), pd.Timestamp(is_end)
        oos_sd, oos_ed = pd.Timestamp(oos_start), pd.Timestamp(oos_end)
        is_mask = np.array([(ds[i] >= is_sd and ds[i] <= is_ed) for i in range(nd)])
        oos_mask = np.array([(ds[i] >= oos_sd and ds[i] <= oos_ed) for i in range(nd)])
        if is_mask.sum() < 20 or oos_mask.sum() < 10: continue
        # IS和OOS分别回测MF主策略
        is_s, is_e = np.where(is_mask)[0][0], np.where(is_mask)[0][-1]
        oos_s, oos_e = np.where(oos_mask)[0][0], np.where(oos_mask)[0][-1]
        # V7
        v7_is = build_mf_signal(z3[is_s:is_e+1], fwd[is_s:is_e+1], dm[is_s:is_e+1],
            cl[is_s:is_e+1], tks, available, is_e-is_s+1, ns, weights_v7, 10, 5)
        v7_oos = build_mf_signal(z3[oos_s:oos_e+1], fwd[oos_s:oos_e+1], dm[oos_s:oos_e+1],
            cl[oos_s:oos_e+1], tks, available, oos_e-oos_s+1, ns, weights_v7, 10, 5)
        # V8
        v8_is = build_mf_signal(z3[is_s:is_e+1], fwd[is_s:is_e+1], dm[is_s:is_e+1],
            cl[is_s:is_e+1], tks, available, is_e-is_s+1, ns, weights_v8, 10, 5)
        v8_oos = build_mf_signal(z3[oos_s:oos_e+1], fwd[oos_s:oos_e+1], dm[oos_s:oos_e+1],
            cl[oos_s:oos_e+1], tks, available, oos_e-oos_s+1, ns, weights_v8, 10, 5)
        results[wname] = {
            "V7_IS": calc_metrics(v7_is), "V7_OOS": calc_metrics(v7_oos),
            "V8_IS": calc_metrics(v8_is), "V8_OOS": calc_metrics(v8_oos),
        }
    return results

def attribution_by_state(sub_rets, states, nd, alloc):
    """各市场状态下子策略贡献归因"""
    attr = {}
    for st in ["bull", "bear", "oscillate", "recovery"]:
        mask = np.array([states[d] == st for d in range(nd)])
        if mask.sum() < 5: continue
        st_ret = np.zeros(mask.sum(), dtype=np.float64)
        idx = 0
        for d in range(nd):
            if states[d] == st:
                for sname, wt in alloc.get(st, []):
                    if sname in sub_rets: st_ret[idx] += wt * sub_rets[sname][d]
                idx += 1
        attr[st] = calc_metrics(st_ret, st)
        # 各子策略在该状态下的独立贡献
        sub_contrib = {}
        for sname, wt in alloc.get(st, []):
            if sname in sub_rets:
                sr = np.array([sub_rets[sname][d] for d in range(nd) if states[d] == st])
                sub_contrib[sname] = {"weight": wt, "metrics": calc_metrics(sr)}
        attr[st]["sub_contrib"] = sub_contrib
    return attr

def correlation_analysis(sub_rets, nd):
    """子策略收益相关性"""
    names = list(sub_rets.keys())
    n = len(names)
    corr = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ri, rj = sub_rets[names[i]], sub_rets[names[j]]
            if np.std(ri) > 1e-10 and np.std(rj) > 1e-10:
                corr[i, j] = np.corrcoef(ri, rj)[0, 1]
            else: corr[i, j] = 0.0
    return names, corr

def monthly_returns(ret, ds, nd):
    """月度收益"""
    df = pd.DataFrame({"date": ds[:len(ret)], "ret": ret[:len(ds)]})
    df["ym"] = df["date"].dt.to_period("M")
    monthly = df.groupby("ym")["ret"].apply(lambda x: np.prod(1 + x) - 1)
    return monthly.to_dict()

def signal_correlation(z3, dm, cl, available, nd, ns, w_v7, w_v8):
    """V7 vs V8 信号相关性"""
    wv7 = np.zeros(len(available), dtype=np.float32)
    wv8 = np.zeros(len(available), dtype=np.float32)
    for i, f in enumerate(available):
        wv7[i] = w_v7.get(f, 0.0)
        wv8[i] = w_v8.get(f, 0.0)
    s7 = z3 @ wv7; s8 = z3 @ wv8
    corrs = []
    for d in range(nd):
        mask = dm[d] & (cl[d] > 1e-10) & (s7[d] != 0) & (s8[d] != 0)
        if mask.sum() > 10:
            c = np.corrcoef(s7[d, mask], s8[d, mask])[0, 1]
            if not np.isnan(c): corrs.append(c)
    return {"mean_corr": round(np.mean(corrs), 4), "median_corr": round(np.median(corrs), 4),
            "pct_same_dir": round(np.mean([c > 0 for c in corrs]), 4), "n_days": len(corrs)}

# ═══════════════ 报告输出 ═══════════════

def fmt_m(m):
    return f"Ret={m['total_ret']:+.2%} Ann={m['ann_ret']:+.2%} Vol={m['ann_vol']:.2%} Sharpe={m['sharpe']:.2f} DD={m['max_dd']:.2%} Calmar={m['calmar']:.2f} WR={m['win_rate']:.1%}"

def fmt_delta(v7, v8, key="calmar"):
    d = v8.get(key, 0) - v7.get(key, 0)
    pct = d / abs(v7.get(key, 1e-10)) * 100 if abs(v7.get(key, 1e-10)) > 1e-6 else 0
    return f"{key}: {v7.get(key,0):.2f}→{v8.get(key,0):.2f} ({d:+.2f}, {pct:+.1f}%)"

# ═══════════════ 主流程 ═══════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2026-06-11")
    args = parser.parse_args()

    t_start = time.time()
    logger.info("=" * 80)
    logger.info("V7 vs V8 全面回测开始")
    logger.info("=" * 80)

    # ── 1. 加载数据 ──
    z3, fwd, dm, cl, tks, available, nd, ns, ds, t2i, per = load_data(args.start, args.end)
    pct = per["pct"]

    # ── 2. 加载权重 ──
    w_v7 = load_weights(V7_CONFIG)
    w_v8 = load_weights(V8_CONFIG)
    logger.info(f"V7权重: {len(w_v7)}因子, V8权重: {len(w_v8)}因子")
    # 列出差异因子
    diff_factors = [f for f in w_v7 if f in w_v8 and abs(w_v7[f] - w_v8[f]) > 0.001]
    logger.info(f"权重差异因子({len(diff_factors)}): {diff_factors}")

    # ── 3. 信号相关性 ──
    logger.info("\n── 信号相关性分析 ──")
    sig_corr = signal_correlation(z3, dm, cl, available, nd, ns, w_v7, w_v8)
    logger.info(f"V7/V8信号日频相关性: mean={sig_corr['mean_corr']:.4f}, median={sig_corr['median_corr']:.4f}")
    logger.info(f"同向比例: {sig_corr['pct_same_dir']:.1%} ({sig_corr['n_days']}天)")

    # ── 4. 市场状态检测 ──
    logger.info("\n── 市场状态检测 ──")
    states = detect_market_state(pct, dm, nd, ns)
    from collections import Counter
    st_cnt = Counter(states)
    logger.info(f"状态分布: {dict(st_cnt)}")

    # ── 5. 子策略独立回测 V7 vs V8 ──
    logger.info("\n── 子策略独立回测 ──")
    mf_subs = ["mf_d10_rp", "mf_vol_d10_rp", "c01_layered_d5", "mf50_chip50"]
    non_mf_subs = ["chip_equal_d3", "osr_d10"]
    all_subs = mf_subs + non_mf_subs
    sub_rets_v7 = {}; sub_rets_v8 = {}; sub_metrics_v7 = {}; sub_metrics_v8 = {}

    for sname in all_subs:
        params = SUB_PARAMS.get(sname, SUB_PARAMS["mf_d10_rp"]).copy()
        # mf50_chip50用chip信号+MF权重混合
        if sname == "mf50_chip50":
            params = {"signal": "mf", "rf": 5, "tn": 10, "mhd": 10, "timing": None}
        if sname in non_mf_subs:
            # 非MF策略不受权重翻转影响
            r = bt_sub_strategy(sname, z3, fwd, dm, cl, pct, tks, available, nd, ns, ds, w_v7, params)
            sub_rets_v7[sname] = r; sub_rets_v8[sname] = r.copy()
            sub_metrics_v7[sname] = calc_metrics(r); sub_metrics_v8[sname] = sub_metrics_v7[sname].copy()
            logger.info(f"  {sname} (非MF): {fmt_m(sub_metrics_v7[sname])}")
        else:
            # MF策略分别用V7和V8权重
            r_v7 = bt_sub_strategy(sname, z3, fwd, dm, cl, pct, tks, available, nd, ns, ds, w_v7, params)
            r_v8 = bt_sub_strategy(sname, z3, fwd, dm, cl, pct, tks, available, nd, ns, ds, w_v8, params)
            sub_rets_v7[sname] = r_v7; sub_rets_v8[sname] = r_v8
            sub_metrics_v7[sname] = calc_metrics(r_v7); sub_metrics_v8[sname] = calc_metrics(r_v8)
            logger.info(f"  {sname} V7: {fmt_m(sub_metrics_v7[sname])}")
            logger.info(f"  {sname} V8: {fmt_m(sub_metrics_v8[sname])}")
            logger.info(f"    → {fmt_delta(sub_metrics_v7[sname], sub_metrics_v8[sname])}")

    # ── 6. MSS混合策略 V7 vs V8 ──
    logger.info("\n── MSS混合策略回测 ──")
    mss_v7 = run_mss(sub_rets_v7, states, nd, V7_ALLOC)
    mss_v8 = run_mss(sub_rets_v8, states, nd, V7_ALLOC)  # 同一分配，不同子策略收益
    mss_m_v7 = calc_metrics(mss_v7); mss_m_v8 = calc_metrics(mss_v8)
    logger.info(f"  MSS V7: {fmt_m(mss_m_v7)}")
    logger.info(f"  MSS V8: {fmt_m(mss_m_v8)}")
    logger.info(f"  → {fmt_delta(mss_m_v7, mss_m_v8)}")

    # ── 7. 分年窗口对比 ──
    logger.info("\n── 分年窗口对比 ──")
    win_v7 = window_analysis(mss_v7, ds, nd, WINDOWS)
    win_v8 = window_analysis(mss_v8, ds, nd, WINDOWS)
    for wname in win_v7:
        if wname in win_v8:
            logger.info(f"  {wname}:")
            logger.info(f"    V7: {fmt_m(win_v7[wname])}")
            logger.info(f"    V8: {fmt_m(win_v8[wname])}")
            logger.info(f"    → {fmt_delta(win_v7[wname], win_v8[wname])}")

    # ── 8. 子策略分年窗口 ──
    logger.info("\n── MF子策略分年窗口(V7 vs V8 Calmar) ──")
    for sname in mf_subs:
        logger.info(f"  {sname}:")
        for wname, wstart, wend in WINDOWS:
            sd = pd.Timestamp(wstart); ed = pd.Timestamp(wend)
            mask = np.array([(ds[i] >= sd and ds[i] <= ed) for i in range(nd)])
            if mask.sum() < 10: continue
            m7 = calc_metrics(sub_rets_v7[sname][mask]) if sname in sub_rets_v7 else {}
            m8 = calc_metrics(sub_rets_v8[sname][mask]) if sname in sub_rets_v8 else {}
            if m7 and m8:
                logger.info(f"    {wname}: V7 Calmar={m7.get('calmar',0):.2f} → V8 Calmar={m8.get('calmar',0):.2f}")

    # ── 9. 状态归因对比 ──
    logger.info("\n── 状态归因对比 ──")
    attr_v7 = attribution_by_state(sub_rets_v7, states, nd, V7_ALLOC)
    attr_v8 = attribution_by_state(sub_rets_v8, states, nd, V7_ALLOC)
    for st in ["bull", "bear", "oscillate", "recovery"]:
        if st in attr_v7 and st in attr_v8:
            logger.info(f"  {st}: V7 Calmar={attr_v7[st].get('calmar',0):.2f} → V8 Calmar={attr_v8[st].get('calmar',0):.2f}")
            if "sub_contrib" in attr_v7[st]:
                for sn in attr_v7[st]["sub_contrib"]:
                    c7 = attr_v7[st]["sub_contrib"][sn]["metrics"].get("calmar", 0)
                    c8 = attr_v8[st]["sub_contrib"][sn]["metrics"].get("calmar", 0) if sn in attr_v8[st].get("sub_contrib", {}) else c7
                    logger.info(f"    {sn}: V7={c7:.2f} → V8={c8:.2f}")

    # ── 10. Walk-forward验证 ──
    logger.info("\n── Walk-forward OOS验证 ──")
    wf_windows = [
        ("WF1: 2018-2021IS/2022OOS", "2018-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
        ("WF2: 2019-2022IS/2023OOS", "2019-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
        ("WF3: 2020-2023IS/2024OOS", "2020-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
        ("WF4: 2021-2024IS/2025OOS", "2021-01-01", "2024-12-31", "2025-01-01", "2026-06-11"),
    ]
    wf_results = walk_forward(z3, fwd, dm, cl, pct, tks, available, nd, ns, ds, w_v7, w_v8, wf_windows)
    for wname, wr in wf_results.items():
        logger.info(f"  {wname}:")
        logger.info(f"    V7 IS: {fmt_m(wr['V7_IS'])}  |  V7 OOS: {fmt_m(wr['V7_OOS'])}")
        logger.info(f"    V8 IS: {fmt_m(wr['V8_IS'])}  |  V8 OOS: {fmt_m(wr['V8_OOS'])}")
        logger.info(f"    OOS Calmar变化: {wr['V7_OOS'].get('calmar',0):.2f} → {wr['V8_OOS'].get('calmar',0):.2f}")

    # ── 11. 相关性分析 ──
    logger.info("\n── 子策略收益相关性(V8) ──")
    names_v8, corr_v8 = correlation_analysis(sub_rets_v8, nd)
    for i, n1 in enumerate(names_v8):
        for j, n2 in enumerate(names_v8):
            if i < j:
                logger.info(f"  {n1} ↔ {n2}: {corr_v8[i,j]:.3f}")

    # ── 12. 月度收益对比 ──
    logger.info("\n── 月度收益对比(近12月) ──")
    mr_v7 = monthly_returns(mss_v7, ds, nd)
    mr_v8 = monthly_returns(mss_v8, ds, nd)
    recent = sorted(mr_v7.keys())[-12:]
    for m in recent:
        r7 = mr_v7.get(m, 0); r8 = mr_v8.get(m, 0)
        logger.info(f"  {m}: V7={r7:+.2%}  V8={r8:+.2%}  Δ={r8-r7:+.2%}")

    # ── 13. 保存结果 ──
    results = {
        "meta": {"version": "V7_vs_V8_full", "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "v7_config": V7_CONFIG, "v8_config": V8_CONFIG, "diff_factors": diff_factors},
        "signal_correlation": sig_corr,
        "mss_v7": mss_m_v7, "mss_v8": mss_m_v8,
        "sub_v7": sub_metrics_v7, "sub_v8": sub_metrics_v8,
        "window_v7": win_v7, "window_v8": win_v8,
        "attribution_v7": {k: {kk: vv for kk, vv in v.items() if kk != "sub_contrib"}
                          for k, v in attr_v7.items()},
        "attribution_v8": {k: {kk: vv for kk, vv in v.items() if kk != "sub_contrib"}
                          for k, v in attr_v8.items()},
        "walk_forward": wf_results,
        "correlation_v8": {n: {names_v8[j]: round(corr_v8[i, j], 4) for j in range(len(names_v8))}
                          for i, n in enumerate(names_v8)},
    }
    out_path = os.path.join(RESULTS_DIR, "v8_full_backtest.json")
    with open(out_path, "w") as f: json.dump(results, f, indent=2, default=str)
    logger.info(f"\n结果已保存: {out_path}")

    # ── 14. 总结 ──
    logger.info("\n" + "=" * 80)
    logger.info("V7 vs V8 全面回测总结")
    logger.info("=" * 80)
    logger.info(f"MSS全区间: V7 Calmar={mss_m_v7['calmar']:.2f} → V8 Calmar={mss_m_v8['calmar']:.2f} (Δ={mss_m_v8['calmar']-mss_m_v7['calmar']:+.2f})")
    logger.info(f"MSS全区间: V7 Sharpe={mss_m_v7['sharpe']:.2f} → V8 Sharpe={mss_m_v8['sharpe']:.2f}")
    logger.info(f"MSS全区间: V7 MaxDD={mss_m_v7['max_dd']:.2%} → V8 MaxDD={mss_m_v8['max_dd']:.2%}")
    logger.info(f"信号相关性: mean={sig_corr['mean_corr']:.4f}, 同向={sig_corr['pct_same_dir']:.1%}")
    # OOS窗口改善统计
    oos_improve = sum(1 for wname, wr in wf_results.items()
                      if wr.get("V8_OOS", {}).get("calmar", 0) > wr.get("V7_OOS", {}).get("calmar", 0))
    logger.info(f"Walk-forward OOS改善: {oos_improve}/{len(wf_results)}窗口")
    logger.info(f"总耗时: {time.time()-t_start:.1f}s")

if __name__ == "__main__":
    main()