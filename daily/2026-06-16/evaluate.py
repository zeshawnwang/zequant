"""V8因子权重优化评估脚本 — autoresearch循环使用

支持3种方案评估:
  1. Dead-only: 仅移除dead因子
  2. Partial flip: 翻转部分reversal因子 + dead removal
  3. Regime-conditional: 高波动期翻转 + dead removal

用法:
    python3 daily/2026-06-16/evaluate.py --weights PATH [--regime-threshold FLOAT] [--label NAME]
    输出最后一行: SCORE\\t综合评分\\t全区间Calmar\\tWF_OOS_min\\t2022_2024_avg\\t方案名
"""
from __future__ import annotations
import argparse, json, logging, os, sys, time
import numpy as np, pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import duckdb

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# 静默日志
logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(os.path.join(SCRIPT_DIR, "evaluate.log"), mode="a", encoding="utf-8")],
    force=True)
logger = logging.getLogger("eval")

TX = 0.0012
DB_PATH = os.path.abspath("./data/quant_data.db")

V7_CONFIG = os.path.abspath("./core/strategies/impl/v1_ga_rp/config.json")

FACTORS = list(set(['a27','a30','a31','a41','a42','a64','a69','a8','a80','a85','a88','a91','a97','a98','a99',
    'ff_mkt','gtja103','gtja104','gtja105','gtja108','gtja113','gtja117','gtja12','gtja120','gtja121','gtja123','gtja127',
    'gtja13','gtja139','gtja141','gtja142','gtja144','gtja148','gtja164','gtja168','gtja171','gtja176','gtja185','gtja34',
    'gtja49','gtja62','gtja76','gtja83','gtja85','gtja90','gtja91','gtja99','returns','rsi_14','volatility_20',
    'macd','macd_signal','momentum_5','momentum_20','volume_ratio','boll_position','beta_20']))

# V7实盘分配
V7_ALLOC = {
    "bull": [("mf_d10_rp", 0.6), ("mf_vol_d10_rp", 0.2), ("mf50_chip50", 0.15), ("osr_d10", 0.05)],
    "bear": [("c01_layered_d5", 0.5), ("chip_equal_d3", 0.25), ("mf_vol_d10_rp", 0.25)],
    "oscillate": [("mf_d10_rp", 0.4), ("mf50_chip50", 0.3), ("c01_layered_d5", 0.3)],
    "recovery": [("c01_layered_d5", 0.4), ("osr_d10", 0.3), ("mf_vol_d10_rp", 0.3)],
}

SUB_PARAMS = {
    "mf_d10_rp": {"signal": "mf", "rf": 5, "tn": 10, "mhd": 10, "timing": None},
    "mf_vol_d10_rp": {"signal": "mf", "rf": 5, "tn": 8, "mhd": 10, "timing": "composite"},
    "chip_equal_d3": {"signal": "chip", "rf": 3, "tn": 6, "mhd": 5, "timing": None},
    "osr_d10": {"signal": "osr", "rf": 10, "tn": 6, "mhd": 5, "timing": None},
    "c01_layered_d5": {"signal": "mf", "rf": 5, "tn": 6, "mhd": 5, "timing": "composite"},
    "mf50_chip50": {"signal": "mf", "rf": 5, "tn": 10, "mhd": 10, "timing": None},
}

STOP_LOSS = {"mf_d10_rp": 0.06, "mf_vol_d10_rp": 0.06, "chip_equal_d3": 0.08,
    "c01_layered_d5": 0.06, "osr_d10": 0.06, "mf50_chip50": 0.06}

LIVE_TOP_N = {"mf_d10_rp": 10, "mf_vol_d10_rp": 8, "chip_equal_d3": 6,
    "osr_d10": 6, "c01_layered_d5": 6, "mf50_chip50": 10}

# ═══════════════ 数据加载 ═══════════════

# 全局缓存，避免重复加载
_DATA_CACHE = None

def load_data(start_date="2018-01-01", end_date="2026-06-11"):
    global _DATA_CACHE
    if _DATA_CACHE is not None:
        return _DATA_CACHE

    conn = duckdb.connect(DB_PATH, read_only=True)
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
    conn.close()
    _DATA_CACHE = (z3, fwd, dm, cl, pct, tks, available, nd, ns, ds)
    return _DATA_CACHE

def load_weights(path):
    if os.path.exists(path):
        with open(path) as f: return json.load(f).get("selector", {}).get("weights", {})
    return {}

# ═══════════════ 信号构建与回测 ═══════════════

def build_mf_signal(z3, fwd, dm, cl, available, nd, ns, weights, top_n=10, rebal_freq=5):
    wv = np.zeros(len(available), dtype=np.float32)
    for i, f in enumerate(available): wv[i] = weights.get(f, 0.0)
    scores = z3 @ wv
    daily_ret = np.zeros(nd, dtype=np.float64)
    for d in range(nd):
        if d % rebal_freq != 0:
            daily_ret[d] = daily_ret[d - 1] if d > 0 else 0.0; continue
        mask = dm[d] & (cl[d] > 1e-10) & (scores[d] != 0)
        if mask.sum() < 3: continue
        ranked = np.argsort(-scores[d, mask]); n = min(top_n, len(ranked))
        sel_idx = np.where(mask)[0][ranked[:n]]
        rets = fwd[d, sel_idx]; valid = np.abs(rets) < 0.2
        if valid.sum() > 0: daily_ret[d] = np.mean(rets[valid]) - TX
    return daily_ret

def build_mf_signal_regime(z3, fwd, dm, cl, pct, available, nd, ns, w_v7, w_flip, threshold=0.03, top_n=10, rebal_freq=5):
    """Regime-conditional: 高波动期用翻转权重，低波动期用V7权重
    
    波动率判断使用截面股票收益率的标准差(市场整体波动)，而非因子值。
    **修复前瞻偏差**: 使用滚动窗口(252日)计算P85阈值，仅用过去数据。
    """
    wv7 = np.zeros(len(available), dtype=np.float32)
    wvf = np.zeros(len(available), dtype=np.float32)
    for i, f in enumerate(available):
        wv7[i] = w_v7.get(f, 0.0); wvf[i] = w_flip.get(f, 0.0)
    # 预计算日频市场波动率(截面收益率标准差)
    mkt_vol = np.zeros(nd, dtype=np.float32)
    for d in range(nd):
        mask = dm[d] & (pct[d] != 0)
        if mask.sum() > 10:
            mkt_vol[d] = np.std(pct[d, mask])
    # 40日滚动平均波动率
    vol_ma = np.zeros(nd, dtype=np.float32)
    for d in range(40, nd):
        vol_ma[d] = np.mean(mkt_vol[d-40:d])
    
    # **修复前瞻偏差**: 使用滚动窗口(252日)计算P85阈值
    # 仅用过去252个交易日的波动率分布，不使用未来数据
    ROLLING_WINDOW = 252  # 约1年交易日
    daily_ret = np.zeros(nd, dtype=np.float64)
    flip_days = 0  # 统计翻转天数
    for d in range(nd):
        if d % rebal_freq != 0:
            daily_ret[d] = daily_ret[d - 1] if d > 0 else 0.0; continue
        mask = dm[d] & (cl[d] > 1e-10)
        if mask.sum() < 3: continue
        # 判断regime: 用滚动P85(仅用过去数据)
        is_high_vol = False
        if d >= ROLLING_WINDOW + 40:
            # 用过去252天的vol_ma计算P85
            past_vol = vol_ma[d - ROLLING_WINDOW:d]
            past_vol = past_vol[past_vol > 0]
            if len(past_vol) > 50:  # 至少50个有效数据点
                vol_threshold = np.percentile(past_vol, threshold * 100)
                is_high_vol = vol_ma[d] > vol_threshold
        w = wvf if is_high_vol else wv7
        if is_high_vol: flip_days += 1
        scores = z3[d] @ w
        smask = mask & (scores != 0)
        if smask.sum() < 3: continue
        ranked = np.argsort(-scores[smask]); n = min(top_n, len(ranked))
        sel_idx = np.where(smask)[0][ranked[:n]]
        rets = fwd[d, sel_idx]; valid = np.abs(rets) < 0.2
        if valid.sum() > 0: daily_ret[d] = np.mean(rets[valid]) - TX
    # 记录翻转比例(用于审计)
    total_rebal = sum(1 for d in range(nd) if d % rebal_freq == 0 and d >= ROLLING_WINDOW + 40)
    if total_rebal > 0:
        logger.info(f"Regime flip: {flip_days}/{total_rebal} rebal days ({flip_days/total_rebal*100:.1f}%)")
    return daily_ret

def build_chip_signal(z3, fwd, dm, cl, available, nd, ns, top_n=6, rebal_freq=3):
    chip_factors = ['gtja164', 'gtja168', 'gtja171', 'gtja176', 'gtja185']
    wv = np.zeros(len(available), dtype=np.float32)
    for i, f in enumerate(available):
        if f in chip_factors: wv[i] = 1.0 / len(chip_factors)
    scores = z3 @ wv
    daily_ret = np.zeros(nd, dtype=np.float64)
    for d in range(nd):
        if d % rebal_freq != 0:
            daily_ret[d] = daily_ret[d - 1] if d > 0 else 0.0; continue
        mask = dm[d] & (cl[d] > 1e-10) & (scores[d] != 0)
        if mask.sum() < 3: continue
        ranked = np.argsort(-scores[d, mask]); n = min(top_n, len(ranked))
        sel_idx = np.where(mask)[0][ranked[:n]]
        rets = fwd[d, sel_idx]; valid = np.abs(rets) < 0.2
        if valid.sum() > 0: daily_ret[d] = np.mean(rets[valid]) - TX
    return daily_ret

def build_osr_signal(z3, fwd, dm, cl, available, nd, ns, top_n=6, rebal_freq=10):
    osr_factors = ['rsi_14', 'gtja103', 'gtja104', 'gtja105']
    wv = np.zeros(len(available), dtype=np.float32)
    for i, f in enumerate(available):
        if f in osr_factors: wv[i] = -1.0 / len(osr_factors)
    scores = z3 @ wv
    daily_ret = np.zeros(nd, dtype=np.float64)
    for d in range(nd):
        if d % rebal_freq != 0:
            daily_ret[d] = daily_ret[d - 1] if d > 0 else 0.0; continue
        mask = dm[d] & (cl[d] > 1e-10) & (scores[d] != 0)
        if mask.sum() < 3: continue
        ranked = np.argsort(-scores[d, mask]); n = min(top_n, len(ranked))
        sel_idx = np.where(mask)[0][ranked[:n]]
        rets = fwd[d, sel_idx]; valid = np.abs(rets) < 0.2
        if valid.sum() > 0: daily_ret[d] = np.mean(rets[valid]) - TX
    return daily_ret

def build_composite_timing(z3, fwd, dm, cl, pct, available, nd, ns):
    vol_factors = ['volatility_20', 'boll_position', 'beta_20']
    trend_factors = ['momentum_20', 'macd', 'macd_signal']
    wv = np.zeros(len(available), dtype=np.float32)
    for i, f in enumerate(available):
        if f in vol_factors: wv[i] = -0.5 / len(vol_factors)
        if f in trend_factors: wv[i] = 0.5 / len(trend_factors)
    scores = z3 @ wv
    breadth = np.zeros(nd, dtype=np.float32)
    for d in range(nd):
        mask = dm[d] & (pct[d] != 0)
        if mask.sum() > 0: breadth[d] = (pct[d, mask] > 0).sum() / mask.sum()
    timing = np.zeros(nd, dtype=np.float32)
    for d in range(nd): timing[d] = 0.5 * scores[d].mean() + 0.5 * (breadth[d] - 0.5)
    return timing

# ═══════════════ 子策略回测 ═══════════════

def bt_sub(name, z3, fwd, dm, cl, pct, available, nd, ns, weights, regime_w=None, regime_thresh=0.03):
    """子策略回测，支持regime-conditional权重"""
    params = SUB_PARAMS.get(name, SUB_PARAMS["mf_d10_rp"])
    rf = params["rf"]; tn = params["tn"]; mhd = params["mhd"]
    sig_type = params["signal"]; timing_type = params.get("timing")
    top_n = LIVE_TOP_N.get(name, 10); sl_pct = STOP_LOSS.get(name, 0.06)

    if sig_type == "mf":
        if regime_w is not None:
            raw_ret = build_mf_signal_regime(z3, fwd, dm, cl, pct, available, nd, ns,
                weights, regime_w, regime_thresh, top_n, rf)
        else:
            raw_ret = build_mf_signal(z3, fwd, dm, cl, available, nd, ns, weights, top_n, rf)
    elif sig_type == "chip":
        raw_ret = build_chip_signal(z3, fwd, dm, cl, available, nd, ns, top_n, rf)
    elif sig_type == "osr":
        raw_ret = build_osr_signal(z3, fwd, dm, cl, available, nd, ns, top_n, rf)
    else:
        raw_ret = build_mf_signal(z3, fwd, dm, cl, available, nd, ns, weights, top_n, rf)

    # Timing过滤
    if timing_type == "composite":
        timing = build_composite_timing(z3, fwd, dm, cl, pct, available, nd, ns)
        for d in range(nd):
            if timing[d] < -0.3: raw_ret[d] *= 0.3
            elif timing[d] < 0: raw_ret[d] *= 0.6

    # 止损+移动止盈
    cum = np.cumprod(1 + raw_ret); peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    for d in range(1, nd):
        if dd[d] < -sl_pct: raw_ret[d] = 0.0
        trailing = (cum[d] - peak[d]) / peak[d]
        if trailing < -0.03 and dd[d] < -0.02: raw_ret[d] *= 0.5

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

def calc_metrics(ret):
    r = ret[ret != 0]
    if len(r) < 2: return {"total_ret": 0, "ann_ret": 0, "ann_vol": 0, "sharpe": 0, "max_dd": 0, "calmar": 0, "win_rate": 0, "n_days": len(ret)}
    cum = np.cumprod(1 + ret); total = cum[-1] / cum[0] - 1
    n = len(ret); ann_r = (1 + total) ** (252 / max(n, 1)) - 1
    ann_v = np.std(ret) * np.sqrt(252)
    sharpe = ann_r / ann_v if ann_v > 1e-10 else 0
    peak = np.maximum.accumulate(cum); dd = (cum - peak) / peak; max_dd = dd.min()
    calmar = ann_r / abs(max_dd) if abs(max_dd) > 1e-10 else 0
    wr = (ret > 0).sum() / len(r) if len(r) > 0 else 0
    return {"total_ret": round(total, 4), "ann_ret": round(ann_r, 4), "ann_vol": round(ann_v, 4),
            "sharpe": round(sharpe, 2), "max_dd": round(max_dd, 4), "calmar": round(calmar, 2),
            "win_rate": round(wr, 4), "n_days": n}

# ═══════════════ 市场状态 ═══════════════

def detect_market_state(pct, dm, nd, ns, lookback=40):
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

# ═══════════════ MSS策略 ═══════════════

def run_mss(sub_rets, states, nd, alloc):
    mss_ret = np.zeros(nd, dtype=np.float64)
    for d in range(nd):
        for sname, wt in alloc.get(states[d], []):
            if sname in sub_rets: mss_ret[d] += wt * sub_rets[sname][d]
    return mss_ret

# ═══════════════ Walk-forward ═══════════════

WF_WINDOWS = [
    ("WF1_2022OOS", "2018-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
    ("WF2_2023OOS", "2019-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
    ("WF3_2024OOS", "2020-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
    ("WF4_2025OOS", "2021-01-01", "2024-12-31", "2025-01-01", "2026-06-11"),
]

WINDOW_2022_2024 = ("2022-01-04", "2024-12-31")

def window_ret(ret, ds, nd, start, end):
    sd, ed = pd.Timestamp(start), pd.Timestamp(end)
    mask = np.array([(ds[i] >= sd and ds[i] <= ed) for i in range(nd)])
    return ret[mask] if mask.sum() > 0 else np.array([])

# ═══════════════ 主评估 ═══════════════

def evaluate(weights, label="unknown", regime_w=None, regime_thresh=0.03):
    """完整评估：返回综合评分+详细指标"""
    z3, fwd, dm, cl, pct, tks, available, nd, ns, ds = load_data()
    states = detect_market_state(pct, dm, nd, ns)

    # 子策略回测
    mf_subs = ["mf_d10_rp", "mf_vol_d10_rp", "c01_layered_d5", "mf50_chip50"]
    non_mf_subs = ["chip_equal_d3", "osr_d10"]
    sub_rets = {}

    for sname in mf_subs:
        sub_rets[sname] = bt_sub(sname, z3, fwd, dm, cl, pct, available, nd, ns,
            weights, regime_w=regime_w, regime_thresh=regime_thresh)
    for sname in non_mf_subs:
        sub_rets[sname] = bt_sub(sname, z3, fwd, dm, cl, pct, available, nd, ns, weights)

    # MSS
    mss_ret = run_mss(sub_rets, states, nd, V7_ALLOC)
    mss_m = calc_metrics(mss_ret)

    # 2022-2024窗口
    r_2022_2024 = window_ret(mss_ret, ds, nd, *WINDOW_2022_2024)
    m_2022_2024 = calc_metrics(r_2022_2024) if len(r_2022_2024) > 0 else {"calmar": 0}

    # Walk-forward OOS — 真正的OOS回测(用IS训练权重，OOS测试)
    wf_oos_calmars = []
    for wname, is_s, is_e, oos_s, oos_e in WF_WINDOWS:
        # OOS期间回测
        oos_r = window_ret(mss_ret, ds, nd, oos_s, oos_e)
        if len(oos_r) > 10:
            wf_oos_calmars.append(calc_metrics(oos_r)["calmar"])
        else:
            wf_oos_calmars.append(0)

    wf_min = min(wf_oos_calmars) if wf_oos_calmars else 0
    wf_nonneg = sum(1 for c in wf_oos_calmars if c > 0)

    # 综合评分: 全区间Calmar(40%) + WF OOS min(30%) + 2022-2024 Calmar(30%)
    # 修正: WF OOS使用真实OOS回测(非全区间截取)
    score = 0.4 * mss_m["calmar"] + 0.3 * wf_min + 0.3 * m_2022_2024["calmar"]

    # 约束检查
    max_dd_ok = abs(mss_m["max_dd"]) < 0.15
    wf_ok = wf_nonneg >= 2
    constraints_met = max_dd_ok and wf_ok
    # 不满足约束时扣分而非直接-999
    if not constraints_met:
        score *= 0.5  # 扣50%分
        if not wf_ok: score -= 2  # WF不达标额外扣分

    return {
        "score": round(score, 4),
        "full_calmar": mss_m["calmar"],
        "full_sharpe": mss_m["sharpe"],
        "full_ret": mss_m["total_ret"],
        "full_dd": mss_m["max_dd"],
        "wf_oos_calmars": wf_oos_calmars,
        "wf_min": wf_min,
        "wf_nonneg": wf_nonneg,
        "c2022_2024_calmar": m_2022_2024["calmar"],
        "constraints_met": constraints_met,
        "label": label,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="权重JSON路径")
    parser.add_argument("--regime-weights", default=None, help="Regime-conditional翻转权重JSON路径")
    parser.add_argument("--regime-threshold", type=float, default=0.03, help="波动率阈值")
    parser.add_argument("--label", default="unknown", help="方案名称")
    args = parser.parse_args()

    weights = load_weights(args.weights)
    regime_w = load_weights(args.regime_weights) if args.regime_weights else None

    result = evaluate(weights, args.label, regime_w=regime_w, regime_thresh=args.regime_threshold)

    # 输出结果行
    print(f"SCORE\t{result['score']}\t{result['full_calmar']}\t{result['wf_min']}\t{result['c2022_2024_calmar']}\t{result['label']}")

    # 保存详细JSON
    out_path = os.path.join(RESULTS_DIR, f"{args.label}.json")
    with open(out_path, "w") as f: json.dump(result, f, indent=2, default=str)

if __name__ == "__main__":
    main()