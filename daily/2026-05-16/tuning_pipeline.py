"""
统一调优管道 — 15个策略参数扫描，针对2022熊市表现改善+全区间Sharpe提升。

每个实验记录全区间、牛市OOS(2024-07~2026-04)、熊市(2022-01~2022-12)三区间结果。
"""
import os, sys, json, logging, gc
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from core.database import Database
from core.positioners import RPPortfolioWeights

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("tuning")
TX = 0.0012

OUT_DIR = os.path.join(os.path.dirname(__file__), "tuning_results")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 因子定义 (V1 50最佳 + 16新技术因子) ──
_V1_BEST = [
    'ff_mkt','gtja142','gtja144','gtja171','gtja103','gtja85','a88','a31',
    'rsi_14','gtja139','gtja123','a42','a41','a97','gtja148','gtja99',
    'gtja117','gtja76','gtja90','volatility_20','gtja113','gtja141','a99',
    'gtja12','gtja83','gtja164','a98','gtja49','gtja121','a85','gtja104',
    'gtja185','gtja176','a80','gtja62','a8','gtja34','returns','gtja168',
    'gtja108','gtja105','gtja127','a27','a64','gtja91','a30','a69','a91',
    'gtja13','gtja120',
]
_NEW_TECH = [
    'ma5','ma20','ma60','ma120',
    'ma_alignment_score','ma60_trend','ma120_trend',
    'macd_above_zero','macd_golden_cross',
    'ma_angle_20','volume_breakout_ratio','volume_contraction',
    'chip_concentration','ma_convergence',
    'box_breakout','breakout_strength',
]
ALL_FACTORS = list(dict.fromkeys(_V1_BEST + _NEW_TECH))

# ── 三区间定义 ──
BEAR_START = np.datetime64("2022-01-04")
BEAR_END = np.datetime64("2022-12-30")
OOS_START = np.datetime64("2024-07-01")
OOS_END = np.datetime64("2026-04-30")


# ════════════════════════════════════════════
# 数据加载
# ════════════════════════════════════════════
def load():
    db = Database()
    all_cols = db.list_factor_columns()
    available = [c for c in ALL_FACTORS if c in all_cols]
    logger.info(f"可用因子: {len(available)}/{len(ALL_FACTORS)}")
    df = db.get_factors(start_date="2018-01-01", end_date="2026-04-30",
                        factor_names=available, with_close=True)
    df['date'] = pd.to_datetime(df['date'])
    ds_arr = sorted(df['date'].unique())
    ds = np.array(ds_arr, dtype='datetime64[ns]')
    tks = db.get_symbols()['symbol'].tolist()
    nd, ns, nf = len(ds), len(tks), len(available)
    t2i = {t: i for i, t in enumerate(tks)}
    d2i = {d: i for i, d in enumerate(ds_arr)}
    v3 = np.full((nd, ns, nf), np.nan, dtype=np.float32)
    dm = np.zeros((nd, ns), dtype=bool)
    cl = np.zeros((nd, ns), dtype=np.float32)
    di = np.array([d2i[d] for d in df['date']], dtype=np.int32)
    si = np.array([t2i.get(s, -1) for s in df['symbol']], dtype=np.int32)
    v = si >= 0; di, si = di[v], si[v]
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
            r = a[d, :]; nz = r[r != 0]
            if len(nz) > 1:
                lo, hi = np.quantile(nz, [0.01, 0.99])
                c = np.clip(r, lo, hi)
                mu, sd = np.mean(c), np.std(c)
                z3[d, :, fi] = (c - mu) / sd if sd > 1e-10 else 0.0
    logger.info(f"数据: {nd}天 × {ns}只 × {nf}因子")
    return z3, fwd, dm, tks, available, nd, ns, ds


# ════════════════════════════════════════════
# 权重加载
# ════════════════════════════════════════════
def v1w():
    p = os.path.join(os.path.dirname(__file__), '..', '2026-05-13', 'v2', 'v1_reference', 'ga_results.json')
    if os.path.exists(p):
        with open(p) as f:
            for it in json.load(f):
                if 'L1中_80代' in it['label']:
                    return it['configs'][0]['weights']
    return {}

def load_ga_weights():
    p1 = "daily/2026-05-16/x4_x5_results/x5_results.json"
    p2 = "daily/2026-05-13/v1/decoupled_results.json"
    ga_w = None
    for p in [p1, p2]:
        fp = os.path.join(os.path.dirname(__file__), '..', '..', p)
        if os.path.exists(fp):
            with open(fp) as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    if 'weights' in data[0]:
                        ga_w = {k: float(v) for k, v in data[0]['weights'].items()}
                        break
                    elif 'configs' in data[0]:
                        ga_w = {k: float(v) for k, v in data[0]['configs'][0]['weights'].items()}
                        break
    return ga_w or {}


# ════════════════════════════════════════════
# 回测函数
# ════════════════════════════════════════════
def bt(sig, fwd, dm, name, rf=3, tn=40, pos_ratio=None, mhd=5, use_limit_filter=False):
    nd, ns = sig.shape
    alloc = RPPortfolioWeights(top_n=tn, min_hold_days=mhd)
    pw = np.zeros(ns, dtype=np.float32)
    hs = np.full(ns, -1, dtype=np.int32)
    rh = 0; eq = np.ones(nd, dtype=np.float64); dr = np.zeros(nd, dtype=np.float64); ttx = 0.0; nt = 0
    for i in range(1, nd):
        rebal = (i % rf == 0)
        if rebal:
            pr = pos_ratio[i] if pos_ratio is not None else 1.0
            nw = alloc.allocate(sig[i], fwd, i, pw, hs, rh)
            # 涨跌停过滤
            if use_limit_filter:
                limit_mask = np.abs(fwd[i]) > 0.095
                if np.any(limit_mask):
                    locked_weight = np.sum(nw[limit_mask])
                    nw[limit_mask] = 0.0
                    unlocked = ~limit_mask & (nw > 0)
                    if np.any(unlocked):
                        nw[unlocked] += locked_weight * nw[unlocked] / np.sum(nw[unlocked])
            to = float(np.sum(np.abs(nw - pw))); txc = 0.5 * to * TX; ttx += txc
            if to > 0.01: nt += 1
            pw = nw
            for j in range(ns):
                if nw[j] > 0 and hs[j] < 0: hs[j] = rh + 1
        else:
            mk = dm[i] & (pw > 0)
            if np.any(mk):
                p2 = pw[mk].copy() / float(np.sum(pw[mk]))
                pw = np.zeros(ns, dtype=np.float32); pw[mk] = p2
        pr = pos_ratio[i] if pos_ratio is not None else 1.0
        rt = pr * float(np.dot(pw, fwd[i]))
        rt = 0.0 if (np.isnan(rt) or np.isinf(rt)) else rt
        dr[i] = rt; eq[i] = eq[i-1] * (1.0 + rt); rh += 1
    ny = nd / 252.0
    ar = (float(eq[-1] / eq[0])) ** (1.0 / max(ny, 0.5)) - 1.0
    lr = np.log(eq[1:] / eq[:-1])
    sp = float(np.mean(lr) / max(np.std(lr), 1e-10) * np.sqrt(252))
    cm = np.maximum.accumulate(eq); dd = (eq - cm) / cm; mdd = float(np.min(dd))
    cal = ar / abs(mdd) if abs(mdd) > 0 else 0
    wr = int(np.sum(dr > 0)) / max(int(np.sum(dr > 0)) + int(np.sum(dr < 0)), 1)
    logger.info(f"  [{name}] 年化={ar*100:.2f}% Sharpe={sp:.3f} 回撤={mdd*100:.2f}% Calmar={cal:.3f}")
    return {"name": name, "annual_return": round(ar, 4), "sharpe": round(sp, 4),
            "max_drawdown": round(mdd, 4), "calmar": round(cal, 4),
            "win_rate": round(wr, 4), "n_trades": nt}


def bt_series(sig, fwd, dm, rf=3, tn=40, pos_ratio=None, mhd=5):
    nd, ns = sig.shape
    alloc = RPPortfolioWeights(top_n=tn, min_hold_days=mhd)
    pw = np.zeros(ns, dtype=np.float32); hs = np.full(ns, -1, dtype=np.int32)
    rh = 0; dr = np.zeros(nd, dtype=np.float64)
    for i in range(1, nd):
        rebal = (i % rf == 0)
        if rebal:
            pr = pos_ratio[i] if pos_ratio is not None else 1.0
            nw = alloc.allocate(sig[i], fwd, i, pw, hs, rh)
            pw = nw
            for j in range(ns):
                if nw[j] > 0 and hs[j] < 0: hs[j] = rh + 1
        else:
            mk = dm[i] & (pw > 0)
            if np.any(mk):
                p2 = pw[mk].copy() / float(np.sum(pw[mk]))
                pw = np.zeros(ns, dtype=np.float32); pw[mk] = p2
        pr = pos_ratio[i] if pos_ratio is not None else 1.0
        rt = pr * float(np.dot(pw, fwd[i]))
        dr[i] = 0.0 if (np.isnan(rt) or np.isinf(rt)) else rt; rh += 1
    return dr


def bt_cov(sig, fwd, dm, name, rf=3, tn=40, cov_lb=20):
    nd, ns = sig.shape
    pw = np.zeros(ns, dtype=np.float32); hs = np.full(ns, -1, dtype=np.int32); rh = 0
    eq = np.ones(nd, dtype=np.float64); dr = np.zeros(nd, dtype=np.float64); ttx = 0.0; nt = 0
    for i in range(1, nd):
        rebal = (i % rf == 0)
        if rebal:
            si = np.argsort(-sig[i])[:tn]
            lb = min(i, cov_lb)
            if lb >= 5:
                sub = fwd[max(0, i - lb):i, :]
                sub = sub[:, si]
                cov = np.cov(sub.T)
                iv = 1.0 / np.sqrt(np.diag(cov) + 1e-10)
                nw = np.zeros(ns); nw[si] = iv / np.sum(iv)
            else:
                nw = np.zeros(ns); nw[si] = 1.0 / len(si)
            to = float(np.sum(np.abs(nw - pw))); txc = 0.5 * to * TX; ttx += txc
            if to > 0.01: nt += 1
            pw = nw
            for j in range(ns):
                if nw[j] > 0 and hs[j] < 0: hs[j] = rh + 1
        else:
            mk = dm[i] & (pw > 0)
            if np.any(mk):
                p2 = pw[mk].copy() / float(np.sum(pw[mk]))
                pw = np.zeros(ns, dtype=np.float32); pw[mk] = p2
        rt = float(np.dot(pw, fwd[i]))
        rt = 0.0 if (np.isnan(rt) or np.isinf(rt)) else rt
        dr[i] = rt; eq[i] = eq[i-1] * (1.0 + rt); rh += 1
    ny = nd / 252.0
    ar = (float(eq[-1] / eq[0])) ** (1.0 / max(ny, 0.5)) - 1.0
    lr = np.log(eq[1:] / eq[:-1])
    sp = float(np.mean(lr) / max(np.std(lr), 1e-10) * np.sqrt(252))
    cm = np.maximum.accumulate(eq); dd = (eq - cm) / cm; mdd = float(np.min(dd))
    cal = ar / abs(mdd) if abs(mdd) > 0 else 0
    wr = int(np.sum(dr > 0)) / max(int(np.sum(dr > 0)) + int(np.sum(dr < 0)), 1)
    logger.info(f"  [{name}] 年化={ar*100:.2f}% Sharpe={sp:.3f} 回撤={mdd*100:.2f}% Calmar={cal:.3f}")
    return {"name": name, "annual_return": round(ar, 4), "sharpe": round(sp, 4),
            "max_drawdown": round(mdd, 4), "calmar": round(cal, 4),
            "win_rate": round(wr, 4), "n_trades": nt}


# ════════════════════════════════════════════
# 三区间评估
# ════════════════════════════════════════════
def eval_from_dr(dr, name):
    nd = len(dr)
    eq = np.ones(nd, dtype=np.float64)
    for i in range(1, nd): eq[i] = eq[i-1] * (1.0 + dr[i])
    ny = nd / 252.0
    ar = (float(eq[-1] / eq[0])) ** (1.0 / max(ny, 0.5)) - 1.0
    lr = np.log(eq[1:] / eq[:-1])
    sp = float(np.mean(lr) / max(np.std(lr), 1e-10) * np.sqrt(252))
    cm = np.maximum.accumulate(eq); dd = (eq - cm) / cm; mdd = float(np.min(dd))
    cal = ar / abs(mdd) if abs(mdd) > 0 else 0
    wr = int(np.sum(dr > 0)) / max(int(np.sum(dr > 0)) + int(np.sum(dr < 0)), 1)
    return {"annual_return": round(ar, 4), "sharpe": round(sp, 4),
            "max_drawdown": round(mdd, 4), "calmar": round(cal, 4),
            "win_rate": round(wr, 4), "n_trades": 0}


def run_bt_triple(sig, fwd, dm, ds, label, rf=3, tn=40, pos_ratio=None, mhd=5,
                  use_limit_filter=False, cov_mode=False, cov_lb=20):
    """全区间 + 熊市 + 牛市OOS 三区间评估。"""
    results = {}
    # 全区间
    if cov_mode:
        r = bt_cov(sig, fwd, dm, f"{label}_full", rf=rf, tn=tn, cov_lb=cov_lb)
    else:
        r = bt(sig, fwd, dm, f"{label}_full", rf=rf, tn=tn, pos_ratio=pos_ratio,
               mhd=mhd, use_limit_filter=use_limit_filter)
    results["full"] = r

    # 熊市 2022
    bear_mask = (ds >= BEAR_START) & (ds <= BEAR_END)
    bear_idx = np.where(bear_mask)[0]
    if len(bear_idx) > 0:
        b_start, b_end = bear_idx[0], bear_idx[-1] + 1
        sig_b = sig[b_start:b_end]
        fwd_b = fwd[b_start:b_end]
        dm_b = dm[b_start:b_end]
        pr_b = pos_ratio[b_start:b_end] if pos_ratio is not None else None
        if cov_mode:
            r_b = bt_cov(sig_b, fwd_b, dm_b, f"{label}_bear", rf=rf, tn=tn, cov_lb=cov_lb)
        else:
            r_b = bt(sig_b, fwd_b, dm_b, f"{label}_bear", rf=rf, tn=tn,
                     pos_ratio=pr_b, mhd=mhd, use_limit_filter=use_limit_filter)
        r_b["period"] = "bear"
        results["bear"] = r_b

    # 牛市OOS 2024-07 ~ 2026-04
    oos_mask = (ds >= OOS_START) & (ds <= OOS_END)
    oos_idx = np.where(oos_mask)[0]
    if len(oos_idx) > 0:
        o_start, o_end = oos_idx[0], oos_idx[-1] + 1
        sig_o = sig[o_start:o_end]
        fwd_o = fwd[o_start:o_end]
        dm_o = dm[o_start:o_end]
        pr_o = pos_ratio[o_start:o_end] if pos_ratio is not None else None
        if cov_mode:
            r_o = bt_cov(sig_o, fwd_o, dm_o, f"{label}_oos", rf=rf, tn=tn, cov_lb=cov_lb)
        else:
            r_o = bt(sig_o, fwd_o, dm_o, f"{label}_oos", rf=rf, tn=tn,
                     pos_ratio=pr_o, mhd=mhd, use_limit_filter=use_limit_filter)
        r_o["period"] = "oos"
        results["oos"] = r_o
    return results


def run_series_triple(sig, fwd, dm, ds, rf=3, tn=40, pos_ratio=None, mhd=5):
    """bt_series版：返回三区间的收益序列字典。"""
    nd = len(sig)
    dr = bt_series(sig, fwd, dm, rf=rf, tn=tn, pos_ratio=pos_ratio, mhd=mhd)

    bear_mask = (ds >= BEAR_START) & (ds <= BEAR_END)
    bear_idx = np.where(bear_mask)[0]
    oos_mask = (ds >= OOS_START) & (ds <= OOS_END)
    oos_idx = np.where(oos_mask)[0]

    dr_bear = dr[bear_idx] if len(bear_idx) > 0 else np.array([])
    dr_oos = dr[oos_idx] if len(oos_idx) > 0 else np.array([])

    return {"full": dr, "bear": dr_bear, "oos": dr_oos}


# ════════════════════════════════════════════
# 记录结果
# ════════════════════════════════════════════
def record_result(all_res, label, triple_results):
    for period_key in ["full", "bear", "oos"]:
        if period_key in triple_results:
            r = triple_results[period_key].copy()
            r["name"] = f"{label}_{period_key}"
            r["period"] = period_key
            all_res.append(r)


def record_from_dr_dict(all_res, label, dr_dict):
    for period_key in ["full", "bear", "oos"]:
        dr = dr_dict.get(period_key)
        if dr is not None and len(dr) > 0:
            r = eval_from_dr(dr, f"{label}_{period_key}")
            r["name"] = f"{label}_{period_key}"
            r["period"] = period_key
            r["n_trades"] = 0
            all_res.append(r)


# ════════════════════════════════════════════
# Main
# ════════════════════════════════════════════
def main():
    z3, fwd, dm, tks, fnames, nd, ns, ds = load()
    fi = {fn: i for i, fn in enumerate(fnames)}
    v1w_dict = v1w()
    all_res = []

    # ── 构建MF信号 (V1权重) ──
    wv = np.zeros(len(fnames), dtype=np.float32)
    for fi_i, fc in enumerate(fnames):
        if fc in v1w_dict: wv[fi_i] = float(v1w_dict[fc])
    s = np.sum(np.abs(wv)); wv /= s if s > 0 else 1
    mf = np.nan_to_num(np.tensordot(z3, wv, axes=(2, 0)), nan=-1e10, neginf=-1e10)

    # ── 构建Chip信号 ──
    vol20_idx = fi.get('volatility_20'); m20_idx = fi.get('momentum_20')
    chip_sig = np.full((nd, ns), -np.inf, dtype=np.float32)
    for d in range(nd):
        s_chip = np.zeros(ns)
        if vol20_idx is not None: s_chip += np.where(z3[d, :, vol20_idx] < -0.3, 1.0, 0.0) * 0.5
        if m20_idx is not None: s_chip += np.where(np.abs(z3[d, :, m20_idx]) < 0.3, 1.0, 0.0) * 0.3
        chip_sig[d] = np.nan_to_num(s_chip, nan=-1e10)

    # ── 构建OSR信号 ──
    rsi_idx = fi.get('rsi_14'); m5_idx = fi.get('momentum_5')
    osr_sig = np.full((nd, ns), -np.inf, dtype=np.float32)
    for d in range(nd):
        s_osr = np.zeros(ns)
        if rsi_idx is not None: s_osr += np.where(z3[d, :, rsi_idx] < -0.5, 1.0, 0.0) * -0.5
        if m5_idx is not None: s_osr += np.where(z3[d, :, m5_idx] > 0.3, 1.0, 0.0) * 0.5
        if 'returns' in fi: s_osr += np.where(z3[d, :, fi['returns']] < -0.5, 1.0, 0.0) * 0.3
        osr_sig[d] = np.nan_to_num(s_osr, nan=-1e10)

    # ── 择时信号 ──
    iv = fi.get('volatility_20')
    vol_p = np.ones(nd, dtype=np.float32)
    if iv is not None:
        vol_p = np.clip(1.0 - np.mean(z3[:, :, iv] > 0.05, axis=1), 0.2, 1.0)

    im, ims, im5, im20, ir = fi.get('macd'), fi.get('macd_signal'), fi.get('momentum_5'), fi.get('momentum_20'), fi.get('rsi_14')
    trend_p = np.full(nd, 0.5, dtype=np.float32)
    for d in range(nd):
        sl = []
        if im is not None and ims is not None:
            sl.append(np.where(z3[d, :, im] > z3[d, :, ims], 1.0, 0.0))
        if im5 is not None and im20 is not None:
            m5v, m20v = z3[d, :, im5], z3[d, :, im20]
            sl.append(np.where((m5v > 0) & (m5v > m20v), 1.0, np.where(m5v < 0, 0.0, 0.5)))
        if ir is not None:
            rv = z3[d, :, ir]
            sl.append(np.where(rv > 70, 0.0, np.where(rv >= 50, 1.0, np.where(rv >= 30, 0.5, 0.0))))
        if sl:
            trend_p[d] = np.clip(np.mean(np.mean(sl, axis=0) >= 0.6) * 2.0, 0.1, 1.0)

    logger.info("=" * 70)
    logger.info("开始15策略统一调优")
    logger.info("=" * 70)

    # ════════════════════════════════════════════════════════════
    # 1. mf_vol_d10_rp v2
    # ════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("策略1: mf_vol_d10_rp v2 — vol_lookback/min_hold_days/high_threshold扫描")
    logger.info("=" * 60)
    for vl in [10, 20, 40, 60]:
        for mhd in [5, 10, 15]:
            for ht in [0.25, 0.30, 0.35]:
                label = f"tune_mfvol_v2_vl{vl}_{mhd}mhd{ht}"
                logger.info(f"\n--- {label} ---")
                # VolTiming: high_threshold对应vol_p的阈值
                # vol_p已经是每日仓位系数，ht控制高波动阈值
                # 这里ht用作vol_p的重新阈值化
                pr = np.clip(1.0 - np.mean(z3[:, :, iv] > ht, axis=1), 0.2, 1.0) if iv is not None else None
                r = run_bt_triple(mf, fwd, dm, ds, label, rf=10, tn=50, pos_ratio=pr, mhd=mhd)
                record_result(all_res, label, r)

    # ════════════════════════════════════════════════════════════
    # 2. mf_d10_rp v2
    # ════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("策略2: mf_d10_rp v2 — top_n/min_hold_days/频率扫描")
    logger.info("=" * 60)
    for tn in [30, 40, 50, 60]:
        for mhd in [5, 10, 15]:
            for rf in [8, 10, 12]:
                label = f"tune_mf_v2_tn{tn}_mhd{mhd}_rf{rf}"
                logger.info(f"\n--- {label} ---")
                r = run_bt_triple(mf, fwd, dm, ds, label, rf=rf, tn=tn, mhd=mhd)
                record_result(all_res, label, r)

    # ════════════════════════════════════════════════════════════
    # 3. chip_covrp v2
    # ════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("策略3: chip_covrp v2 — cov_window扫描")
    logger.info("=" * 60)
    for cl in [10, 20, 40, 60]:
        label = f"tune_chipcovrp_v2_cl{cl}"
        logger.info(f"\n--- {label} ---")
        r = run_bt_triple(chip_sig, fwd, dm, ds, label, rf=3, tn=40, cov_mode=True, cov_lb=cl)
        record_result(all_res, label, r)

    # ════════════════════════════════════════════════════════════
    # 4. chip_equal_vol_d3 (新策略)
    # ════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("策略4: chip_equal_vol_d3 — ht扫描")
    logger.info("=" * 60)
    for ht in [0.20, 0.25, 0.30]:
        label = f"tune_chip_equal_vol_ht{ht}"
        logger.info(f"\n--- {label} ---")
        pr = np.clip(1.0 - np.mean(z3[:, :, iv] > ht, axis=1), 0.2, 1.0) if iv is not None else None
        r = run_bt_triple(chip_sig, fwd, dm, ds, label, rf=3, tn=40, pos_ratio=pr, mhd=5)
        record_result(all_res, label, r)

    # ════════════════════════════════════════════════════════════
    # 5. mf_dynamic_combo (新策略)
    # ════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("策略5: mf_dynamic_combo — 动态牛熊权重")
    logger.info("=" * 60)
    label = "tune_dynamic_combo"
    logger.info(f"\n--- {label} ---")
    dr_mf = bt_series(mf, fwd, dm, rf=10, tn=50, mhd=10)
    dr_chip = bt_series(chip_sig, fwd, dm, rf=3, tn=40, mhd=5)
    dr_dyn = np.zeros(nd, dtype=np.float64)
    for i in range(nd):
        if trend_p[i] > 0.5:
            dr_dyn[i] = dr_mf[i] * 0.7 + dr_chip[i] * 0.3
        else:
            dr_dyn[i] = dr_mf[i] * 0.3 + dr_chip[i] * 0.7
    dr_dict = {
        "full": dr_dyn,
        "bear": dr_dyn[np.where((ds >= BEAR_START) & (ds <= BEAR_END))[0]],
        "oos": dr_dyn[np.where((ds >= OOS_START) & (ds <= OOS_END))[0]],
    }
    record_from_dr_dict(all_res, label, dr_dict)

    # ════════════════════════════════════════════════════════════
    # 6. v1_ga_rp + 涨跌停过滤
    # ════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("策略6: v1_ga_rp + 涨跌停过滤")
    logger.info("=" * 60)
    label = "tune_v1_plus_limit"
    logger.info(f"\n--- {label} ---")
    r = run_bt_triple(mf, fwd, dm, ds, label, rf=3, tn=40, mhd=5, use_limit_filter=True)
    record_result(all_res, label, r)

    # ════════════════════════════════════════════════════════════
    # 7. osr_d10 + MA200过滤 (新策略)
    # ════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("策略7: osr_d10_ma200 — MA200市场过滤")
    logger.info("=" * 60)
    # 计算全市场平均close的MA200
    cl_all = np.zeros((nd, ns), dtype=np.float32)
    # 从z3重建close(近似) — 实际是用fwd累乘得到价格
    close_approx = np.zeros((nd, ns), dtype=np.float32)
    close_approx[0] = 1.0
    for d in range(1, nd):
        close_approx[d] = close_approx[d-1] * (1.0 + fwd[d])
    close_avg = np.nanmean(close_approx, axis=1)
    # MA200
    ma200 = np.zeros(nd, dtype=np.float32)
    for d in range(nd):
        if d >= 199:
            ma200[d] = np.mean(close_avg[d-199:d+1])
        else:
            ma200[d] = close_avg[d]
    # 仓位系数: close_avg > ma200 = 1.0, else 0.5
    ma200_pr = np.where(close_avg > ma200, 1.0, 0.5).astype(np.float32)

    label = "tune_osr_ma200"
    logger.info(f"\n--- {label} ---")
    r = run_bt_triple(osr_sig, fwd, dm, ds, label, rf=10, tn=40, pos_ratio=ma200_pr, mhd=5)
    record_result(all_res, label, r)

    # ════════════════════════════════════════════════════════════
    # 8. chip_rp v2: 频率扫描
    # ════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("策略8: chip_rp v2 — 频率扫描")
    logger.info("=" * 60)
    for rf in [2, 3, 5, 7, 10]:
        label = f"tune_chip_rp_rf{rf}"
        logger.info(f"\n--- {label} ---")
        r = run_bt_triple(chip_sig, fwd, dm, ds, label, rf=rf, tn=40, mhd=5)
        record_result(all_res, label, r)

    # ════════════════════════════════════════════════════════════
    # 9. chip_vol_rp v2: ht扫描
    # ════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("策略9: chip_vol_rp v2 — ht扫描")
    logger.info("=" * 60)
    for ht in [0.15, 0.20, 0.25, 0.30, 0.35]:
        label = f"tune_chipvol_ht{ht}"
        logger.info(f"\n--- {label} ---")
        pr = np.clip(1.0 - np.mean(z3[:, :, iv] > ht, axis=1), 0.2, 1.0) if iv is not None else None
        r = run_bt_triple(chip_sig, fwd, dm, ds, label, rf=3, tn=40, pos_ratio=pr, mhd=5)
        record_result(all_res, label, r)

    # ════════════════════════════════════════════════════════════
    # 10. mf_trend_d5_rp v2
    # ════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("策略10: mf_trend_d5_rp v2 — buy_threshold扫描")
    logger.info("=" * 60)
    for bt_val in [0.5, 0.6, 0.7, 0.8]:
        label = f"tune_mftrend_bt{bt_val}"
        logger.info(f"\n--- {label} ---")
        # 重新计算trend_p，使用bt_val作为阈值
        tp = np.full(nd, 0.5, dtype=np.float32)
        for d in range(nd):
            sl = []
            if im is not None and ims is not None:
                sl.append(np.where(z3[d, :, im] > z3[d, :, ims], 1.0, 0.0))
            if im5 is not None and im20 is not None:
                m5v, m20v = z3[d, :, im5], z3[d, :, im20]
                sl.append(np.where((m5v > 0) & (m5v > m20v), 1.0, np.where(m5v < 0, 0.0, 0.5)))
            if ir is not None:
                rv = z3[d, :, ir]
                sl.append(np.where(rv > 70, 0.0, np.where(rv >= 50, 1.0, np.where(rv >= 30, 0.5, 0.0))))
            if sl:
                tp[d] = np.clip(np.mean(np.mean(sl, axis=0) >= bt_val) * 2.0, 0.1, 1.0)
        r = run_bt_triple(mf, fwd, dm, ds, label, rf=5, tn=40, pos_ratio=tp, mhd=5)
        record_result(all_res, label, r)

    # ════════════════════════════════════════════════════════════
    # 11. mf60_chip40_combo v2: 权重扫描
    # ════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("策略11: mf60_chip40_combo v2 — 权重扫描")
    logger.info("=" * 60)
    dr_mf_d10 = bt_series(mf, fwd, dm, rf=10, tn=50, mhd=10)
    dr_chip_d3 = bt_series(chip_sig, fwd, dm, rf=3, tn=40, mhd=5)
    for w_mf in [0.3, 0.5, 0.6, 0.7, 0.8]:
        label = f"tune_mfchip_weight{w_mf}"
        w_chip = round(1.0 - w_mf, 1)
        logger.info(f"\n--- {label} (MF={w_mf}, Chip={w_chip}) ---")
        dr_c = dr_mf_d10 * w_mf + dr_chip_d3 * w_chip
        bear_idx = np.where((ds >= BEAR_START) & (ds <= BEAR_END))[0]
        oos_idx = np.where((ds >= OOS_START) & (ds <= OOS_END))[0]
        dr_dict_c = {
            "full": dr_c,
            "bear": dr_c[bear_idx],
            "oos": dr_c[oos_idx],
        }
        record_from_dr_dict(all_res, label, dr_dict_c)

    # ════════════════════════════════════════════════════════════
    # 12. mf50_chip50_combo + Vol (新策略)
    # ════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("策略12: mf50_chip50_combo + Vol — VolTiming作用在总仓位")
    logger.info("=" * 60)
    label = "tune_mf50chip50_vol"
    logger.info(f"\n--- {label} ---")
    dr_combo_50 = dr_mf_d10 * 0.5 + dr_chip_d3 * 0.5
    dr_combo_vol = dr_combo_50 * vol_p
    bear_idx = np.where((ds >= BEAR_START) & (ds <= BEAR_END))[0]
    oos_idx = np.where((ds >= OOS_START) & (ds <= OOS_END))[0]
    dr_dict_12 = {
        "full": dr_combo_vol,
        "bear": dr_combo_vol[bear_idx],
        "oos": dr_combo_vol[oos_idx],
    }
    record_from_dr_dict(all_res, label, dr_dict_12)

    # ════════════════════════════════════════════════════════════
    # 13. c01_layered_d5 v2: 择时阈值扫描
    # ════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("策略13: c01_layered_d5 v2 — 择时阈值扫描")
    logger.info("=" * 60)
    for bt_val in [0.5, 0.6, 0.7]:
        label = f"tune_c01_bt{bt_val}"
        logger.info(f"\n--- {label} ---")
        # c01: MF信号被择时过滤
        c01_sig = mf.copy()
        for d in range(nd):
            sl = []
            if im is not None and ims is not None:
                sl.append(np.where(z3[d, :, im] > z3[d, :, ims], 1.0, 0.0))
            if im5 is not None and im20 is not None:
                m5v, m20v = z3[d, :, im5], z3[d, :, im20]
                sl.append(np.where((m5v > 0) & (m5v > m20v), 1.0, np.where(m5v < 0, 0.0, 0.5)))
            if ir is not None:
                rv = z3[d, :, ir]
                sl.append(np.where(rv > 70, 0.0, np.where(rv >= 50, 1.0, np.where(rv >= 30, 0.5, 0.0))))
            ts = np.mean(sl, axis=0) if sl else np.full(ns, 0.5)
            c01_sig[d] = np.where(ts >= bt_val, mf[d], np.full(ns, -np.inf))
        r = run_bt_triple(c01_sig, fwd, dm, ds, label, rf=5, tn=40, mhd=5)
        record_result(all_res, label, r)

    # ════════════════════════════════════════════════════════════
    # 14. OSR + 择时组合
    # ════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("策略14: OSR + 择时组合 — 频率×择时类型")
    logger.info("=" * 60)
    timing_types = {
        "vol": vol_p,
        "trend": trend_p,
    }
    for rf_osr in [5, 10, 15]:
        for timing_name, pr_signal in timing_types.items():
            label = f"tune_osr_freq{rf_osr}_timing{timing_name}"
            logger.info(f"\n--- {label} ---")
            r = run_bt_triple(osr_sig, fwd, dm, ds, label, rf=rf_osr, tn=40, pos_ratio=pr_signal, mhd=5)
            record_result(all_res, label, r)

    # ════════════════════════════════════════════════════════════
    # 15. Chip系列复合 (新策略)
    # ════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("策略15: Chip系列复合 — chip_covrp(40%)+chip_equal(40%)+chip_vol(20%)")
    logger.info("=" * 60)
    label = "tune_chip_combo"
    logger.info(f"\n--- {label} ---")
    # chip_covrp 40%
    dr_cc = bt_series(chip_sig, fwd, dm, rf=3, tn=40, mhd=5, pos_ratio=None)
    # chip_equal 40% (等权 = tn=40, 但用RP分配器... 等权需要另外实现)
    # 手写等权分配
    dr_ce = np.zeros(nd, dtype=np.float64)
    pw_eq = np.zeros(ns, dtype=np.float32); hs_eq = np.full(ns, -1, dtype=np.int32); rh_eq = 0
    for i in range(1, nd):
        rebal = (i % 3 == 0)
        if rebal:
            si_eq = np.argsort(-chip_sig[i])[:40]
            nw_eq = np.zeros(ns, dtype=np.float32)
            nw_eq[si_eq] = 1.0 / len(si_eq)
            pw_eq = nw_eq
            for j in range(ns):
                if nw_eq[j] > 0 and hs_eq[j] < 0: hs_eq[j] = rh_eq + 1
        else:
            mk = dm[i] & (pw_eq > 0)
            if np.any(mk):
                p2 = pw_eq[mk].copy() / float(np.sum(pw_eq[mk]))
                pw_eq = np.zeros(ns, dtype=np.float32); pw_eq[mk] = p2
        rt = float(np.dot(pw_eq, fwd[i]))
        dr_ce[i] = 0.0 if (np.isnan(rt) or np.isinf(rt)) else rt; rh_eq += 1
    # chip_vol 20%
    dr_cv = bt_series(chip_sig, fwd, dm, rf=3, tn=40, mhd=5, pos_ratio=vol_p)
    dr_chip_combo = dr_cc * 0.4 + dr_ce * 0.4 + dr_cv * 0.2
    bear_idx = np.where((ds >= BEAR_START) & (ds <= BEAR_END))[0]
    oos_idx = np.where((ds >= OOS_START) & (ds <= OOS_END))[0]
    dr_dict_15 = {
        "full": dr_chip_combo,
        "bear": dr_chip_combo[bear_idx],
        "oos": dr_chip_combo[oos_idx],
    }
    record_from_dr_dict(all_res, label, dr_dict_15)

    # ════════════════════════════════════════════════════════════
    # 保存结果
    # ════════════════════════════════════════════════════════════
    out_path = os.path.join(OUT_DIR, "results.json")
    with open(out_path, 'w') as f:
        json.dump(all_res, f, indent=2, ensure_ascii=False)
    logger.info(f"\n结果已保存至: {out_path}")
    logger.info(f"共 {len(all_res)} 条记录")

    # ════════════════════════════════════════════════════════════
    # 输出最佳Top20
    # ════════════════════════════════════════════════════════════
    # 筛选全区间+熊市+OOS都有记录的实验
    full_results = [r for r in all_res if r.get("period") == "full"]
    bear_results = {r["name"].replace("_bear", "").replace("_full", ""): r
                    for r in all_res if r.get("period") == "bear"}
    oos_results = {r["name"].replace("_oos", "").replace("_full", ""): r
                   for r in all_res if r.get("period") == "oos"}

    # 综合评分: 熊市Sharpe × 0.4 + OOS Sharpe × 0.3 + 全区间Sharpe × 0.3
    scored = []
    for r in full_results:
        base_name = r["name"].replace("_full", "")
        b_r = bear_results.get(base_name)
        o_r = oos_results.get(base_name)
        if b_r is not None and o_r is not None:
            score = b_r["sharpe"] * 0.4 + o_r["sharpe"] * 0.3 + r["sharpe"] * 0.3
            scored.append({
                "name": base_name,
                "score": round(score, 4),
                "full_sharpe": r["sharpe"],
                "full_return": r["annual_return"],
                "full_mdd": r["max_drawdown"],
                "bear_sharpe": b_r["sharpe"],
                "bear_return": b_r["annual_return"],
                "bear_mdd": b_r["max_drawdown"],
                "oos_sharpe": o_r["sharpe"],
                "oos_return": o_r["annual_return"],
                "oos_mdd": o_r["max_drawdown"],
            })

    scored.sort(key=lambda x: x["score"], reverse=True)

    print(f"\n{'=' * 120}")
    print(f"{'🏆 综合评分Top20 (熊市Sharpe×0.4 + OOS Sharpe×0.3 + 全区间Sharpe×0.3)':^120}")
    print(f"{'=' * 120}")
    print(f"{'排名':<4} {'策略':<30} {'综合分':<8} {'全Sharpe':<9} {'全年化%':<8} {'全回撤%':<8} "
          f"{'熊Sharpe':<9} {'熊年化%':<8} {'熊回撤%':<8} "
          f"{'OOS Sharpe':<10} {'OOS年化%':<9} {'OOS回撤%':<8}")
    print('-' * 120)
    for i, s in enumerate(scored[:20]):
        print(f"{i+1:<4} {s['name']:<30} {s['score']:<8.3f} "
              f"{s['full_sharpe']:<9.3f} {s['full_return']*100:<8.2f} {s['full_mdd']*100:<8.2f} "
              f"{s['bear_sharpe']:<9.3f} {s['bear_return']*100:<8.2f} {s['bear_mdd']*100:<8.2f} "
              f"{s['oos_sharpe']:<10.3f} {s['oos_return']*100:<9.2f} {s['oos_mdd']*100:<8.2f}")
    print(f"{'=' * 120}")

    # 也输出熊市Sharpe排名
    print(f"\n{'=' * 100}")
    print(f"{'🐻 熊市Sharpe排名Top20':^100}")
    print(f"{'=' * 100}")
    bear_sorted = sorted(scored, key=lambda x: x["bear_sharpe"], reverse=True)
    print(f"{'排名':<4} {'策略':<30} {'熊Sharpe':<9} {'熊年化%':<8} {'熊回撤%':<8} "
          f"{'全Sharpe':<9} {'OOS Sharpe':<10}")
    print('-' * 100)
    for i, s in enumerate(bear_sorted[:20]):
        print(f"{i+1:<4} {s['name']:<30} {s['bear_sharpe']:<9.3f} {s['bear_return']*100:<8.2f} "
              f"{s['bear_mdd']*100:<8.2f} {s['full_sharpe']:<9.3f} {s['oos_sharpe']:<10.3f}")
    print(f"{'=' * 100}")

    logger.info("\n调优完成!")


if __name__ == "__main__":
    main()
