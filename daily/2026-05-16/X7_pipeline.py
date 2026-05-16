"""
X7 — ga_d10 + chip_covrp 组合沉淀 & ga_d10 参数微调

两大任务:
  任务1: ga_d10 + chip_covrp 组合扫描, 找最佳资金分配权重
  任务2: ga_d10 参数微调 (top_n, min_hold_days, vol_lookback, rf)

依赖: x5_results.json (GA rf=10 最优权重, 66因子)

说明:
  - ga_d10: GA信号 + RP仓位分配器 (标准bt)
  - chip_covrp: 低波动+低动量信号 + covRP仓位分配器 (协方差风险平价)
"""
import os, sys, json, logging, gc
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from core.database import Database
from core.positioners import RPPortfolioWeights

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("x7")

TX = 0.0012

OUT_DIR = os.path.join(os.path.dirname(__file__), "x7_results")
os.makedirs(OUT_DIR, exist_ok=True)

# === 66因子 (复用X5) ===
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
    'ma5','ma20','ma60','ma120','ma_alignment_score','ma60_trend','ma120_trend',
    'macd_above_zero','macd_golden_cross','ma_angle_20','volume_breakout_ratio',
    'volume_contraction','chip_concentration','ma_convergence','box_breakout','breakout_strength',
]
ALL_FACTORS = list(dict.fromkeys(_V1_BEST + _NEW_TECH))


# ============================================================
# 数据加载
# ============================================================
def load():
    db = Database()
    df = db.get_factors(start_date="2018-01-01", end_date="2026-04-30",
                        factor_names=ALL_FACTORS, with_close=True)
    df['date'] = pd.to_datetime(df['date'])
    ds = sorted(df['date'].unique())
    tks = db.get_symbols()['symbol'].tolist()
    nd, ns, nf = len(ds), len(tks), len(ALL_FACTORS)
    t2i = {t: i for i, t in enumerate(tks)}
    d2i = {d: i for i, d in enumerate(ds)}

    v3 = np.full((nd, ns, nf), np.nan, dtype=np.float32)
    dm = np.zeros((nd, ns), dtype=bool)
    cl = np.zeros((nd, ns), dtype=np.float32)
    di = np.array([d2i[d] for d in df['date']], dtype=np.int32)
    si = np.array([t2i.get(s, -1) for s in df['symbol']], dtype=np.int32)
    v = si >= 0; di, si = di[v], si[v]
    for fi, fc in enumerate(ALL_FACTORS):
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
                lo, hi = np.quantile(nz, [0.01, 0.99]); c = np.clip(r, lo, hi)
                mu, sd = np.mean(c), np.std(c)
                z3[d, :, fi] = (c - mu) / sd if sd > 1e-10 else 0.0

    fi_map = {fn: i for i, fn in enumerate(ALL_FACTORS)}
    logger.info(f"数据: {nd}天 × {ns}只 × {nf}因子")
    return z3, fwd, dm, tks, ALL_FACTORS, nd, ns, ds, fi_map


# ============================================================
# 标准RP回测函数 (同V6~V9风格, 扩展mhd和vol_lookback)
# ============================================================
def bt(sig, fwd, dm, name, rf=3, tn=40, mhd=5, vol_lb=20):
    nd, ns = sig.shape
    alloc = RPPortfolioWeights(top_n=tn, min_hold_days=mhd, vol_lookback=vol_lb)
    pw = np.zeros(ns, dtype=np.float32)
    hs = np.full(ns, -1, dtype=np.int32)
    rh = 0; dr = np.zeros(nd, dtype=np.float64)
    nt = 0
    for i in range(1, nd):
        rebal = (i % rf == 0)
        txc = 0.0
        if rebal:
            nw = alloc.allocate(sig[i], fwd, i, pw, hs, rh)
            to = float(np.sum(np.abs(nw - pw)))
            txc = 0.5 * to * TX
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
        dr[i] = 0.0 if (np.isnan(rt) or np.isinf(rt)) else rt - txc
        rh += 1

    eq = np.ones(nd, dtype=np.float64)
    for i in range(1, nd): eq[i] = eq[i-1] * (1.0 + dr[i])
    ny = nd/252.0
    ar = (float(eq[-1]/eq[0]))**(1.0/max(ny, 0.5)) - 1.0
    lr = np.log(eq[1:]/eq[:-1])
    sp = float(np.mean(lr)/max(np.std(lr), 1e-10)*np.sqrt(252))
    cm = np.maximum.accumulate(eq); dd = (eq-cm)/cm; mdd = float(np.min(dd))
    cal = ar/abs(mdd) if abs(mdd) > 0 else 0
    wr = int(np.sum(dr>0))/max(int(np.sum(dr>0))+int(np.sum(dr<0)), 1)
    logger.info(f"  [{name}] 年化={ar*100:.2f}% Sharpe={sp:.3f} 回撤={abs(mdd)*100:.2f}% Calmar={cal:.3f}")
    return {"name": name, "annual_return": round(ar,4), "sharpe": round(sp,4),
            "max_drawdown": round(mdd,4), "calmar": round(cal,4),
            "win_rate": round(wr,4), "n_trades": nt}


# ============================================================
# CovRP回测函数 (协方差风险平价, vol_lookback可配)
# ============================================================
def bt_covrp(sig, fwd, dm, name, rf=3, tn=40, vol_lb=20):
    nd, ns = sig.shape
    pw = np.zeros(ns, dtype=np.float32)
    hs = np.full(ns, -1, dtype=np.int32)
    rh = 0; dr = np.zeros(nd, dtype=np.float64)
    nt = 0
    for i in range(1, nd):
        rebal = (i % rf == 0)
        txc = 0.0
        if rebal:
            si = np.argsort(-sig[i])[:tn]
            nw = np.zeros(ns)
            if i >= vol_lb:
                seg = fwd[max(0, i - vol_lb):i, :]
                sub = seg[:, si]
                sub = sub[:, ~np.any(np.isnan(sub) | np.isinf(sub), axis=0)]
                if sub.shape[1] >= 2:
                    try:
                        cov = np.cov(sub.T)
                        iv = 1.0 / np.sqrt(np.diag(cov) + 1e-10)
                    except Exception:
                        iv = np.ones(sub.shape[1])
                else:
                    iv = np.ones(sub.shape[1])
            else:
                iv = np.ones(min(tn, ns))
            sidx = si[:len(iv)]
            if len(sidx) > 0:
                nw[sidx] = iv / np.sum(iv)

            to = float(np.sum(np.abs(nw - pw)))
            txc = 0.5 * to * TX
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
        dr[i] = 0.0 if (np.isnan(rt) or np.isinf(rt)) else rt - txc
        rh += 1

    eq = np.ones(nd, dtype=np.float64)
    for i in range(1, nd): eq[i] = eq[i-1] * (1.0 + dr[i])
    ny = nd/252.0
    ar = (float(eq[-1]/eq[0]))**(1.0/max(ny, 0.5)) - 1.0
    lr = np.log(eq[1:]/eq[:-1])
    sp = float(np.mean(lr)/max(np.std(lr), 1e-10)*np.sqrt(252))
    cm = np.maximum.accumulate(eq); dd = (eq-cm)/cm; mdd = float(np.min(dd))
    cal = ar/abs(mdd) if abs(mdd) > 0 else 0
    wr = int(np.sum(dr>0))/max(int(np.sum(dr>0))+int(np.sum(dr<0)), 1)
    logger.info(f"  [{name}] 年化={ar*100:.2f}% Sharpe={sp:.3f} 回撤={abs(mdd)*100:.2f}% Calmar={cal:.3f}")
    return {"name": name, "annual_return": round(ar,4), "sharpe": round(sp,4),
            "max_drawdown": round(mdd,4), "calmar": round(cal,4),
            "win_rate": round(wr,4), "n_trades": nt}


# ============================================================
# 获取GA最优权重 (rf=10)
# ============================================================
def load_ga_weights():
    p = os.path.join(os.path.dirname(__file__), "x4_x5_results", "x5_results.json")
    with open(p) as f:
        data = json.load(f)
    for r in data["ga_results"]:
        if r["rebal_freq"] == 10:
            return r["weights"]
    return data["ga_results"][0]["weights"]


# ============================================================
# 构建ga_d10信号 (GA权重合成)
# ============================================================
def build_ga_signal(z3, fi_map, ga_weights):
    nf = len(ALL_FACTORS)
    wv = np.zeros(nf, dtype=np.float32)
    for fn, fw in ga_weights.items():
        idx = fi_map.get(fn)
        if idx is not None:
            wv[idx] = float(fw)
    s = np.sum(np.abs(wv))
    if s > 0: wv /= s
    sig = np.nan_to_num(np.tensordot(z3, wv, axes=(2, 0)), nan=-1e10, neginf=-1e10)
    return sig


# ============================================================
# 构建chip_covrp信号 (低波动+低动量, 同V8)
# ============================================================
def build_chip_signal(z3, fi_map, nd, ns):
    vol20_idx = fi_map.get('volatility_20')
    m20_idx = fi_map.get('momentum_20')
    sig = np.full((nd, ns), -np.inf, dtype=np.float32)
    for d in range(nd):
        s = np.zeros(ns)
        if vol20_idx is not None:
            s += np.where(z3[d, :, vol20_idx] < -0.3, 1.0, 0.0) * 0.5
        if m20_idx is not None:
            s += np.where(np.abs(z3[d, :, m20_idx]) < 0.3, 1.0, 0.0) * 0.3
        sig[d] = np.nan_to_num(s, nan=-1e10)
    return sig


# ============================================================
# 任务2: ga_d10 参数微调
# ============================================================
def run_tuning(ga_sig, fwd, dm):
    logger.info("\n" + "="*70)
    logger.info("任务2: ga_d10 参数微调")
    logger.info("="*70)

    top_ns = [20, 30, 40, 50, 60]
    mhds = [1, 5, 10, 15]
    vol_lbs = [10, 20, 40, 60]
    rfs = [7, 8, 10, 12, 15]

    results = []
    total = len(top_ns) * len(mhds) * len(vol_lbs) * len(rfs)
    idx = 0
    for tn in top_ns:
        for mhd in mhds:
            for vl in vol_lbs:
                for rf in rfs:
                    idx += 1
                    label = f"ga_d10_tn{tn}_mhd{mhd}_vl{vl}_rf{rf}"
                    logger.info(f"[{idx}/{total}] {label}")
                    r = bt(ga_sig, fwd, dm, label, rf=rf, tn=tn, mhd=mhd, vol_lb=vl)
                    r["params"] = {"top_n": tn, "min_hold_days": mhd, "vol_lookback": vl, "rebal_freq": rf}
                    results.append(r)
                    gc.collect()

    logger.info(f"\n参数扫描完成: {len(results)}组")
    return results


# ============================================================
# 任务1: ga_d10 + chip_covrp 组合扫描
# ============================================================
def run_combo(ga_sig, chip_sig, fwd, dm):
    logger.info("\n" + "="*70)
    logger.info("任务1: ga_d10 + chip_covrp 组合扫描")
    logger.info("="*70)

    rf, tn, mhd, vl = 10, 40, 5, 20
    logger.info(f"基准配置: rf={rf} tn={tn} mhd={mhd} vol_lb={vl}")

    # ga_d10: RP仓位分配器
    r_ga = bt(ga_sig, fwd, dm, "ga_d10_单独", rf=rf, tn=tn, mhd=mhd, vol_lb=vl)

    # chip_covrp: covRP仓位分配器 (名称含covrp即使用协方差风险平价)
    r_chip = bt_covrp(chip_sig, fwd, dm, "chip_covrp_单独", rf=rf, tn=tn, vol_lb=vl)

    # 获取日收益率序列
    def get_dr_rp(sig, rf=rf, tn=tn, mhd=mhd, vl=vl):
        nd, ns = sig.shape
        alloc = RPPortfolioWeights(top_n=tn, min_hold_days=mhd, vol_lookback=vl)
        pw = np.zeros(ns, dtype=np.float32)
        hs = np.full(ns, -1, dtype=np.int32)
        rh = 0; dr = np.zeros(nd, dtype=np.float64)
        for i in range(1, nd):
            rebal = (i % rf == 0)
            if rebal:
                nw = alloc.allocate(sig[i], fwd, i, pw, hs, rh)
                pw = nw
                for j in range(ns):
                    if nw[j] > 0 and hs[j] < 0: hs[j] = rh + 1
            else:
                mk = dm[i] & (pw > 0)
                if np.any(mk):
                    p2 = pw[mk].copy() / float(np.sum(pw[mk]))
                    pw = np.zeros(ns, dtype=np.float32); pw[mk] = p2
            rt = float(np.dot(pw, fwd[i]))
            dr[i] = 0.0 if (np.isnan(rt) or np.isinf(rt)) else rt
            rh += 1
        return dr

    def get_dr_covrp(sig, rf=rf, tn=tn, vl=vl):
        nd, ns = sig.shape
        pw = np.zeros(ns, dtype=np.float32)
        hs = np.full(ns, -1, dtype=np.int32)
        rh = 0; dr = np.zeros(nd, dtype=np.float64)
        for i in range(1, nd):
            rebal = (i % rf == 0)
            if rebal:
                si = np.argsort(-sig[i])[:tn]
                nw = np.zeros(ns)
                if i >= vl:
                    seg = fwd[max(0, i - vl):i, :]
                    sub = seg[:, si]
                    sub = sub[:, ~np.any(np.isnan(sub) | np.isinf(sub), axis=0)]
                    if sub.shape[1] >= 2:
                        try:
                            cov = np.cov(sub.T)
                            iv = 1.0 / np.sqrt(np.diag(cov) + 1e-10)
                        except Exception:
                            iv = np.ones(sub.shape[1])
                    else:
                        iv = np.ones(sub.shape[1])
                else:
                    iv = np.ones(min(tn, ns))
                sidx = si[:len(iv)]
                if len(sidx) > 0:
                    nw[sidx] = iv / np.sum(iv)
                pw = nw
                for j in range(ns):
                    if nw[j] > 0 and hs[j] < 0: hs[j] = rh + 1
            else:
                mk = dm[i] & (pw > 0)
                if np.any(mk):
                    p2 = pw[mk].copy() / float(np.sum(pw[mk]))
                    pw = np.zeros(ns, dtype=np.float32); pw[mk] = p2
            rt = float(np.dot(pw, fwd[i]))
            dr[i] = 0.0 if (np.isnan(rt) or np.isinf(rt)) else rt
            rh += 1
        return dr

    dr_ga = get_dr_rp(ga_sig, rf, tn, mhd, vl)
    dr_chip = get_dr_covrp(chip_sig, rf, tn, vl)

    # 组合扫描: w_ga 0.1~0.9
    from core.strategies.pipeline import StrategyPipeline
    combos = []
    for w in [round(x * 0.1, 1) for x in range(1, 10)]:
        nd = len(dr_ga)
        dr = dr_ga * w + dr_chip * (1.0 - w)
        eq = np.ones(nd)
        for i in range(1, nd):
            eq[i] = eq[i-1] * (1.0 + dr[i])
        ny = nd/252.0
        ar = (float(eq[-1]/eq[0]))**(1.0/max(ny, 0.5)) - 1.0
        lr = np.log(eq[1:]/eq[:-1])
        sp = float(np.mean(lr)/max(np.std(lr), 1e-10)*np.sqrt(252))
        cm = np.maximum.accumulate(eq)
        dd = (eq-cm)/cm
        mdd = float(np.min(dd))
        cal = ar/abs(mdd) if abs(mdd) > 0 else 0
        wr = int(np.sum(dr>0))/max(int(np.sum(dr>0))+int(np.sum(dr<0)), 1)
        entry = {
            "w_ga": w, "w_chip": round(1.0 - w, 1),
            "annual_return": round(ar, 4), "sharpe": round(sp, 4),
            "max_drawdown": round(mdd, 4), "calmar": round(cal, 4),
            "win_rate": round(wr, 4),
        }
        combos.append(entry)
        has_qual = (abs(mdd) < 0.20 and ar > 0.20) or sp > 1.3
        qual = "✓" if has_qual else ""
        logger.info(f"  w_ga={w:.1f}: 年化={ar*100:.2f}% Sharpe={sp:.3f} 回撤={abs(mdd)*100:.2f}% {qual}")

    best_sharpe = max(combos, key=lambda x: x['sharpe'])
    qualified = [c for c in combos if (abs(c['max_drawdown']) < 0.20 and c['annual_return'] > 0.20) or c['sharpe'] > 1.3]
    best_qualified = max(qualified, key=lambda x: x['sharpe']) if qualified else best_sharpe

    return {
        "ga_solo": r_ga, "chip_solo": r_chip,
        "combos": combos, "best_sharpe": best_sharpe, "best_qualified": best_qualified,
    }


def main():
    logger.info("="*70)
    logger.info("X7: ga_d10 + chip_covrp 组合沉淀 & ga_d10 参数微调")
    logger.info("="*70)

    z3, fwd, dm, tks, fnames, nd, ns, ds, fi_map = load()
    ga_weights = load_ga_weights()
    logger.info(f"GA权重非零因子: {len(ga_weights)}个")

    ga_sig = build_ga_signal(z3, fi_map, ga_weights)
    chip_sig = build_chip_signal(z3, fi_map, nd, ns)
    logger.info("信号构建完成")

    tuning_results = run_tuning(ga_sig, fwd, dm)
    combo_results = run_combo(ga_sig, chip_sig, fwd, dm)

    # 保存结果
    output = {
        "data_info": {"nd": nd, "ns": ns, "n_factors": len(ALL_FACTORS)},
        "ga_weights_count": len(ga_weights),
        "tuning": {
            "param_grid": {
                "top_n": [20, 30, 40, 50, 60],
                "min_hold_days": [1, 5, 10, 15],
                "vol_lookback": [10, 20, 40, 60],
                "rebal_freq": [7, 8, 10, 12, 15],
            },
            "n_results": len(tuning_results),
            "best": max(tuning_results, key=lambda x: x['sharpe']),
            "best_calmar": max(tuning_results, key=lambda x: x['calmar']),
            "qualified": sorted(
                [r for r in tuning_results if abs(r['max_drawdown']) < 0.20 and r['annual_return'] > 0.20],
                key=lambda x: x['sharpe'], reverse=True,
            )[:10],
        },
        "combo": {
            "ga_solo": combo_results["ga_solo"],
            "chip_solo": combo_results["chip_solo"],
            "combos": combo_results["combos"],
            "best_sharpe": combo_results["best_sharpe"],
            "best_qualified": combo_results["best_qualified"],
        },
    }

    out_path = os.path.join(OUT_DIR, "results.json")
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"\n结果已保存至: {out_path}")

    # 汇总打印
    print(f"\n{'='*110}")
    print("【X7 结果汇总】")
    print(f"{'='*110}")

    t_best = output["tuning"]["best"]
    t_cal = output["tuning"]["best_calmar"]
    print(f"\n--- 任务2: ga_d10 参数微调 ---")
    print(f"  ★ 最高Sharpe: tn={t_best['params']['top_n']} mhd={t_best['params']['min_hold_days']} "
          f"vl={t_best['params']['vol_lookback']} rf={t_best['params']['rebal_freq']}")
    print(f"    年化={t_best['annual_return']*100:.2f}% Sharpe={t_best['sharpe']:.3f} "
          f"回撤={abs(t_best['max_drawdown'])*100:.2f}% Calmar={t_best['calmar']:.3f}")
    print(f"  ★ 最高Calmar: tn={t_cal['params']['top_n']} mhd={t_cal['params']['min_hold_days']} "
          f"vl={t_cal['params']['vol_lookback']} rf={t_cal['params']['rebal_freq']}")
    print(f"    年化={t_cal['annual_return']*100:.2f}% Sharpe={t_cal['sharpe']:.3f} "
          f"回撤={abs(t_cal['max_drawdown'])*100:.2f}% Calmar={t_cal['calmar']:.3f}")

    qual = output["tuning"]["qualified"]
    if qual:
        print(f"\n  达标前{len(qual)} (回撤<20% 年化>20%):")
        for r in qual[:5]:
            p = r['params']
            print(f"    tn={p['top_n']} mhd={p['min_hold_days']} vl={p['vol_lookback']} rf={p['rebal_freq']} | "
                  f"年化={r['annual_return']*100:.2f}% Sharpe={r['sharpe']:.3f} "
                  f"回撤={abs(r['max_drawdown'])*100:.2f}%")
    else:
        print(f"\n  (无参数组满足 回撤<20% 且 年化>20% 的严苛条件)")

    c_bs = combo_results["best_sharpe"]
    c_bq = combo_results["best_qualified"]
    print(f"\n--- 任务1: ga_d10 + chip_covrp 组合 ---")
    print(f"  ga_d10单独(RP):  年化={combo_results['ga_solo']['annual_return']*100:.2f}% "
          f"Sharpe={combo_results['ga_solo']['sharpe']:.3f} "
          f"回撤={abs(combo_results['ga_solo']['max_drawdown'])*100:.2f}%")
    print(f"  chip_covrp单独(covRP):  年化={combo_results['chip_solo']['annual_return']*100:.2f}% "
          f"Sharpe={combo_results['chip_solo']['sharpe']:.3f} "
          f"回撤={abs(combo_results['chip_solo']['max_drawdown'])*100:.2f}%")
    print(f"  最佳Sharpe: w_ga={c_bs['w_ga']:.1f} w_chip={c_bs['w_chip']:.1f} | "
          f"年化={c_bs['annual_return']*100:.2f}% Sharpe={c_bs['sharpe']:.3f} "
          f"回撤={abs(c_bs['max_drawdown'])*100:.2f}%")
    print(f"  最佳达标: w_ga={c_bq['w_ga']:.1f} w_chip={c_bq['w_chip']:.1f} | "
          f"年化={c_bq['annual_return']*100:.2f}% Sharpe={c_bq['sharpe']:.3f} "
          f"回撤={abs(c_bq['max_drawdown'])*100:.2f}%")

    print(f"\n{'='*110}")
    logger.info("X7 全部完成!")


if __name__ == "__main__":
    main()
