"""因子翻转对全策略影响评估 — 量化A4翻转方案对每个已沉淀策略及实盘MSS混合策略的影响

用法:
    python3 daily/2026-06-15/strategy_impact.py
    python3 daily/2026-06-15/strategy_impact.py --start 2026-01-01 --end 2026-06-11
    python3 daily/2026-06-15/strategy_impact.py --oos-only   # 仅OOS期(2026-04-01起)

输出:
    1. results.tsv — 每个策略原始 vs 翻转的回测指标对比
    2. strategy_impact.json — 详细结果JSON
    3. stdout最后一行 — MSS混合策略加权Calmar变化百分比
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np, pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import duckdb
from core.positioners.impl.rp_weights import RPPortfolioWeights

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.abspath("./data/quant_data.db")
TX = 0.0012

# ── Reversal/Dead因子(6-12消融实验确认) ──
REVERSAL_FACTORS = {"gtja144", "a41", "gtja117", "gtja49", "gtja13", "gtja34", "a69", "gtja113"}
DEAD_FACTORS = {"gtja142", "gtja141"}

# ── 策略注册表 ──
# name: (config_path, signal_type, rebal_freq, top_n, stop_loss)
STRATEGY_REGISTRY = {
    # MF策略(共享权重，受翻转影响)
    "mf_d10_rp":       ("core/strategies/impl/mf_d10_rp/config.json",       "mf",       5, 10, 0.06),
    "mf_vol_d10_rp":   ("core/strategies/impl/mf_vol_d10_rp/config.json",   "mf_vol",   5,  8, 0.06),
    "c01_layered_d5":  ("core/strategies/impl/c01_layered_d5/config.json",   "mf_trend", 5,  6, 0.06),
    "mf50_chip50":     ("core/strategies/impl/mf50_chip50_combo/config.json","combo_50", 5,  8, 0.06),
    "mf_trend_d5_rp":  ("core/strategies/impl/mf_trend_d5_rp/config.json",  "mf_trend", 5,  6, 0.06),
    "v1_ga_rp":        ("core/strategies/impl/v1_ga_rp/config.json",         "mf",       3, 40, 0.06),
    # 非MF策略(不受翻转影响)
    "chip_equal_d3":   ("core/strategies/impl/chip_equal_d3/config.json",   "chip",     3,  6, 0.08),
    "osr_d10":         ("core/strategies/impl/osr_d10/config.json",         "osr",      10, 6, 0.06),
}

# ── V6A实盘分配 ──
V6A_ALLOCATION = {
    "bull":      [("mf_d10_rp", 0.6), ("mf_vol_d10_rp", 0.2), ("mf50_chip50", 0.15), ("osr_d10", 0.05)],
    "bear":      [("c01_layered_d5", 0.5), ("chip_equal_d3", 0.25), ("mf_vol_d10_rp", 0.25)],
    "oscillate": [("mf_d10_rp", 0.4), ("mf50_chip50", 0.3), ("c01_layered_d5", 0.3)],
    "recovery":  [("c01_layered_d5", 0.4), ("osr_d10", 0.3), ("mf_vol_d10_rp", 0.3)],
}


def load_data(start_date="2026-01-01", end_date="2026-06-11"):
    """加载因子和行情数据"""
    t0 = time.time()
    conn = duckdb.connect(DB_PATH, read_only=True)

    # 获取因子列
    all_cols = [r[0] for r in conn.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name='factors_wide'"
    ).fetchall()]

    # 收集所有策略用到的因子
    all_factors = set()
    for cfg_path, _, _, _, _ in STRATEGY_REGISTRY.values():
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
            w = cfg.get("selector", {}).get("weights", {})
            all_factors.update(w.keys())

    available = [c for c in sorted(all_factors) if c in all_cols]
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
    valid = (di >= 0) & (si >= 0)
    di, si = di[valid], si[valid]

    for fi, fc in enumerate(available):
        if fc in df.columns:
            v3[di, si, fi] = df[fc].values[valid].astype(np.float32)
    cl[di, si] = df['close'].values[valid].astype(np.float32)
    if 'pct_change' in df.columns:
        pct[di, si] = df['pct_change'].values[valid].astype(np.float32)
    dm[di, si] = True

    for a in [v3, cl, pct]:
        np.nan_to_num(a, nan=0.0, copy=False)

    # 前向收益
    fwd = np.zeros((nd, ns), dtype=np.float32)
    for d in range(nd - 1):
        b = (cl[d] > 1e-10) & (cl[d + 1] > 1e-10)
        fwd[d, b] = (cl[d + 1, b] - cl[d, b]) / cl[d, b]

    # 截面标准化
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
    print(f"[INFO] 数据加载: {nd}天×{ns}只×{nf}因子 ({time.time()-t0:.1f}s)", file=sys.stderr)
    return z3, fwd, dm, cl, tks, available, nd, ns, ds


def build_signal(z3, fnames, weights_dict, signal_type="mf"):
    """构建策略信号"""
    wv = np.zeros(len(fnames), dtype=np.float32)
    for fi, fc in enumerate(fnames):
        if fc in weights_dict:
            wv[fi] = float(weights_dict[fc])
    s = np.sum(np.abs(wv))
    if s > 0:
        wv /= s
    mf = np.nan_to_num(np.tensordot(z3, wv, axes=(2, 0)), nan=-1e10, neginf=-1e10)
    return mf


def flip_weights(weights_dict):
    """翻转reversal因子权重 + dead因子降为0"""
    w = dict(weights_dict)
    for f in REVERSAL_FACTORS:
        if f in w:
            w[f] = -w[f]
    for f in DEAD_FACTORS:
        if f in w:
            w[f] = 0.0
    return w


def backtest(sig, fwd, dm, cl, nd, ns, rebal_freq=5, top_n=10,
             min_hold_days=5, stop_loss_pct=0.06, tx_cost=TX):
    """纯多因子策略回测"""
    alloc = RPPortfolioWeights(top_n=top_n, min_hold_days=min_hold_days)
    pw = np.zeros(ns, dtype=np.float32)
    hs = np.full(ns, -1, dtype=np.int32)
    rh = 0
    dr = np.zeros(nd, dtype=np.float64)
    entry_px = np.zeros(ns, dtype=np.float32)

    for i in range(1, nd):
        # 止损
        if stop_loss_pct > 0 and np.any(pw > 0):
            for j in range(ns):
                if pw[j] > 0 and hs[j] >= 0 and entry_px[j] > 0:
                    if fwd[i, j] < -stop_loss_pct and fwd[i, j] > -0.95:
                        pw[j] = 0.0
                        hs[j] = -1
                        entry_px[j] = 0.0
        # 换仓
        rebal = (i % rebal_freq == 0)
        if rebal:
            nw = alloc.allocate(sig[i], fwd, i, pw, hs, rh)
            for j in range(ns):
                if nw[j] > 0 and entry_px[j] <= 0:
                    entry_px[j] = max(1.0, 1.0 + fwd[i, j])
            to = float(np.sum(np.abs(nw - pw)))
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
            to = 0.0
        for j in range(ns):
            if pw[j] > 0 and entry_px[j] <= 0:
                entry_px[j] = max(1.0, 1.0 + fwd[i, j])
        rt = float(np.dot(pw, fwd[i])) - 0.5 * to * tx_cost
        dr[i] = 0.0 if (np.isnan(rt) or np.isinf(rt)) else rt
        rh += 1
    return dr


def compute_metrics(dr):
    """计算回测指标"""
    nd = len(dr)
    eq = np.ones(nd, dtype=np.float64)
    for i in range(1, nd):
        eq[i] = eq[i - 1] * (1.0 + dr[i])
    total_ret = float(eq[-1] / eq[0] - 1.0)
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
        "annual_return": round(float(ar), 4),
        "sharpe": round(float(sp), 4),
        "max_drawdown": round(float(mdd), 4),
        "calmar": round(float(cal), 4),
        "win_rate": round(float(wr), 4),
    }


def main():
    parser = argparse.ArgumentParser(description="因子翻转对全策略影响评估")
    parser.add_argument("--start", default="2026-04-01", help="回测起始日期")
    parser.add_argument("--end", default="2026-06-11", help="回测结束日期")
    parser.add_argument("--oos-only", action="store_true", help="仅OOS期")
    args = parser.parse_args()

    start_date = args.start
    end_date = args.end

    # ── 1. 加载数据 ──
    data_start = pd.Timestamp(start_date) - pd.Timedelta(days=30)
    z3, fwd, dm, cl, tks, fnames, nd, ns, ds = load_data(
        start_date=data_start.strftime("%Y-%m-%d"),
        end_date=end_date
    )
    bt_start = pd.Timestamp(start_date)
    start_idx = next((i for i, d in enumerate(ds) if d >= bt_start), 0)

    # ── 2. 遍历策略 ──
    results = {}
    tsv_lines = ["策略\t类型\t变体\tCalmar\tSharpe\tMDD%\tAR%\tWinRate\tCalmar变化"]

    for name, (cfg_path, sig_type, rebal, topn, sl) in STRATEGY_REGISTRY.items():
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"[{name}] {sig_type} | rebal={rebal} topn={topn} sl={sl}", file=sys.stderr)

        # 判断是否MF策略
        is_mf = sig_type in ("mf", "mf_vol", "mf_trend", "combo_50")

        # 加载权重
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
            orig_weights = cfg.get("selector", {}).get("weights", {})
        else:
            print(f"  [WARN] 配置不存在: {cfg_path}", file=sys.stderr)
            orig_weights = {}

        if not orig_weights:
            print(f"  [SKIP] 无权重", file=sys.stderr)
            continue

        # ── 原始权重回测 ──
        sig_orig = build_signal(z3, fnames, orig_weights, sig_type)
        dr_orig = backtest(sig_orig[start_idx:], fwd[start_idx:], dm[start_idx:],
                           cl[start_idx:], nd - start_idx, ns,
                           rebal_freq=rebal, top_n=topn, stop_loss_pct=sl)
        m_orig = compute_metrics(dr_orig)
        print(f"  原始: Calmar={m_orig['calmar']:.4f} Sharpe={m_orig['sharpe']:.4f} "
              f"MDD={m_orig['max_drawdown']:.4f} AR={m_orig['annual_return']:.4f}", file=sys.stderr)

        # ── 翻转权重回测(MF策略) ──
        m_flip = None
        if is_mf:
            flip_w = flip_weights(orig_weights)
            sig_flip = build_signal(z3, fnames, flip_w, sig_type)
            dr_flip = backtest(sig_flip[start_idx:], fwd[start_idx:], dm[start_idx:],
                               cl[start_idx:], nd - start_idx, ns,
                               rebal_freq=rebal, top_n=topn, stop_loss_pct=sl)
            m_flip = compute_metrics(dr_flip)
            calmar_delta = m_flip['calmar'] - m_orig['calmar']
            print(f"  翻转: Calmar={m_flip['calmar']:.4f} Sharpe={m_flip['sharpe']:.4f} "
                  f"MDD={m_flip['max_drawdown']:.4f} AR={m_flip['annual_return']:.4f} "
                  f"(ΔCalmar={calmar_delta:+.4f})", file=sys.stderr)
        else:
            calmar_delta = 0.0
            print(f"  非MF策略，不受翻转影响", file=sys.stderr)

        # 记录结果
        results[name] = {
            "signal_type": sig_type,
            "is_mf": is_mf,
            "rebal_freq": rebal,
            "top_n": topn,
            "stop_loss": sl,
            "original": m_orig,
            "flipped": m_flip,
            "calmar_delta": round(float(calmar_delta), 4) if is_mf else 0.0,
        }

        # TSV行
        tsv_lines.append(f"{name}\t{sig_type}\t原始\t{m_orig['calmar']}\t{m_orig['sharpe']}\t"
                         f"{abs(m_orig['max_drawdown'])*100:.2f}\t{m_orig['annual_return']*100:.2f}\t"
                         f"{m_orig['win_rate']*100:.1f}\t---")
        if m_flip:
            tsv_lines.append(f"{name}\t{sig_type}\t翻转\t{m_flip['calmar']}\t{m_flip['sharpe']}\t"
                             f"{abs(m_flip['max_drawdown'])*100:.2f}\t{m_flip['annual_return']*100:.2f}\t"
                             f"{m_flip['win_rate']*100:.1f}\t{calmar_delta:+.4f}")

    # ── 3. MSS混合策略加权影响 ──
    print(f"\n{'='*60}", file=sys.stderr)
    print("[MSS混合策略加权影响分析]", file=sys.stderr)

    mss_impact = {}
    for state, allocs in V6A_ALLOCATION.items():
        weighted_calmar_orig = 0.0
        weighted_calmar_flip = 0.0
        mf_weight_total = 0.0

        for sname, weight in allocs:
            if sname in results:
                r = results[sname]
                c_orig = r["original"]["calmar"]
                c_flip = r["flipped"]["calmar"] if r["flipped"] else c_orig
                weighted_calmar_orig += weight * c_orig
                weighted_calmar_flip += weight * c_flip
                if r["is_mf"]:
                    mf_weight_total += weight

        delta = weighted_calmar_flip - weighted_calmar_orig
        pct_change = (delta / abs(weighted_calmar_orig) * 100) if abs(weighted_calmar_orig) > 0.001 else 0

        mss_impact[state] = {
            "weighted_calmar_orig": round(float(weighted_calmar_orig), 4),
            "weighted_calmar_flip": round(float(weighted_calmar_flip), 4),
            "calmar_delta": round(float(delta), 4),
            "pct_change": round(float(pct_change), 2),
            "mf_weight_pct": round(float(mf_weight_total * 100), 1),
        }
        print(f"  {state}: Calmar {weighted_calmar_orig:.4f}→{weighted_calmar_flip:.4f} "
              f"(Δ={delta:+.4f}, {pct_change:+.1f}%) MF占比={mf_weight_total*100:.0f}%",
              file=sys.stderr)

        tsv_lines.append(f"MSS_{state}\t混合\t原始\t{weighted_calmar_orig:.4f}\t---\t---\t---\t---\t---")
        tsv_lines.append(f"MSS_{state}\t混合\t翻转\t{weighted_calmar_flip:.4f}\t---\t---\t---\t---\t{delta:+.4f}")

    # ── 4. 输出结果 ──
    # TSV
    tsv_path = os.path.join(SCRIPT_DIR, "results.tsv")
    with open(tsv_path, "w") as f:
        f.write("\n".join(tsv_lines) + "\n")
    print(f"\n[INFO] TSV已写入 {tsv_path}", file=sys.stderr)

    # JSON
    json_path = os.path.join(SCRIPT_DIR, "strategy_impact.json")
    with open(json_path, "w") as f:
        json.dump({
            "meta": {
                "start_date": start_date,
                "end_date": end_date,
                "reversal_factors": sorted(REVERSAL_FACTORS),
                "dead_factors": sorted(DEAD_FACTORS),
            },
            "strategies": results,
            "mss_impact": mss_impact,
        }, f, indent=2, ensure_ascii=False)
    print(f"[INFO] JSON已写入 {json_path}", file=sys.stderr)

    # stdout最后一行: MSS加权Calmar平均变化百分比
    avg_pct = np.mean([v["pct_change"] for v in mss_impact.values()])
    print(f"{avg_pct:.2f}")


if __name__ == "__main__":
    main()