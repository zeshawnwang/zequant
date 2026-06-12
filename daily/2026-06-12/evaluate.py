"""因子权重评估脚本 — 输出Calmar比率到stdout最后一行

用法:
    python3 daily/2026-06-12/evaluate.py --weights daily/2026-06-12/weights.json
    python3 daily/2026-06-12/evaluate.py --weights daily/2026-06-12/weights.json --start 2026-04-01 --end 2026-06-11

输出: 最后一行为 Calmar比率 数值
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np, pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import duckdb
from core.positioners.impl.rp_weights import RPPortfolioWeights

DB_PATH = os.path.abspath("./data/quant_data.db")
TX = 0.0012

# 实盘因子列表(与config.json一致)
LIVE_FACTORS = [
    'ff_mkt','gtja142','gtja144','gtja171','gtja103','gtja85','a88','a31',
    'rsi_14','gtja139','gtja123','a42','a41','a97','gtja148','gtja99',
    'gtja117','gtja76','gtja90','volatility_20','gtja113','gtja141','a99',
    'gtja12','gtja83','gtja164','a98','gtja49','gtja121','a85','gtja104',
    'gtja185','gtja176','a80','gtja62','a8','gtja34','returns','gtja168',
    'gtja108','gtja105','gtja127','a27','a64','gtja91','a30','a69','a91',
    'gtja13','gtja120'
]

def load_data(start_date="2026-01-01", end_date="2026-06-11"):
    """加载因子和行情数据"""
    t0 = time.time()
    conn = duckdb.connect(DB_PATH, read_only=True)
    all_cols = [r[0] for r in conn.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name='factors_wide'"
    ).fetchall()]
    available = [c for c in LIVE_FACTORS if c in all_cols]
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


def build_mf_signal(z3, fnames, weights_dict):
    """用给定权重构建多因子信号"""
    wv = np.zeros(len(fnames), dtype=np.float32)
    for fi, fc in enumerate(fnames):
        if fc in weights_dict:
            wv[fi] = float(weights_dict[fc])
    s = np.sum(np.abs(wv))
    if s > 0:
        wv /= s
    mf = np.nan_to_num(np.tensordot(z3, wv, axes=(2, 0)), nan=-1e10, neginf=-1e10)
    return mf


def backtest_mf(sig, fwd, dm, cl, nd, ns, rebal_freq=5, top_n=10,
                min_hold_days=10, stop_loss_pct=0.06, tx_cost=TX):
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
        "total_return": round(total_ret, 4),
        "annual_return": round(float(ar), 4),
        "sharpe": round(float(sp), 4),
        "max_drawdown": round(float(mdd), 4),
        "calmar": round(float(cal), 4),
        "win_rate": round(float(wr), 4),
    }


def main():
    parser = argparse.ArgumentParser(description="因子权重评估")
    parser.add_argument("--weights", required=True, help="权重JSON文件路径")
    parser.add_argument("--start", default="2026-04-01", help="回测起始日期")
    parser.add_argument("--end", default="2026-06-11", help="回测结束日期")
    parser.add_argument("--rebal", type=int, default=5, help="换仓频率(天)")
    parser.add_argument("--top_n", type=int, default=10, help="持仓数量")
    parser.add_argument("--stop_loss", type=float, default=0.06, help="止损比例")
    args = parser.parse_args()

    # 加载权重
    with open(args.weights) as f:
        weights_dict = json.load(f)

    # 加载数据(从更早开始以获得预热期)
    data_start = pd.Timestamp(args.start) - pd.Timedelta(days=30)
    z3, fwd, dm, cl, tks, fnames, nd, ns, ds = load_data(
        start_date=data_start.strftime("%Y-%m-%d"),
        end_date=args.end
    )

    # 找到回测起始日索引
    bt_start = pd.Timestamp(args.start)
    start_idx = next((i for i, d in enumerate(ds) if d >= bt_start), 0)

    # 构建信号
    sig = build_mf_signal(z3, fnames, weights_dict)

    # 回测
    dr = backtest_mf(sig, fwd, dm, cl, nd, ns,
                     rebal_freq=args.rebal, top_n=args.top_n,
                     stop_loss_pct=args.stop_loss)

    # 只取回测期内的收益
    dr_bt = dr[start_idx:]
    metrics = compute_metrics(dr_bt)

    # 输出结果到stderr(详细信息)
    print(f"[RESULT] AR={metrics['annual_return']*100:.2f}% "
          f"SR={metrics['sharpe']:.3f} "
          f"MDD={abs(metrics['max_drawdown'])*100:.2f}% "
          f"Calmar={metrics['calmar']:.3f} "
          f"WR={metrics['win_rate']*100:.1f}%", file=sys.stderr)

    # stdout最后一行只输出Calmar
    print(f"{metrics['calmar']:.4f}")


if __name__ == "__main__":
    main()