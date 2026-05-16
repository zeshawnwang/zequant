"""
X8 — TrendBreakoutSelector 真实信号验证

用 TrendBreakoutSelector 所需的全部因子（ma5/ma20/ma60/ma120/
macd_above_zero/volume_breakout_ratio 等）构建真实信号。

优化：向量化预计算选股信号矩阵（等价于逐日调用 select()，但快数百倍），
避免 1776 天 × groupby 的开销。

回测扫描：
  - 频率: D3 / D5 / D10
  - 择时: ×VolTiming / ×无择时
"""
import os, sys, json, logging, numpy as np, pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from core.database import Database
from core.positioners import RPPortfolioWeights

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("X8")
TX = 0.0012

# TrendBreakoutSelector 所需的因子列
TB_FACTORS = [
    "ma5", "ma20", "ma60", "ma120",
    "ma_alignment_score", "ma60_trend", "ma120_trend",
    "macd_above_zero", "volume_breakout_ratio",
]

# 择时因子（VolTiming 需要 volatility_20）
TIMING_FACTORS = ["volatility_20"]

# close 通过 with_close=True 额外获取，不作为 tensor 因子维
ALL_FACTORS = list(set(TB_FACTORS + TIMING_FACTORS))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "x8_results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load():
    """加载 TB 因子数据 + 收盘价，构建原始值 tensor（不做 zscore 归一化）。"""
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
    v = si >= 0
    di, si = di[v], si[v]

    for fi, fc in enumerate(ALL_FACTORS):
        if fc in df.columns:
            v3[di, si, fi] = df[fc].values[v].astype(np.float32)

    cl[di, si] = df['close'].values[v].astype(np.float32)
    dm[di, si] = True

    np.nan_to_num(v3, nan=0.0, copy=False)
    np.nan_to_num(cl, nan=0.0, copy=False)

    # 前向收益率
    fwd = np.zeros((nd, ns), dtype=np.float32)
    for d in range(nd - 1):
        b = (cl[d] > 1e-10) & (cl[d + 1] > 1e-10)
        fwd[d, b] = (cl[d + 1, b] - cl[d, b]) / cl[d, b]

    logger.info(f"数据: {nd}天 × {ns}只 × {nf}因子")
    return v3, fwd, dm, tks, ALL_FACTORS, nd, ns, ds, t2i, d2i, cl


def build_tb_signal(v3, dm, cl, fi, nd, ns, t2i, top_n=40):
    """向量化预计算 TrendBreakoutSelector 信号矩阵。

    逻辑完全等价于 TrendBreakoutSelector.select() 的逐日调用：
      1. 用前一日数据（避免前视 bias）
      2. MA5 > MA20 > MA60（多头排列）
      3. close > MA60（价格在60日线上方）
      4. MA60_trend > 0 且 MA120_trend > 0（长短均线上行）
      5. MACD > 0（零轴上方）
      6. volume_breakout_ratio >= 1.5（放量突破）
      7. 剩余候选按 volume_breakout_ratio 降序取 top_n
    """
    # 提取各因子列（原始值）
    ma5 = v3[:, :, fi['ma5']]
    ma20 = v3[:, :, fi['ma20']]
    ma60 = v3[:, :, fi['ma60']]
    ma60_trend = v3[:, :, fi['ma60_trend']]
    ma120_trend = v3[:, :, fi['ma120_trend']]
    macd_above_zero = v3[:, :, fi['macd_above_zero']]
    vol_breakout = v3[:, :, fi['volume_breakout_ratio']]
    min_vol_ratio = 1.5

    tb_sig = np.full((nd, ns), -np.inf, dtype=np.float32)

    # 逐日预计算（仅用前一日数据）
    for d in range(1, nd):
        prev = d - 1

        mask = dm[prev].copy()

        if 'ma5' in fi and 'ma20' in fi and 'ma60' in fi:
            mask &= (ma5[prev] > ma20[prev]) & (ma20[prev] > ma60[prev])

        if 'close' in fi or True:
            mask &= cl[prev] > ma60[prev]

        if 'ma60_trend' in fi:
            mask &= ma60_trend[prev] > 0
        if 'ma120_trend' in fi:
            mask &= ma120_trend[prev] > 0

        if 'macd_above_zero' in fi:
            mask &= macd_above_zero[prev] > 0

        if 'volume_breakout_ratio' in fi:
            mask &= vol_breakout[prev] >= min_vol_ratio

        candidates = np.where(mask)[0]
        if len(candidates) > 0:
            scores = vol_breakout[prev, candidates]
            k = min(top_n, len(candidates))
            top_idx = candidates[np.argsort(-scores)[:k]]
            tb_sig[d, top_idx] = 1.0

    logger.info(f"TrendBreakout 信号预计算完成：{nd}天，top_n={top_n}")
    return tb_sig


def bt(sig, fwd, dm, name, rf=3, tn=40, pos_ratio=None, mhd=5):
    """回测函数，与 V6/X2 系列一致。"""
    nd, ns = sig.shape
    alloc = RPPortfolioWeights(top_n=tn, min_hold_days=mhd)
    pw = np.zeros(ns, dtype=np.float32)
    hs = np.full(ns, -1, dtype=np.int32)
    rh = 0
    eq = np.ones(nd, dtype=np.float64)
    dr = np.zeros(nd, dtype=np.float64)
    ttx = 0.0
    nt = 0
    for i in range(1, nd):
        rebal = (i % rf == 0)
        if rebal:
            pr = pos_ratio[i] if pos_ratio is not None else 1.0
            nw = alloc.allocate(sig[i], fwd, i, pw, hs, rh)
            to = float(np.sum(np.abs(nw - pw)))
            txc = 0.5 * to * TX
            ttx += txc
            if to > 0.01:
                nt += 1
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
        rt = 0.0 if (np.isnan(rt) or np.isinf(rt)) else rt
        dr[i] = rt
        eq[i] = eq[i - 1] * (1.0 + rt)
        rh += 1
    tr = float(eq[-1] / eq[0] - 1.0)
    ny = nd / 252.0
    ar = (float(eq[-1] / eq[0])) ** (1.0 / max(ny, 0.5)) - 1.0
    lr = np.log(eq[1:] / eq[:-1])
    sp = float(np.mean(lr) / max(np.std(lr), 1e-10) * np.sqrt(252))
    cm = np.maximum.accumulate(eq)
    dd = (eq - cm) / cm
    mdd = float(np.min(dd))
    cal = ar / abs(mdd) if abs(mdd) > 0 else 0
    wr = int(np.sum(dr > 0)) / max(int(np.sum(dr > 0)) + int(np.sum(dr < 0)), 1)
    logger.info(f"  [{name}] 年化={ar*100:.2f}% Sharpe={sp:.3f} 回撤={mdd*100:.2f}% Calmar={cal:.3f}")
    return {
        "name": name,
        "annual_return": round(ar, 4),
        "sharpe": round(sp, 4),
        "max_drawdown": round(mdd, 4),
        "calmar": round(cal, 4),
        "win_rate": round(wr, 4),
        "n_trades": nt,
    }


def main():
    logger.info("=" * 60)
    logger.info("X8 — TrendBreakoutSelector 真实信号验证")
    logger.info("=" * 60)

    # 1. 加载数据
    v3, fwd, dm, tks, fnames, nd, ns, ds, t2i, d2i, cl = load()
    fi = {fn: i for i, fn in enumerate(fnames)}

    # 2. 预计算 TB 信号矩阵（向量化，不等价于逐日 select() 但逻辑完全一致）
    logger.info("\n=== 预计算 TrendBreakoutSelector 信号 ===")
    tb_sig = build_tb_signal(v3, dm, cl, fi, nd, ns, t2i, top_n=40)

    # 3. 择时信号（同 V6/V8）
    iv = fi.get('volatility_20')
    vol_p = np.ones(nd, dtype=np.float32)
    if iv is not None:
        vol_p = np.clip(1.0 - np.mean(v3[:, :, iv] > 0.05, axis=1), 0.2, 1.0)

    # 4. 回测扫描：频率 D3/D5/D10 × 择时 无择时/VolTiming
    all_results = []
    freqs = [("D3", 3), ("D5", 5), ("D10", 10)]
    timings = [("无择时", None), ("VolTiming", vol_p)]

    logger.info("\n=== TB 信号回测 ===")
    for fname, fd in freqs:
        for pname, ppos in timings:
            label = f"TB_{pname}_{fname}"
            all_results.append(
                bt(tb_sig, fwd, dm, label, rf=fd, tn=40, pos_ratio=ppos, mhd=5)
            )

    # 5. 保存结果
    out_path = os.path.join(RESULTS_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info(f"\n结果已保存至: {out_path}")

    # 6. 打印汇总
    print(f"\n{'=' * 110}")
    print(f"{'实验':<30} {'年化%':<8} {'Sharpe':<8} {'回撤%':<8} {'Calmar':<8} {'胜率':<6} {'交易':<5}")
    print("-" * 110)
    for r in sorted(all_results, key=lambda x: x['sharpe'], reverse=True):
        ok = r['max_drawdown'] < 0.20 and r['annual_return'] > 0.05
        cls = "🏆" if ok else "  "
        print(
            f"{cls} {r['name']:<28} {r['annual_return'] * 100:>6.2f}% "
            f"{r['sharpe']:>7.3f} {r['max_drawdown'] * 100:>6.2f}% "
            f"{r['calmar']:>7.3f} {r['win_rate'] * 100:>5.1f}% {r['n_trades']:>4}"
        )
    print("=" * 110)


if __name__ == "__main__":
    main()
