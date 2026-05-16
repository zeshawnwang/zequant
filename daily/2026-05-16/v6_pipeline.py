"""
V6 — 择时器+频率+分配器完整扫描（修复择时bug后的正确版本）
修复：择时器的仓位系数应单独作用于收益计算，而非改变选股排名
"""
import os, sys, json, logging, gc, numpy as np, pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from core.database import Database
from core.positioners import RPPortfolioWeights

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("v6")

TX_COST = 0.0012
FACTORS = list(set(['a27','a30','a31','a41','a42','a64','a69','a8','a80','a85',
    'a88','a91','a97','a98','a99','ff_mkt','gtja103','gtja104','gtja105',
    'gtja108','gtja113','gtja117','gtja12','gtja120','gtja121','gtja123',
    'gtja127','gtja13','gtja139','gtja141','gtja142','gtja144','gtja148',
    'gtja164','gtja168','gtja171','gtja176','gtja185','gtja34','gtja49',
    'gtja62','gtja76','gtja83','gtja85','gtja90','gtja91','gtja99',
    'returns','rsi_14','volatility_20','macd','macd_signal','momentum_5',
    'momentum_20','volume_ratio','boll_position']))

V6_DIR = os.path.join(os.path.dirname(__file__), "v6_results")
os.makedirs(V6_DIR, exist_ok=True)

def load():
    db = Database()
    df = db.get_factors(start_date="2018-01-01", end_date="2026-04-30", factor_names=FACTORS, with_close=True)
    df['date'] = pd.to_datetime(df['date'])
    ds = sorted(df['date'].unique())
    tks = db.get_symbols()['symbol'].tolist()
    nd, ns, nf = len(ds), len(tks), len(FACTORS)
    t2i = {t:i for i,t in enumerate(tks)}; d2i = {d:i for i,d in enumerate(ds)}
    v3 = np.full((nd,ns,nf), np.nan, dtype=np.float32)
    dm = np.zeros((nd,ns), dtype=bool); cl = np.zeros((nd,ns), dtype=np.float32)
    di = np.array([d2i[d] for d in df['date']], dtype=np.int32)
    si = np.array([t2i.get(s,-1) for s in df['symbol']], dtype=np.int32)
    v = si >= 0; di, si = di[v], si[v]
    for fi,fc in enumerate(FACTORS):
        if fc in df.columns: v3[di,si,fi] = df[fc].values[v].astype(np.float32)
    cl[di,si] = df['close'].values[v].astype(np.float32); dm[di,si] = True
    np.nan_to_num(v3, nan=0.0, copy=False); np.nan_to_num(cl, nan=0.0, copy=False)
    fwd = np.zeros((nd,ns), dtype=np.float32)
    for d in range(nd-1):
        b = (cl[d] > 1e-10) & (cl[d+1] > 1e-10)
        fwd[d,b] = (cl[d+1,b] - cl[d,b]) / cl[d,b]
    z3 = np.zeros_like(v3)
    for fi in range(nf):
        a = v3[:,:,fi]
        for d in range(nd):
            r = a[d,:]; nz = r[r!=0]
            if len(nz) > 1:
                lo,hi = np.quantile(nz,[0.01,0.99]); c = np.clip(r,lo,hi)
                mu,sd = np.mean(c), np.std(c)
                z3[d,:,fi] = (c-mu)/sd if sd > 1e-10 else 0.0
    logger.info(f"数据: {nd}天×{ns}只×{nf}因子")
    return z3, fwd, dm, tks, FACTORS, nd, ns

def v1w():
    p = os.path.join(os.path.dirname(__file__),'..','2026-05-13','v2','v1_reference','ga_results.json')
    if os.path.exists(p):
        with open(p) as f:
            for it in json.load(f):
                if 'L1中_80代' in it['label']:
                    return it['configs'][0]['weights']
    return {}

def bt(sig, fwd, dm, name, rf=3, tn=40, pos_ratio=None):
    """回测，支持 pos_ratio (nd,) 作为每日仓位系数。"""
    nd, ns = sig.shape
    alloc = RPPortfolioWeights(top_n=tn, min_hold_days=5)
    pw = np.zeros(ns, dtype=np.float32)
    hs = np.full(ns, -1, dtype=np.int32)
    rh = 0; eq = np.ones(nd, dtype=np.float64); dr = np.zeros(nd, dtype=np.float64)
    ttx = 0.0; nt = 0
    for i in range(1, nd):
        rebal = (i % rf == 0)
        if rebal:
            pr = pos_ratio[i] if pos_ratio is not None else 1.0
            nw = alloc.allocate(sig[i], fwd, i, pw, hs, rh)
            to = float(np.sum(np.abs(nw - pw)))
            txc = 0.5 * to * TX_COST
            ttx += txc
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
    tr = float(eq[-1]/eq[0] - 1.0)
    ny = nd/252.0; ar = (float(eq[-1]/eq[0]))**(1.0/max(ny,0.5)) - 1.0
    lr = np.log(eq[1:]/eq[:-1])
    sp = float(np.mean(lr)/max(np.std(lr),1e-10)*np.sqrt(252))
    cm = np.maximum.accumulate(eq); dd = (eq-cm)/cm; mdd = float(np.min(dd))
    cal = ar/abs(mdd) if abs(mdd) > 0 else 0
    wr = int(np.sum(dr>0))/max(int(np.sum(dr>0))+int(np.sum(dr<0)),1)
    logger.info(f"  [{name}] 年化={ar*100:.2f}% Sharpe={sp:.3f} 回撤={mdd*100:.2f}% Calmar={cal:.3f}")
    return {"name":name, "annual_return":round(ar,4), "sharpe":round(sp,4),
            "max_drawdown":round(mdd,4), "calmar":round(cal,4), "win_rate":round(wr,4), "n_trades":nt}

def main():
    z3, fwd, dm, tks, fnames, nd, ns = load()
    fi = {fn:i for i,fn in enumerate(fnames)}
    v1w_dict = v1w()
    wv = np.zeros(len(fnames), dtype=np.float32)
    for fi_i, fc in enumerate(fnames):
        if fc in v1w_dict: wv[fi_i] = float(v1w_dict[fc])
    s = np.sum(np.abs(wv))
    if s > 0: wv /= s
    mf = np.nan_to_num(np.tensordot(z3, wv, axes=(2,0)), nan=-1e10, neginf=-1e10)

    # 预计算择时信号
    im, ims = fi.get('macd'), fi.get('macd_signal')
    im5, im20 = fi.get('momentum_5'), fi.get('momentum_20')
    ir, iv = fi.get('rsi_14'), fi.get('volatility_20')
    trend_signal = np.full((nd, ns), 0.5, dtype=np.float32)
    for d in range(nd):
        s_l = []
        if im is not None and ims is not None:
            s_l.append(np.where(z3[d,:,im] > z3[d,:,ims], 1.0, 0.0))
        if im5 is not None and im20 is not None:
            m5v, m20v = z3[d,:,im5], z3[d,:,im20]
            s_l.append(np.where((m5v>0)&(m5v>m20v), 1.0, np.where(m5v<0, 0.0, 0.5)))
        if ir is not None:
            rv = z3[d,:,ir]
            s_l.append(np.where(rv>70, 0.0, np.where(rv>=50, 1.0, np.where(rv>=30, 0.5, 0.0))))
        if s_l: trend_signal[d] = np.mean(s_l, axis=0)

    # 仓位系数: 每日BUY信号比例
    trend_pos = np.clip(np.mean(trend_signal >= 0.6, axis=1) * 2.0, 0.1, 1.0)
    tv_signal = trend_signal.copy()
    if iv is not None:
        tv_signal[z3[:,:,iv] > 0.05] = -1.0
    tv_pos = np.clip(np.mean(tv_signal >= 0.6, axis=1) * 2.0, 0.1, 1.0)

    # VolTiming仓位: 高波动比例决定
    vol_pos = np.ones(nd, dtype=np.float32)
    if iv is not None:
        hv_ratio = np.mean(z3[:,:,iv] > 0.05, axis=1)
        vol_pos = np.clip(1.0 - hv_ratio, 0.2, 1.0)

    # MR仓位: 市场状态
    mr_pos = np.full(nd, 0.6, dtype=np.float32)
    if iv is not None:
        mkt_vol = np.nanmean(z3[:,:,iv], axis=1)
        mr_pos[mkt_vol < 0.04] = 1.0
        mr_pos[mkt_vol > 0.08] = 0.3

    res = []
    logger.info("="*60)
    logger.info("V6 — 择时器序列(修复后) + 分配器序列(补充) + 最佳组合探索")
    logger.info("="*60)

    # ── 择时器序列 (修复后) ──
    logger.info("\n择时器序列 (MF + 各择时器 + RP)")
    res.append(bt(mf, fwd, dm, "A01_MF_无择时(基准)", rf=3))
    res.append(bt(mf, fwd, dm, "A08_MF+TrendTiming(位置)", rf=3, pos_ratio=trend_pos))
    res.append(bt(mf, fwd, dm, "A09_MF+TVTiming(位置)", rf=3, pos_ratio=tv_pos))
    res.append(bt(mf, fwd, dm, "A10_MF+VolTiming(位置)", rf=3, pos_ratio=vol_pos))
    res.append(bt(mf, fwd, dm, "A11_MF+MRTiming(位置)", rf=3, pos_ratio=mr_pos))

    # ── 频率×择时交叉 (B区间) ──
    logger.info("\n频率×择时交叉 (MF + 各择时器 + 各频率 + RP)")
    for pos_ratio, plabel, pname in [(trend_pos, "TrendTiming", "B01"), (tv_pos, "TVTiming", "B02"),
                                       (vol_pos, "VolTiming", "B03"), (mr_pos, "MRTiming", "B04")]:
        for fd in [2, 5, 10]:
            res.append(bt(mf, fwd, dm, f"{pname}_MF+{plabel}+D{fd}", rf=fd, pos_ratio=pos_ratio))

    # ── 频率×择时×分配器: D10最佳频率的最佳组合 ──
    logger.info("\nD10双周频 × 择时器交叉")
    for pos_ratio, plabel in [(None, "无择时"), (trend_pos, "TrendTiming"),
                              (tv_pos, "TVTiming"), (vol_pos, "VolTiming"), (mr_pos, "MRTiming")]:
        res.append(bt(mf, fwd, dm, f"X01_MF+{plabel}+D10", rf=10, pos_ratio=pos_ratio))

    # ── 保存 ──
    with open(os.path.join(V6_DIR, "results.json"), 'w') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*100}")
    print(f"{'实验':<28} {'年化%':<8} {'Sharpe':<8} {'回撤%':<8} {'Calmar':<8} {'胜率':<6} {'换手':<5}")
    print('-'*100)
    for r in sorted(res, key=lambda x: x['sharpe'], reverse=True):
        cls = "🏆" if r['max_drawdown'] < 0.20 and r['annual_return'] > 0.05 else "  "
        print(f"{cls} {r['name']:<26} {r['annual_return']*100:>6.2f}% {r['sharpe']:>7.3f} {r['max_drawdown']*100:>6.2f}% {r['calmar']:>7.3f} {r['win_rate']*100:>5.1f}% {r['n_trades']:>4}")
    print('='*100)

    logger.info("\n达标策略(回撤<20%且年化>5%):")
    for r in res:
        if r['max_drawdown'] < 0.20 and r['annual_return'] > 0.05:
            logger.info(f"  🏆 {r['name']}: 年化={r['annual_return']*100:.2f}% 回撤={r['max_drawdown']*100:.2f}% Sharpe={r['sharpe']:.3f}")

if __name__ == "__main__":
    main()
