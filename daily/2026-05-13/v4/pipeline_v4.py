import os, sys, json, time, logging, gc
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from core.database import Database

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
RUN_DAY = "2026-05-13"
VERSION = "v4"
OUT_DIR = os.path.join(PROJ_ROOT, "daily", RUN_DAY, VERSION)
os.makedirs(OUT_DIR, exist_ok=True)

LOG_FILE = os.path.join(OUT_DIR, "pipeline.log")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("pipeline_v4")

logger.info("=" * 60)
logger.info("V4 Pipeline — 择时器对比: TV / MR / Composite / 无择时")
logger.info("=" * 60)

TX_COST_RATE = 0.0012

FACTOR_NAMES = ['a27','a30','a31','a41','a42','a64','a69','a8','a80','a85',
    'a88','a91','a97','a98','a99','ff_mkt','gtja103','gtja104','gtja105',
    'gtja108','gtja113','gtja117','gtja12','gtja120','gtja121','gtja123',
    'gtja127','gtja13','gtja139','gtja141','gtja142','gtja144','gtja148',
    'gtja164','gtja168','gtja171','gtja176','gtja185','gtja34','gtja49',
    'gtja62','gtja76','gtja83','gtja85','gtja90','gtja91','gtja99',
    'returns','rsi_14','volatility_20',
    'macd','macd_signal','momentum_5','momentum_20','volume_ratio','boll_position']

FACTOR_NAMES = list(set(FACTOR_NAMES))


def load_data():
    logger.info("加载数据...")
    db = Database()
    symbols_df = db.get_symbols()
    tickers = symbols_df['symbol'].tolist()
    logger.info(f"股票池: {len(tickers)}只")

    factor_df = db.get_factors(start_date="2018-01-01", end_date="2026-04-30",
                               factor_names=FACTOR_NAMES, with_close=True)
    logger.info(f"因子数据: {factor_df.shape}")
    factor_df['date'] = pd.to_datetime(factor_df['date'])
    all_dates = sorted(factor_df['date'].unique())
    logger.info(f"日期: {all_dates[0]}~{all_dates[-1]} ({len(all_dates)}天)")

    n_dates, n_symbols = len(all_dates), len(tickers)
    n_factors = len(FACTOR_NAMES)
    vals_3d = np.full((n_dates, n_symbols, n_factors), np.nan, dtype=np.float32)
    data_mask = np.zeros((n_dates, n_symbols), dtype=bool)
    close_arr = np.zeros((n_dates, n_symbols), dtype=np.float32)
    ticker_to_idx = {t: i for i, t in enumerate(tickers)}
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    di_map = np.array([date_to_idx[d] for d in factor_df['date']], dtype=np.int32)
    si_map = np.array([ticker_to_idx.get(s, -1) for s in factor_df['symbol']], dtype=np.int32)
    valid = si_map >= 0
    di_map, si_map = di_map[valid], si_map[valid]

    for fi, fc in enumerate(FACTOR_NAMES):
        if fc in factor_df.columns:
            vals = factor_df[fc].values[valid].astype(np.float32)
            vals_3d[di_map, si_map, fi] = vals

    cv = factor_df['close'].values[valid].astype(np.float32)
    close_arr[di_map, si_map] = cv
    data_mask[di_map, si_map] = True
    np.nan_to_num(vals_3d, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    np.nan_to_num(close_arr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    logger.info("计算收益率...")
    fwd_rets = np.zeros((n_dates, n_symbols), dtype=np.float32)
    for di in range(n_dates - 1):
        both = (close_arr[di] > 1e-10) & (close_arr[di + 1] > 1e-10)
        fwd_rets[di, both] = (close_arr[di + 1, both] - close_arr[di, both]) / close_arr[di, both]

    logger.info(f"数据: {n_dates}天×{n_symbols}只×{n_factors}因子")
    return vals_3d, fwd_rets, tickers, all_dates, data_mask, close_arr, n_dates, n_symbols


def compute_sma(arr_2d, w):
    n_dates, n_sym = arr_2d.shape
    out = np.zeros_like(arr_2d)
    for di in range(n_dates):
        if di < w - 1:
            out[di] = np.mean(arr_2d[:di+1], axis=0) if di > 0 else arr_2d[0]
        else:
            out[di] = np.mean(arr_2d[di-w+1:di+1], axis=0)
    return out


def precompute_mf_signal(vals_3d, v1wp):
    logger.info("预计算MF信号...")
    v1_w = {}
    if os.path.exists(v1wp):
        with open(v1wp) as f:
            vd = json.load(f)
        for item in vd:
            if 'L1中_80代' in item['label']:
                v1_w = item['configs'][0]['weights']
                break
    logger.info(f"V1权重: {len(v1_w)}个因子")

    wv = np.zeros(vals_3d.shape[2], dtype=np.float32)
    for fi, fc in enumerate(FACTOR_NAMES):
        if fc in v1_w:
            wv[fi] = float(v1_w[fc])
    s = np.sum(np.abs(wv))
    if s > 0:
        wv /= s

    z_3d = np.zeros_like(vals_3d)
    for fi in range(vals_3d.shape[2]):
        arr = vals_3d[:, :, fi]
        for di in range(arr.shape[0]):
            row = arr[di, :]
            lo, hi = np.quantile(row[row != 0], [0.01, 0.99]) if np.any(row != 0) else (0, 0)
            clipped = np.clip(row, lo, hi)
            mu, sd = np.mean(clipped), np.std(clipped)
            z_3d[di, :, fi] = (clipped - mu) / sd if sd > 1e-10 else 0.0

    sig = np.tensordot(z_3d, wv, axes=(2, 0))
    sig = np.nan_to_num(sig, nan=-1e10, neginf=-1e10)
    logger.info(f"MF信号: {sig.shape}")
    return sig


def precompute_position_ratios(vals_3d, close_arr, all_dates):
    n_dates, n_sym = vals_3d.shape[:2]
    fi_map = {fn: i for i, fn in enumerate(FACTOR_NAMES)}
    def fi(n): return fi_map.get(n)

    im, ims = fi('macd'), fi('macd_signal')
    im5, im20 = fi('momentum_5'), fi('momentum_20')
    ir = fi('rsi_14')
    iv = fi('volatility_20')
    hv_t = 0.05

    logger.info("预计算TV择时仓位系数...")
    tv_ratio = np.full(n_dates, 0.5, dtype=np.float32)
    for di in range(n_dates):
        scores = []
        nc = 0
        if im is not None and ims is not None:
            m, ms = vals_3d[di, :, im], vals_3d[di, :, ims]
            scores.append(np.where(m > ms, 1.0, 0.0)); nc += 1
        if im5 is not None and im20 is not None:
            m5, m20 = vals_3d[di, :, im5], vals_3d[di, :, im20]
            s = np.where((m5 > 0) & (m5 > m20), 1.0, np.where(m5 < 0, 0.0, 0.5))
            scores.append(s); nc += 1
        if ir is not None:
            r = vals_3d[di, :, ir]
            s = np.where(r > 70, 0.0, np.where(r >= 50, 1.0, np.where(r >= 30, 0.5, 0.0)))
            scores.append(s); nc += 1
        trend = np.mean(scores, axis=0) if nc > 0 else np.full(n_sym, 0.5)
        if iv is not None:
            trend[vals_3d[di, :, iv] > hv_t] = -1.0
        valid = trend[(trend >= 0) & ~np.isnan(trend)]
        if len(valid) > 0:
            br = float(np.mean(valid >= 0.6))
            tv_ratio[di] = np.clip(br * 1.5, 0.2, 1.0)

    logger.info("预计算MR择时仓位系数...")
    ma5 = compute_sma(close_arr, 5)
    ma20 = compute_sma(close_arr, 20)
    ma60 = compute_sma(close_arr, 60)
    mr_ratio = np.full(n_dates, 0.6, dtype=np.float32)
    for di in range(n_dates):
        votes = []
        if np.any(ma60[di] > 0):
            br = np.mean((ma5[di] > ma20[di]) & (ma20[di] > ma60[di]))
            votes.append((br - 0.5) * 2)
        if np.any(ma20[di] > 0):
            ab = np.mean(close_arr[di] > ma60[di]) if np.any(ma60[di] > 0) else 0.5
            votes.append((ab - 0.5) * 2)
        if votes:
            as_ = np.mean(votes)
            mr_ratio[di] = 1.0 if as_ > 0.3 else (0.3 if as_ < -0.3 else 0.6)

    logger.info("预计算Composite择时仓位系数...")
    comp_ratio = np.full(n_dates, 0.6, dtype=np.float32)
    for di in range(n_dates):
        tv_v = 1 if tv_ratio[di] >= 0.6 else (-1 if tv_ratio[di] < 0.3 else 0)
        mr_v = 1 if mr_ratio[di] >= 0.8 else (-1 if mr_ratio[di] <= 0.4 else 0)
        vs = tv_v + mr_v
        comp_ratio[di] = 1.0 if vs >= 1 else (0.3 if vs <= -1 else 0.6)

    logger.info(f"TV平均仓位: {np.mean(tv_ratio)*100:.0f}%, MR: {np.mean(mr_ratio)*100:.0f}%, Comp: {np.mean(comp_ratio)*100:.0f}%")
    return {
        "none": None,
        "tv": lambda i: float(tv_ratio[min(i, n_dates-1)]),
        "mr": lambda i: float(mr_ratio[min(i, n_dates-1)]),
        "composite": lambda i: float(comp_ratio[min(i, n_dates-1)]),
    }


def run_bt(sig, fwd_rets, dm, name, rf=3, pos_fn=None, tn=40):
    n_d, n_s = sig.shape
    pw = np.zeros(n_s, dtype=np.float32)
    hs = np.full(n_s, -1, dtype=np.int32)
    rh = 0
    eq = np.ones(n_d, dtype=np.float64)
    dr = np.zeros(n_d, dtype=np.float64)
    ttx = 0.0
    nt = 0
    prs = []
    cur_pr = 1.0

    for i in range(1, n_d):
        rebal = (i % rf == 0)
        st = sig[i]

        if rebal:
            cur_pr = pos_fn(i) if pos_fn else 1.0
            prs.append(cur_pr)

            lo = _lkw(pw, hs, rh)
            lw = float(np.sum(pw[lo]))
            si = np.argsort(-st)[:tn]
            vm = np.zeros(n_s, dtype=bool)
            vm[si] = True
            av = vm & ~lo
            ai = np.where(av)[0]
            if len(ai) == 0:
                ai = np.where(vm)[0]
            rm = 1.0 - lw
            if rm <= 0:
                nw = pw.copy()
            else:
                vol = np.nanstd(fwd_rets[max(0,i-20):i], axis=0) + 1e-10 if i >= 20 else np.ones(n_s)
                iv = 1.0 / vol[ai]
                ivs = float(np.sum(iv))
                tgt = (iv / ivs) * rm if ivs > 0 else np.ones(len(ai)) * rm / len(ai)
                nw = np.zeros(n_s, dtype=np.float32)
                nw[lo] = pw[lo]
                nw[ai] = tgt

            to = float(np.sum(np.abs(nw - pw)))
            txc = 0.5 * to * TX_COST_RATE
            ttx += txc
            if to > 0.01:
                nt += 1

            pw = nw
            for j in range(n_s):
                if nw[j] > 0 and hs[j] < 0:
                    hs[j] = rh + 1
        else:
            mk = dm[i] & (pw > 0)
            if np.any(mk):
                ps = pw[mk].copy()
                p2 = ps / float(np.sum(ps))
                pw = np.zeros(n_s, dtype=np.float32)
                pw[mk] = p2

        rt = cur_pr * float(np.dot(pw, fwd_rets[i]))
        rt = 0.0 if (np.isnan(rt) or np.isinf(rt)) else rt
        if rebal and to > 0.01:
            rt -= txc
        dr[i] = rt
        eq[i] = eq[i-1] * (1.0 + rt)
        rh += 1

    tr = float(eq[-1] / eq[0] - 1.0)
    ny = n_d / 252.0
    ar = (float(eq[-1] / eq[0])) ** (1.0 / max(ny, 0.5)) - 1.0
    lr = np.log(eq[1:] / eq[:-1])
    sp = float(np.mean(lr) / max(np.std(lr), 1e-10) * np.sqrt(252))
    cm = np.maximum.accumulate(eq)
    ddn = (eq - cm) / cm
    mdd = float(np.min(ddn))
    cal = ar / abs(mdd) if abs(mdd) > 0 else 0
    wd = int(np.sum(dr > 0))
    ld = int(np.sum(dr < 0))
    wr = wd / max(wd + ld, 1)
    apr = float(np.mean(prs)) if prs else 1.0

    logger.info(f"  [{name}] 年化={ar*100:.2f}% Sharpe={sp:.3f} 回撤={mdd*100:.2f}% "
                f"Calmar={cal:.3f} 胜率={wr*100:.1f}% 仓位={apr*100:.0f}%")
    return {"config_name": name, "total_return": tr, "annual_return": ar, "sharpe": sp,
            "max_drawdown": mdd, "calmar": cal, "win_rate": wr, "n_trades": nt,
            "total_tx_cost": ttx, "avg_position": apr}


def _lkw(pw, hs, rh, md=5):
    lo = np.zeros(len(pw), dtype=bool)
    for j in range(len(pw)):
        if hs[j] > 0 and pw[j] > 0 and (rh - hs[j]) < md:
            lo[j] = True
    return lo


def gen_report(rr):
    h = """<!DOCTYPE html><html lang="zh-CN">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Zequant V4 择时器对比报告</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#f5f7fa;color:#1a1a2e;line-height:1.6}
.header{background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);color:white;padding:40px 60px}
.header h1{font-size:28px;margin-bottom:8px}.header .subtitle{color:#a0aec0;font-size:14px}
.container{max-width:1200px;margin:0 auto;padding:30px}
.section{background:white;border-radius:12px;padding:24px;margin-bottom:24px;box-shadow:0 2px 8px rgba(0,0,0,0.06)}
.section h2{font-size:18px;color:#2d3748;margin-bottom:16px;padding-bottom:8px;border-bottom:2px solid #e2e8f0}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{padding:10px 12px;text-align:right;border-bottom:1px solid #e2e8f0}
th{background:#f7fafc;font-weight:600;color:#4a5568;font-size:12px}
td:first-child,th:first-child{text-align:left}
tr:hover{background:#f7fafc}
.best{background:#c6f6d5!important;font-weight:600}
</style></head><body>
<div class="header"><h1>Zequant V4 择时器对比报告</h1>
<div class="subtitle">MF选股+择时器仓位系数+周频+风险平价</div></div>
<div class="container">"""
    h += '<div class="section"><h2>全部结果（按Sharpe）</h2><table>'
    h += '<tr><th>#</th><th>择时器</th><th>年化%</th><th>Sharpe</th><th>回撤%</th><th>Calmar</th><th>胜率%</th><th>仓位</th><th>换手</th></tr>'
    for i, r in enumerate(sorted(rr, key=lambda x: x["sharpe"], reverse=True), 1):
        c = "best" if i == 1 else ""
        h += f'<tr class="{c}"><td>{i}</td><td>{r["config_name"]}</td>'
        h += f'<td>{r["annual_return"]*100:.2f}%</td><td>{r["sharpe"]:.3f}</td>'
        h += f'<td style="color:#e53e3e">{r["max_drawdown"]*100:.2f}%</td>'
        h += f'<td>{r["calmar"]:.3f}</td><td>{r["win_rate"]*100:.1f}%</td>'
        h += f'<td>{r["avg_position"]*100:.0f}%</td><td>{r["n_trades"]}</td></tr>'
    h += '</table></div>'
    h += '</div></body></html>'
    rp = os.path.join(OUT_DIR, "timing_compare_report.html")
    with open(rp, 'w') as f:
        f.write(h)
    logger.info(f"报告: {rp}")


def main():
    v3d, fwr, tks, ads, dm, cl, nd, ns = load_data()
    logger.info(f"矩阵: {v3d.shape}")

    v1wp = os.path.join(PROJ_ROOT, 'daily', RUN_DAY, 'v2', 'v1_reference', 'ga_results.json')
    mf = precompute_mf_signal(v3d, v1wp)
    pf = precompute_position_ratios(v3d, cl, ads)

    cfgs = [("无择时", "none"), ("TrendVolatilityTiming", "tv"),
            ("MarketRegimeTiming", "mr"), ("Composite(TV+MR)", "composite")]

    res = []
    for lb, k in cfgs:
        logger.info(f"\n回测: {lb}")
        res.append(run_bt(mf, fwr, dm, lb, rf=3, pos_fn=pf[k], tn=40))
        gc.collect()

    json.dump({"version":"v4","results":res,"timestamp":datetime.now().isoformat()},
              open(os.path.join(OUT_DIR,"results.json"),'w'), indent=2)

    print(f"\n{'='*100}")
    print(f"{'#':<3} {'择时器':<26} {'年化%':<8} {'Sharpe':<8} {'回撤%':<8} {'Calmar':<8} {'胜率':<6} {'仓位':<6} {'换手':<5}")
    print('-'*100)
    for i, r in enumerate(sorted(res, key=lambda x: x["sharpe"], reverse=True), 1):
        print(f"{i:<3} {r['config_name']:<26} {r['annual_return']*100:>6.2f}% {r['sharpe']:>7.3f} "
              f"{r['max_drawdown']*100:>6.2f}% {r['calmar']:>7.3f} {r['win_rate']*100:>5.1f}% "
              f"{r['avg_position']*100:>4.0f}% {r['n_trades']:>4}")
    print('='*100)
    gen_report(res)


if __name__ == "__main__":
    main()
