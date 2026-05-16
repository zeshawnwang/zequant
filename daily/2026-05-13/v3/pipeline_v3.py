import os, sys, json, time, logging, gc
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from core.database import Database

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
RUN_DAY = "2026-05-13"
VERSION = "v3"
OUT_DIR = os.path.join(PROJ_ROOT, "daily", RUN_DAY, VERSION)
os.makedirs(OUT_DIR, exist_ok=True)

LOG_FILE = os.path.join(OUT_DIR, "pipeline.log")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("pipeline_v3")

logger.info("=" * 60)
logger.info("V3 Pipeline — 信号源×调频×分配器 交叉对比（实际类逻辑，向量化加速）")
logger.info("=" * 60)

TX_COST_RATE = 0.0012

V1_FACTOR_NAMES = ['a27','a30','a31','a41','a42','a64','a69','a8','a80','a85',
    'a88','a91','a97','a98','a99','ff_mkt','gtja103','gtja104','gtja105',
    'gtja108','gtja113','gtja117','gtja12','gtja120','gtja121','gtja123',
    'gtja127','gtja13','gtja139','gtja141','gtja142','gtja144','gtja148',
    'gtja164','gtja168','gtja171','gtja176','gtja185','gtja34','gtja49',
    'gtja62','gtja76','gtja83','gtja85','gtja90','gtja91','gtja99',
    'returns','rsi_14','volatility_20']

TV_EXTRA = ['macd', 'macd_signal', 'momentum_5', 'momentum_20', 'volume_ratio', 'boll_position']

ALL_FACTOR_NAMES = list(set(V1_FACTOR_NAMES + TV_EXTRA))


def load_data():
    logger.info("加载数据...")
    db = Database()
    symbols_df = db.get_symbols()
    tickers = symbols_df['symbol'].tolist()
    logger.info(f"股票池: {len(tickers)}只")

    factor_df = db.get_factors(
        start_date="2018-01-01", end_date="2026-04-30",
        factor_names=ALL_FACTOR_NAMES, with_close=True
    )
    logger.info(f"因子数据: {factor_df.shape}")

    factor_df['date'] = pd.to_datetime(factor_df['date'])
    all_dates = sorted(factor_df['date'].unique())
    dates_arr = np.array(all_dates, dtype='datetime64')
    logger.info(f"日期: {all_dates[0]} ~ {all_dates[-1]} ({len(all_dates)}天)")

    ticker_to_idx = {t: i for i, t in enumerate(tickers)}
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    n_dates, n_symbols = len(all_dates), len(tickers)

    vals_3d = np.full((n_dates, n_symbols, len(ALL_FACTOR_NAMES)), np.nan, dtype=np.float32)
    data_mask = np.zeros((n_dates, n_symbols), dtype=bool)
    close_arr = np.zeros((n_dates, n_symbols), dtype=np.float32)

    logger.info("构建因子矩阵...")
    di_map = np.array([date_to_idx[d] for d in factor_df['date']], dtype=np.int32)
    si_map = np.array([ticker_to_idx.get(s, -1) for s in factor_df['symbol']], dtype=np.int32)
    valid = si_map >= 0
    di_map, si_map = di_map[valid], si_map[valid]

    for fi, fc in enumerate(ALL_FACTOR_NAMES):
        if fc in factor_df.columns:
            vals = factor_df[fc].values[valid].astype(np.float32)
            vals_3d[di_map, si_map, fi] = vals

    close_vals = factor_df['close'].values[valid].astype(np.float32)
    close_arr[di_map, si_map] = close_vals
    data_mask[di_map, si_map] = True

    np.nan_to_num(vals_3d, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    np.nan_to_num(close_arr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    logger.info("计算收益率...")
    fwd_rets = np.zeros((n_dates, n_symbols), dtype=np.float32)
    for di in range(n_dates - 1):
        both = (close_arr[di] > 1e-10) & (close_arr[di + 1] > 1e-10)
        fwd_rets[di, both] = (close_arr[di + 1, both] - close_arr[di, both]) / close_arr[di, both]

    logger.info(f"数据: {n_dates}天×{n_symbols}只×{len(ALL_FACTOR_NAMES)}因子")
    return vals_3d, fwd_rets, tickers, dates_arr, all_dates, data_mask, factor_df, all_dates, n_dates, n_symbols


def precompute_mf_signal(vals_3d, factor_names, v1_weights_path):
    logger.info("预计算 MultiFactorSelector 信号...")
    v1_w = {}
    if os.path.exists(v1_weights_path):
        with open(v1_weights_path) as f:
            v1_data = json.load(f)
        for item in v1_data:
            if 'L1中_80代' in item['label']:
                v1_w = item['configs'][0]['weights']
                break
    logger.info(f"V1权重: {len(v1_w)}个因子, {sum(1 for v in v1_w.values() if abs(v)>1e-6)}个非零")

    weights_vec = np.zeros(vals_3d.shape[2], dtype=np.float32)
    used_factors = []
    for fi, fc in enumerate(ALL_FACTOR_NAMES):
        if fc in v1_w and fc in factor_names:
            weights_vec[fi] = float(v1_w[fc])
            used_factors.append(fc)
    s = np.sum(np.abs(weights_vec))
    if s > 1e-10:
        weights_vec = weights_vec / s
    logger.info(f"归一化权重: {len(used_factors)}个有效因子")

    z_3d = np.zeros_like(vals_3d)
    for fi in range(vals_3d.shape[2]):
        arr = vals_3d[:, :, fi]
        for di in range(arr.shape[0]):
            row = arr[di, :]
            lo, hi = np.quantile(row[row != 0], [0.01, 0.99]) if np.any(row != 0) else (0, 0)
            clipped = np.clip(row, lo, hi)
            mu, sd = np.mean(clipped), np.std(clipped)
            if sd > 1e-10:
                z_3d[di, :, fi] = (clipped - mu) / sd
            else:
                z_3d[di, :, fi] = 0.0

    signal = np.tensordot(z_3d, weights_vec, axes=(2, 0))
    signal = np.nan_to_num(signal, nan=-1e10, neginf=-1e10)
    logger.info(f"MF信号矩阵: {signal.shape}")
    return signal


def precompute_tv_signal(vals_3d, factor_names):
    logger.info("预计算 TrendVolatilityTiming 信号...")
    fi_map = {fn: i for i, fn in enumerate(ALL_FACTOR_NAMES)}

    def _get(field):
        return fi_map[field] if field in fi_map else None

    idx_macd, idx_macd_sig = _get('macd'), _get('macd_signal')
    idx_m5, idx_m20 = _get('momentum_5'), _get('momentum_20')
    idx_rsi = _get('rsi_14')
    idx_vol = _get('volatility_20')
    high_vol = 0.05

    hv_tr = high_vol * 3
    n_dates, n_symbols = vals_3d.shape[:2]
    signal = np.full((n_dates, n_symbols), -np.inf, dtype=np.float32)

    for di in range(n_dates):
        scores = []
        n_components = 0

        if idx_macd is not None and idx_macd_sig is not None:
            macd = vals_3d[di, :, idx_macd]
            macd_sig = vals_3d[di, :, idx_macd_sig]
            s = np.where((macd > macd_sig) & ~np.isnan(macd) & ~np.isnan(macd_sig), 1.0, 0.0)
            scores.append(s)
            n_components += 1

        if idx_m5 is not None and idx_m20 is not None:
            m5 = vals_3d[di, :, idx_m5]
            m20 = vals_3d[di, :, idx_m20]
            s = np.where((m5 > 0) & (m5 > m20) & ~np.isnan(m5) & ~np.isnan(m20), 1.0,
                        np.where(m5 < 0, 0.0, 0.5))
            scores.append(s)
            n_components += 1

        if idx_rsi is not None:
            rsi = vals_3d[di, :, idx_rsi]
            s = np.where(rsi > 70, 0.0,
                        np.where(rsi >= 50, 1.0,
                                np.where(rsi >= 30, 0.5, 0.0)))
            scores.append(s)
            n_components += 1

        if n_components > 0:
            trend = np.mean(scores, axis=0)
        else:
            continue

        if idx_vol is not None:
            vol = vals_3d[di, :, idx_vol]
            high_vol_mask = (vol > high_vol) & ~np.isnan(vol)
            trend[high_vol_mask] = -1.0

        data_ok = ~np.isnan(trend)
        signal[di, data_ok] = trend[data_ok]

    logger.info(f"TV信号矩阵: {signal.shape}")
    return signal


def compute_rp_weights(scores, fwd_ret, i, prev_weights, hold_since, ret_history_len,
                       top_n=30, min_hold_days=5):
    n_sym = len(scores)
    locked = np.zeros(n_sym, dtype=bool)
    for j in range(n_sym):
        if hold_since[j] > 0 and prev_weights[j] > 0 and (ret_history_len - hold_since[j]) < min_hold_days:
            locked[j] = True
    locked_w = float(np.sum(prev_weights[locked]))

    sidx = np.argsort(-scores)
    top_idx = sidx[:top_n]
    valid_m = np.zeros(n_sym, dtype=bool)
    valid_m[top_idx] = True
    avail = valid_m & ~locked
    aidx = np.where(avail)[0]
    if len(aidx) == 0:
        aidx = np.where(valid_m)[0]

    remaining = 1.0 - locked_w
    if remaining <= 0.0:
        return prev_weights.copy()

    if i >= 20:
        lookback = min(20, i)
        vol = np.nanstd(fwd_ret[i - lookback:i, :], axis=0) + 1e-10
    else:
        vol = np.ones(n_sym, dtype=np.float32)

    inv_v = 1.0 / vol[aidx]
    iv_sum = float(np.sum(inv_v))
    tgt = (inv_v / iv_sum) * remaining if iv_sum > 0 else np.ones(len(aidx), dtype=np.float32) * remaining / len(aidx)

    nw = np.zeros(n_sym, dtype=np.float32)
    nw[locked] = prev_weights[locked]
    nw[aidx] = tgt
    return nw


def compute_rp_hyst(scores, fwd_ret, i, pw, hold_since, rh_len, top_n=30, min_hold=5,
                    lpt=0.10, mad=0.02, kr=0.70):
    target = compute_rp_weights(scores, fwd_ret, i, pw, hold_since, rh_len, top_n, min_hold)
    delta = target - pw
    delta[(pw >= lpt) & (np.abs(delta) < mad)] = 0.0
    nz = np.where(np.abs(delta) > 1e-10)[0]
    if len(nz) > 0:
        ad = np.abs(delta[nz])
        nk = max(1, int(len(nz) * kr))
        tk = np.argpartition(-ad, nk)[:nk]
        delta[np.setdiff1d(nz, nz[tk])] = 0.0
    adj = target.copy()
    adj += delta - (target - pw)
    adj = np.maximum(adj, 0.0)
    t = float(np.sum(adj))
    return (adj / t).astype(np.float32) if t > 0 else pw.copy()


def run_bt(signal, fwd_rets, data_mask, config_name, rebal_freq, use_hyst, top_n=40):
    n_dates, n_sym = signal.shape
    pw = np.zeros(n_sym, dtype=np.float32)
    hold_since = np.full(n_sym, -1, dtype=np.int32)
    rh_len = 0
    equity = np.ones(n_dates, dtype=np.float64)
    dr = np.zeros(n_dates, dtype=np.float64)
    total_tx = 0.0
    n_trades = 0

    for i in range(1, n_dates):
        rebal = (i % rebal_freq == 0)
        scores_t = signal[i]

        if rebal:
            nw = compute_rp_hyst(scores_t, fwd_rets, i, pw, hold_since, rh_len,
                                 top_n=top_n, min_hold=5, lpt=0.10, mad=0.02, kr=0.70) if use_hyst else \
                 compute_rp_weights(scores_t, fwd_rets, i, pw, hold_since, rh_len,
                                    top_n=top_n, min_hold_days=5)
            to = float(np.sum(np.abs(nw - pw)))
            txc = 0.5 * to * TX_COST_RATE
            total_tx += txc
            if to > 0.01:
                n_trades += 1
            pw = nw
            for j in range(n_sym):
                if nw[j] > 0 and hold_since[j] < 0:
                    hold_since[j] = rh_len + 1
        else:
            mk = data_mask[i] & (pw > 0)
            if np.any(mk):
                p2 = pw[mk].copy() / float(np.sum(pw[mk]))
                pw = np.zeros(n_sym, dtype=np.float32)
                pw[mk] = p2

        ret = float(np.dot(pw, fwd_rets[i]))
        ret = 0.0 if (np.isnan(ret) or np.isinf(ret)) else ret
        if rebal and to > 0.01:
            ret -= txc
            txc = 0.0
        dr[i] = ret
        equity[i] = equity[i - 1] * (1.0 + ret)
        rh_len += 1

    total_ret = float(equity[-1] / equity[0] - 1.0)
    ny = n_dates / 252.0
    ar = (float(equity[-1] / equity[0])) ** (1.0 / max(ny, 0.5)) - 1.0
    lr = np.log(equity[1:] / equity[:-1])
    sp = float(np.mean(lr) / max(np.std(lr), 1e-10) * np.sqrt(252))
    cm = np.maximum.accumulate(equity)
    dd = (equity - cm) / cm
    mdd = float(np.min(dd))
    cal = ar / abs(mdd) if abs(mdd) > 0 else 0
    wd = int(np.sum(dr > 0))
    ld = int(np.sum(dr < 0))
    wr = wd / max(wd + ld, 1)

    logger.info(f"  [{config_name}] 年化={ar*100:.2f}% Sharpe={sp:.3f} 回撤={mdd*100:.2f}% Calmar={cal:.3f} 胜率={wr*100:.1f}% 换手={n_trades}")
    return {"config_name": config_name, "total_return": total_ret, "annual_return": ar,
            "sharpe": sp, "max_drawdown": mdd, "calmar": cal, "win_rate": wr,
            "n_trades": n_trades, "total_tx_cost": total_tx}


def generate_report(all_results):
    html = """<!DOCTYPE html><html lang="zh-CN">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Zequant V3 交叉对比报告（项目实际类）</title>
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
.badge-mf{display:inline-block;padding:2px 10px;border-radius:4px;font-size:11px;font-weight:600;background:#bee3f8;color:#2b6cb0}
.badge-tv{display:inline-block;padding:2px 10px;border-radius:4px;font-size:11px;font-weight:600;background:#fefcbf;color:#975a16}
</style></head><body>
<div class="header"><h1>Zequant V3 交叉对比报告</h1>
<div class="subtitle">实际项目类逻辑：MultiFactorSelector × TrendVolatilityTiming</div></div>
<div class="container">"""
    html += '<div class="section"><h2>全部结果排名（按Sharpe）</h2><table>'
    html += '<tr><th>排名</th><th>信号源</th><th>调频</th><th>分配器</th><th>年化%</th><th>Sharpe</th><th>回撤%</th><th>Calmar</th><th>胜率%</th><th>换手</th></tr>'
    for rank, r in enumerate(sorted(all_results, key=lambda x: x["sharpe"], reverse=True), 1):
        parts = r["config_name"].split("_")
        badge = '<span class="badge-mf">多因子</span>' if parts[0] == "MF" else '<span class="badge-tv">趋势+波动率</span>'
        freq = "周频(3d)" if parts[1] == "W" else "高频(1d)"
        alloc = "迟滞" if parts[2] == "H" else "原始"
        cls = "best" if rank == 1 else ""
        html += f'<tr class="{cls}"><td>{rank}</td><td>{badge}</td><td>{freq}</td><td>{alloc}</td>'
        html += f'<td>{r["annual_return"]*100:.2f}%</td><td>{r["sharpe"]:.3f}</td>'
        html += f'<td style="color:#e53e3e">{r["max_drawdown"]*100:.2f}%</td>'
        html += f'<td>{r["calmar"]:.3f}</td><td>{r["win_rate"]*100:.1f}%</td><td>{r["n_trades"]}</td></tr>'
    html += '</table></div>'
    for pk, pl in [("MF", "MultiFactorSelector"), ("TV", "TrendVolatilityTiming")]:
        html += f'<div class="section"><h2>{pl}</h2><table>'
        html += '<tr><th>调频</th><th>分配器</th><th>年化%</th><th>Sharpe</th><th>回撤%</th><th>Calmar</th><th>胜率%</th><th>换手</th></tr>'
        for r in sorted([x for x in all_results if x["config_name"].startswith(pk)], key=lambda x: x["sharpe"], reverse=True):
            parts = r["config_name"].split("_")
            cls = "best" if r["sharpe"] == max(x["sharpe"] for x in all_results if x["config_name"].startswith(pk)) else ""
            html += f'<tr class="{cls}"><td>{"周频(3d)" if parts[1]=="W" else "高频(1d)"}</td>'
            html += f'<td>{"迟滞" if parts[2]=="H" else "原始"}</td>'
            html += f'<td>{r["annual_return"]*100:.2f}%</td><td>{r["sharpe"]:.3f}</td>'
            html += f'<td style="color:#e53e3e">{r["max_drawdown"]*100:.2f}%</td>'
            html += f'<td>{r["calmar"]:.3f}</td><td>{r["win_rate"]*100:.1f}%</td><td>{r["n_trades"]}</td></tr>'
        html += '</table></div>'
    html += '<div class="section"><h2>参数</h2><ul style="font-size:13px;color:#4a5568;padding-left:20px">'
    html += '<li>MultiFactorSelector: V1权重, 截面Z-Score+Winsorize(1%)</li>'
    html += '<li>TrendVolatilityTiming: MACD+动量5/20+RSI打分; vol>5%→卖出</li>'
    html += '<li>风险平价: 20日波动率倒数加权+5日最低持仓</li>'
    html += '<li>迟滞: 大仓10%/调整2%/保留70%</li>'
    html += '<li>报告: ' + datetime.now().strftime("%Y-%m-%d %H:%M") + '</li>'
    html += '</ul></div></div></body></html>'
    rp = os.path.join(OUT_DIR, "cross_compare_report.html")
    with open(rp, 'w') as f:
        f.write(html)
    logger.info(f"报告: {rp}")


def main():
    vals_3d, fwd_rets, tickers, dates_arr, all_dates, data_mask, factor_df, _, n_dates, n_symbols = load_data()

    v1wp = os.path.join(PROJ_ROOT, 'daily', RUN_DAY, 'v2', 'v1_reference', 'ga_results.json')
    mf_sig = precompute_mf_signal(vals_3d, ALL_FACTOR_NAMES, v1wp)
    tv_sig = precompute_tv_signal(vals_3d, ALL_FACTOR_NAMES)

    configs = [
        ("MF_W_RP", 3, False), ("MF_W_H", 3, True),
        ("MF_D_RP", 1, False), ("MF_D_H", 1, True),
        ("TV_W_RP", 3, False), ("TV_W_H", 3, True),
        ("TV_D_RP", 1, False), ("TV_D_H", 1, True),
    ]

    results = []
    for ck, rf, uh in configs:
        sig = mf_sig if ck.startswith("MF") else tv_sig
        logger.info(f"\n回测: {ck}")
        results.append(run_bt(sig, fwd_rets, data_mask, ck, rf, uh, top_n=40))
        gc.collect()

    bt_out = {"version": "v3", "results": results, "timestamp": datetime.now().isoformat()}
    with open(os.path.join(OUT_DIR, "results.json"), 'w') as f:
        json.dump(bt_out, f, indent=2)

    print(f"\n{'='*90}")
    print(f"{'排名':<4} {'信号源':<22} {'调频':<6} {'分配器':<8} {'年化%':<8} {'Sharpe':<8} {'回撤%':<8} {'Calmar':<8} {'胜率':<6}")
    print('-'*90)
    for i, r in enumerate(sorted(results, key=lambda x: x["sharpe"], reverse=True), 1):
        p = r["config_name"].split("_")
        sig = "MultiFactor" if p[0]=="MF" else "TrendVolTiming"
        print(f"{i:<4} {sig:<22} {'周频' if p[1]=='W' else '高频':<6} {'迟滞' if p[2]=='H' else '原始':<8} {r['annual_return']*100:>6.2f}% {r['sharpe']:>7.3f} {r['max_drawdown']*100:>6.2f}% {r['calmar']:>7.3f} {r['win_rate']*100:>5.1f}%")
    print('='*90)

    generate_report(results)


if __name__ == "__main__":
    main()
