"""
通用实验管道 — Type A/B/C/D/E 全部实验的统一入口。

一个脚本跑任意实验组合，输出结果并更新 SUMMARY.md。
"""
import os, sys, json, logging, gc
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from core.database import Database
from core.optimization import VectorizedEvaluator, EvalResult
from core.positioners import RPPortfolioWeights

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("experiment")

TX_COST_RATE = 0.0012
V1_FACTOR_NAMES = ['a27','a30','a31','a41','a42','a64','a69','a8','a80','a85',
    'a88','a91','a97','a98','a99','ff_mkt','gtja103','gtja104','gtja105',
    'gtja108','gtja113','gtja117','gtja12','gtja120','gtja121','gtja123',
    'gtja127','gtja13','gtja139','gtja141','gtja142','gtja144','gtja148',
    'gtja164','gtja168','gtja171','gtja176','gtja185','gtja34','gtja49',
    'gtja62','gtja76','gtja83','gtja85','gtja90','gtja91','gtja99',
    'returns','rsi_14','volatility_20']

TV_EXTRA = ['macd','macd_signal','momentum_5','momentum_20','volume_ratio','boll_position']
ALL_FACTOR_NAMES = list(set(V1_FACTOR_NAMES + TV_EXTRA))


def load_data():
    """加载全量因子数据，返回截面Z-Score矩阵+前向收益+DataMask+因子列名单。"""
    db = Database()
    factor_df = db.get_factors(start_date="2018-01-01", end_date="2026-04-30",
                               factor_names=ALL_FACTOR_NAMES, with_close=True)
    factor_df['date'] = pd.to_datetime(factor_df['date'])
    all_dates = sorted(factor_df['date'].unique())
    tickers = db.get_symbols()['symbol'].tolist()
    n_dates, n_sym, n_factors = len(all_dates), len(tickers), len(ALL_FACTOR_NAMES)
    t2i = {t:i for i,t in enumerate(tickers)}
    d2i = {d:i for i,d in enumerate(all_dates)}

    vals_3d = np.full((n_dates, n_sym, n_factors), np.nan, dtype=np.float32)
    dm = np.zeros((n_dates, n_sym), dtype=bool)
    cl = np.zeros((n_dates, n_sym), dtype=np.float32)

    di = np.array([d2i[d] for d in factor_df['date']], dtype=np.int32)
    si = np.array([t2i.get(s,-1) for s in factor_df['symbol']], dtype=np.int32)
    v = si >= 0; di, si = di[v], si[v]

    for fi, fc in enumerate(ALL_FACTOR_NAMES):
        if fc in factor_df.columns:
            vals_3d[di, si, fi] = factor_df[fc].values[v].astype(np.float32)
    cl[di, si] = factor_df['close'].values[v].astype(np.float32)
    dm[di, si] = True
    np.nan_to_num(vals_3d, nan=0.0, copy=False); np.nan_to_num(cl, nan=0.0, copy=False)

    fwd = np.zeros((n_dates, n_sym), dtype=np.float32)
    for d in range(n_dates-1):
        b = (cl[d] > 1e-10) & (cl[d+1] > 1e-10)
        fwd[d, b] = (cl[d+1, b] - cl[d, b]) / cl[d, b]

    z_3d = np.zeros_like(vals_3d)
    for fi in range(n_factors):
        a = vals_3d[:,:,fi]
        for d in range(n_dates):
            r = a[d,:]; nz = r[r!=0]
            if len(nz) > 1:
                lo, hi = np.quantile(nz, [0.01, 0.99])
                c = np.clip(r, lo, hi)
                mu, sd = np.mean(c), np.std(c)
                z_3d[d,:,fi] = (c - mu) / sd if sd > 1e-10 else 0.0

    logger.info(f"数据: {n_dates}天×{n_sym}只×{n_factors}因子")
    return z_3d, fwd, dm, tickers, ALL_FACTOR_NAMES, factor_df, all_dates


def load_v1_weights():
    v1wp = os.path.join(os.path.dirname(__file__), '..', '..', 'daily',
                        '2026-05-13', 'v2', 'v1_reference', 'ga_results.json')
    if not os.path.exists(v1wp):
        logger.warning("V1权重文件不存在，返回空")
        return {}
    with open(v1wp) as f:
        for item in json.load(f):
            if 'L1中_80代' in item['label']:
                return item['configs'][0]['weights']
    return {}


def build_mf_signal(z_3d, v1_w, factor_names):
    """构建MultiFactorSelector加权信号 (截面Z-Score+Winsorize+权重Sum)。"""
    wv = np.zeros(z_3d.shape[2], dtype=np.float32)
    for fi, fc in enumerate(factor_names):
        if fc in v1_w:
            wv[fi] = float(v1_w[fc])
    s = np.sum(np.abs(wv))
    if s > 0: wv /= s
    sig = np.tensordot(z_3d, wv, axes=(2, 0))
    return np.nan_to_num(sig, nan=-1e10, neginf=-1e10)


def build_trend_signal(z_3d, factor_names):
    """TrendTiming评分：MACD+动量5/20+RSI三因素均值，0~1。"""
    fi_map = {fn:i for i,fn in enumerate(factor_names)}
    n_d, n_s = z_3d.shape[:2]
    sig = np.full((n_d, n_s), 0.5, dtype=np.float32)
    im, ims = fi_map.get('macd'), fi_map.get('macd_signal')
    im5, im20 = fi_map.get('momentum_5'), fi_map.get('momentum_20')
    ir = fi_map.get('rsi_14')
    for d in range(n_d):
        scores = []
        if im is not None and ims is not None:
            scores.append(np.where(z_3d[d,:,im] > z_3d[d,:,ims], 1.0, 0.0))
        if im5 is not None and im20 is not None:
            m5, m20 = z_3d[d,:,im5], z_3d[d,:,im20]
            s = np.where((m5 > 0) & (m5 > m20), 1.0, np.where(m5 < 0, 0.0, 0.5))
            scores.append(s)
        if ir is not None:
            r = z_3d[d,:,ir]
            s = np.where(r > 70, 0.0, np.where(r >= 50, 1.0, np.where(r >= 30, 0.5, 0.0)))
            scores.append(s)
        if scores:
            sig[d] = np.mean(scores, axis=0)
    return sig


def run_experiment(name, signal, fwd_rets, data_mask, rebal_freq=3, top_n=40):
    """运行单次实验，返回结果dict。"""
    ev = VectorizedEvaluator(tx_cost_rate=TX_COST_RATE,
        portfolio_builder=RPPortfolioWeights(top_n=top_n, min_hold_days=5))
    w = np.ones(1, dtype=np.float32)  # dummy, signal已经是综合得分
    result = ev.evaluate(w, signal.reshape(*signal.shape, 1), fwd_rets, data_mask,
                         rebal_freq=rebal_freq, top_n=top_n)
    logger.info(f"[{name}] 年化={result.annual_return*100:.2f}% Sharpe={result.sharpe:.3f} "
                f"回撤={result.max_drawdown*100:.2f}% Calmar={result.calmar:.3f} 胜率={result.win_rate*100:.1f}%")
    return {
        "name": name, "annual_return": round(result.annual_return, 4),
        "sharpe": round(result.sharpe, 4), "max_drawdown": round(result.max_drawdown, 4),
        "calmar": round(result.calmar, 4), "win_rate": round(result.win_rate, 4),
        "n_trades": result.n_trades,
    }


def update_summary(results, version_dir):
    """将实验结果追加到 SUMMARY.md 和 results.json。"""
    rp = os.path.join(version_dir, "results.json")
    if os.path.exists(rp):
        with open(rp) as f:
            existing = json.load(f)
    else:
        existing = []
    existing.extend(results)
    with open(rp, 'w') as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    summary_path = os.path.join(os.path.dirname(version_dir), "results", "SUMMARY.md")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            content = f.read()
    else:
        content = ""

    for r in results:
        line = f"| 🟡 | {r['name']} | — | — | — | — | {r['annual_return']*100:.2f} | {r['sharpe']:.3f} | {r['max_drawdown']*100:.2f} | {r['calmar']:.3f} | — | 0516 |\n"
        content += line

    with open(summary_path, 'w') as f:
        f.write(content)


def main():
    """入口：加载数据后，执行指定实验。"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", nargs="+", default=["A01"])
    args = parser.parse_args()

    z3d, fwd, dm, tickers, fnames, factor_df, dates = load_data()
    v1_w = load_v1_weights()
    mf_sig = build_mf_signal(z3d, v1_w, fnames)
    trend_sig = build_trend_signal(z3d, fnames)

    version_dir = os.path.join(os.path.dirname(__file__), "v_a")
    os.makedirs(version_dir, exist_ok=True)

    results = {}
    for exp in args.experiments:
        if exp == "A01":
            r = run_experiment("A01_MF_无择时_RP_3d", mf_sig, fwd, dm, 3)
        elif exp == "A07_TrendTiming":
            r = run_experiment("A07_MF_TrendTiming_RP_3d", mf_sig, fwd, dm, 3)
        else:
            logger.warning(f"未知实验: {exp}")
            continue
        results[exp] = r
        gc.collect()

    update_summary(list(results.values()), version_dir)
    print("\n" + "=" * 60)
    print("实验结果:")
    for k, v in results.items():
        print(f"  {k}: 年化={v['annual_return']*100:.2f}% Sharpe={v['sharpe']:.3f} 回撤={v['max_drawdown']*100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
