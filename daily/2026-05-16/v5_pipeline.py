"""
V5 — Type A系列实验 (单变量扫描)
A01~A21: 选股器/择时器/分配器/频率/权重 五个维度的单变量扫描
"""
import os, sys, json, logging, gc
from datetime import datetime
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from core.database import Database
from core.optimization import VectorizedEvaluator
from core.positioners import RPPortfolioWeights
from core.screening.impl.factor_rank import FactorRankSelector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("v5")

TX_COST_RATE = 0.0012
V1_FACTOR_NAMES = ['a27','a30','a31','a41','a42','a64','a69','a8','a80','a85',
    'a88','a91','a97','a98','a99','ff_mkt','gtja103','gtja104','gtja105',
    'gtja108','gtja113','gtja117','gtja12','gtja120','gtja121','gtja123',
    'gtja127','gtja13','gtja139','gtja141','gtja142','gtja144','gtja148',
    'gtja164','gtja168','gtja171','gtja176','gtja185','gtja34','gtja49',
    'gtja62','gtja76','gtja83','gtja85','gtja90','gtja91','gtja99',
    'returns','rsi_14','volatility_20']
TV_EXTRA = ['macd','macd_signal','momentum_5','momentum_20','volume_ratio','boll_position','beta_20']
ALL_FACTOR_NAMES = list(set(V1_FACTOR_NAMES + TV_EXTRA))

VERSION_DIR = os.path.join(os.path.dirname(__file__), "v5_results")
os.makedirs(VERSION_DIR, exist_ok=True)

def load_and_prepare():
    db = Database()
    factor_df = db.get_factors(start_date="2018-01-01", end_date="2026-04-30",
                               factor_names=ALL_FACTOR_NAMES, with_close=True)
    factor_df['date'] = pd.to_datetime(factor_df['date'])
    all_dates = sorted(factor_df['date'].unique())
    tickers = db.get_symbols()['symbol'].tolist()
    n_dates, n_sym = len(all_dates), len(tickers)
    t2i = {t:i for i,t in enumerate(tickers)}
    d2i = {d:i for i,d in enumerate(all_dates)}
    n_factors = len(ALL_FACTOR_NAMES)
    factor_cols_in_df = [c for c in ALL_FACTOR_NAMES if c in factor_df.columns]

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
    return z_3d, fwd, dm, tickers, factor_df, all_dates, ALL_FACTOR_NAMES, n_dates, n_sym

def load_v1_w():
    p = os.path.join(os.path.dirname(__file__), '..', '2026-05-13', 'v2', 'v1_reference', 'ga_results.json')
    if os.path.exists(p):
        with open(p) as f:
            for item in json.load(f):
                if 'L1中_80代' in item['label']:
                    return item['configs'][0]['weights']
    return {}

def run_bt(signal, fwd_rets, dm, name="", rf=3, tn=40):
    ev = VectorizedEvaluator(tx_cost_rate=TX_COST_RATE,
        portfolio_builder=RPPortfolioWeights(top_n=tn, min_hold_days=5))
    w = np.ones(1, dtype=np.float32)
    r = ev.evaluate(w, signal.reshape(*signal.shape, 1), fwd_rets, dm, rebal_freq=rf, top_n=tn)
    logger.info(f"  [{name}] 年化={r.annual_return*100:.2f}% Sharpe={r.sharpe:.3f} 回撤={r.max_drawdown*100:.2f}% Calmar={r.calmar:.3f}")
    return {"name": name, "annual_return": round(r.annual_return,4), "sharpe": round(r.sharpe,4),
            "max_drawdown": round(r.max_drawdown,4), "calmar": round(r.calmar,4),
            "win_rate": round(r.win_rate,4), "n_trades": r.n_trades}


def make_mf_signal(z_3d, fnames):
    v1_w = load_v1_w()
    wv = np.zeros(z_3d.shape[2], dtype=np.float32)
    for fi, fc in enumerate(fnames):
        if fc in v1_w: wv[fi] = float(v1_w[fc])
    s = np.sum(np.abs(wv))
    if s > 0: wv /= s
    return np.nan_to_num(np.tensordot(z_3d, wv, axes=(2,0)), nan=-1e10, neginf=-1e10)

def make_trend_signal(z_3d, fnames):
    fi_map = {fn:i for i,fn in enumerate(fnames)}
    n_d, n_s = z_3d.shape[:2]
    sig = np.full((n_d, n_s), 0.5, dtype=np.float32)
    im, ims = fi_map.get('macd'), fi_map.get('macd_signal')
    im5, im20 = fi_map.get('momentum_5'), fi_map.get('momentum_20')
    ir = fi_map.get('rsi_14')
    for d in range(n_d):
        s = []
        if im is not None and ims is not None:
            s.append(np.where(z_3d[d,:,im] > z_3d[d,:,ims], 1.0, 0.0))
        if im5 is not None and im20 is not None:
            m5, m20 = z_3d[d,:,im5], z_3d[d,:,im20]
            s.append(np.where((m5>0)&(m5>m20), 1.0, np.where(m5<0, 0.0, 0.5)))
        if ir is not None:
            rv = z_3d[d,:,ir]
            s.append(np.where(rv>70, 0.0, np.where(rv>=50, 1.0, np.where(rv>=30, 0.5, 0.0))))
        if s: sig[d] = np.mean(s, axis=0)
    return sig

def make_tv_signal(z_3d, fnames):
    sig = make_trend_signal(z_3d, fnames)
    fi_map = {fn:i for i,fn in enumerate(fnames)}
    iv = fi_map.get('volatility_20')
    if iv is not None:
        sig[z_3d[:,:,iv] > 0.05] = -1.0
    return sig

def make_factorrank_signal(z_3d, fnames, factor_name='momentum_20', ascending=False):
    fi = fnames.index(factor_name) if factor_name in fnames else 0
    return np.nan_to_num(z_3d[:,:,fi] * (1 if not ascending else -1), nan=-1e10, neginf=-1e10)

def make_mr_signal(z_3d, fnames):
    """MarketRegimeTiming仓位系数"""
    n_d, n_s = z_3d.shape[:2]
    ratio = np.full(n_d, 0.6, dtype=np.float32)
    return ratio.reshape(-1, 1)

def run_a_series():
    logger.info("=" * 60)
    logger.info("V5 — Type A 单变量扫描 (21个实验)")
    logger.info("=" * 60)

    z3d, fwd, dm, tickers, factor_df, dates, fnames, nd, ns = load_and_prepare()
    fi_map = {fn:i for i,fn in enumerate(fnames)}
    res = []

    # ── 预计算信号 ──
    mf = make_mf_signal(z3d, fnames)
    trend = make_trend_signal(z3d, fnames)
    tv = make_tv_signal(z3d, fnames)
    m20 = make_factorrank_signal(z3d, fnames, 'momentum_20', False)
    vol20_inv = make_factorrank_signal(z3d, fnames, 'volatility_20', True)  # 低波动
    rsi = make_factorrank_signal(z3d, fnames, 'rsi_14', False)

    # ── A01-A06: 选股器扫描 ──
    logger.info("\nA01-A06 选股器扫描 (固定: 无择时+RP+3d)")
    res.append(run_bt(mf, fwd, dm, "A01_MF", rf=3))
    res.append(run_bt(m20, fwd, dm, "A02_FactorRank(momentum20)", rf=3))
    res.append(run_bt(vol20_inv, fwd, dm, "A02b_FactorRank(vol20_asc)", rf=3))
    res.append(run_bt(rsi, fwd, dm, "A02c_FactorRank(rsi14)", rf=3))

    # ── TrendBreakout 信号 ──
    logger.info("  构建TrendBreakout信号...")
    tb_sig = np.full((nd, ns), -np.inf, dtype=np.float32)
    for d in range(nd):
        m5, m20v, m60 = np.zeros(ns), np.zeros(ns), np.zeros(ns)
        if d >= 4:
            close_5 = np.full(ns, np.nan, dtype=np.float32)
            close_20 = np.full(ns, np.nan, dtype=np.float32)
            for s in range(ns):
                pass  # 简化版: 使用已有的因子
        if 'a64' in fnames:
            tb_sig[d] = z3d[d,:,fi_map['a64']]  # 动量20日作为突破代理
    tb_sig = np.nan_to_num(tb_sig, nan=-1e10, neginf=-1e10)
    res.append(run_bt(tb_sig, fwd, dm, "A04_TrendBreakout(proxy)", rf=3))

    # ── A07-A12: 择时器扫描 ──
    logger.info("\nA07-A12 择时器扫描 (固定: MF+RP+3d)")
    # A07: 无择时 → 就是A01
    # A08: TrendTiming择时 → 仓位系数 = trend信号的平均值
    trend_pos = np.clip(np.nanmean(trend, axis=1), 0.1, 1.0)
    mf_trend = mf * trend_pos[:, np.newaxis]
    res.append(run_bt(mf_trend, fwd, dm, "A08_MF+TrendTiming", rf=3))

    # A09: TVTiming择时
    tv_pos = np.clip(np.nanmean(tv, axis=1) * 1.5, 0.1, 1.0)
    mf_tv = mf * tv_pos[:, np.newaxis]
    res.append(run_bt(mf_tv, fwd, dm, "A09_MF+TVTiming", rf=3))

    # A10: VolTiming(只出SELL)
    iv = fi_map.get('volatility_20')
    if iv is not None:
        vol_sell_ratio = np.mean(z3d[:,:,iv] > 0.05, axis=1)  # 高波动占比高→减仓
        vol_pos = np.clip(1.0 - vol_sell_ratio * 2, 0.2, 1.0)
        mf_vol = mf * vol_pos[:, np.newaxis]
        res.append(run_bt(mf_vol, fwd, dm, "A10_MF+VolTiming", rf=3))

    # A11: MR择时 → 大盘仓位
    mr_pos = np.full(nd, 0.6, dtype=np.float32)
    if 'volatility_20' in fnames:
        mkt_vol = np.nanmean(z3d[:,:,fi_map['volatility_20']], axis=1)
        mr_pos[mkt_vol < 0.04] = 1.0
        mr_pos[mkt_vol > 0.08] = 0.3
    mf_mr = mf * mr_pos[:, np.newaxis]
    res.append(run_bt(mf_mr, fwd, dm, "A11_MF+MRTiming", rf=3))

    # ── A13-A19: 分配器扫描 (用不同的分配器参数) ──
    logger.info("\nA13-A19 分配器扫描 (固定: MF+无择时+3d)")
    # A14: EqualWeight = top_n只等权
    def bt_eq(signal, name):
        ev = VectorizedEvaluator(tx_cost_rate=TX_COST_RATE,
            portfolio_builder=RPPortfolioWeights(top_n=40, min_hold_days=5))
        w = np.ones(1, dtype=np.float32)
        r = ev.evaluate(w, signal.reshape(*signal.shape, 1), fwd, dm, rebal_freq=3, top_n=40)
        logger.info(f"  [{name}] 年化={r.annual_return*100:.2f}% Sharpe={r.sharpe:.3f} 回撤={r.max_drawdown*100:.2f}%")
        return {"name": name, "annual_return": round(r.annual_return,4), "sharpe": round(r.sharpe,4),
                "max_drawdown": round(r.max_drawdown,4), "calmar": round(r.calmar,4),
                "win_rate": round(r.win_rate,4), "n_trades": r.n_trades}
    res.append(bt_eq(mf, "A13_MF_EqualWeight(proxy)"))

    # ── A16-A19: 频率扫描 ──
    logger.info("\nA16-A19 频率扫描 (固定: MF+无择时+RP)")
    for fd, label in [(1,"A01b_MF_D1"), (2,"A16_MF_D2"), (3,"A01_MF_D3"), (5,"A17_MF_D5"), (10,"A18_MF_D10"), (21,"A19_MF_D21")]:
        res.append(run_bt(mf, fwd, dm, label, rf=fd))

    # ── A20-A21: 权重扫描 ──
    logger.info("\nA20-A21 权重扫描")
    # A20: 等权 (50因子各1/50)
    ew = np.ones(len(fnames), dtype=np.float32) / len(fnames)
    mf_ew = np.nan_to_num(np.tensordot(z3d, ew, axes=(2,0)), nan=-1e10, neginf=-1e10)
    res.append(run_bt(mf_ew, fwd, dm, "A20_MF_EqualWeight(50f)", rf=3))

    # A21: 单因子 momentum_20
    res.append(run_bt(m20, fwd, dm, "A21_FactorRank(momentum20)", rf=3))

    # ── 保存 ──
    with open(os.path.join(VERSION_DIR, "results.json"), 'w') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*90}")
    print(f"{'实验':<22} {'年化%':<8} {'Sharpe':<8} {'回撤%':<8} {'Calmar':<8} {'胜率':<6} {'换手':<5}")
    print('-'*90)
    for r in sorted(res, key=lambda x: x['sharpe'], reverse=True):
        print(f"{r['name']:<22} {r['annual_return']*100:>6.2f}% {r['sharpe']:>7.3f} {r['max_drawdown']*100:>6.2f}% {r['calmar']:>7.3f} {r['win_rate']*100:>5.1f}% {r['n_trades']:>4}")
    print('='*90)

    # 标记达标策略
    for r in res:
        if r['max_drawdown'] < 0.20 and r['annual_return'] > 0.10:
            logger.info(f"🏆 达标: {r['name']} 年化={r['annual_return']*100:.2f}% 回撤={r['max_drawdown']*100:.2f}%")

if __name__ == "__main__":
    run_a_series()
