"""
X9 — 17策略全窗口排名

Task1: 用StrategyPipeline跑全部17+策略的7窗口分析,输出排名矩阵
Task2: BacktestEngine vs bt() 交叉验证 (MF_D10, 2024Q1)
"""
import os, sys, json, logging, gc
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from core.database import Database
from core.positioners import RPPortfolioWeights
from core.strategies.pipeline import StrategyPipeline
from core.execution.impl.backtest import BacktestEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("X9")

TX = 0.0012
TOP_N = 40
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "x9_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

FACTORS = [
    'a27','a30','a31','a41','a42','a64','a69','a8','a80','a85',
    'a88','a91','a97','a98','a99','ff_mkt','gtja103','gtja104','gtja105',
    'gtja108','gtja113','gtja117','gtja12','gtja120','gtja121','gtja123',
    'gtja127','gtja13','gtja139','gtja141','gtja142','gtja144','gtja148',
    'gtja164','gtja168','gtja171','gtja176','gtja185','gtja34','gtja49',
    'gtja62','gtja76','gtja83','gtja85','gtja90','gtja91','gtja99',
    'returns','rsi_14','volatility_20','macd','macd_signal','momentum_5',
    'momentum_20','volume_ratio','boll_position','beta_20',
]

X9_WINDOWS = [
    ("W1_2019", "2019-01-02", "2019-12-31"),
    ("W2_2020", "2020-01-02", "2020-12-31"),
    ("W3_2021", "2021-01-04", "2021-12-31"),
    ("W4_2022", "2022-01-04", "2022-12-30"),
    ("W5_2023", "2023-01-03", "2023-12-29"),
    ("W6_2024", "2024-01-02", "2024-12-31"),
    ("W7_2025", "2025-01-02", "2026-04-30"),
]

# ════════════════════════════════════════════
# 数据加载
# ════════════════════════════════════════════
def load():
    db = Database()
    df = db.get_factors(start_date="2018-01-01", end_date="2026-04-30",
                        factor_names=FACTORS, with_close=True)
    df['date'] = pd.to_datetime(df['date'])
    ds = sorted(df['date'].unique())
    tks = db.get_symbols()['symbol'].tolist()
    nd, ns, nf = len(ds), len(tks), len(FACTORS)
    t2i = {t: i for i, t in enumerate(tks)}
    d2i = {d: i for i, d in enumerate(ds)}
    v3 = np.full((nd, ns, nf), np.nan, dtype=np.float32)
    dm = np.zeros((nd, ns), dtype=bool)
    cl = np.zeros((nd, ns), dtype=np.float32)
    di = np.array([d2i[d] for d in df['date']], dtype=np.int32)
    si = np.array([t2i.get(s, -1) for s in df['symbol']], dtype=np.int32)
    v = si >= 0; di, si = di[v], si[v]
    for fi, fc in enumerate(FACTORS):
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
    fi = {fn: i for i, fn in enumerate(FACTORS)}
    logger.info(f"数据: {nd}天 × {ns}只 × {nf}因子")
    return z3, fwd, dm, tks, FACTORS, nd, ns, ds, fi

# ════════════════════════════════════════════
# 权重加载
# ════════════════════════════════════════════
def load_v1_weights():
    p = os.path.join(os.path.dirname(__file__), '..', '2026-05-13',
                     'v2', 'v1_reference', 'ga_results.json')
    if os.path.exists(p):
        with open(p) as f:
            for it in json.load(f):
                if 'L1中_80代' in it['label']:
                    return it['configs'][0]['weights']
    return {}

def load_x5_weights(rf_target):
    p = os.path.join(os.path.dirname(__file__), 'x4_x5_results', 'x5_results.json')
    if not os.path.exists(p):
        logger.warning(f"x5_results.json not found: {p}")
        return {}
    with open(p) as f:
        data = json.load(f)
    for entry in data.get('ga_results', []):
        if entry.get('rebal_freq') == rf_target:
            return entry.get('weights', {})
    return {}

def build_weight_vector(weight_dict, fnames):
    wv = np.zeros(len(fnames), dtype=np.float32)
    for i, fc in enumerate(fnames):
        if fc in weight_dict:
            wv[i] = float(weight_dict[fc])
    s = np.sum(np.abs(wv))
    if s > 0:
        wv /= s
    return wv

# ════════════════════════════════════════════
# 信号构建
# ════════════════════════════════════════════
def build_mf_signal(z3, v1_weights, fnames):
    wv = build_weight_vector(v1_weights, fnames)
    return np.nan_to_num(np.tensordot(z3, wv, axes=(2, 0)), nan=-1e10, neginf=-1e10)

def build_ga_signal(z3, ga_weights, fnames):
    wv = build_weight_vector(ga_weights, fnames)
    return np.nan_to_num(np.tensordot(z3, wv, axes=(2, 0)), nan=-1e10, neginf=-1e10)

def build_chip_signal(z3, fi, nd, ns, mom_weight=0.3):
    sig = np.full((nd, ns), -np.inf, dtype=np.float32)
    vol20_idx = fi.get('volatility_20')
    m20_idx = fi.get('momentum_20')
    for d in range(nd):
        s = np.zeros(ns)
        if vol20_idx is not None:
            s += np.where(z3[d, :, vol20_idx] < -0.3, 1.0, 0.0) * 0.5
        if m20_idx is not None:
            s += np.where(np.abs(z3[d, :, m20_idx]) < 0.3, 1.0, 0.0) * mom_weight
        sig[d] = np.nan_to_num(s, nan=-1e10)
    return sig

def build_osr_signal(z3, fi, nd, ns):
    sig = np.full((nd, ns), -np.inf, dtype=np.float32)
    rsi_idx = fi.get('rsi_14')
    m5_idx = fi.get('momentum_5')
    for d in range(nd):
        s = np.zeros(ns)
        if rsi_idx is not None:
            s += np.where(z3[d, :, rsi_idx] < -0.5, 1.0, 0.0) * -0.5
        if m5_idx is not None:
            s += np.where(z3[d, :, m5_idx] > 0.3, 1.0, 0.0) * 0.5
        sig[d] = np.nan_to_num(s, nan=-1e10)
    return sig

# ════════════════════════════════════════════
# 择时信号
# ════════════════════════════════════════════
def build_timing_signals(z3, fi, nd, ns):
    im, ims = fi.get('macd'), fi.get('macd_signal')
    im5, im20 = fi.get('momentum_5'), fi.get('momentum_20')
    ir = fi.get('rsi_14'); iv = fi.get('volatility_20')

    tsig = np.full((nd, ns), 0.5, dtype=np.float32)
    for d in range(nd):
        s_l = []
        if im is not None and ims is not None:
            s_l.append(np.where(z3[d, :, im] > z3[d, :, ims], 1.0, 0.0))
        if im5 is not None and im20 is not None:
            m5v, m20v = z3[d, :, im5], z3[d, :, im20]
            s_l.append(np.where((m5v > 0) & (m5v > m20v), 1.0,
                                np.where(m5v < 0, 0.0, 0.5)))
        if ir is not None:
            rv = z3[d, :, ir]
            s_l.append(np.where(rv > 70, 0.0,
                                np.where(rv >= 50, 1.0,
                                         np.where(rv >= 30, 0.5, 0.0))))
        if s_l:
            tsig[d] = np.mean(s_l, axis=0)

    trend_pos = np.clip(np.mean(tsig >= 0.6, axis=1) * 2.0, 0.1, 1.0)

    tv_sig = tsig.copy()
    if iv is not None:
        tv_sig[z3[:, :, iv] > 0.05] = -1.0
    tv_pos = np.clip(np.mean(tv_sig >= 0.6, axis=1) * 2.0, 0.1, 1.0)

    vol_pos = np.ones(nd, dtype=np.float32)
    if iv is not None:
        hv_ratio = np.mean(z3[:, :, iv] > 0.05, axis=1)
        vol_pos = np.clip(1.0 - hv_ratio, 0.2, 1.0)

    return trend_pos, tv_pos, vol_pos

def build_c01_signal(z3, fi, nd, ns, mf):
    im, ims = fi.get('macd'), fi.get('macd_signal')
    im5, im20 = fi.get('momentum_5'), fi.get('momentum_20')
    ir = fi.get('rsi_14')
    sig = mf.copy().astype(np.float32)
    for d in range(nd):
        s_l = []
        if im is not None and ims is not None:
            s_l.append(np.where(z3[d, :, im] > z3[d, :, ims], 1.0, 0.0))
        if im5 is not None and im20 is not None:
            m5v, m20v = z3[d, :, im5], z3[d, :, im20]
            s_l.append(np.where((m5v > 0) & (m5v > m20v), 1.0,
                                np.where(m5v < 0, 0.0, 0.5)))
        if ir is not None:
            rv = z3[d, :, ir]
            s_l.append(np.where(rv > 70, 0.0,
                                np.where(rv >= 50, 1.0,
                                         np.where(rv >= 30, 0.5, 0.0))))
        ts = np.mean(s_l, axis=0) if s_l else np.full(z3.shape[1], 0.5)
        sig[d] = np.where(ts >= 0.6, mf[d], np.full(ns, -np.inf))
    return sig

# ════════════════════════════════════════════
# TimingPipeline — 支持择时的子类
# ════════════════════════════════════════════
class TimingPipeline(StrategyPipeline):
    def __init__(self, pos_ratio=None, **kwargs):
        super().__init__(**kwargs)
        self._pos_ratio = pos_ratio

    def _backtest_rp(self, sig, fwd, dm, nd, ns):
        alloc = RPPortfolioWeights(top_n=self.top_n, min_hold_days=self.min_hold_days)
        pw = np.zeros(ns, dtype=np.float32)
        hs = np.full(ns, -1, dtype=np.int32)
        rh = 0; dr = np.zeros(nd, dtype=np.float64); nt = 0
        for i in range(1, nd):
            rebal = (i % self.rebal_freq == 0)
            txc = 0.0
            if rebal:
                nw = alloc.allocate(sig[i], fwd, i, pw, hs, rh)
                to = float(np.sum(np.abs(nw - pw)))
                txc = 0.5 * to * self.tx_cost
                if to > 0.01: nt += 1
                pw = nw
                for j in range(ns):
                    if nw[j] > 0 and hs[j] < 0: hs[j] = rh + 1
            else:
                mk = dm[i] & (pw > 0)
                if np.any(mk):
                    p2 = pw[mk].copy() / float(np.sum(pw[mk]))
                    pw = np.zeros(ns, dtype=np.float32)
                    pw[mk] = p2
            pr = self._pos_ratio[i] if self._pos_ratio is not None else 1.0
            rt = pr * float(np.dot(pw, fwd[i]))
            dr[i] = 0.0 if (np.isnan(rt) or np.isinf(rt)) else rt - txc
            rh += 1
        return dr, nt

    def _backtest_covrp(self, sig, fwd, dm, nd, ns):
        pw = np.zeros(ns, dtype=np.float32)
        hs = np.full(ns, -1, dtype=np.int32)
        rh = 0; dr = np.zeros(nd, dtype=np.float64); nt = 0
        for i in range(1, nd):
            rebal = (i % self.rebal_freq == 0)
            txc = 0.0
            if rebal:
                si = np.argsort(-sig[i])[:self.top_n]
                if i >= 20:
                    seg = fwd[max(0, i - 20):i, :]
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
                    iv = np.ones(min(self.top_n, ns))
                nw = np.zeros(ns)
                sidx = si[:len(iv)]
                if len(sidx) > 0:
                    nw[sidx] = iv / np.sum(iv)
                to = float(np.sum(np.abs(nw - pw)))
                txc = 0.5 * to * self.tx_cost
                if to > 0.01: nt += 1
                pw = nw
                for j in range(ns):
                    if nw[j] > 0 and hs[j] < 0: hs[j] = rh + 1
            else:
                mk = dm[i] & (pw > 0)
                if np.any(mk):
                    p2 = pw[mk].copy() / float(np.sum(pw[mk]))
                    pw = np.zeros(ns, dtype=np.float32)
                    pw[mk] = p2
            pr = self._pos_ratio[i] if self._pos_ratio is not None else 1.0
            rt = pr * float(np.dot(pw, fwd[i]))
            dr[i] = 0.0 if (np.isnan(rt) or np.isinf(rt)) else rt - txc
            rh += 1
        return dr, nt

def run_pipeline(sig, name, rf, pos_type='rp', pos_ratio=None,
                 z3=None, fwd=None, dm=None, tks=None, fnames=None,
                 nd=None, ns=None, ds=None, fi=None):
    PipeClass = TimingPipeline if pos_ratio is not None else StrategyPipeline
    pipe = PipeClass(
        signal_builder=None, name=name, rebal_freq=rf,
        top_n=TOP_N, min_hold_days=5, positioner_type=pos_type,
        tx_cost=TX, factor_names=fnames,
    )
    pipe.z3 = z3; pipe.fwd = fwd; pipe.dm = dm; pipe.tks = tks
    pipe.nd = nd; pipe.ns = ns; pipe.ds = ds; pipe.fi = fi
    pipe._data_loaded = True
    pipe.sig = sig; pipe._signal_built = True
    if pos_ratio is not None:
        pipe._pos_ratio = pos_ratio
    return pipe.window_analysis(windows=X9_WINDOWS)

def get_metrics_dict(metrics_list):
    d = {}
    for m in metrics_list:
        if m.window == "全区间":
            d["全区间"] = m
        elif m.window.startswith("W1"):
            d["W1"] = m
        elif m.window.startswith("W2"):
            d["W2"] = m
        elif m.window.startswith("W3"):
            d["W3"] = m
        elif m.window.startswith("W4"):
            d["W4"] = m
        elif m.window.startswith("W5"):
            d["W5"] = m
        elif m.window.startswith("W6"):
            d["W6"] = m
        elif m.window.startswith("W7"):
            d["W7"] = m
    return d

# ════════════════════════════════════════════
# Task2: BacktestEngine vs bt() 交叉验证
# ════════════════════════════════════════════
def cross_validate_engine(z3_all, fwd_all, dm_all, ds, fi, v1_weights, fnames, tks):
    logger.info("\n" + "=" * 60)
    logger.info("Task2: BacktestEngine vs bt() 交叉验证 (MF_D10, 2024Q1)")
    logger.info("=" * 60)

    start_str = "2024-01-02"
    end_str = "2024-03-29"

    mf_all = build_mf_signal(z3_all, v1_weights, fnames)

    sidx = next((i for i, d in enumerate(ds) if d >= pd.Timestamp(start_str)), 0)
    eidx = next((i for i, d in enumerate(ds) if d > pd.Timestamp(end_str)), len(ds))
    nd_sub = eidx - sidx
    sig_sub = mf_all[sidx:eidx]
    fwd_sub = fwd_all[sidx:eidx]
    dm_sub = dm_all[sidx:eidx]
    ds_sub = ds[sidx:eidx]

    # — bt()手写回测 —
    alloc_bt = RPPortfolioWeights(top_n=TOP_N, min_hold_days=5)
    pw = np.zeros(sig_sub.shape[1], dtype=np.float32)
    hs = np.full(sig_sub.shape[1], -1, dtype=np.int32)
    rh = 0; dr = np.zeros(nd_sub, dtype=np.float64); nt = 0
    for i in range(1, nd_sub):
        rebal = (i % 10 == 0)
        txc = 0.0
        if rebal:
            nw = alloc_bt.allocate(sig_sub[i], fwd_sub, i, pw, hs, rh)
            to = float(np.sum(np.abs(nw - pw)))
            txc = 0.5 * to * TX
            if to > 0.01: nt += 1
            pw = nw
            for j in range(sig_sub.shape[1]):
                if nw[j] > 0 and hs[j] < 0: hs[j] = rh + 1
        else:
            mk = dm_sub[i] & (pw > 0)
            if np.any(mk):
                p2 = pw[mk].copy() / float(np.sum(pw[mk]))
                pw = np.zeros(sig_sub.shape[1], dtype=np.float32)
                pw[mk] = p2
        rt = float(np.dot(pw, fwd_sub[i]))
        dr[i] = 0.0 if (np.isnan(rt) or np.isinf(rt)) else rt
        rh += 1

    eq = np.ones(nd_sub)
    for i in range(1, nd_sub):
        eq[i] = eq[i-1] * (1.0 + dr[i])
    tr = float(eq[-1] / eq[0] - 1.0)
    ny = nd_sub / 252.0
    ar = max(float((eq[-1]/eq[0]) ** (1.0/max(ny, 0.5)) - 1.0), -1.0)
    lr = np.log(eq[1:] / eq[:-1])
    lr = lr[~np.isnan(lr) & ~np.isinf(lr)]
    sp = float(np.mean(lr) / max(np.std(lr), 1e-10) * np.sqrt(252))
    cm = np.maximum.accumulate(eq)
    mdd = float(np.min((eq - cm) / cm))

    bt_res = {"method": "bt()手写回测", "interval": f"{start_str}~{end_str}",
              "total_return": round(tr, 6), "annual_return": round(ar, 6),
              "sharpe": round(sp, 4), "max_drawdown": round(mdd, 6),
              "n_trades": nt, "n_days": nd_sub}
    logger.info(f"bt()手写回测: 总收益={bt_res['total_return']*100:.4f}%  "
                f"年化={bt_res['annual_return']*100:.2f}%  "
                f"Sharpe={bt_res['sharpe']:.3f}  "
                f"回撤={bt_res['max_drawdown']*100:.2f}%  "
                f"交易={bt_res['n_trades']}次")

    # — BacktestEngine —
    try:
        from core.strategies.impl.mf_d10_rp import build_mf_d10_rp
        strategy = build_mf_d10_rp(top_n=TOP_N)
    except Exception as e:
        logger.error(f"build_mf_d10_rp 失败: {e}")
        return [bt_res]

    db = Database()
    factor_df = db.get_factors(start_date=start_str, end_date=end_str,
                                factor_names=fnames, with_close=True)
    if factor_df.empty:
        logger.error("BacktestEngine: 无数据")
        return [bt_res, {"method": "BacktestEngine", "error": "无数据"}]

    try:
        engine = BacktestEngine(initial_capital=1_000_000)
        report = engine.run(
            strategy=strategy,
            factor_data=factor_df,
            start_date=start_str,
            end_date=end_str,
            rebalance_freq='10d',
        )
        eng_res = {"method": "BacktestEngine", "interval": f"{start_str}~{end_str}",
                   "total_return": round(report.total_return, 6),
                   "annual_return": round(report.annualized_return, 6),
                   "sharpe": round(report.sharpe_ratio, 4),
                   "max_drawdown": round(report.max_drawdown, 6),
                   "n_trades": report.total_trades,
                   "final_value": round(report.final_value, 2)}
        logger.info(f"BacktestEngine:  总收益={eng_res['total_return']*100:.4f}%  "
                    f"年化={eng_res['annual_return']*100:.2f}%  "
                    f"Sharpe={eng_res['sharpe']:.3f}  "
                    f"回撤={eng_res['max_drawdown']*100:.2f}%  "
                    f"交易={eng_res['n_trades']}次")

        diff_ar = abs(bt_res['annual_return'] - eng_res['annual_return']) * 100
        diff_sp = abs(bt_res['sharpe'] - eng_res['sharpe'])
        eng_res["diff_annual_return_pct"] = round(diff_ar, 4)
        eng_res["diff_sharpe"] = round(diff_sp, 4)
        eng_res["verdict"] = "PASS" if (diff_ar < 5 and diff_sp < 0.5) else "DIFFER"
        logger.info(f"  年化差={diff_ar:.2f}%  Sharpe差={diff_sp:.4f}  "
                    f"{'✅一致' if eng_res['verdict']=='PASS' else '⚠️有差异'}")

        # 打印BacktestEngine内置报告
        print("\n--- BacktestEngine 内置报告 ---")
        print(report.pretty_print(top_positions=5))
        print("--- end ---\n")
    except Exception as e:
        logger.error(f"BacktestEngine运行失败: {e}", exc_info=True)
        eng_res = {"method": "BacktestEngine", "error": str(e)}

    with open(os.path.join(RESULTS_DIR, "cv_comparison.json"), 'w') as f:
        json.dump([bt_res, eng_res], f, indent=2, ensure_ascii=False)
    return [bt_res, eng_res]

# ════════════════════════════════════════════
# Main
# ════════════════════════════════════════════
def main():
    logger.info("=" * 60)
    logger.info("X9 — 17策略全窗口排名 + BacktestEngine交叉验证")
    logger.info("=" * 60)

    z3, fwd, dm, tks, fnames, nd, ns, ds, fi = load()

    v1_w = load_v1_weights()
    ga_w_10 = load_x5_weights(10)
    ga_w_5 = load_x5_weights(5)
    logger.info(f"V1权重={len(v1_w)}因子  GA(rf=10)={len(ga_w_10)} GA(rf=5)={len(ga_w_5)}")

    trend_pos, tv_pos, vol_p = build_timing_signals(z3, fi, nd, ns)

    # — 构建所有策略信号 —
    mf = build_mf_signal(z3, v1_w, fnames)
    ga10 = build_ga_signal(z3, ga_w_10, fnames)
    ga5 = build_ga_signal(z3, ga_w_5, fnames)
    chip = build_chip_signal(z3, fi, nd, ns, mom_weight=0.3)
    chip_eq = build_chip_signal(z3, fi, nd, ns, mom_weight=0.5)
    osr = build_osr_signal(z3, fi, nd, ns)
    c01 = build_c01_signal(z3, fi, nd, ns, mf)

    strategy_configs = [
        ("mf_vol_d10_rp",  mf,      10, 'rp',   vol_p),
        ("mf_d10_rp",      mf,      10, 'rp',   None),
        ("ga_d10",         ga10,    10, 'rp',   None),
        ("ga_d5",          ga5,      5, 'rp',   None),
        ("chip_rp",        chip,     3, 'rp',   None),
        ("chip_vol_rp",    chip,     3, 'rp',   vol_p),
        ("chip_covrp",     chip,     3, 'covrp', None),
        ("chip_equal_d3",  chip_eq,  3, 'rp',   None),
        ("mf50_chip50",    0.5*mf + 0.5*chip,  10, 'rp',   None),
        ("mf60_chip40",    0.6*mf + 0.4*chip,  10, 'rp',   None),
        ("mf50_chipcovrp", 0.5*mf + 0.5*chip,  10, 'covrp', None),
        ("osr_d10",        osr,     10, 'rp',   None),
        ("osr_vol_eq_d10", osr,     10, 'rp',   vol_p),
        ("v1_ga_rp",       mf,       3, 'rp',   None),
        ("v4_mf_rp",       mf,       3, 'rp',   None),
        ("v4_mf_tv_rp",    mf,       3, 'rp',   tv_pos),
        ("c01_layered_d5", c01,      5, 'rp',   None),
    ]

    all_results = {}
    for sname, sig, rf, ptype, pr in strategy_configs:
        logger.info(f"\n▶ {sname}  (rf={rf}, pos={ptype}, timing={'Y' if pr is not None else 'N'})")
        try:
            metrics = run_pipeline(sig, sname, rf, ptype, pr,
                z3=z3, fwd=fwd, dm=dm, tks=tks, fnames=fnames,
                nd=nd, ns=ns, ds=ds, fi=fi)
            all_results[sname] = metrics
            logger.info(f"  ✓ 完成 ({len(metrics)} 窗口)")
        except Exception as e:
            logger.error(f"  ✗ 失败: {e}", exc_info=True)
            all_results[sname] = None
        gc.collect()

    # — 构建输出矩阵 —
    summary_rows = []
    for sname, metrics in all_results.items():
        if metrics is None: continue
        d = get_metrics_dict(metrics)
        row = {"策略": sname}
        row["全区间年化"] = d["全区间"].annual_return if "全区间" in d else 0
        row["全区间Sharpe"] = d["全区间"].sharpe if "全区间" in d else 0
        pos_wins = 0
        for wk in ["W1", "W2", "W3", "W4", "W5", "W6", "W7"]:
            if wk in d:
                a = d[wk].annual_return
                s = d[wk].sharpe
                row[f"{wk}年化"] = a; row[f"{wk}Sharpe"] = s
                if a > 0: pos_wins += 1
            else:
                row[f"{wk}年化"] = 0; row[f"{wk}Sharpe"] = 0
        row["正窗口数"] = f"{pos_wins}/7"
        row["全通"] = pos_wins >= 7
        summary_rows.append(row)

    sdf = pd.DataFrame(summary_rows).sort_values("全区间Sharpe", ascending=False).reset_index(drop=True)

    # CSV
    csv_path = os.path.join(RESULTS_DIR, "all_window_results.csv")
    sdf.to_csv(csv_path, index=False, encoding='utf-8-sig')
    logger.info(f"\n结果已保存: {csv_path}")

    # — 打印矩阵 —
    WK = ["W1","W2","W3","W4","W5","W6","W7"]
    print("\n" + "=" * 170)
    h = f"{'#':<3} {'策略':<20} {'年化%':>8} {'Sharpe':>8}"
    for w in WK: h += f"  {w}年化%  {w}Shrp"
    h += f"  {'正窗口':>6}  {'全通'}"
    print(h)
    print("=" * 170)
    for idx, row in sdf.iterrows():
        ps = f"{row['年度化']*100:.2f}%" if '年度化' in row else f"{row['全区间年化']*100:.2f}%"
        ss = f"{row['全区间Sharpe']:.3f}"
        line = f"{idx+1:<3} {row['策略']:<20} {ps:>8} {ss:>8}"
        pw = 0
        for w in WK:
            a = row.get(f"{w}年化", 0)
            s = row.get(f"{w}Sharpe", 0)
            line += f"  {a*100:>+7.2f}% {s:>+7.3f}" if not isinstance(s, str) else f"  {a*100:>+7.2f}% {'N/A':>7}"
            if a > 0 and not np.isnan(a): pw += 1
        line += f"  {pw}/7"
        line += "  ✅" if pw >= 7 else "  ❌"
        print(line)
    print("=" * 170)

    print("\n★ 全窗口通过(7/7)策略:")
    passed = [r for r in summary_rows if r["全通"]]
    if passed:
        for r in passed:
            print(f"  ✅ {r['策略']:<22} 年化={r['全区间年化']*100:.2f}% Sharpe={r['全区间Sharpe']:.3f}")
    else:
        print("  (无)")

    print("\n★ Top5 (按全区间Sharpe):")
    for idx, row in sdf.head(5).iterrows():
        print(f"  #{idx+1} {row['策略']:<22} 年化={row['全区间年化']*100:.2f}% Sharpe={row['全区间Sharpe']:.3f}  {row['正窗口数']}")

    print("\n★ Top5 (按正窗口数):")
    sdf2 = sdf.sort_values(["正窗口数", "全区间Sharpe"], ascending=[False, False]).reset_index(drop=True)
    for idx, row in sdf2.head(5).iterrows():
        print(f"  #{idx+1} {row['策略']:<22} {row['正窗口数']} 年化={row['全区间年化']*100:.2f}% Sharpe={row['全区间Sharpe']:.3f}")

    # — Task2 —
    logger.info("\n" + "=" * 60)
    logger.info("Task2: BacktestEngine 交叉验证")
    logger.info("=" * 60)
    cv = cross_validate_engine(z3, fwd, dm, ds, fi, v1_w, fnames, tks)

    print("\nBacktestEngine vs bt() 对比:")
    for r in cv:
        if "error" in r:
            print(f"  {r['method']}: ERROR — {r['error']}")
        else:
            print(f"  {r['method']:<24}  总收益={r.get('total_return',0)*100:.4f}%  "
                  f"年化={r.get('annual_return',0)*100:.2f}%  "
                  f"Sharpe={r.get('sharpe',0):.3f}  "
                  f"回撤={r.get('max_drawdown',0)*100:.2f}%  "
                  f"交易={r.get('n_trades','?')}")

    json_path = os.path.join(RESULTS_DIR, "x9_summary.json")
    with open(json_path, 'w') as f:
        json.dump({"ranking": sdf.to_dict(orient='records'),
                   "passed_77": passed,
                   "top5_by_sharpe": sdf.head(5).to_dict(orient='records'),
                   "cv_comparison": cv},
                  f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"最终报告: {json_path}")
    logger.info("X9 完成 ✓")

if __name__ == "__main__":
    main()
