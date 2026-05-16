"""
X4_pipeline — 真实选股器重跑 + Walk-Forward 验证

任务1: 使用DB新因子列 (ma5/ma20/ma_alignment_score 等), 
    对 TrendBreakoutSelector / OversoldReboundSelector / ChipConcentrationSelector 
    用select()真实选股, 构建信号向量并用bt()回测。

任务3: Walk-Forward滚动验证 (MF+Vol_D10配置的时变稳定性)

用法:
    python3 daily/2026-05-16/X4_pipeline.py 2>&1 | tee daily/2026-05-16/x4_x5_results/x4.log
"""
import os, sys, json, logging, gc, time, bisect
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from core.database import Database
from core.positioners import RPPortfolioWeights
from core.screening.impl.momentum_breakout import (
    TrendBreakoutSelector,
    OversoldReboundSelector,
    ChipConcentrationSelector,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("x4")
TX = 0.0012

# ============================================================
# 输出目录
# ============================================================
OUT_DIR = os.path.join(os.path.dirname(__file__), "x4_x5_results")
os.makedirs(OUT_DIR, exist_ok=True)

# V1 MF信号用的因子 (55个)
MF_FACTORS = list(set([
    'a27','a30','a31','a41','a42','a64','a69','a8','a80','a85',
    'a88','a91','a97','a98','a99','ff_mkt','gtja103','gtja104','gtja105',
    'gtja108','gtja113','gtja117','gtja12','gtja120','gtja121','gtja123',
    'gtja127','gtja13','gtja139','gtja141','gtja142','gtja144','gtja148',
    'gtja164','gtja168','gtja171','gtja176','gtja185','gtja34','gtja49',
    'gtja62','gtja76','gtja83','gtja85','gtja90','gtja91','gtja99',
    'returns','rsi_14','volatility_20','macd','macd_signal','momentum_5',
    'momentum_20','volume_ratio','boll_position','beta_20',
]))

# 选股器需要的因子列
SELECTOR_FACTORS = list(set([
    'ma5','ma10','ma20','ma21','ma60','ma120',
    'ma_alignment_score','ma60_trend','ma120_trend',
    'macd_above_zero','macd_golden_cross',
    'volume_breakout_ratio','volume_contraction',
    'ma_convergence','chip_concentration','ma_angle_20',
    'macd','macd_signal','volume_ratio',
    'box_breakout','breakout_strength',
    'close',
]))

# ============================================================
# 数据加载
# ============================================================
def load():
    """加载MF信号数据 + 选股器因子数据"""
    db = Database()

    # 1. 加载MF信号用因子
    df_mf = db.get_factors(start_date="2018-01-01", end_date="2026-04-30",
                           factor_names=MF_FACTORS, with_close=True)
    df_mf['date'] = pd.to_datetime(df_mf['date'])
    ds = sorted(df_mf['date'].unique())
    tks = db.get_symbols()['symbol'].tolist()
    nd, ns, nf = len(ds), len(tks), len(MF_FACTORS)
    t2i = {t: i for i, t in enumerate(tks)}
    d2i = {d: i for i, d in enumerate(ds)}

    # 2. MF信号 3D矩阵
    v3 = np.full((nd, ns, nf), np.nan, dtype=np.float32)
    dm = np.zeros((nd, ns), dtype=bool)
    cl = np.zeros((nd, ns), dtype=np.float32)
    di = np.array([d2i[d] for d in df_mf['date']], dtype=np.int32)
    si = np.array([t2i.get(s, -1) for s in df_mf['symbol']], dtype=np.int32)
    v = si >= 0; di, si = di[v], si[v]
    for fi, fc in enumerate(MF_FACTORS):
        if fc in df_mf.columns:
            v3[di, si, fi] = df_mf[fc].values[v].astype(np.float32)
    cl[di, si] = df_mf['close'].values[v].astype(np.float32)
    dm[di, si] = True
    np.nan_to_num(v3, nan=0.0, copy=False)
    np.nan_to_num(cl, nan=0.0, copy=False)

    # 3. 前向收益
    fwd = np.zeros((nd, ns), dtype=np.float32)
    for d in range(nd - 1):
        b = (cl[d] > 1e-10) & (cl[d + 1] > 1e-10)
        fwd[d, b] = (cl[d + 1, b] - cl[d, b]) / cl[d, b]

    # 4. 截面Z-Score
    z3 = np.zeros_like(v3)
    for fi in range(nf):
        a = v3[:, :, fi]
        for d in range(nd):
            r = a[d, :]; nz = r[r != 0]
            if len(nz) > 1:
                lo, hi = np.quantile(nz, [0.01, 0.99]); c = np.clip(r, lo, hi)
                mu, sd = np.mean(c), np.std(c)
                z3[d, :, fi] = (c - mu) / sd if sd > 1e-10 else 0.0

    # 5. 加载选股器因子数据 (包含close)
    sel_factor_names = [f for f in SELECTOR_FACTORS if f != 'close']
    df_sel = db.get_factors(start_date="2018-01-01", end_date="2026-04-30",
                            factor_names=sel_factor_names, with_close=True)
    df_sel['date'] = pd.to_datetime(df_sel['date'])
    # merge close from df_mf if needed
    if 'close' not in df_sel.columns and 'close' in df_mf.columns:
        df_sel = df_sel.merge(df_mf[['date', 'symbol', 'close']], on=['date', 'symbol'], how='left')

    logger.info(f"数据: {nd}天 × {ns}只 × {nf}因子(MF) + {len(SELECTOR_FACTORS)}因子(选股器)")
    return z3, fwd, dm, tks, list(MF_FACTORS), nd, ns, ds, t2i, d2i, df_sel


# ============================================================
# V1 权重加载
# ============================================================
def v1w():
    p = os.path.join(os.path.dirname(__file__), '..', '2026-05-13', 'v2', 'v1_reference', 'ga_results.json')
    if os.path.exists(p):
        with open(p) as f:
            for it in json.load(f):
                if 'L1中_80代' in it['label']:
                    return it['configs'][0]['weights']
    return {}


# ============================================================
# bt() — 回测函数 (同V6/V7/X2)
# ============================================================
def bt(sig, fwd, dm, name, rf=3, tn=40, pos_ratio=None, mhd=5):
    nd, ns = sig.shape
    alloc = RPPortfolioWeights(top_n=tn, min_hold_days=mhd)
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
            txc = 0.5 * to * TX; ttx += txc
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
    ny = nd / 252.0
    ar = (float(eq[-1] / eq[0])) ** (1.0 / max(ny, 0.5)) - 1.0
    lr = np.log(eq[1:] / eq[:-1])
    sp = float(np.mean(lr) / max(np.std(lr), 1e-10) * np.sqrt(252))
    cm = np.maximum.accumulate(eq); dd = (eq - cm) / cm; mdd = float(np.min(dd))
    cal = ar / abs(mdd) if abs(mdd) > 0 else 0
    wr = int(np.sum(dr > 0)) / max(int(np.sum(dr > 0)) + int(np.sum(dr < 0)), 1)
    logger.info(f"  [{name}] 年化={ar*100:.2f}% Sharpe={sp:.3f} 回撤={mdd*100:.2f}% Calmar={cal:.3f}")
    return {"name": name, "annual_return": round(ar, 4), "sharpe": round(sp, 4),
            "max_drawdown": round(mdd, 4), "calmar": round(cal, 4),
            "win_rate": round(wr, 4), "n_trades": nt}


# ============================================================
# 构建选股器信号矩阵
# ============================================================
def build_selector_signal(selector_cls, df_sel, ds, t2i, nd, ns, top_n=40,
                          selector_kwargs=None, warmup=120):
    """对每个交易日调用 selector.select(), 构建信号矩阵。

    selector.select() 内部通过 _latest_per_symbol() 取 date 之前的最新数据,
    所以传入的 DataFrame 必须包含 date 列。我们使用增量更新方法:
      遍历日期 → 维护 {symbol: {factor: val}} 的 latest 快照 → 构建 DataFrame 传给 select()

    Returns:
        sig: (nd, ns) float32, 选中=1.0, 未选中=-inf
    """
    t0 = time.time()
    if selector_kwargs is None:
        selector_kwargs = {}
    selector = selector_cls(**selector_kwargs)

    # 选股器需要的因子列
    try:
        needed = selector.factor_names
    except AttributeError:
        needed = list(df_sel.columns)
    needed = [c for c in needed if c in df_sel.columns]

    sig = np.full((nd, ns), -np.inf, dtype=np.float32)

    # 增量更新: current_vals[symbol] = {col: val, ...}
    current_vals = {}
    all_symbols_set = set(t2i.keys())

    # 预计算 date → day_data 映射, 避免反复过滤
    date_to_data = {}
    for d, grp in df_sel.groupby('date'):
        date_to_data[d] = grp

    for d_idx, date in enumerate(ds):
        day_data = date_to_data.get(date)
        if day_data is not None:
            for _, row in day_data.iterrows():
                sym = row['symbol']
                if sym not in current_vals:
                    current_vals[sym] = {}
                for c in needed:
                    v = row.get(c)
                    if pd.notna(v):
                        current_vals[sym][c] = v

        if d_idx < warmup:
            continue

        # 构建 latest DataFrame: 每个 symbol 一行, 包含 date 列 (设为 date - 1天)
        valid_rows = []
        for sym in all_symbols_set:
            cv = current_vals.get(sym)
            if cv is None:
                continue
            row_data = {'symbol': sym, 'date': date - pd.Timedelta(days=1)}
            has_data = False
            for c in needed:
                v = cv.get(c)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    row_data[c] = v
                    has_data = True
                else:
                    row_data[c] = np.nan
            if has_data:
                valid_rows.append(row_data)

        if len(valid_rows) < 5:
            continue

        latest_df = pd.DataFrame(valid_rows)

        try:
            selected = selector.select(latest_df, date, top_n)
            indices = [t2i.get(s) for s in selected if s in t2i]
            indices = [i for i in indices if i is not None]
            if indices:
                sig[d_idx, indices] = 1.0
        except Exception as e:
            logger.warning(f"  [{selector_cls.__name__}] date={date} select failed: {e}")

        if (d_idx + 1) % 500 == 0:
            elapsed = time.time() - t0
            logger.info(f"  [{selector_cls.__name__}] 已处理 {d_idx+1}/{nd} 日, 耗时 {elapsed:.0f}s")

    elapsed = time.time() - t0
    logger.info(f"  [{selector_cls.__name__}] 信号构建完成, 耗时 {elapsed:.0f}s")
    return sig


# ============================================================
# 构建择时信号
# ============================================================
def build_timing_signals(z3, fi, nd, ns):
    """构建 TrendTiming / VolTiming 仓位系数 (同V6/V7)"""
    im, ims = fi.get('macd'), fi.get('macd_signal')
    im5, im20 = fi.get('momentum_5'), fi.get('momentum_20')
    ir, iv = fi.get('rsi_14'), fi.get('volatility_20')

    # TrendTiming 仓位系数
    trend_signal = np.full((nd, ns), 0.5, dtype=np.float32)
    for d in range(nd):
        sl = []
        if im is not None and ims is not None:
            sl.append(np.where(z3[d, :, im] > z3[d, :, ims], 1.0, 0.0))
        if im5 is not None and im20 is not None:
            m5v, m20v = z3[d, :, im5], z3[d, :, im20]
            sl.append(np.where((m5v > 0) & (m5v > m20v), 1.0, np.where(m5v < 0, 0.0, 0.5)))
        if ir is not None:
            rv = z3[d, :, ir]
            sl.append(np.where(rv > 70, 0.0, np.where(rv >= 50, 1.0, np.where(rv >= 30, 0.5, 0.0))))
        if sl:
            trend_signal[d] = np.mean(sl, axis=0)

    trend_p = np.clip(np.mean(trend_signal >= 0.6, axis=1) * 2.0, 0.1, 1.0)

    # VolTiming 仓位系数
    vol_p = np.ones(nd, dtype=np.float32)
    if iv is not None:
        hv_ratio = np.mean(z3[:, :, iv] > 0.05, axis=1)
        vol_p = np.clip(1.0 - hv_ratio, 0.2, 1.0)

    # MR 仓位系数
    mr_p = np.full(nd, 0.6, dtype=np.float32)
    if iv is not None:
        mkt_vol = np.nanmean(z3[:, :, iv], axis=1)
        mr_p[mkt_vol < 0.04] = 1.0
        mr_p[mkt_vol > 0.08] = 0.3

    return trend_p, vol_p, mr_p


# ============================================================
# 任务1: 真实选股器重跑
# ============================================================
def run_selector_experiments(z3, fwd, dm, tks, fnames, nd, ns, ds, t2i, d2i, df_sel):
    """任务1: 真实选股器 vs 代理信号, 频率扫描, 择时叠加"""
    logger.info("\n" + "="*70)
    logger.info("任务1: 真实选股器重跑")
    logger.info("="*70)

    all_results = []
    fi = {fn: i for i, fn in enumerate(fnames)}
    v1w_dict = v1w()

    # 构建MF信号 (代理信号基准)
    wv = np.zeros(len(fnames), dtype=np.float32)
    for fi_i, fc in enumerate(fnames):
        if fc in v1w_dict:
            wv[fi_i] = float(v1w_dict[fc])
    s = np.sum(np.abs(wv))
    if s > 0:
        wv /= s
    mf = np.nan_to_num(np.tensordot(z3, wv, axes=(2, 0)), nan=-1e10, neginf=-1e10)

    # 构建择时信号
    trend_p, vol_p, mr_p = build_timing_signals(z3, fi, nd, ns)

    # ── 构建真实选股器信号 ──
    selectors_config = [
        ("TrendBreakout", TrendBreakoutSelector, {}),
        ("OversoldRebound", OversoldReboundSelector, {}),
        ("ChipConcentration", ChipConcentrationSelector, {
            'max_volume_contraction': 0.5,
            'max_chip_concentration': 0.05,
            'max_ma_convergence': 0.05,
            'min_breakout_volume': 1.5,
        }),
    ]

    selector_signals = {}
    for sname, scls, skwargs in selectors_config:
        logger.info(f"\n--- 构建 {sname} 信号 ---")
        sig = build_selector_signal(scls, df_sel, ds, t2i, nd, ns,
                                    top_n=40, selector_kwargs=skwargs, warmup=120)
        selector_signals[sname] = sig
        # 基准: 无择时
        for fd in [3, 5, 10]:
            r = bt(sig, fwd, dm, f"真实_{sname}_无择时_D{fd}", rf=fd, tn=40, mhd=5)
            all_results.append(r)
        # VolTiming
        for fd in [3, 5, 10]:
            r = bt(sig, fwd, dm, f"真实_{sname}_VolTiming_D{fd}", rf=fd, tn=40, pos_ratio=vol_p, mhd=5)
            all_results.append(r)
        # TrendTiming
        for fd in [3, 5, 10]:
            r = bt(sig, fwd, dm, f"真实_{sname}_TrendTiming_D{fd}", rf=fd, tn=40, pos_ratio=trend_p, mhd=5)
            all_results.append(r)
        gc.collect()

    # ── MF代理信号对比基准 ──
    logger.info("\n--- MF代理信号 (对比基准) ---")
    for fd in [3, 5, 10]:
        r = bt(mf, fwd, dm, f"代理_MF_无择时_D{fd}", rf=fd, tn=40, mhd=5)
        all_results.append(r)
    for fd in [3, 5, 10]:
        r = bt(mf, fwd, dm, f"代理_MF_VolTiming_D{fd}", rf=fd, tn=40, pos_ratio=vol_p, mhd=5)
        all_results.append(r)
    for fd in [3, 5, 10]:
        r = bt(mf, fwd, dm, f"代理_MF_TrendTiming_D{fd}", rf=fd, tn=40, pos_ratio=trend_p, mhd=5)
        all_results.append(r)

    return all_results


# ============================================================
# 任务3: Walk-Forward滚动验证
# ============================================================
def run_walk_forward(z3, fwd, dm, ds, fnames, nd, ns):
    """Walk-Forward滚动验证 MF+Vol_D10 配置的时变稳定性"""
    logger.info("\n" + "="*70)
    logger.info("任务3: Walk-Forward滚动验证 (MF+Vol_D10)")
    logger.info("="*70)

    fi = {fn: i for i, fn in enumerate(fnames)}
    v1w_dict = v1w()

    # MF信号
    wv = np.zeros(len(fnames), dtype=np.float32)
    for fi_i, fc in enumerate(fnames):
        if fc in v1w_dict:
            wv[fi_i] = float(v1w_dict[fc])
    s = np.sum(np.abs(wv))
    if s > 0:
        wv /= s
    mf = np.nan_to_num(np.tensordot(z3, wv, axes=(2, 0)), nan=-1e10, neginf=-1e10)

    # VolTiming信号
    _, vol_p, _ = build_timing_signals(z3, fi, nd, ns)

    # 窗口定义
    windows = [
        ("W1_2019-01~2020-06_train", "2019-01-01", "2020-06-30",
         "2020-07-01", "2021-06-30"),
        ("W2_2020-07~2021-12_train", "2020-07-01", "2021-12-31",
         "2022-01-01", "2022-12-31"),
        ("W3_2022-01~2023-06_train", "2022-01-01", "2023-06-30",
         "2023-07-01", "2024-06-30"),
        ("W4_全量_2019-01~2024-12_train", "2019-01-01", "2024-12-31",
         "2025-01-01", "2026-04-30"),
    ]

    def slice_data(sig_arr, fwd_arr, dm_arr, ds_list, start, end):
        d0 = bisect.bisect_left(ds_list, pd.Timestamp(start))
        d1 = bisect.bisect_right(ds_list, pd.Timestamp(end)) - 1
        if d0 >= len(ds_list) or d1 < 0 or d0 > d1:
            return None, None, None
        return sig_arr[d0:d1+1], fwd_arr[d0:d1+1], dm_arr[d0:d1+1]

    ds_list = list(ds)
    all_results = []

    for wname, ws_train, we_train, ws_val, we_val in windows:
        logger.info(f"\n--- Walk-Forward: {wname} ---")

        # 训练期 (只用于参考, 这里不重新训练, 直接验证)
        s_tr, f_tr, d_tr = slice_data(mf, fwd, dm, ds_list, ws_train, we_train)
        # 验证期
        s_val, f_val, d_val = slice_data(mf, fwd, dm, ds_list, ws_val, we_val)

        if s_val is None:
            logger.info(f"  [{wname}] 无验证期数据,跳过")
            continue

        # 截取vol_p
        d0_val = bisect.bisect_left(ds_list, pd.Timestamp(ws_val))
        d1_val = bisect.bisect_right(ds_list, pd.Timestamp(we_val)) - 1
        vol_p_slice = vol_p[d0_val:d1_val+1] if d0_val <= d1_val else None

        # 验证期用 MF+Vol_D10
        label_val = f"WF_{wname}_val"
        r_val = bt(s_val, f_val, d_val, label_val, rf=10, tn=40, pos_ratio=vol_p_slice, mhd=5)
        r_val['window'] = wname
        r_val['phase'] = 'validation'
        all_results.append(r_val)

        if s_tr is not None:
            d0_tr = bisect.bisect_left(ds_list, pd.Timestamp(ws_train))
            d1_tr = bisect.bisect_right(ds_list, pd.Timestamp(we_train)) - 1
            vol_p_tr = vol_p[d0_tr:d1_tr+1] if d0_tr <= d1_tr else None
            label_tr = f"WF_{wname}_train"
            r_tr = bt(s_tr, f_tr, d_tr, label_tr, rf=10, tn=40, pos_ratio=vol_p_tr, mhd=5)
            r_tr['window'] = wname
            r_tr['phase'] = 'training'
            all_results.append(r_tr)

    return all_results


# ============================================================
# 保存与报告
# ============================================================
def save_and_report(all_results, prefix="x4"):
    """保存结果并打印汇总"""
    out_path = os.path.join(OUT_DIR, f"{prefix}_results.json")
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info(f"\n结果已保存至: {out_path}")

    print(f"\n{'='*130}")
    print(f"{'实验名':<48} {'年化%':<8} {'Sharpe':<8} {'回撤%':<8} {'Calmar':<8} {'胜率':<6} {'交易':<5}")
    print('-'*130)
    valid = [r for r in all_results if r.get('annual_return', -999) != -999]
    for r in sorted(valid, key=lambda x: x.get('sharpe', -999), reverse=True):
        cls = "🏆" if r['max_drawdown'] < 0.20 and r['annual_return'] > 0.05 else "  "
        print(f"{cls} {r['name']:<46} {r['annual_return']*100:>6.2f}% {r['sharpe']:>7.3f} {r['max_drawdown']*100:>6.2f}% {r['calmar']:>7.3f} {r['win_rate']*100:>5.1f}% {r['n_trades']:>4}")
    print('='*130)

    qualified = [r for r in valid if r['max_drawdown'] < 0.20 and r['annual_return'] > 0.05]
    logger.info(f"\n🏆 达标(回撤<20% 年化>5%): {len(qualified)}个")
    for r in sorted(qualified, key=lambda x: x['sharpe'], reverse=True)[:15]:
        logger.info(f"  🏆 {r['name']}: 年化={r['annual_return']*100:.2f}% 回撤={r['max_drawdown']*100:.2f}% Sharpe={r['sharpe']:.3f}")
    return valid


# ============================================================
# Main
# ============================================================
def main():
    logger.info("="*70)
    logger.info("X4 Pipeline — 真实选股器重跑 + Walk-Forward验证")
    logger.info("="*70)

    t_start = time.time()

    # 数据加载
    z3, fwd, dm, tks, fnames, nd, ns, ds, t2i, d2i, df_sel = load()
    logger.info(f"加载完成: {nd}天 {ns}只 {len(fnames)}个MF因子 | 选股器数据 {df_sel.shape}")

    all_results = []

    # ── 任务1: 真实选股器重跑 ──
    t1 = time.time()
    r1 = run_selector_experiments(z3, fwd, dm, tks, fnames, nd, ns, ds, t2i, d2i, df_sel)
    all_results.extend(r1)
    logger.info(f"任务1 完成, 耗时 {(time.time()-t1)/60:.1f}分")

    # ── 任务3: Walk-Forward ──
    t3 = time.time()
    r3 = run_walk_forward(z3, fwd, dm, ds, fnames, nd, ns)
    all_results.extend(r3)
    logger.info(f"任务3 完成, 耗时 {(time.time()-t3)/60:.1f}分")

    # ── 保存与报告 ──
    save_and_report(all_results, "x4")

    elapsed = (time.time() - t_start) / 60
    logger.info(f"\nX4 全部完成! 总耗时 {elapsed:.1f}分")


if __name__ == "__main__":
    main()
