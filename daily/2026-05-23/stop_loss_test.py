"""止损线参数扫描：对比 5%, 8%, 10%, 15% 对 mf_d10_rp 的影响。

用法：
    cd /Users/wangzeshang1/MyProjects/zequant
    python3 daily/2026-05-23/stop_loss_test.py
"""
from __future__ import annotations
import json, logging, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from core.database import Database
from core.strategies.pipeline import StrategyPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stop_loss_test")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

FULL_RANGE = ("2019-01-02", "2026-04-30")
WINDOWS = [
    ("全区间", "2019-01-02", "2026-04-30"),
    ("2022熊市", "2022-01-04", "2022-12-30"),
    ("修复牛OOS", "2024-07-01", "2026-04-30"),
]

# mf_d10_rp 真实因子权重
MF_WEIGHTS = {
    "ff_mkt": 0.0413, "gtja142": 0.3005, "gtja144": 0.2045,
    "gtja171": -0.0443, "gtja103": -0.0147, "gtja85": -0.0258,
    "a88": -0.035, "a31": -0.0249, "rsi_14": 0.0253,
    "gtja139": -0.0112, "gtja123": 0.1666, "a42": 0.1999,
    "a41": 0.2152, "a97": -0.0734, "gtja148": -0.0048,
    "gtja99": -0.059, "gtja117": 0.2324, "gtja76": 0.0032,
    "gtja90": 0.0437, "volatility_20": -0.1127, "gtja113": -0.0874,
    "gtja141": 0.2104, "a99": -0.072, "gtja12": -0.1859,
    "gtja83": 0.1429, "gtja164": 0.0235, "a98": 0.0657,
    "gtja49": -0.2478, "gtja121": -0.0095, "a85": 0.1419,
    "gtja104": -0.1303, "gtja185": -0.0565, "gtja176": -0.075,
    "a80": 0.1689, "gtja62": 0.1181, "a8": 0.0657,
    "gtja34": -0.0816, "returns": -0.0508, "gtja168": 0.3003,
    "gtja108": -0.0791, "gtja105": 0.0686, "gtja127": -0.0506,
    "a27": -0.0627, "a64": 0.0874, "gtja91": -0.0399,
    "a30": -0.0666, "a69": -0.0961, "a91": -0.0582,
    "gtja13": 0.0903, "gtja120": 0.055,
}


def make_mf_signal_builder(weights):
    def _builder(z3, fi, nd, ns):
        w = np.zeros(z3.shape[2], dtype=np.float32)
        for fn, wt in weights.items():
            if fn in fi:
                w[fi[fn]] = wt
        return np.nan_to_num(np.tensordot(z3, w, axes=(2, 0)), nan=-1e10, neginf=-1e10)
    return _builder


class StrategyPipelineV2(StrategyPipeline):
    """带止损+增强ST的Pipeline。"""

    def __init__(self, stop_loss_pct: float = 0.08, **kwargs):
        super().__init__(**kwargs)
        self.stop_loss_pct = stop_loss_pct

    def _build_universe_mask(self, db, t2i, d2i):
        um = super()._build_universe_mask(db, t2i, d2i)
        try:
            bars = db.get_daily_bars(
                columns=['symbol', 'date', 'pct_change', 'close', 'volume'],
                start_date=str(self.ds[0].date()),
                end_date=str(self.ds[-1].date()),
            )
            if bars is not None and not bars.empty:
                bars['date'] = pd.to_datetime(bars['date'])
                flagged = set()
                for sym in t2i:
                    sym_bars = bars[bars['symbol'] == sym].sort_values('date')
                    if len(sym_bars) < 20:
                        continue
                    pcts = sym_bars['pct_change'].values.astype(float)
                    closes = sym_bars['close'].values.astype(float)
                    dates = sym_bars['date'].values
                    for j in range(len(pcts) - 1):
                        if pcts[j] < -9.5 and pcts[j + 1] < -9.5:
                            limit_date = pd.Timestamp(dates[j + 1])
                            for di, d in enumerate(self.ds):
                                if d >= limit_date:
                                    um[di, t2i[sym]] = False
                            flagged.add(sym)
                            break
                    recent = closes[-min(20, len(closes)):]
                    if len(recent) >= 5:
                        if np.mean(recent[-5:]) < 3.0 and np.mean(pcts[-5:]) < -2.0:
                            start_dt = pd.Timestamp(dates[-5])
                            for di, d in enumerate(self.ds):
                                if d >= start_dt:
                                    um[di, t2i[sym]] = False
                            flagged.add(sym)
                logger.info("  增强ST: %d 只额外排除", len(flagged))
        except Exception as e:
            logger.warning("  增强ST失败: %s", e)
        return um

    def _backtest_rp(self, sig, fwd, dm, nd, ns):
        from core.positioners import RPPortfolioWeights
        alloc = RPPortfolioWeights(top_n=self.top_n, min_hold_days=self.min_hold_days)
        pw = np.zeros(ns, dtype=np.float32)
        hs = np.full(ns, -1, dtype=np.int32)
        rh = 0
        dr = np.zeros(nd, dtype=np.float64)
        nt = 0
        stop_triggered = 0
        entry_price = np.zeros(ns, dtype=np.float32)

        for i in range(1, nd):
            rebal = (i % self.rebal_freq == 0)
            txc = 0.0

            if self.stop_loss_pct > 0 and np.any(pw > 0):
                for j in range(ns):
                    if pw[j] > 0 and hs[j] >= 0 and entry_price[j] > 0:
                        if fwd[i, j] < -self.stop_loss_pct:
                            pw[j] = 0.0
                            hs[j] = -1
                            stop_triggered += 1
                            nt += 1

            if rebal:
                masked_sig = sig[i].copy()
                if self.use_universe_filter and self.um is not None:
                    abs_i = getattr(self, '_backtest_sidx', 0) + i
                    masked_sig[~self.um[abs_i]] = -1e10
                nw = alloc.allocate(masked_sig, fwd, i, pw, hs, rh)
                to = float(np.sum(np.abs(nw - pw)))
                txc = 0.5 * to * self.tx_cost
                if to > 0.01:
                    nt += 1
                for j in range(ns):
                    if nw[j] > 0 and pw[j] <= 0:
                        entry_price[j] = 1.0 + max(fwd[i, j], 0)
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

            for j in range(ns):
                if pw[j] > 0 and entry_price[j] <= 0:
                    entry_price[j] = 1.0 + max(fwd[i, j], 0)

            rt = float(np.dot(pw, fwd[i]))
            dr[i] = 0.0 if (np.isnan(rt) or np.isinf(rt)) else rt - txc
            rh += 1

        logger.info(f"  止损触发: {stop_triggered} 次")
        return dr, nt


def run_test(stop_loss: float, label: str) -> dict:
    """运行一次止损参数测试。"""
    logger.info(f"\n=== 止损线: {label} ({stop_loss*100:.0f}%) ===")
    db = Database()
    all_factors = db.list_factor_columns()
    db.close()

    mf_weights = {k: v for k, v in MF_WEIGHTS.items() if k in all_factors}
    builder = make_mf_signal_builder(mf_weights)

    import pandas as pd
    p = StrategyPipelineV2(
        stop_loss_pct=stop_loss,
        signal_builder=builder,
        name=f"mf_d10_rp_stop{int(stop_loss*100)}",
        rebal_freq=10, top_n=20, min_hold_days=5,
        positioner_type="rp", tx_cost=0.002,
        factor_names=list(mf_weights.keys()),
        use_universe_filter=True,
    )
    result = p.run(start=FULL_RANGE[0], end=FULL_RANGE[1])
    windows = p.window_analysis(WINDOWS)

    return {
        "stop_loss_pct": stop_loss,
        "label": label,
        "full_range": {
            "annual_return": round(result.annual_return, 4),
            "sharpe": round(result.sharpe, 4),
            "max_drawdown": round(result.max_drawdown, 4),
            "calmar": round(result.calmar, 4),
            "win_rate": round(result.win_rate, 4),
            "n_trades": result.n_trades,
        },
        "windows": [
            {"window": w.window, "annual_return": round(w.annual_return, 4),
             "sharpe": round(w.sharpe, 4), "max_drawdown": round(w.max_drawdown, 4)}
            for w in windows
        ],
    }


def print_table(results):
    """打印对比表格。"""
    print(f"\n{'='*90}")
    print(f"  止损线对比 — mf_d10_rp (2019-01 ~ 2026-04)")
    print(f"{'='*90}")

    header = f"{'止损线':<10} {'年化%':<10} {'Sharpe':<10} {'回撤%':<10} {'Calmar':<10} {'胜率':<8} {'交易数':<8}"
    print(header)
    print("-" * 90)

    for r in results:
        f = r["full_range"]
        sl = r["label"]
        print(f"{sl:<10} {f['annual_return']*100:>+7.2f}% {f['sharpe']:>8.3f} {-f['max_drawdown']*100:>7.2f}% {f['calmar']:>8.3f} {f['win_rate']*100:>5.1f}% {f['n_trades']:>6d}")

    print(f"\n  --- 分窗口年化收益 ---")
    print(f"  {'止损线':<10} {'全区间%':<10} {'2022熊市%':<12} {'复苏牛OOS%':<12}")
    print(f"  {'-'*44}")
    for r in results:
        ws = {w["window"]: w for w in r["windows"]}
        ar_all = ws.get("全区间", {}).get("annual_return", 0)
        ar_bear = ws.get("2022熊市", {}).get("annual_return", 0)
        ar_oos = ws.get("修复牛OOS", {}).get("annual_return", 0)
        print(f"  {r['label']:<10} {ar_all*100:>+7.2f}% {ar_bear*100:>+9.2f}% {ar_oos*100:>+9.2f}%")

    print(f"\n  --- 分窗口夏普比率 ---")
    print(f"  {'止损线':<10} {'全区间':<10} {'2022熊市':<12} {'复苏牛OOS':<12}")
    print(f"  {'-'*44}")
    for r in results:
        ws = {w["window"]: w for w in r["windows"]}
        sp_all = ws.get("全区间", {}).get("sharpe", 0)
        sp_bear = ws.get("2022熊市", {}).get("sharpe", 0)
        sp_oos = ws.get("修复牛OOS", {}).get("sharpe", 0)
        print(f"  {r['label']:<10} {sp_all:>8.3f} {sp_bear:>9.3f} {sp_oos:>9.3f}")

    print(f"{'='*90}\n")


def main():
    test_cases = [
        (0.00, "无止损"),
        (0.05, "-5%"),
        (0.08, "-8%"),
        (0.10, "-10%"),
        (0.15, "-15%"),
    ]

    results = []
    for sl, label in test_cases:
        r = run_test(sl, label)
        results.append(r)
        with open(os.path.join(RESULTS_DIR, f"stop_loss_{int(sl*100)}.json"), "w") as f:
            json.dump(r, f, indent=2, ensure_ascii=False)

    print_table(results)

    with open(os.path.join(RESULTS_DIR, "stop_loss_comparison.json"), "w") as f:
        json.dump({"test_cases": results}, f, indent=2, ensure_ascii=False)
    logger.info(f"结果已保存至: {RESULTS_DIR}/")


if __name__ == "__main__":
    import pandas as pd
    main()
