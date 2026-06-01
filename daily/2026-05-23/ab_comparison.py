"""A/B 对比回测：baseline vs V2（ST过滤增强 + 个股止损 -8%）

用法：
    cd /Users/wangzeshang1/MyProjects/zequant
    python3 daily/2026-05-23/ab_comparison.py --strategy mf_d10_rp   # MF策略对比
    python3 daily/2026-05-23/ab_comparison.py --strategy chip_covrp  # Chip策略对比
    python3 daily/2026-05-23/ab_comparison.py --strategy both        # 全部（默认）
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from core.strategies.pipeline import StrategyPipeline, BacktestMetrics
from core.database import Database
from core.positioners import RPPortfolioWeights

logger = logging.getLogger("ab_comparison")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

FULL_RANGE = ("2019-01-02", "2026-04-30")
BEAR_RANGE = ("2022-01-04", "2022-12-30")
OOS_RANGE = ("2024-07-01", "2026-04-30")
WINDOWS = [
    ("全区间", "2019-01-02", "2026-04-30"),
    ("2022熊市", "2022-01-04", "2022-12-30"),
    ("修复牛OOS", "2024-07-01", "2026-04-30"),
]

# mf_d10_rp 的 48 个真实因子权重（从 config.json 提取）
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

# chip_covrp 使用的技术因子
CHIP_FACTORS = [
    "volatility_20", "momentum_20", "rsi_14", "volume_ratio",
    "boll_position", "macd", "macd_signal",
    "momentum_5", "returns",
]


def make_mf_signal_builder(weights: Dict[str, float]):
    """创建 mf_d10_rp 真实因子权重的 signal_builder。"""
    def _builder(z3, fi, nd, ns):
        w = np.zeros(z3.shape[2], dtype=np.float32)
        for fn, wt in weights.items():
            if fn in fi:
                w[fi[fn]] = wt
        sig = np.nan_to_num(
            np.tensordot(z3, w, axes=(2, 0)),
            nan=-1e10, neginf=-1e10
        )
        return sig
    return _builder


class StrategyPipelineV2(StrategyPipeline):
    """改良版 Pipeline：增加个股止损 + 增强ST检测。"""

    def __init__(self, stop_loss_pct: float = 0.08, **kwargs):
        super().__init__(**kwargs)
        self.stop_loss_pct = stop_loss_pct

    def _build_universe_mask(self, db, t2i, d2i):
        """覆盖父类：增强ST检测。"""
        um = super()._build_universe_mask(db, t2i, d2i)

        # 额外ST检测：连续跌停 + 低价
        try:
            bars = db.get_daily_bars(
                columns=['symbol', 'date', 'pct_change', 'close', 'volume'],
                start_date=str(self.ds[0].date()),
                end_date=str(self.ds[-1].date()),
            )
            if bars is not None and not bars.empty:
                bars['date'] = pd.to_datetime(bars['date'])

                sym_df = db.get_symbols()
                st_set = set()
                if not sym_df.empty and 'name' in sym_df.columns:
                    st_msk = sym_df['name'].fillna('').str.upper().str.contains('ST', na=False)
                    st_set = set(sym_df.loc[st_msk, 'symbol'].tolist())

                flagged = set()
                for sym in t2i:
                    sym_bars = bars[bars['symbol'] == sym].sort_values('date')
                    if len(sym_bars) < 20:
                        continue
                    pcts = sym_bars['pct_change'].values.astype(float)
                    closes = sym_bars['close'].values.astype(float)
                    dates = sym_bars['date'].values

                    # 特征1：最近20天内有连续2天跌停（<-9.5%）
                    for j in range(len(pcts) - 1):
                        if pcts[j] < -9.5 and pcts[j + 1] < -9.5:
                            limit_date = pd.Timestamp(dates[j + 1])
                            for di, d in enumerate(self.ds):
                                if d >= limit_date:
                                    um[di, t2i[sym]] = False
                            flagged.add(sym)
                            break

                    # 特征2：价格 < 3 元且持续下跌
                    recent = closes[-min(20, len(closes)):]
                    recent_dates = dates[-min(20, len(dates)):]
                    if len(recent) >= 5:
                        avg_price = np.mean(recent[-5:])
                        if avg_price < 3.0 and np.mean(pcts[-5:]) < -2.0:
                            start_dt = pd.Timestamp(recent_dates[-5])
                            for di, d in enumerate(self.ds):
                                if d >= start_dt:
                                    um[di, t2i[sym]] = False
                            flagged.add(sym)

                logger.info("  增强ST检测: %d 只额外标记为不可交易", len(flagged))
        except Exception as e:
            logger.warning("  增强ST检测失败: %s", e)

        return um

    def _backtest_rp(self, sig, fwd, dm, nd, ns):
        """覆盖父类：增加个股止损。"""
        alloc = RPPortfolioWeights(
            top_n=self.top_n, min_hold_days=self.min_hold_days
        )
        pw = np.zeros(ns, dtype=np.float32)
        hs = np.full(ns, -1, dtype=np.int32)
        rh = 0
        dr = np.zeros(nd, dtype=np.float64)
        nt = 0
        entry_price = np.zeros(ns, dtype=np.float32)

        for i in range(1, nd):
            rebal = (i % self.rebal_freq == 0)
            txc = 0.0

            # 止损检查：持仓中亏损 > stop_loss_pct 的立即卖出
            stop_sells = []
            if self.stop_loss_pct > 0 and np.any(pw > 0):
                for j in range(ns):
                    if pw[j] > 0 and hs[j] >= 0 and entry_price[j] > 0:
                        if fwd[i, j] < -self.stop_loss_pct:
                            stop_sells.append(j)
                            pw[j] = 0.0
                            hs[j] = -1
                            nt += 1

            if stop_sells:
                remaining = pw.sum()
                if remaining > 0:
                    pw = pw / remaining * remaining
                for j in stop_sells:
                    pass

            if rebal:
                masked_sig = sig[i].copy()
                if self.use_universe_filter and self.um is not None:
                    abs_i = self._backtest_sidx + i
                    masked_sig[~self.um[abs_i]] = -1e10

                nw = alloc.allocate(masked_sig, fwd, i, pw, hs, rh)
                to = float(np.sum(np.abs(nw - pw)))
                txc = 0.5 * to * self.tx_cost
                if to > 0.01:
                    nt += 1

                for j in range(ns):
                    if nw[j] > 0 and pw[j] <= 0:
                        entry_price[j] = 1.0 + fwd[i, j]
                    elif nw[j] > 0 and pw[j] > 0:
                        pass

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
                    entry_price[j] = 1.0 + fwd[i, j]

            rt = float(np.dot(pw, fwd[i]))
            dr[i] = 0.0 if (np.isnan(rt) or np.isinf(rt)) else rt - txc
            rh += 1

        return dr, nt

    def _backtest_covrp(self, sig, fwd, dm, nd, ns):
        dr_base, nt = super()._backtest_covrp(sig, fwd, dm, nd, ns)

        if self.stop_loss_pct <= 0:
            return dr_base, nt

        dr = dr_base.copy()
        pw = np.zeros(ns, dtype=np.float32)
        hs = np.full(ns, -1, dtype=np.int32)
        rh = 0
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
                            nt += 1

            if rebal:
                masked_sig = sig[i].copy()
                if self.use_universe_filter and self.um is not None:
                    abs_i = self._backtest_sidx + i
                    masked_sig[~self.um[abs_i]] = -1e10

                si = np.argsort(-masked_sig)[:self.top_n]
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

                for j in range(ns):
                    if nw[j] > 0 and hs[j] < 0:
                        hs[j] = rh + 1
                    if nw[j] > 0 and pw[j] <= 0:
                        entry_price[j] = 1.0 + fwd[i, j]

                pw = nw
            else:
                mk = dm[i] & (pw > 0)
                if np.any(mk):
                    p2 = pw[mk].copy() / float(np.sum(pw[mk]))
                    pw = np.zeros(ns, dtype=np.float32)
                    pw[mk] = p2

            for j in range(ns):
                if pw[j] > 0 and entry_price[j] <= 0:
                    entry_price[j] = 1.0 + fwd[i, j]

            rt = float(np.dot(pw, fwd[i]))
            dr[i] = 0.0 if (np.isnan(rt) or np.isinf(rt)) else rt - txc
            rh += 1

        return dr, nt


def run_pipeline(
    name: str,
    factor_names: List[str],
    weights: Dict[str, float] = None,
    top_n: int = 40,
    rebal_freq: int = 10,
    min_hold_days: int = 5,
    positioner_type: str = "rp",
    tx_cost: float = 0.002,
    use_v2: bool = False,
    stop_loss_pct: float = 0.08,
) -> Tuple[dict, List[dict]]:
    """运行单次回测，返回指标字典。"""
    signal_builder = None
    if weights:
        signal_builder = make_mf_signal_builder(weights)

    cls = StrategyPipelineV2 if use_v2 else StrategyPipeline
    extra_kwargs = {}
    if use_v2:
        extra_kwargs["stop_loss_pct"] = stop_loss_pct

    p = cls(
        signal_builder=signal_builder,
        name=name,
        rebal_freq=rebal_freq,
        top_n=top_n,
        min_hold_days=min_hold_days,
        positioner_type=positioner_type,
        tx_cost=tx_cost,
        factor_names=factor_names,
        use_universe_filter=True,
        **extra_kwargs,
    )
    result = p.run(start=FULL_RANGE[0], end=FULL_RANGE[1])
    windows = p.window_analysis(WINDOWS)

    result_dict = {
        "annual_return": round(result.annual_return, 4),
        "sharpe": round(result.sharpe, 4),
        "max_drawdown": round(result.max_drawdown, 4),
        "calmar": round(result.calmar, 4),
        "win_rate": round(result.win_rate, 4),
        "n_trades": result.n_trades,
    }

    windows_dict = []
    for w in windows:
        windows_dict.append({
            "window": w.window,
            "annual_return": round(w.annual_return, 4),
            "sharpe": round(w.sharpe, 4),
            "max_drawdown": round(w.max_drawdown, 4),
            "calmar": round(w.calmar, 4),
        })

    return result_dict, windows_dict


def compare_strategy(
    name: str,
    factor_names: List[str],
    weights: Dict[str, float] = None,
    top_n: int = 20,
    rebal_freq: int = 10,
    min_hold_days: int = 5,
    positioner_type: str = "rp",
    label: str = "mf_d10_rp",
):
    """运行 baseline vs V2 对比。"""
    logger.info(f"\n{'='*60}")
    logger.info(f"  {label}  A/B 对比")
    logger.info(f"  配置: top_n={top_n} rf={rebal_freq} mhd={min_hold_days} {positioner_type}")
    logger.info(f"{'='*60}")

    # Baseline
    logger.info(f"\n--- Baseline (原始ST过滤, 无止损) ---")
    base_result, base_windows = run_pipeline(
        name=f"{label}_baseline",
        factor_names=factor_names,
        weights=weights,
        top_n=top_n,
        rebal_freq=rebal_freq,
        min_hold_days=min_hold_days,
        positioner_type=positioner_type,
        use_v2=False,
    )

    # V2: 增强ST + 止损
    logger.info(f"\n--- V2 (增强ST过滤 + 个股-8%止损) ---")
    v2_result, v2_windows = run_pipeline(
        name=f"{label}_v2",
        factor_names=factor_names,
        weights=weights,
        top_n=top_n,
        rebal_freq=rebal_freq,
        min_hold_days=min_hold_days,
        positioner_type=positioner_type,
        use_v2=True,
        stop_loss_pct=0.08,
    )

    return {
        "label": label,
        "config": {
            "top_n": top_n,
            "rebal_freq": rebal_freq,
            "min_hold_days": min_hold_days,
            "positioner": positioner_type,
        },
        "baseline": {
            "full_range": base_result,
            "windows": base_windows,
        },
        "v2_stop_08": {
            "full_range": v2_result,
            "windows": v2_windows,
        },
    }


def print_comparison(comp: dict):
    """打印 A/B 对比结果。"""
    label = comp["label"]
    cfg = comp["config"]
    b = comp["baseline"]
    v2 = comp["v2_stop_08"]

    print(f"\n{'='*72}")
    print(f"  {label}  A/B 对比结果")
    print(f"  配置: top_n={cfg['top_n']} rf={cfg['rebal_freq']} mhd={cfg['min_hold_days']} {cfg['positioner']}")
    print(f"{'='*72}")

    header = f"{'指标':<16} {'Baseline':<18} {'V2(停损+ST)':<18} {'差值':<12}"
    print(header)
    print("-" * 72)

    rows = [
        ("年化收益", "annual_return", "%", lambda x: f"{x*100:.2f}%"),
        ("夏普比率", "sharpe", "", lambda x: f"{x:.3f}"),
        ("最大回撤", "max_drawdown", "%", lambda x: f"{-x*100:.2f}%"),
        ("卡玛比率", "calmar", "", lambda x: f"{x:.3f}"),
        ("胜率", "win_rate", "%", lambda x: f"{x*100:.1f}%"),
        ("交易次数", "n_trades", "", lambda x: f"{x:.0f}"),
    ]

    base_full = b["full_range"]
    v2_full = v2["full_range"]

    for label, key, unit, fmt in rows:
        bv = base_full[key]
        vv = v2_full[key]
        diff = vv - bv if key != "max_drawdown" else bv - vv
        diff_str = f"{diff*100 if key in ('annual_return', 'max_drawdown') else diff * (100 if key == 'win_rate' else 1):+.2f}{'%' if key in ('annual_return', 'max_drawdown', 'win_rate') else ''}" if key != "n_trades" else f"{diff:+d}"
        print(f"  {label:<14} {fmt(bv):<18} {fmt(vv):<18} {diff_str:<>14}")

    print(f"\n  --- 分窗口对比 ---")
    print(f"  {'窗口':<14} {'指标':<12} {'Baseline':<16} {'V2':<16} {'差值':<12}")
    print(f"  {'-'*70}")

    for bw, vw in zip(b["windows"], v2["windows"]):
        wn = bw["window"]
        for key in ("annual_return", "sharpe", "max_drawdown"):
            bv = bw[key]
            vv = vw[key]
            if key == "max_drawdown":
                diff = bv - vv
                unit = "%"
            elif key == "annual_return":
                diff = vv - bv
                unit = "%"
            else:
                diff = vv - bv
                unit = ""

            bfmt = f"{bv*100:.2f}%" if key != "sharpe" else f"{bv:.3f}"
            vfmt = f"{vv*100:.2f}%" if key != "sharpe" else f"{vv:.3f}"
            dfmt = f"{diff*100:+.2f}%" if key != "sharpe" else f"{diff:+.3f}"
            print(f"  {wn:<14} {key:<12} {bfmt:<16} {vfmt:<16} {dfmt:<12}")

    print(f"{'='*72}\n")


def export_results(comp: dict, filename: str):
    """保存结果为 JSON。"""
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(comp, f, indent=2, ensure_ascii=False)
    logger.info(f"已保存: {path}")


def main():
    parser = argparse.ArgumentParser(description="A/B 对比回测")
    parser.add_argument("--strategy", choices=["mf_d10_rp", "chip_covrp", "both"],
                        default="both", help="策略选择")
    args = parser.parse_args()

    db = Database()
    all_factors = db.list_factor_columns()
    db.close()

    from core.strategies.pipeline import DEFAULT_FACTORS
    available = [f for f in DEFAULT_FACTORS if f in all_factors]
    logger.info(f"可用因子: {len(all_factors)} 个, 默认因子可用: {len(available)}/{len(DEFAULT_FACTORS)}")

    chip_available = [f for f in CHIP_FACTORS if f in all_factors]
    logger.info(f"Chip因子可用: {len(chip_available)}/{len(CHIP_FACTORS)}")

    results = []

    # mf_d10_rp 对比
    if args.strategy in ("mf_d10_rp", "both"):
        mf_available_weights = {k: v for k, v in MF_WEIGHTS.items() if k in all_factors}
        logger.info(f"MF权重因子可用: {len(mf_available_weights)}/{len(MF_WEIGHTS)}")

        comp = compare_strategy(
            name="mf_d10_rp",
            factor_names=list(mf_available_weights.keys()),
            weights=mf_available_weights,
            top_n=20,
            rebal_freq=10,
            min_hold_days=5,
            positioner_type="rp",
            label="mf_d10_rp",
        )
        print_comparison(comp)
        export_results(comp, "ab_mf_d10_rp.json")
        results.append(comp)

    # chip_covrp 对比
    if args.strategy in ("chip_covrp", "both"):
        comp = compare_strategy(
            name="chip_covrp",
            factor_names=chip_available,
            weights=None,
            top_n=40,
            rebal_freq=3,
            min_hold_days=3,
            positioner_type="covrp",
            label="chip_covrp",
        )
        print_comparison(comp)
        export_results(comp, "ab_chip_covrp.json")
        results.append(comp)

    logger.info("\n✅ A/B 对比完成!")
    logger.info(f"结果保存至: {RESULTS_DIR}/")
    for r in results:
        logger.info(f"  - {r['label']}: baseline.json + v2.json")


if __name__ == "__main__":
    main()
