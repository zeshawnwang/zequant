"""
2026-05-17 策略重评估管道。

基于修复后的 StrategyPipeline（Universe过滤 + 真实交易成本）重新评估所有策略。

用法：
    # MF参数扫描
    python3 daily/2026-05-17/tuning_pipeline.py --experiment mf_params

    # Chip参数扫描
    python3 daily/2026-05-17/tuning_pipeline.py --experiment chip_params

    # 验证单个已沉淀策略
    python3 daily/2026-05-17/tuning_pipeline.py --experiment validate --strategy mf_d10_rp

    # 生成汇总报告
    python3 daily/2026-05-17/tuning_pipeline.py --experiment summary
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
from datetime import datetime
from itertools import product
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.strategies.pipeline import StrategyPipeline, BacktestMetrics
from core.database import Database

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("tuning_0517")

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


def run_pipeline(
    name: str,
    factor_names: List[str],
    top_n: int = 40,
    rebal_freq: int = 10,
    min_hold_days: int = 5,
    positioner_type: str = "rp",
    tx_cost: float = 0.002,
) -> Tuple[BacktestMetrics, List[BacktestMetrics]]:
    """使用修复后的 Pipeline 回测一个策略配置。"""
    p = StrategyPipeline(
        signal_builder=None,
        name=name,
        rebal_freq=rebal_freq,
        top_n=top_n,
        min_hold_days=min_hold_days,
        positioner_type=positioner_type,
        tx_cost=tx_cost,
        factor_names=factor_names,
        use_universe_filter=True,
    )
    result = p.run(start=FULL_RANGE[0], end=FULL_RANGE[1])
    windows = p.window_analysis(WINDOWS)
    return result, windows


def experiment_mf_params():
    """实验1：MF核心参数扫描。"""
    db = Database()
    all_factors = db.list_factor_columns()
    logger.info("可用因子: %d 个", len(all_factors))

    # 使用 DEFAULT_FACTORS (从 pipeline 导入)
    from core.strategies.pipeline import DEFAULT_FACTORS
    factor_names = [f for f in DEFAULT_FACTORS if f in all_factors]
    logger.info("使用 %d/%d 个默认因子", len(factor_names), len(DEFAULT_FACTORS))

    params = list(product(
        [20, 30, 40, 50],    # top_n
        [3, 5, 10],           # rebal_freq
        [3, 5, 10],           # min_hold_days
        ['rp', 'covrp'],      # positioner_type
    ))

    results = []
    logger.info("MF参数扫描: %d 个实验", len(params))

    for top_n, rf, mhd, ptype in params:
        name = f"mf_tn{top_n}_rf{rf}_mhd{mhd}_{ptype}"
        try:
            result, windows = run_pipeline(
                name=name,
                factor_names=factor_names,
                top_n=top_n,
                rebal_freq=rf,
                min_hold_days=mhd,
                positioner_type=ptype,
            )
            entry = {
                "name": name,
                "top_n": top_n,
                "rebal_freq": rf,
                "min_hold_days": mhd,
                "positioner": ptype,
                "annual_return": round(result.annual_return, 4),
                "sharpe": round(result.sharpe, 4),
                "max_drawdown": round(result.max_drawdown, 4),
                "calmar": round(result.calmar, 4),
                "win_rate": round(result.win_rate, 4),
                "n_trades": result.n_trades,
            }
            # 熊市与OOS窗口
            for w in windows:
                if "熊市" in w.window:
                    entry["bear_return"] = round(w.annual_return, 4)
                    entry["bear_sharpe"] = round(w.sharpe, 4)
                    entry["bear_drawdown"] = round(w.max_drawdown, 4)
                elif "OOS" in w.window:
                    entry["oos_return"] = round(w.annual_return, 4)
                    entry["oos_sharpe"] = round(w.sharpe, 4)
                    entry["oos_drawdown"] = round(w.max_drawdown, 4)

            results.append(entry)
            logger.info(f"  ✅ {name}: 年化={result.annual_return*100:.1f}% "
                        f"Sharpe={result.sharpe:.3f} 回撤={result.max_drawdown*100:.1f}%")
        except Exception as e:
            logger.error(f"  ❌ {name}: {e}")

    results.sort(key=lambda r: r.get("sharpe", 0), reverse=True)
    out_path = os.path.join(RESULTS_DIR, "mf_params_scan.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("MF扫描完成，结果写入 %s", out_path)
    logger.info("Top 5 按Sharpe:")
    for r in results[:5]:
        logger.info("  %s: Sharpe=%.3f 年化=%.1f%% 回撤=%.1f%%",
                    r["name"], r["sharpe"],
                    r["annual_return"]*100, r["max_drawdown"]*100)


def experiment_validate(strategy_name: str):
    """验证单个已沉淀策略在修正Pipeline下的表现。"""
    from core.strategies.impl import hub as strategy_hub

    # 获取策略的因子列表（简化：使用 DEFAULT_FACTORS）
    from core.strategies.pipeline import DEFAULT_FACTORS
    db = Database()
    all_factors = db.list_factor_columns()
    factor_names = [f for f in DEFAULT_FACTORS if f in all_factors]

    logger.info("验证策略: %s", strategy_name)

    result, windows = run_pipeline(
        name=strategy_name,
        factor_names=factor_names,
        top_n=30,
        rebal_freq=10,
        min_hold_days=5,
    )

    logger.info("=" * 60)
    logger.info("策略: %s (修正Pipeline)", strategy_name)
    logger.info("全区间: 年化=%.2f%% Sharpe=%.3f 回撤=%.2f%%",
                result.annual_return*100, result.sharpe, result.max_drawdown*100)
    for w in windows:
        if w.n_days == 0:
            continue
        logger.info("  %s: 年化=%.2f%% Sharpe=%.3f 回撤=%.2f%%",
                    w.window, w.annual_return*100, w.sharpe, w.max_drawdown*100)
    logger.info("=" * 60)

    out_data = {
        "strategy": strategy_name,
        "pipeline_version": "fixed_0517",
        "full_range": {
            "annual_return": round(result.annual_return, 4),
            "sharpe": round(result.sharpe, 4),
            "max_drawdown": round(result.max_drawdown, 4),
        },
        "windows": [{
            "name": w.window,
            "annual_return": round(w.annual_return, 4),
            "sharpe": round(w.sharpe, 4),
            "max_drawdown": round(w.max_drawdown, 4),
        } for w in windows if w.n_days > 0],
    }

    out_path = os.path.join(RESULTS_DIR, f"validate_{strategy_name}.json")
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)


def experiment_summary():
    """汇总所有验证结果。"""
    import glob
    files = glob.glob(os.path.join(RESULTS_DIR, "validate_*.json"))
    rows = []
    for fp in files:
        with open(fp) as f:
            data = json.load(f)
        full = data.get("full_range", {})
        windows = {w["name"]: w for w in data.get("windows", [])}
        bear = windows.get("2022熊市", {})
        oos = windows.get("修复牛OOS", {})

        rows.append({
            "策略": data["strategy"],
            "年化%": round(full.get("annual_return", 0) * 100, 2),
            "Sharpe": full.get("sharpe", 0),
            "回撤%": round(full.get("max_drawdown", 0) * 100, 2),
            "熊市年化%": round(bear.get("annual_return", 0) * 100, 2),
            "OOS年化%": round(oos.get("annual_return", 0) * 100, 2),
            "OOS Sharpe": oos.get("sharpe", 0),
        })

    rows.sort(key=lambda r: r["Sharpe"], reverse=True)

    lines = []
    lines.append("# ZEquant 策略重评估汇总 (2026-05-17)")
    lines.append("")
    lines.append(f"> 修正项: Universe过滤(ST/新股/涨跌停/停牌) + 真实tx_cost=0.002")
    lines.append("")
    lines.append("| 排名 | 策略 | 年化% | Sharpe | 回撤% | 熊市年化% | OOS年化% | OOS Sharpe |")
    lines.append("|:---:|:----|:----:|:-----:|:----:|:---------:|:--------:|:----------:|")
    for i, r in enumerate(rows, 1):
        lines.append(f"| {i} | {r['策略']} | {r['年化%']} | {r['Sharpe']:.3f} | {r['回撤%']} | {r['熊市年化%']} | {r['OOS年化%']} | {r['OOS Sharpe']:.3f} |")

    summary = "\n".join(lines)
    out_path = os.path.join(RESULTS_DIR, "SUMMARY.md")
    with open(out_path, "w") as f:
        f.write(summary)
    logger.info("汇总报告已写入 %s", out_path)
    print(summary)


def main():
    parser = argparse.ArgumentParser(description="2026-05-17 策略重评估管道")
    parser.add_argument("--experiment", choices=["mf_params", "chip_params", "validate", "summary"],
                        default="summary")
    parser.add_argument("--strategy", default="mf_d10_rp", help="验证的策略名")
    args = parser.parse_args()

    if args.experiment == "mf_params":
        experiment_mf_params()
    elif args.experiment == "chip_params":
        logger.info("Chip参数扫描待实现（需定义Chip因子列表）")
    elif args.experiment == "validate":
        experiment_validate(args.strategy)
    elif args.experiment == "summary":
        experiment_summary()


if __name__ == "__main__":
    main()
