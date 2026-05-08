#!/usr/bin/env python3
"""因子评估脚本:批量算 IC/IR/分组收益/换手率,落库到 factor_registry。

设计要点
--------
- 默认值统一从 [`config/config.yaml`](../config/config.yaml) 的 `factors` / `database`
  / `backtest` 段读取,命令行参数仅作为按需覆盖
- category / description 直接从 [`FactorHub`](../core/factor_hub.py:1) 元信息反查,
  不再写死映射,新增因子无需改本脚本

用法
----
    python3 scripts/evaluate_factors.py
    python3 scripts/evaluate_factors.py --start 2024-01-01 --end 2024-06-30 --days 10
    python3 scripts/evaluate_factors.py --names momentum_20 rsi_14 a3
"""
from __future__ import annotations
import sys
import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

import argparse
from typing import Dict

import pandas as pd

from core.config import load_config, get_db_path
from core.database import Database
from core.factor_evaluator import FactorEvaluator
from core.factor_hub import FactorHub
import factors  # noqa: F401  触发 @register_factor 注册


def _build_meta_maps() -> tuple:
    """从 FactorHub 反查 category / description,作为 registry 的元信息源。"""
    cat_map: Dict[str, str] = {}
    desc_map: Dict[str, str] = {}
    for name in FactorHub.list_all():
        meta = FactorHub.get(name)
        cat_map[name] = meta.category or "misc"
        desc_map[name] = meta.description or ""
    return cat_map, desc_map


def main() -> None:
    parser = argparse.ArgumentParser(description="因子评估(IC/IR/分组/换手)")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--start", default=None, help="评估起始日,默认读 config.backtest.start_date")
    parser.add_argument("--end",   default=None, help="评估结束日,默认读 config.backtest.end_date")
    parser.add_argument("--days",  type=int, default=None,
                        help="前瞻收益窗口,默认读 config.factors.forward_days")
    parser.add_argument("--groups", type=int, default=5, help="分组数")
    parser.add_argument("--ir_threshold", type=float, default=None,
                        help="|IR|<阈值的因子 enabled=False,默认读 config.factors.ir_threshold")
    parser.add_argument("--names", nargs="*", default=None,
                        help="只评估这些因子;省略则评估库中全部因子列")
    args = parser.parse_args()

    cfg = load_config(args.config)
    start = args.start or cfg["backtest"]["start_date"]
    end = args.end or cfg["backtest"]["end_date"]
    days = args.days if args.days is not None else int(cfg["factors"]["forward_days"])
    ir_thr = args.ir_threshold if args.ir_threshold is not None else float(cfg["factors"]["ir_threshold"])

    db = Database(get_db_path(cfg))
    ev = FactorEvaluator(db)

    factor_names = args.names or db.list_factor_columns()
    if not factor_names:
        print("[error] 库中无因子列,请先 compute_factors / compute_alpha101_full")
        db.close()
        sys.exit(1)

    print(f"准备评估 {len(factor_names)} 个因子(前 10 个: {factor_names[:10]}{'...' if len(factor_names) > 10 else ''})")
    print(f"期间: {start} ~ {end},前瞻 {days} 日,分组 {args.groups},|IR|阈值 {ir_thr}")
    print()

    summary = ev.evaluate_all(
        factor_names=factor_names,
        start_date=start,
        end_date=end,
        forward_days=days,
        n_groups=args.groups,
    )

    print("\n===== 评估汇总(按 |IR| 降序) =====")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(summary.to_string(index=False))

    cat_map, desc_map = _build_meta_maps()
    records = ev.to_registry_records(
        summary,
        category_map=cat_map,
        description_map=desc_map,
        ir_threshold=ir_thr,
    )
    db.upsert_factor_registry(records)

    enabled = db.get_enabled_factors(min_abs_ir=ir_thr)
    print(f"\n[OK] 已写入 factor_registry,启用 |IR|≥{ir_thr} 的因子 {len(enabled)} 个: {enabled[:20]}{'...' if len(enabled) > 20 else ''}")

    db.close()


if __name__ == "__main__":
    main()