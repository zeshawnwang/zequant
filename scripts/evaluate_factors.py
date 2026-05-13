#!/usr/bin/env python3
"""基于 FactorHub + FactorEvaluator 的因子评估与入库脚本。

使用:
    python3 scripts/evaluate_factors.py                                          # 评估所有 technical 因子
    python3 scripts/evaluate_factors.py --category alpha101 --names a1 a3 a5     # 指定因子
    python3 scripts/evaluate_factors.py --category alpha101 --years 3            # 近3年
    python3 scripts/evaluate_factors.py --ir 0.3                                 # 只注册 IC>0.3
    python3 scripts/evaluate_factors.py --save                                   # 将结果写入数据库
"""
from __future__ import annotations
import sys
import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

import argparse
from datetime import datetime, timedelta
from typing import Dict

from core.config import load_config
from core.database import Database
from core.factors.base.factor_hub import FactorHub
from core.research.impl.evaluation import FactorEvaluator

import core.factors.impl.technical  # noqa: F401  触发注册
import core.factors.impl.alpha101_full  # noqa: F401
import core.factors.impl.gtja191_full  # noqa: F401
import core.factors.impl.fama_french  # noqa: F401


def _build_meta_maps() -> tuple[Dict[str, str], Dict[str, str]]:
    cat_map: Dict[str, str] = {}
    desc_map: Dict[str, str] = {}
    for name in FactorHub.list_all():
        meta = FactorHub.get(name)
        cat_map[name] = meta.category or "misc"
        desc_map[name] = meta.description or ""
    return cat_map, desc_map


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--category", default=None,
                        help="评估某分类,默认所有因子")
    parser.add_argument("--names", nargs="*", default=None,
                        help="指定因子名,覆盖 --category")
    parser.add_argument("--start", default=None,
                        help="开始日期,默认3年前")
    parser.add_argument("--end", default=None,
                        help="结束日期,默认昨天")
    parser.add_argument("--groups", type=int, default=5,
                        help="分组数,默认5")
    parser.add_argument("--years", type=int, default=None,
                        help="评估窗口(年),覆盖 --start")
    parser.add_argument("--ir", type=float, default=-999,
                        help="写入数据库的 IC IR 阈值")
    parser.add_argument("--save", action="store_true",
                        help="将评估结果写入 factor_registry")
    args = parser.parse_args()

    cfg = load_config(args.config)
    db_path = cfg["database"]["path"]
    db = Database(db_path)

    end = args.end or (datetime.now() - timedelta(1)).strftime("%Y-%m-%d")
    if args.years:
        y = int(end[:4]) - args.years
        start = f"{y}-01-01"
    else:
        start = args.start or (datetime.now() - timedelta(365 * 3)).strftime("%Y-%m-%d")

    if args.names:
        factor_names = args.names
    elif args.category:
        factor_names = FactorHub.list_by_category(args.category)
    else:
        factor_names = FactorHub.list_all()

    cat_map, desc_map = _build_meta_maps()

    print(f"评估 {len(factor_names)} 个因子: {start} ~ {end}")

    ev = FactorEvaluator(db)
    summary = ev.evaluate_all(
        factor_names=factor_names,
        start_date=start,
        end_date=end,
        forward_days=1,
        n_groups=args.groups,
    )
    ev.print_summary(summary, desc_map)

    if args.save:
        ir_used = args.ir if args.ir > -999 else None
        records = ev.to_registry_records(
            summary,
            category_map=cat_map,
            description_map=desc_map,
            ir_threshold=ir_used,
        )
        ev.write_to_registry(records)
        print(f"写入 {len(records)} 条因子登记记录")

    db.close()


if __name__ == "__main__":
    main()
