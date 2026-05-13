"""
全量 Alpha101 因子计算 + 落库 + IC 评估 + 排序

流程:
  1. 加载 daily_bars(指定区间)
  2. FactorHub.compute_all() 全量计算 101 个因子(long 中间格式)
  3. 写入 factors_wide(宽表),save_factors 自动 long->wide
  4. 调用 FactorEvaluator 评估 IC/IR
  5. 按 |IR| 排序输出 Top-N

用法:
  python3 scripts/compute_alpha101_full.py \\
      --start 2023-01-01 --end 2024-06-30 \\
      --eval-start 2024-01-15 --eval-end 2024-06-30 \\
      --top-n 30
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from core.config import load_config
from core.database import Database
from core.factors.base.factor_hub import FactorHub
import core.factors.impl.alpha101_full  # noqa: F401  触发注册


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--db", default=None, help="覆盖 config.database.path")
    ap.add_argument("--start", default=None, help="计算区间起,默认 config.backtest.start_date")
    ap.add_argument("--end",   default=None, help="计算区间止,默认 config.backtest.end_date")
    ap.add_argument("--eval-start", default=None,
                    help="评估期起,默认 config.backtest.start_date")
    ap.add_argument("--eval-end",   default=None,
                    help="评估期止,默认 config.backtest.end_date")
    ap.add_argument("--top-n", type=int, default=30, help="输出 Top-N IR 因子")
    ap.add_argument("--names", default="", help="只算这些因子,逗号分隔;空=全部")
    ap.add_argument("--skip-write", action="store_true", help="不写库,只评估")
    args = ap.parse_args()

    cfg = load_config(args.config)
    db_path = args.db or cfg["database"]["path"]
    start = args.start or cfg["backtest"]["start_date"]
    end = args.end or cfg["backtest"]["end_date"]
    eval_start = args.eval_start or start
    eval_end = args.eval_end or end

    print(f"[1/4] loading bars {start} ~ {end} ...")
    db = Database(db_path)
    bars = db.get_daily_bars(start_date=start, end_date=end)
    print(f"      bars: {len(bars):,} rows  symbols={bars['symbol'].nunique()}")

    if args.names:
        names = [n.strip() for n in args.names.split(",") if n.strip()]
    else:
        names = sorted(FactorHub.list_by_category("alpha101"),
                       key=lambda x: int(x[1:]))
    print(f"[2/4] computing {len(names)} factors via FactorHub ...")
    t0 = time.time()
    long_df = FactorHub.compute_all(bars, names=names, verbose=True)
    print(f"      done. rows: {len(long_df):,}  ({time.time()-t0:.1f}s)")

    if not args.skip_write:
        print(f"[3/4] writing to factors_wide ...")
        t0 = time.time()
        # save_factors 内部识别 long 格式并 pivot 到宽表,
        # 同时自动 ALTER TABLE ADD COLUMN 扩展新因子
        db.save_factors(long_df)
        print(f"      written {len(long_df):,} rows ({time.time()-t0:.1f}s)")
    else:
        print("[3/4] --skip-write,跳过落库")

    # ---------- 评估 ----------
    print(f"[4/4] evaluating IC/IR on {eval_start} ~ {eval_end} ...")
    from core.research.impl.evaluation import FactorEvaluator
    ev = FactorEvaluator(db)
    summary = ev.evaluate_all(
        factor_names=names,
        start_date=eval_start,
        end_date=eval_end,
    )
    if summary is None or len(summary) == 0:
        print("      [WARN] empty summary, skip ranking")
        return

    if not isinstance(summary, pd.DataFrame):
        summary = pd.DataFrame(summary)

    # 按 |IR| 排序
    if "ir" not in summary.columns:
        print("      [WARN] no 'ir' column, columns =", list(summary.columns))
        print(summary.head())
        return

    summary["abs_ir"] = summary["ir"].abs()
    ranked = summary.sort_values("abs_ir", ascending=False)

    print()
    print("=" * 78)
    print(f"Top-{args.top_n} alpha factors by |IR|  (eval period {eval_start} ~ {eval_end})")
    print("=" * 78)
    cols = [c for c in ["factor_name", "ic_mean", "ic_std", "ir",
                        "t_stat", "ic_positive_ratio", "n_periods"]
            if c in ranked.columns]
    print(ranked[cols].head(args.top_n).to_string(index=False))
    print()
    print("=" * 78)
    print(f"Bottom-10 (|IR| 最低)")
    print("=" * 78)
    print(ranked[cols].tail(10).to_string(index=False))

    # 输出 csv 给后续脚本用
    out_path = ROOT / "data" / "alpha101_eval_summary.csv"
    out_path.parent.mkdir(exist_ok=True)
    ranked.to_csv(out_path, index=False)
    print(f"\n  saved -> {out_path}")


if __name__ == "__main__":
    main()