"""清理 factors_wide:按白名单保留因子列,其余 DROP COLUMN,然后 CHECKPOINT 压缩回收磁盘。

用法:
  python3 scripts/cleanup_factors.py --keep a1,a29,a57,a60,a96,a98
  python3 scripts/cleanup_factors.py --clear-all
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.datasourcesourcebase import Database


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/quant_data.db")
    ap.add_argument("--keep", default="", help="保留的因子列名,逗号分隔")
    ap.add_argument("--clear-all", action="store_true", help="删除所有因子列")
    args = ap.parse_args()

    db = Database(args.db)
    existing = db.list_factor_columns()
    print(f"[before] factors_wide columns = {len(existing)}  -> {existing[:10]}...")

    if args.clear_all:
        db.delete_factors(existing)
        print(f"  dropped all {len(existing)} factor columns")
    elif args.keep:
        keep = set(s.strip() for s in args.keep.split(",") if s.strip())
        to_drop = [c for c in existing if c not in keep]
        db.delete_factors(to_drop)
        print(f"  dropped {len(to_drop)} columns,kept: {sorted(keep & set(existing))}")
    else:
        print("nothing to do (use --keep or --clear-all)")
        return

    after = db.list_factor_columns()
    print(f"[after ] factors_wide columns = {len(after)}")

    print("[checkpoint] compacting DuckDB file ...")
    db.conn.execute("CHECKPOINT")
    print("  done.")


if __name__ == "__main__":
    main()