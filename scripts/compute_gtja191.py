"""
GTJA 191 因子计算 + 落库 + IC 评估 + 检查点恢复

流程:
  1. 检查已有因子（检查点）
  2. 加载 daily_bars(指定区间)
  3. FactorHub.compute_all() 计算缺失的 GTJA 因子
  4. 先保存到临时文件，再写入数据库
  5. 写入失败时保留临时文件供恢复

用法:
  # 正常计算（自动跳过已完成的因子）:
  python3 scripts/compute_gtja191.py

  # 强制重算指定因子:
  python3 scripts/compute_gtja191.py --force gtja1,gtja2

  # 从临时文件恢复并写入数据库:
  python3 scripts/restore_gtja_checkpoint.py

  # 查看状态:
  python3 scripts/compute_gtja191.py --status
"""
from __future__ import annotations
import argparse
import glob
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from core.config import load_config, get_db_path
from core.database import Database
from core.factor_hub import FactorHub, list_by_category, compute_all
import factors.gtja191_full  # noqa: F401  触发注册


def get_temp_dir():
    """获取临时文件目录"""
    temp_dir = ROOT / "data" / "gtja_checkpoints"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def get_completed_factors(db: Database) -> set:
    """获取数据库中已有数据的因子集合"""
    completed = set()
    try:
        cols = db.conn.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'factors_wide'
            AND column_name LIKE 'gtja%'
        """).fetchall()
        for (col,) in cols:
            cnt = db.conn.execute(f"""
                SELECT COUNT(*) FROM factors_wide WHERE "{col}" IS NOT NULL
            """).fetchone()[0]
            if cnt > 0:
                completed.add(col)
    except Exception:
        pass
    return completed


def get_temp_factors() -> set:
    """获取临时文件中已有的因子"""
    temp_dir = get_temp_dir()
    temp_files = glob.glob(str(temp_dir / "*.parquet"))
    return {Path(f).stem for f in temp_files}


def save_checkpoint(factor_name: str, df: pd.DataFrame) -> Path:
    """保存因子到临时文件"""
    temp_dir = get_temp_dir()
    path = temp_dir / f"{factor_name}.parquet"
    df.to_parquet(path, compression="zstd")
    return path


def load_checkpoint(factor_name: str) -> pd.DataFrame:
    """从临时文件加载因子"""
    path = get_temp_dir() / f"{factor_name}.parquet"
    return pd.read_parquet(path)


def remove_checkpoint(factor_name: str):
    """删除临时文件"""
    path = get_temp_dir() / f"{factor_name}.parquet"
    if path.exists():
        path.unlink()


def write_factor_to_db(db: Database, factor_name: str, df: pd.DataFrame):
    """写入因子到数据库"""
    db.save_factors(df)


def compute_and_save_factor(
    db: Database,
    bars: pd.DataFrame,
    factor_name: str,
    force: bool = False
) -> bool:
    """计算单个因子并保存到数据库"""
    # 检查数据库
    if not force:
        cnt = db.conn.execute(f"""
            SELECT COUNT(*) FROM factors_wide WHERE "{factor_name}" IS NOT NULL
        """).fetchone()[0]
        if cnt > 0:
            return True

    # 检查临时文件
    temp_path = get_temp_dir() / f"{factor_name}.parquet"
    if temp_path.exists() and not force:
        df = pd.read_parquet(temp_path)
    else:
        # 计算因子
        t0 = time.time()
        hub = FactorHub()
        result = hub.compute(factor_name, bars)
        elapsed = time.time() - t0

        if result is None or result.empty:
            print(f"  [WARN] {factor_name} computed empty, skipping")
            return False

        df = pd.DataFrame({
            "date": result.index,
            "symbol": result.columns.repeat(len(result)),
        })
        idx = pd.MultiIndex.from_product(
            [result.index, result.columns], names=["date", "symbol"]
        )
        df["factor_name"] = factor_name
        df["value"] = result.values.ravel(order="C")
        df = df[["date", "symbol", "factor_name", "value"]]
        print(f"  {factor_name}  shape={result.shape}  ({elapsed:.2f}s)")

    # 先保存到临时文件
    save_checkpoint(factor_name, df)

    # 写入数据库
    try:
        write_factor_to_db(db, factor_name, df)
        # 成功写入后删除临时文件
        remove_checkpoint(factor_name)
        return True
    except Exception as e:
        print(f"  [ERROR] {factor_name} write failed: {e}")
        print(f"  [INFO]  Saved to {temp_path}, run restore script to retry")
        return False


def print_status(db: Database, all_factors: list):
    """打印因子计算状态"""
    completed = get_completed_factors(db)
    temp_factors = get_temp_factors()

    print(f"\n{'='*60}")
    print(f"GTJA 因子计算状态")
    print(f"{'='*60}")
    print(f"总因子数: {len(all_factors)}")
    print(f"已完成（数据库）: {len(completed)}")
    print(f"待恢复（临时文件）: {len(temp_factors)}")
    print(f"待计算: {len(all_factors) - len(completed) - len(temp_factors)}")
    print()

    if temp_factors:
        print("临时文件中的因子（需恢复）:")
        for f in sorted(temp_factors):
            path = get_temp_dir() / f"{f}.parquet"
            size = path.stat().st_size / 1024 / 1024
            print(f"  - {f} ({size:.1f} MB)")
        print()
        print("运行以下命令恢复:")
        print("  python3 scripts/restore_gtja_checkpoint.py")
        print()

    if completed:
        print("已完成（数据库中）:")
        completed_sorted = sorted(completed, key=lambda x: int(x[4:]) if x[4:].isdigit() else 0)
        print("  " + ", ".join(completed_sorted[:50]))
        if len(completed_sorted) > 50:
            print(f"  ... 等共 {len(completed_sorted)} 个")
        print()


def main():
    ap = argparse.ArgumentParser(description="GTJA 191 因子计算与评估")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--db", default=None, help="覆盖 config.database.path")
    ap.add_argument("--start", default=None, help="计算区间起,默认 config.backtest.start_date")
    ap.add_argument("--end", default=None, help="计算区间止,默认 config.backtest.end_date")
    ap.add_argument("--names", default="", help="只算这些因子,逗号分隔;空=全部")
    ap.add_argument("--force", default="",
                    help="强制重算这些因子,逗号分隔")
    ap.add_argument("--n-jobs", type=int, default=1,
                    help="并行计算的进程数,默认1(串行),-1=全部CPU核心")
    ap.add_argument("--status", action="store_true", help="只打印状态,不计算")
    ap.add_argument("--skip-write", action="store_true", help="不写库,只评估")
    args = ap.parse_args()

    cfg = load_config(args.config)
    db_path = args.db or get_db_path(cfg)
    start = args.start or cfg["backtest"]["start_date"]
    end = args.end or cfg["backtest"]["end_date"]

    db = Database(db_path)

    # 获取所有 GTJA 因子
    all_gtja = list_by_category("gtja191")
    all_factors = sorted(all_gtja, key=lambda x: int(x[4:]) if x[4:].isdigit() else 0)

    # 只打印状态
    if args.status:
        print_status(db, all_factors)
        return

    # 解析要计算的因子
    if args.names:
        names = [n.strip() for n in args.names.split(",") if n.strip()]
    else:
        names = all_factors

    # 解析强制重算的因子
    force_set = set()
    if args.force:
        force_set = {n.strip() for n in args.force.split(",") if n.strip()}

    print(f"[1/3] loading bars {start} ~ {end} ...")
    bars = db.get_daily_bars(start_date=start, end_date=end)
    print(f"      bars: {len(bars):,} rows  symbols={bars['symbol'].nunique()}")

    # 检查已完成和临时文件的因子
    completed = get_completed_factors(db)
    temp_factors = get_temp_factors()

    # 过滤要计算的因子
    to_compute = []
    for name in names:
        if name in force_set:
            to_compute.append(name)
        elif name not in completed and name not in temp_factors:
            to_compute.append(name)

    print(f"[2/3] computing {len(to_compute)}/{len(names)} GTJA factors ...")
    print(f"      parallel: n_jobs={args.n_jobs}")
    print(f"      completed in db: {len(completed)}")
    print(f"      pending in temp: {len(temp_factors)}")
    print(f"      will compute: {len(to_compute)}")

    if not to_compute:
        print("\n所有因子已完成，无需计算")
    else:
        t0 = time.time()

        # 使用 FactorHub 并行计算
        long_df = compute_all(
            bars,
            names=to_compute,
            verbose=True,
            n_jobs=args.n_jobs
        )
        elapsed = time.time() - t0
        print(f"      computed. rows: {len(long_df):,}  ({elapsed:.1f}s)")

        if long_df is None or long_df.empty:
            print("[ERROR] no factors computed")
            return

        if not args.skip_write:
            print(f"[3/3] writing to database + temp files ...")
            t0 = time.time()

            # 按因子分组保存
            successful = []
            failed = []
            for fname in to_compute:
                factor_df = long_df[long_df["factor_name"] == fname]
                if factor_df.empty:
                    continue

                # 保存到临时文件
                temp_path = save_checkpoint(fname, factor_df)

                # 尝试写入数据库
                try:
                    write_factor_to_db(db, fname, factor_df)
                    # 成功写入后删除临时文件
                    remove_checkpoint(fname)
                    successful.append(fname)
                    print(f"  ✓ {fname} saved ({len(factor_df):,} rows)")
                except Exception as e:
                    print(f"  ✗ {fname} write failed: {e}")
                    print(f"    temp file kept: {temp_path}")
                    failed.append((fname, str(e)))

            elapsed = time.time() - t0
            print(f"\n      done. successful: {len(successful)}, failed: {len(failed)}")
            print(f"      total time: {elapsed:.1f}s")

            if failed:
                print("\n      [WARN] 以下因子写入失败，保留临时文件:")
                for fname, err in failed:
                    print(f"        - {fname}: {err}")
                print("\n      运行以下命令重试:")
                print("        python3 scripts/restore_gtja_checkpoint.py")
        else:
            print("[3/3] --skip-write,跳过落库")

    # 打印最终状态
    print()
    print_status(db, all_factors)


if __name__ == "__main__":
    main()
