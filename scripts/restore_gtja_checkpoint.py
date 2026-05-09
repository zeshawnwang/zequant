"""
从临时文件恢复并写入数据库

用法:
  python3 scripts/restore_gtja_checkpoint.py

功能:
  1. 扫描 data/gtja_checkpoints/ 目录下的临时文件
  2. 逐个写入数据库
  3. 成功后删除临时文件
  4. 记录失败的文件供下次重试
"""
from __future__ import annotations
import argparse
import glob
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from core.config import load_config, get_db_path
from core.database import Database


def get_temp_dir():
    """获取临时文件目录"""
    return ROOT / "data" / "gtja_checkpoints"


def get_pending_checkpoints() -> list:
    """获取待恢复的临时文件列表"""
    temp_dir = get_temp_dir()
    temp_dir.mkdir(parents=True, exist_ok=True)
    files = glob.glob(str(temp_dir / "*.parquet"))
    return sorted(files, key=lambda x: Path(x).stem)


def write_checkpoint(db: Database, fpath: Path) -> bool:
    """写入单个临时文件到数据库"""
    factor_name = fpath.stem
    df = pd.read_parquet(fpath)

    if df.empty:
        print(f"  [WARN] {factor_name}: empty file, removing")
        fpath.unlink()
        return True

    try:
        db.save_factors(df)
        fpath.unlink()
        return True
    except Exception as e:
        print(f"  [ERROR] {factor_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="恢复 GTJA 因子临时文件")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--db", default=None, help="覆盖 config.database.path")
    parser.add_argument("--dry-run", action="store_true",
                        help="只显示待恢复的文件，不实际写入")
    args = parser.parse_args()

    cfg = load_config(args.config)
    db_path = args.db or get_db_path(cfg)
    temp_dir = get_temp_dir()

    pending = get_pending_checkpoints()

    if not pending:
        print("没有待恢复的临时文件")
        return

    print(f"\n{'='*60}")
    print(f"恢复 GTJA 因子临时文件")
    print(f"{'='*60}")
    print(f"临时文件目录: {temp_dir}")
    print(f"待恢复文件数: {len(pending)}")
    print()

    for i, fpath in enumerate(pending):
        factor_name = fpath.stem
        size_mb = fpath.stat().st_size / 1024 / 1024
        print(f"[{i+1}/{len(pending)}] {factor_name} ({size_mb:.1f} MB)")

        if args.dry_run:
            print(f"  [DRY RUN] would restore {factor_name}")
            continue

        success = write_checkpoint(Database(db_path), fpath)
        if success:
            print(f"  ✓ {factor_name} restored and removed")
        else:
            print(f"  ✗ {factor_name} failed, will retry next time")

    print()
    remaining = get_pending_checkpoints()
    if remaining:
        print(f"[WARN] {len(remaining)} 个文件恢复失败:")
        for f in remaining:
            print(f"  - {Path(f).stem}")
        print("\n下次运行 'python3 scripts/restore_gtja_checkpoint.py' 重试")
    else:
        print("✓ 所有临时文件已恢复")


if __name__ == "__main__":
    main()
