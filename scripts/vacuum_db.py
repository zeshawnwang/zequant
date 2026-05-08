"""回收 DuckDB 磁盘空间(VACUUM / 缩库)。

DuckDB 在 DROP TABLE / 大量 UPDATE 后,磁盘文件并不会自动收缩。
本脚本通过 EXPORT DATABASE → 新建空库 → IMPORT 的方式重建,
输出一份干净紧凑的 .db 文件。

流程:
  1. 把现有 quant_data.db EXPORT 到临时 parquet 目录
  2. 新建空白 quant_data.db.new,IMPORT 导入所有表
  3. 把旧库重命名为 quant_data.db.old(保留备份,用户可手动删除)
  4. 把新库重命名为 quant_data.db
  5. 清理临时 export 目录

用法:
  python3 scripts/vacuum_db.py
  python3 scripts/vacuum_db.py --keep-old    # 保留 .old 备份(默认保留)
  python3 scripts/vacuum_db.py --delete-old  # 缩库成功后立即删除 .old 备份
"""
from __future__ import annotations
import argparse
import shutil
import sys
import time
from pathlib import Path

import duckdb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/quant_data.db")
    ap.add_argument("--export-dir", default="data/_export_tmp",
                    help="临时 EXPORT 目录(完成后清理)")
    ap.add_argument("--delete-old", action="store_true",
                    help="缩库成功后立即删除 .old 备份,默认保留")
    args = ap.parse_args()

    db_path = Path(args.db)
    new_path = db_path.with_suffix(db_path.suffix + ".new")
    old_path = db_path.with_suffix(db_path.suffix + ".old")
    export_dir = Path(args.export_dir)

    if not db_path.exists():
        print(f"[error] 数据库 {db_path} 不存在")
        sys.exit(1)

    size_before = db_path.stat().st_size
    print(f"[1/5] 当前库大小: {size_before / 2**30:.2f} GB  ({db_path})")

    # 清理已有的临时输出
    for p in (new_path, export_dir):
        if p.exists():
            shutil.rmtree(p) if p.is_dir() else p.unlink()

    t0 = time.time()
    print(f"[2/5] EXPORT DATABASE -> {export_dir} (parquet) ...")
    con = duckdb.connect(str(db_path), read_only=True)
    con.execute(f"EXPORT DATABASE '{export_dir}' (FORMAT PARQUET)")
    con.close()
    print(f"      用时 {time.time()-t0:.1f}s")

    t0 = time.time()
    print(f"[3/5] 新建空库 + IMPORT DATABASE -> {new_path} ...")
    con = duckdb.connect(str(new_path))
    con.execute(f"IMPORT DATABASE '{export_dir}'")
    con.execute("CHECKPOINT")
    con.close()
    print(f"      用时 {time.time()-t0:.1f}s")

    size_after = new_path.stat().st_size
    print(f"[4/5] 新库大小: {size_after / 2**30:.2f} GB  "
          f"(节省 {(size_before - size_after) / 2**30:.2f} GB, "
          f"压缩率 {100 * (1 - size_after / size_before):.1f}%)")

    print(f"[5/5] 替换旧文件 ...")
    if old_path.exists():
        old_path.unlink()
    db_path.rename(old_path)
    new_path.rename(db_path)
    shutil.rmtree(export_dir)
    print(f"      旧库备份: {old_path}")
    if args.delete_old:
        old_path.unlink()
        print(f"      已删除 {old_path}")
    print("[done] 完成")


if __name__ == "__main__":
    main()