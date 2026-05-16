#!/usr/bin/env python3
"""数据库交互式查询工具。

用法:
  python3 scripts/db_query.py                 # 打印数据库摘要
  python3 scripts/db_query.py "SQL语句"       # 执行任意 SQL
  python3 scripts/db_query.py --shell         # 进入交互式 SQL 命令行

示例:
  python3 scripts/db_query.py "SELECT * FROM daily_bars WHERE symbol='000001' LIMIT 5"
  python3 scripts/db_query.py "SELECT symbol, close FROM factors_wide WHERE date='2025-05-08' LIMIT 10"
"""
import sys
import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

import pandas as pd
from core.datasourcesourcebase import Database


def print_summary(db: Database) -> None:
    """打印数据库摘要:表名、行数、字段结构、数据覆盖。"""
    conn = db.conn
    tables = conn.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema='main' ORDER BY table_name"
    ).df()["table_name"].tolist()

    print(f"\n数据库:{db.db_path}  共 {len(tables)} 张表\n")
    for t in tables:
        n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"  - {t:<20s} {n:>12,} 行")

    print("\n" + "=" * 60)
    for t in tables:
        cols = conn.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name = ? AND table_schema='main' "
            "ORDER BY ordinal_position",
            [t],
        ).df()
        print(f"\n[{t}]")
        # factors_wide 列数可能上百,只展示前 10 列
        if t == "factors_wide" and len(cols) > 12:
            print(cols.head(10).to_string(index=False))
            print(f"... (共 {len(cols)} 列,省略 {len(cols)-10} 个因子列)")
        else:
            print(cols.to_string(index=False))

    print("\n" + "=" * 60)
    print("数据覆盖")
    print("=" * 60)
    if "daily_bars" in tables:
        r = conn.execute(
            "SELECT COUNT(DISTINCT symbol), COUNT(*), MIN(date), MAX(date) "
            "FROM daily_bars"
        ).fetchone()
        print(f"  daily_bars:   {r[0]} 只股票, {r[1]:,} 条, {r[2]} ~ {r[3]}")
    if "factors_wide" in tables:
        r = conn.execute(
            "SELECT COUNT(DISTINCT symbol), COUNT(*), MIN(date), MAX(date) "
            "FROM factors_wide"
        ).fetchone()
        factor_cols = db.list_factor_columns()
        print(f"  factors_wide: {r[0]} 只股票, {r[1]:,} 条, "
              f"{r[2]} ~ {r[3]}, {len(factor_cols)} 个因子列")
    if "symbols" in tables:
        r = conn.execute("SELECT COUNT(*) FROM symbols").fetchone()
        print(f"  symbols:      {r[0]} 只股票元信息")
    if "factor_registry" in tables:
        r = conn.execute(
            "SELECT COUNT(*), SUM(CASE WHEN enabled THEN 1 ELSE 0 END) "
            "FROM factor_registry"
        ).fetchone()
        print(f"  factor_registry: {r[0]} 个因子评估记录,{r[1] or 0} 个已启用")
    print()


def run_sql(db: Database, sql: str) -> None:
    """执行一条 SQL 并打印结果。"""
    try:
        df = db.conn.execute(sql).df()
        if df.empty:
            print("(无结果)")
        else:
            pd.set_option("display.max_columns", None)
            pd.set_option("display.width", 200)
            print(df.to_string(index=False))
            print(f"\n[{len(df)} 行]")
    except Exception as e:
        print(f"SQL 错误: {e}")


def interactive_shell(db: Database) -> None:
    """简易交互式 SQL shell。"""
    print("进入 SQL shell(输入 .exit 退出,.tables 列出表)")
    while True:
        try:
            sql = input("sql> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not sql:
            continue
        if sql in (".exit", ".quit", "exit", "quit"):
            break
        if sql == ".tables":
            run_sql(db, "SELECT table_name FROM information_schema.tables "
                        "WHERE table_schema='main'")
            continue
        run_sql(db, sql.rstrip(";"))


def main() -> None:
    db = Database()
    try:
        if len(sys.argv) == 1:
            print_summary(db)
        elif sys.argv[1] == "--shell":
            interactive_shell(db)
        else:
            run_sql(db, " ".join(sys.argv[1:]))
    finally:
        db.close()


if __name__ == "__main__":
    main()