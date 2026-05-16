#!/usr/bin/env python3
"""
初始化数据库
创建所有表结构。
"""
import sys
import os
# 使用 append 而非 insert(0, ...) 以免遮蔽 Python 标准库同名包(如 selectors)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

from pathlib import Path
from core.datasourcesourcebase import Database


def main():
    db_path = "./data/quant_data.db"
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    db = Database(db_path)
    print(f"数据库已初始化: {db_path}")

    # 验证表(使用 information_schema 更可靠)
    tables = db.conn.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema='main' ORDER BY table_name"
    ).df()
    print("表列表:", tables['table_name'].tolist())

    db.close()

if __name__ == "__main__":
    main()
