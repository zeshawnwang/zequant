#!/usr/bin/env python3
"""
初始化数据库
创建所有表结构。
"""
import sys
sys.path.insert(0, '.')

from core.database import Database
from pathlib import Path

def main():
    db_path = "./data/quant_data.db"
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    db = Database(db_path)
    print(f"数据库已初始化: {db_path}")

    # 验证表
    tables = db.conn.execute("SHOW TABLES").df()
    print("表列表:", tables['name'].tolist())

    db.close()

if __name__ == "__main__":
    main()
