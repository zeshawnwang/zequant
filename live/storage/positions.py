"""持仓快照管理 — 每日收盘记录持仓到独立数据库。"""
from __future__ import annotations
import json
import logging
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from core.database import Database

logger = logging.getLogger(__name__)


class PositionStorage:
    LIVE_DB_PATH = "./data_live/live_data.db"

    def __init__(self):
        self.db = Database(self.LIVE_DB_PATH)
        self._init_tables()

    def _init_tables(self):
        """初始化实盘数据库表。"""
        self.db.conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_snapshots (
                date         DATE,
                strategy     VARCHAR,
                total_value  DOUBLE,
                cash         DOUBLE,
                positions    JSON,
                orders       JSON,
                created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.db.conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                trade_id    VARCHAR PRIMARY KEY,
                date        DATE,
                symbol      VARCHAR,
                direction   VARCHAR,
                price       DOUBLE,
                shares      INT,
                amount      DOUBLE,
                fee         DOUBLE,
                strategy    VARCHAR,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.db.conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_performance (
                date            DATE PRIMARY KEY,
                total_value     DOUBLE,
                daily_return    DOUBLE,
                cumulative      DOUBLE,
                max_drawdown    DOUBLE,
                positions_count INT,
                turnover        DOUBLE,
                benchmark_ret   DOUBLE,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

    def save_snapshot(self, strategy: str = "default", total_value: float = 0.0,
                      cash: float = 0.0, positions: dict = None, orders: list = None):
        """保存当日持仓快照。"""
        today = datetime.now().strftime("%Y-%m-%d")
        self.db.conn.execute("""
            INSERT INTO daily_snapshots (date, strategy, total_value, cash, positions, orders)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (date, strategy) DO NOTHING
        """, [today, strategy, total_value, cash,
              json.dumps(positions or {}), json.dumps(orders or [])])
        logger.info("持仓快照已保存: %s", today)
