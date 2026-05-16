"""逐笔成交记录。"""
from __future__ import annotations
import logging
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)


class TradeRecorder:
    def __init__(self, db):
        self.db = db

    def record(self, symbol: str, direction: str, price: float,
               shares: int, fee: float, strategy: str = "") -> str:
        """记录一笔成交。"""
        trade_id = f"{datetime.now().strftime('%Y%m%d')}_{symbol}_{uuid.uuid4().hex[:8]}"
        amount = price * shares
        try:
            self.db.conn.execute("""
                INSERT INTO trades (trade_id, date, symbol, direction, price, shares, amount, fee, strategy)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [trade_id, datetime.now().strftime("%Y-%m-%d"), symbol,
                  direction, price, shares, amount, fee, strategy])
        except Exception as e:
            logger.error("记录成交失败: %s", e)
        return trade_id
