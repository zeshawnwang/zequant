from __future__ import annotations
import json
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

from core.database import Database

logger = logging.getLogger("record_live")

LIVE_DB_PATH = "./data_live/live_data.db"
QUANT_DB_PATH = "./data/quant_data.db"


@dataclass
class Trade:
    symbol: str
    direction: str
    shares: int
    price: float


def _init_live_db(db: Database):
    db.conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_snapshots (
            date         DATE, strategy     VARCHAR,
            total_value  DOUBLE, cash       DOUBLE,
            positions    JSON, orders       JSON,
            created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    db.conn.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_snapshot_date_strategy
        ON daily_snapshots (date, strategy)
    """)
    db.conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            trade_id    VARCHAR PRIMARY KEY, date   DATE,
            symbol      VARCHAR, direction VARCHAR,
            price       DOUBLE, shares     INT,
            amount      DOUBLE, fee        DOUBLE DEFAULT 0,
            strategy    VARCHAR DEFAULT '', created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    db.conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_performance (
            date            DATE PRIMARY KEY, total_value     DOUBLE,
            daily_return    DOUBLE, cumulative      DOUBLE DEFAULT 0,
            max_drawdown    DOUBLE DEFAULT 0, positions_count INT DEFAULT 0,
            turnover        DOUBLE DEFAULT 0, benchmark_ret   DOUBLE DEFAULT 0,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    db.conn.commit()


def get_last_snapshot(live_db: Database, strategy: Optional[str] = None) -> dict:
    if strategy:
        row = live_db.conn.execute("""
            SELECT date, total_value, cash, positions, orders
            FROM daily_snapshots WHERE strategy=? ORDER BY date DESC LIMIT 1
        """, [strategy]).fetchone()
    else:
        row = live_db.conn.execute("""
            SELECT date, total_value, cash, positions, orders
            FROM daily_snapshots ORDER BY date DESC LIMIT 1
        """).fetchone()
    if row:
        return {
            "date": row[0], "total_value": row[1], "cash": row[2],
            "positions": json.loads(row[3]) if row[3] else {},
            "orders": json.loads(row[4]) if row[4] else [],
        }
    return {"date": None, "total_value": 0, "cash": 0, "positions": {}, "orders": []}


def get_closing_prices(quant_db: Database, symbols: List[str], trade_date: str) -> Dict[str, float]:
    if not symbols:
        return {}
    ph = ",".join("?" for _ in symbols)
    rows = quant_db.conn.execute(
        f"SELECT symbol, close FROM daily_bars WHERE date=? AND symbol IN ({ph})",
        [trade_date] + symbols
    ).fetchall()
    return {r[0]: float(r[1]) for r in rows}


def get_latest_close(quant_db: Database, symbol: str, before_date: str) -> Optional[float]:
    row = quant_db.conn.execute("""
        SELECT close FROM daily_bars
        WHERE symbol=? AND date<=? ORDER BY date DESC LIMIT 1
    """, [symbol, before_date]).fetchone()
    return float(row[0]) if row else None


def get_benchmark_return(quant_db: Database, trade_date: str) -> float:
    row = quant_db.conn.execute("""
        SELECT pct_change FROM daily_bars
        WHERE symbol='000300' AND date=?
    """, [trade_date]).fetchone()
    return float(row[0]) / 100.0 if row else 0.0


def calc_position_value(positions: Dict[str, int], prices: Dict[str, float],
                        quant_db: Database, trade_date: str) -> float:
    total = 0.0
    for sym, shares in positions.items():
        p = prices.get(sym) or get_latest_close(quant_db, sym, trade_date) or 0
        total += p * shares
    return total


def save_trades(live_db: Database, trades: List[Trade], strategy: str, trade_date: str):
    for t in trades:
        tid = f"{trade_date}_{t.symbol}_{t.direction}_{abs(t.shares)}"
        amount = t.price * t.shares
        live_db.conn.execute("""
            INSERT INTO trades (trade_id,date,symbol,direction,price,shares,amount,strategy)
            VALUES (?,?,?,?,?,?,?,?) ON CONFLICT (trade_id) DO NOTHING
        """, [tid, trade_date, t.symbol, t.direction, t.price, t.shares, amount, strategy])


def save_snapshot(live_db: Database, trade_date: str, strategy: str,
                  total_value: float, cash: float, positions: dict, orders: list):
    live_db.conn.execute("""
        INSERT INTO daily_snapshots (date,strategy,total_value,cash,positions,orders)
        VALUES (?,?,?,?,?,?) ON CONFLICT (date,strategy) DO UPDATE SET
            total_value=EXCLUDED.total_value, cash=EXCLUDED.cash,
            positions=EXCLUDED.positions, orders=EXCLUDED.orders
    """, [trade_date, strategy, total_value, cash,
          json.dumps(positions), json.dumps(orders)])


def save_performance(live_db: Database, trade_date: str, total_value: float,
                     last_value: float, position_count: int, turnover: float,
                     benchmark_ret: float, net_cashflow: float = 0):
    net_change = total_value - net_cashflow
    daily_ret = (net_change - last_value) / last_value if last_value > 0 else 0.0
    prev = live_db.conn.execute("""
        SELECT cumulative FROM daily_performance ORDER BY date DESC LIMIT 1
    """).fetchone()
    prev_cum = float(prev[0]) if prev else 1.0
    cumulative = prev_cum * (1 + daily_ret)
    max_cum = live_db.conn.execute("SELECT MAX(cumulative) FROM daily_performance").fetchone()
    peak = max(float(max_cum[0]), cumulative) if max_cum and max_cum[0] else cumulative
    max_dd = (cumulative - peak) / peak if peak > 0 else 0.0
    live_db.conn.execute("""
        INSERT INTO daily_performance (date,total_value,daily_return,cumulative,
            max_drawdown,positions_count,turnover,benchmark_ret)
        VALUES (?,?,?,?,?,?,?,?) ON CONFLICT (date) DO UPDATE SET
            total_value=EXCLUDED.total_value, daily_return=EXCLUDED.daily_return,
            cumulative=EXCLUDED.cumulative, max_drawdown=EXCLUDED.max_drawdown,
            positions_count=EXCLUDED.positions_count, turnover=EXCLUDED.turnover,
            benchmark_ret=EXCLUDED.benchmark_ret
    """, [trade_date, total_value, daily_ret, cumulative, max_dd,
          position_count, turnover, benchmark_ret])


def parse_trade(text: str) -> Optional[Trade]:
    parts = text.strip().split()
    if len(parts) < 4:
        return None
    sym = parts[0].zfill(6)
    d = parts[1].upper()
    direction = "B" if d in ("B", "BUY", "买入") else "S" if d in ("S", "SELL", "卖出", "卖") else ""
    if not direction:
        return None
    try:
        return Trade(symbol=sym, direction=direction, shares=int(parts[2]), price=float(parts[3]))
    except ValueError:
        return None
