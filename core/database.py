"""
DuckDB Database Manager
Single-file database for all data: daily bars, factors, positions, signals.
"""
import os
import duckdb
import pandas as pd
from pathlib import Path


class Database:
    """DuckDB single-file database manager."""

    def __init__(self, db_path: str = "./data/quant_data.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(db_path)
        self._init_tables()

    def _init_tables(self):
        """Initialize all tables."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_bars (
                symbol      VARCHAR,
                date        DATE,
                open        DECIMAL(10,3),
                high        DECIMAL(10,3),
                low         DECIMAL(10,3),
                close       DECIMAL(10,3),
                volume      BIGINT,
                amount      DECIMAL(20,3),
                pct_change  DECIMAL(10,4),
                PRIMARY KEY (symbol, date)
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS factors (
                date        DATE,
                symbol      VARCHAR,
                close       DECIMAL(10,3),
                returns     DECIMAL(10,6),
                momentum_5  DECIMAL(10,6),
                momentum_20 DECIMAL(10,6),
                rsi_14      DECIMAL(10,4),
                macd        DECIMAL(10,4),
                macd_signal DECIMAL(10,4),
                macd_hist   DECIMAL(10,4),
                boll_upper  DECIMAL(10,3),
                boll_middle DECIMAL(10,3),
                boll_lower  DECIMAL(10,3),
                boll_position DECIMAL(10,4),
                volume_ratio DECIMAL(10,4),
                volatility_20 DECIMAL(10,6),
                PRIMARY KEY (date, symbol)
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS symbols (
                symbol      VARCHAR PRIMARY KEY,
                name        VARCHAR,
                list_date   DATE,
                delist_date DATE,
                sector      VARCHAR,
                market      VARCHAR
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS factor_registry (
                factor_name VARCHAR PRIMARY KEY,
                category    VARCHAR,
                description VARCHAR,
                params      VARCHAR,
                last_update TIMESTAMP,
                is_active   BOOLEAN DEFAULT true
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS update_log (
                table_name  VARCHAR,
                last_update TIMESTAMP,
                records_updated INT,
                status      VARCHAR
            )
        """)

    def upsert_daily_bars(self, df: pd.DataFrame):
        """Upsert daily bars (INSERT OR REPLACE)."""
        if df.empty:
            return
        self.conn.execute("DELETE FROM daily_bars WHERE (symbol, date) IN (SELECT symbol, date FROM daily_bars)")
        self.conn.execute(
            "INSERT INTO daily_bars SELECT * FROM df",
            params={"df": df}
        )

    def get_daily_bars(self, symbol: str = None, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Query daily bars with optional filters."""
        sql = "SELECT * FROM daily_bars WHERE 1=1"
        params = {}
        if symbol:
            sql += " AND symbol = :symbol"
            params["symbol"] = symbol
        if start_date:
            sql += " AND date >= :start_date"
            params["start_date"] = start_date
        if end_date:
            sql += " AND date <= :end_date"
            params["end_date"] = end_date
        sql += " ORDER BY date"
        return self.conn.execute(sql, params=params).df()

    def get_max_date(self, table: str, column: str = "date") -> str:
        """Get max date from a table."""
        try:
            result = self.conn.execute(f"SELECT MAX({column}) FROM {table}").fetchone()
            return result[0] if result and result[0] else None
        except:
            return None

    def save_factors(self, df: pd.DataFrame):
        """Save factor data."""
        if df.empty:
            return
        self.conn.execute(
            "INSERT INTO factors (date, symbol, close, returns, momentum_5, momentum_20, "
            "rsi_14, macd, macd_signal, macd_hist, boll_upper, boll_middle, boll_lower, "
            "boll_position, volume_ratio, volatility_20) "
            "VALUES (:date, :symbol, :close, :returns, :momentum_5, :momentum_20, "
            ":rsi_14, :macd, :macd_signal, :macd_hist, :boll_upper, :boll_middle, :boll_lower, "
            ":boll_position, :volume_ratio, :volatility_20) "
            "ON CONFLICT (date, symbol) DO UPDATE SET "
            "close=excluded.close, returns=excluded.returns, momentum_5=excluded.momentum_5, "
            "momentum_20=excluded.momentum_20, rsi_14=excluded.rsi_14, macd=excluded.macd, "
            "macd_signal=excluded.macd_signal, macd_hist=excluded.macd_hist, "
            "boll_upper=excluded.boll_upper, boll_middle=excluded.boll_middle, "
            "boll_lower=excluded.boll_lower, boll_position=excluded.boll_position, "
            "volume_ratio=excluded.volume_ratio, volatility_20=excluded.volatility_20",
            params={"df": df}
        )

    def get_factors(self, symbols: list = None, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Get factor data."""
        sql = "SELECT * FROM factors WHERE 1=1"
        params = {}
        if symbols:
            sql += " AND symbol IN :symbols"
            params["symbols"] = symbols
        if start_date:
            sql += " AND date >= :start_date"
            params["start_date"] = start_date
        if end_date:
            sql += " AND date <= :end_date"
            params["end_date"] = end_date
        sql += " ORDER BY date, symbol"
        return self.conn.execute(sql, params=params).df()

    def save_symbols(self, df: pd.DataFrame):
        """Save symbol master data."""
        if df.empty:
            return
        self.conn.execute(
            "INSERT INTO symbols (symbol, name, list_date, delist_date, sector, market) "
            "VALUES (:symbol, :name, :list_date, :delist_date, :sector, :market) "
            "ON CONFLICT (symbol) DO UPDATE SET "
            "name=excluded.name, list_date=excluded.list_date, delist_date=excluded.delist_date, "
            "sector=excluded.sector, market=excluded.market",
            params={"df": df}
        )

    def get_symbols(self) -> pd.DataFrame:
        """Get all active symbols."""
        return self.conn.execute("SELECT * FROM symbols").df()

    def execute(self, sql: str, **kwargs):
        """Execute raw SQL."""
        return self.conn.execute(sql, **kwargs)

    def close(self):
        self.conn.close()
