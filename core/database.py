"""
DuckDB Database Manager
Single-file database for all data: daily bars, factors, positions, signals.

修复说明(相对旧版):
1. upsert_daily_bars 旧版会在写入前 DELETE 全表(子查询条件恒真),导致数据自毁。
   现改为 INSERT ... ON CONFLICT DO UPDATE,走主键冲突合并。
2. DuckDB Python API 的 DataFrame 写入使用 `conn.register("view", df)` 注册临时视图,
   再在 SQL 里引用;不要用 `params={"df": df}`,那是错误的用法。
3. 查询过滤使用位置参数(`?`)而非命名参数,兼容性更好。
"""
import duckdb
import pandas as pd
from pathlib import Path


# 数据库中 factors 表的有序列名,save_factors 依此挑选字段,防止串列
FACTOR_COLUMNS = [
    "date", "symbol", "close", "returns",
    "momentum_5", "momentum_20",
    "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "boll_upper", "boll_middle", "boll_lower", "boll_position",
    "volume_ratio",
    "volatility_20",
]

DAILY_BAR_COLUMNS = [
    "symbol", "date", "open", "high", "low", "close",
    "volume", "amount", "pct_change",
]


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
                date          DATE,
                symbol        VARCHAR,
                close         DECIMAL(10,3),
                returns       DECIMAL(10,6),
                momentum_5    DECIMAL(10,6),
                momentum_20   DECIMAL(10,6),
                rsi_14        DECIMAL(10,4),
                macd          DECIMAL(10,4),
                macd_signal   DECIMAL(10,4),
                macd_hist     DECIMAL(10,4),
                boll_upper    DECIMAL(10,3),
                boll_middle   DECIMAL(10,3),
                boll_lower    DECIMAL(10,3),
                boll_position DECIMAL(10,4),
                volume_ratio  DECIMAL(10,4),
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

    # ---------- daily_bars ----------

    def upsert_daily_bars(self, df: pd.DataFrame):
        """
        Upsert daily bars by (symbol, date) primary key.
        使用 INSERT ... ON CONFLICT 合并,不会抹掉其他分区数据。
        """
        if df is None or df.empty:
            return
        df = df[[c for c in DAILY_BAR_COLUMNS if c in df.columns]].copy()
        self.conn.register("_stg_bars", df)
        try:
            self.conn.execute("""
                INSERT INTO daily_bars
                    (symbol, date, open, high, low, close, volume, amount, pct_change)
                SELECT symbol, date, open, high, low, close, volume, amount, pct_change
                FROM _stg_bars
                ON CONFLICT (symbol, date) DO UPDATE SET
                    open=excluded.open, high=excluded.high, low=excluded.low,
                    close=excluded.close, volume=excluded.volume, amount=excluded.amount,
                    pct_change=excluded.pct_change
            """)
        finally:
            self.conn.unregister("_stg_bars")

    def get_daily_bars(self, symbol: str = None,
                       start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Query daily bars with optional filters."""
        sql = "SELECT * FROM daily_bars WHERE 1=1"
        params = []
        if symbol:
            sql += " AND symbol = ?"
            params.append(symbol)
        if start_date:
            sql += " AND date >= ?"
            params.append(start_date)
        if end_date:
            sql += " AND date <= ?"
            params.append(end_date)
        sql += " ORDER BY symbol, date"
        return self.conn.execute(sql, params).df()

    def get_max_date(self, table: str, column: str = "date"):
        """Get max date from a table. Returns date/datetime or None."""
        try:
            row = self.conn.execute(f"SELECT MAX({column}) FROM {table}").fetchone()
            return row[0] if row and row[0] is not None else None
        except Exception:
            return None

    def get_symbol_max_date(self, symbol: str):
        """按单只股票取 daily_bars 最大日期(增量起点)。"""
        try:
            row = self.conn.execute(
                "SELECT MAX(date) FROM daily_bars WHERE symbol = ?", [symbol]
            ).fetchone()
            return row[0] if row and row[0] is not None else None
        except Exception:
            return None

    # ---------- factors ----------

    def save_factors(self, df: pd.DataFrame):
        """Save factor data with ON CONFLICT upsert."""
        if df is None or df.empty:
            return
        # 只取 schema 列,避免原始 K 线列污染 INSERT
        cols = [c for c in FACTOR_COLUMNS if c in df.columns]
        if "date" not in cols or "symbol" not in cols:
            raise ValueError("factor df must contain date and symbol")
        df = df[cols].copy()
        self.conn.register("_stg_factors", df)
        set_clause = ", ".join(
            f"{c}=excluded.{c}" for c in cols if c not in ("date", "symbol")
        )
        col_list = ", ".join(cols)
        try:
            self.conn.execute(f"""
                INSERT INTO factors ({col_list})
                SELECT {col_list} FROM _stg_factors
                ON CONFLICT (date, symbol) DO UPDATE SET {set_clause}
            """)
        finally:
            self.conn.unregister("_stg_factors")

    def get_factors(self, symbols: list = None,
                    start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Get factor data."""
        sql = "SELECT * FROM factors WHERE 1=1"
        params = []
        if symbols:
            placeholders = ",".join(["?"] * len(symbols))
            sql += f" AND symbol IN ({placeholders})"
            params.extend(symbols)
        if start_date:
            sql += " AND date >= ?"
            params.append(start_date)
        if end_date:
            sql += " AND date <= ?"
            params.append(end_date)
        sql += " ORDER BY date, symbol"
        return self.conn.execute(sql, params).df()

    # ---------- symbols ----------

    def save_symbols(self, df: pd.DataFrame):
        if df is None or df.empty:
            return
        cols = ["symbol", "name", "list_date", "delist_date", "sector", "market"]
        df = df[[c for c in cols if c in df.columns]].copy()
        self.conn.register("_stg_symbols", df)
        col_list = ", ".join(df.columns)
        set_clause = ", ".join(
            f"{c}=excluded.{c}" for c in df.columns if c != "symbol"
        )
        try:
            self.conn.execute(f"""
                INSERT INTO symbols ({col_list})
                SELECT {col_list} FROM _stg_symbols
                ON CONFLICT (symbol) DO UPDATE SET {set_clause}
            """)
        finally:
            self.conn.unregister("_stg_symbols")

    def get_symbols(self) -> pd.DataFrame:
        return self.conn.execute("SELECT * FROM symbols ORDER BY symbol").df()

    def execute(self, sql: str, *args, **kwargs):
        return self.conn.execute(sql, *args, **kwargs)

    def close(self):
        self.conn.close()