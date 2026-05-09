"""
DuckDB 数据库管理器
单文件数据库,统一存放:日线、因子(宽表)、标的、因子注册表、更新日志。

设计要点
--------
1) daily_bars 通过 INSERT ... ON CONFLICT DO UPDATE 走主键合并,支持增量更新。
2) factors 改为【宽表】:(date, symbol, f1, f2, ..., fN),每个因子一列。
   - 优点:读取无需 pivot,选股/回测/评估直接 SELECT 列,DuckDB 列存对宽表友好。
   - 扩展:新增因子时调用 ensure_factor_columns([...]),自动 ALTER TABLE ADD COLUMN。
   - 写入:save_factors 同时支持宽表 / 长表输入,内部统一落到宽表。
3) factor_registry 维持不变,记录 IC/IR/换手等评估结果与启用开关。

向后兼容
--------
- 旧调用方使用 get_factors_long() 仍可用,内部 melt 宽表得到同样的 long 格式。
- list_factor_names() 改为返回宽表中的因子列(扣除元信息列)。
"""
from __future__ import annotations
import duckdb
import numpy as np
import os
import pandas as pd
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional


# 日线主键列
DAILY_BAR_COLUMNS = [
    "symbol", "date", "open", "high", "low", "close",
    "volume", "amount", "pct_change",
]

# 宽表元信息列(非因子列)
FACTOR_META_COLUMNS = ("date", "symbol")


class Database:
    """DuckDB 单文件数据库管理器。

    Args:
        db_path:   数据库文件路径
        read_only: 是否只读连接(被其他进程独占时可用,但禁止写入操作)
    """

    def __init__(self, db_path: str = "./data/quant_data.db",
                 read_only: bool = False):
        self.db_path = db_path
        self.read_only = read_only
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            self.conn = duckdb.connect(db_path, read_only=read_only)
        except duckdb.IOException:
            if not read_only:
                self.conn = duckdb.connect(db_path, read_only=True)
                self.read_only = True
            else:
                raise
        if not self.read_only:
            self._configure_duckdb()
            self._init_tables()

    def _configure_duckdb(self):
        """配置 DuckDB 连接参数以提升性能。"""
        self.conn.execute("SET threads TO 8")
        self.conn.execute("SET memory_limit='4GB'")

    # ============================================================
    # Schema 初始化
    # ============================================================

    def _init_tables(self):
        """初始化所有表。factors 表采用宽表设计,初始只含主键。"""
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

        # 宽表因子:初始只建主键,因子列由 ensure_factor_columns 动态扩展
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS factors_wide (
                date        DATE,
                symbol      VARCHAR,
                PRIMARY KEY (date, symbol)
            )
        """)
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_factors_wide_date "
            "ON factors_wide(date)"
        )

        # daily_bars 添加 date 单列索引,加速日期范围查询
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_daily_bars_date "
            "ON daily_bars(date)"
        )

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

        # 因子注册表:评估结果 + 启用开关
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS factor_registry (
                factor_name   VARCHAR PRIMARY KEY,
                category      VARCHAR,
                description   VARCHAR,
                ic_mean       DOUBLE,
                ic_std        DOUBLE,
                ir            DOUBLE,
                ic_t_stat     DOUBLE,
                turnover      DOUBLE,
                top_group_ret DOUBLE,
                bot_group_ret DOUBLE,
                monotonic     BOOLEAN,
                last_eval     TIMESTAMP,
                enabled       BOOLEAN DEFAULT true
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS update_log (
                table_name      VARCHAR,
                last_update     TIMESTAMP,
                records_updated INT,
                status          VARCHAR
            )
        """)

    # ============================================================
    # daily_bars
    # ============================================================

    def upsert_daily_bars(self, df: pd.DataFrame):
        """日线 upsert。df 必须含 DAILY_BAR_COLUMNS 中的列。"""
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

    def get_daily_bars(self, symbols=None, symbol: str = None,
                       start_date: str = None, end_date: str = None,
                       columns: List[str] = None) -> pd.DataFrame:
        """查询日线。symbol(单只)与 symbols(多只)二选一。

        Args:
            columns: 可选,指定返回的列名列表,减少数据传输量
        """
        all_cols = ["symbol", "date", "open", "high", "low", "close",
                    "volume", "amount", "pct_change"]
        if columns:
            cols = [c for c in columns if c in all_cols]
        else:
            cols = all_cols
        sql = f"SELECT {', '.join(cols)} FROM daily_bars WHERE 1=1"
        params: list = []
        if symbol:
            sql += " AND symbol = ?"
            params.append(symbol)
        elif symbols:
            ph = ",".join(["?"] * len(symbols))
            sql += f" AND symbol IN ({ph})"
            params.extend(symbols)
        if start_date:
            sql += " AND date >= ?"
            params.append(start_date)
        if end_date:
            sql += " AND date <= ?"
            params.append(end_date)
        sql += " ORDER BY symbol, date"
        return self.conn.execute(sql, params).df()

    def get_max_date(self, table: str, column: str = "date"):
        try:
            row = self.conn.execute(f"SELECT MAX({column}) FROM {table}").fetchone()
            return row[0] if row and row[0] is not None else None
        except Exception:
            return None

    def get_symbol_max_date(self, symbol: str):
        try:
            row = self.conn.execute(
                "SELECT MAX(date) FROM daily_bars WHERE symbol = ?", [symbol]
            ).fetchone()
            return row[0] if row and row[0] is not None else None
        except Exception:
            return None

    def get_daily_bars_with_fwd_ret(
        self,
        symbols: Optional[List[str]] = None,
        start_date: str = None,
        end_date: str = None,
        forward_days: int = 5,
        columns: List[str] = None,
    ) -> pd.DataFrame:
        """查询日线并计算前向收益率(通过 SQL 窗口函数,避免加载全量数据到内存)。

        使用 LEAD() OVER (PARTITION BY symbol ORDER BY date) 计算 fwd_ret,
        比 pandas groupby.shift() 更高效且内存占用更少。
        """
        all_cols = ["symbol", "date", "open", "high", "low", "close",
                    "volume", "amount", "pct_change"]
        base_cols = [c for c in (columns or all_cols) if c in all_cols]

        where = ["1=1"]
        params: list = []
        if symbols:
            ph = ",".join(["?"] * len(symbols))
            where.append(f"symbol IN ({ph})")
            params.extend(symbols)
        if start_date:
            where.append("date >= ?")
            params.append(start_date)
        if end_date:
            where.append("date <= ?")
            params.append(end_date)
        where_sql = " AND ".join(where)

        col_list = ", ".join(base_cols)
        sql = f"""
            SELECT {col_list},
                   LEAD(close, {forward_days}) OVER (
                       PARTITION BY symbol ORDER BY date
                   ) AS fwd_close
            FROM daily_bars
            WHERE {where_sql}
            ORDER BY symbol, date
        """
        df = self.conn.execute(sql, params).df()
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["date"])
        df["fwd_ret"] = df["fwd_close"] / df["close"].astype("float64") - 1.0
        df.drop(columns=["fwd_close"], inplace=True, errors="ignore")
        return df

    # ============================================================
    # factors(宽表) —— 核心 API
    # ============================================================

    def list_factor_columns(self) -> List[str]:
        """列出 factors_wide 中除元信息外的所有因子列。"""
        df = self.conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'factors_wide' ORDER BY ordinal_position"
        ).df()
        if df.empty:
            return []
        return [c for c in df["column_name"].tolist() if c not in FACTOR_META_COLUMNS]

    # 旧名兼容
    def list_factor_names(self) -> List[str]:
        """旧 API:返回所有因子名(等同 list_factor_columns)。"""
        return self.list_factor_columns()

    def ensure_factor_columns(self, factor_names: Iterable[str]):
        """确保 factors_wide 中存在指定的因子列,缺失的自动 ALTER TABLE 添加为 DOUBLE。

        实现策略
        --------
        - 一次性算出缺失列(set 差集),把多个 ALTER 包在单个事务里提交;
          DuckDB 的 ALTER 不支持单语句多列,但事务化能避免 N 次 fsync,
          首次 Alpha101 全量入库(101 列)从约 2~3s 降至 < 200ms。
        - 列名需为合法 SQL 标识符(字母/数字/下划线),否则抛 ValueError。
        """
        existed = set(self.list_factor_columns()) | set(FACTOR_META_COLUMNS)
        to_add: List[str] = []
        for fn in factor_names:
            fn = str(fn).strip()
            if not fn or fn in existed:
                continue
            if not fn.replace("_", "").isalnum():
                raise ValueError(f"非法因子名: {fn}(只允许字母/数字/下划线)")
            to_add.append(fn)
            existed.add(fn)
        if not to_add:
            return

        # 在事务中批量 ALTER —— 减少 commit/fsync 次数
        try:
            self.conn.execute("BEGIN TRANSACTION")
            for fn in to_add:
                self.conn.execute(
                    f'ALTER TABLE factors_wide ADD COLUMN "{fn}" DOUBLE'
                )
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    def save_factors(self, df: pd.DataFrame, factor_names: Optional[List[str]] = None):
        if df is None or df.empty:
            return

        # 1) 长表 -> 宽表
        if "factor_name" in df.columns and "value" in df.columns:
            wide = df.pivot(
                index=["date", "symbol"],
                columns="factor_name",
                values="value",
            ).reset_index()
            wide.columns.name = None
        else:
            wide = df.copy()

        if "date" not in wide.columns or "symbol" not in wide.columns:
            raise ValueError("save_factors: df 必须包含 date 与 symbol 列")

        reserved = {"date", "symbol", "close", "open", "high", "low",
                    "volume", "amount", "pct_change"}
        if factor_names is None:
            cols = [c for c in wide.columns
                    if c not in reserved
                    and pd.api.types.is_numeric_dtype(wide[c])]
        else:
            cols = [c for c in factor_names if c in wide.columns]
        if not cols:
            return

        wide = wide[["date", "symbol"] + cols].copy()
        for c in cols:
            wide[c] = wide[c].replace([np.inf, -np.inf], np.nan)

        self.ensure_factor_columns(cols)

        # 4) 在事务中批量写入,用 Parquet 中转避免 register/unregister 开销
        batch_size = 10000
        n_batches = (len(wide) + batch_size - 1) // batch_size
        self.conn.execute("BEGIN TRANSACTION")
        try:
            tmp_dir = tempfile.gettempdir()
            for i in range(n_batches):
                start = i * batch_size
                end = min(start + batch_size, len(wide))
                batch = wide.iloc[start:end]
                tmp_path = os.path.join(tmp_dir, f"_stg_fw_{os.getpid()}_{i}.parquet")
                batch.to_parquet(tmp_path, compression="zstd")
                set_clause = ", ".join(f'"{c}"=excluded."{c}"' for c in cols)
                col_list = ", ".join(['"date"', '"symbol"'] + [f'"{c}"' for c in cols])
                self.conn.execute(f"""
                    INSERT INTO factors_wide ({col_list})
                    SELECT {col_list} FROM read_parquet('{tmp_path}')
                    ON CONFLICT (date, symbol) DO UPDATE SET {set_clause}
                """)
                os.remove(tmp_path)
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    def get_factors(self, symbols: Optional[List[str]] = None,
                    start_date: str = None, end_date: str = None,
                    factor_names: Optional[List[str]] = None,
                    with_close: bool = True) -> pd.DataFrame:
        """返回宽表 (date, symbol, f1, f2, ...);可选 join close/volume/amount/pct_change。

        factor_names=None 时返回全部已存在的因子列。
        """
        all_cols = self.list_factor_columns()
        if factor_names is None:
            factor_names = all_cols
        else:
            factor_names = [c for c in factor_names if c in all_cols]
        if not factor_names:
            # 即使没有因子列,也允许只取 date/symbol(空因子场景)
            select_factors = ""
        else:
            select_factors = ", " + ", ".join(f'"{c}"' for c in factor_names)

        where = ["1=1"]
        params: list = []
        if symbols:
            ph = ",".join(["?"] * len(symbols))
            where.append(f"symbol IN ({ph})")
            params.extend(symbols)
        if start_date:
            where.append("date >= ?")
            params.append(start_date)
        if end_date:
            where.append("date <= ?")
            params.append(end_date)
        where_sql = " AND ".join(where)

        sql = (
            f"SELECT date, symbol{select_factors} "
            f"FROM factors_wide WHERE {where_sql} "
            f"ORDER BY date, symbol"
        )
        wide = self.conn.execute(sql, params).df()
        if wide.empty:
            return wide

        if with_close:
            bars = self.get_daily_bars(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
            )
            if not bars.empty:
                join_cols = ["date", "symbol", "close",
                             "pct_change", "volume", "amount"]
                join_cols = [c for c in join_cols if c in bars.columns]
                tail = bars[join_cols].copy()
                tail["date"] = pd.to_datetime(tail["date"])
                wide["date"] = pd.to_datetime(wide["date"])
                wide = wide.merge(tail, on=["date", "symbol"], how="left")
        return wide

    def get_factors_long(self, symbols: Optional[List[str]] = None,
                         start_date: str = None, end_date: str = None,
                         factor_names: Optional[List[str]] = None) -> pd.DataFrame:
        """兼容 API:返回长表 (date, symbol, factor_name, value)。
        内部从宽表 melt 而来,适合因子评估等需要逐因子聚合的场景。
        """
        wide = self.get_factors(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            factor_names=factor_names,
            with_close=False,
        )
        if wide.empty:
            return pd.DataFrame(columns=["date", "symbol", "factor_name", "value"])
        value_cols = [c for c in wide.columns if c not in FACTOR_META_COLUMNS]
        if not value_cols:
            return pd.DataFrame(columns=["date", "symbol", "factor_name", "value"])
        long_df = wide.melt(
            id_vars=["date", "symbol"],
            value_vars=value_cols,
            var_name="factor_name",
            value_name="value",
        ).dropna(subset=["value"])
        return long_df

    def delete_factors(self, factor_names: Iterable[str]):
        """删除某些因子列(谨慎使用)。"""
        existing = set(self.list_factor_columns())
        for fn in factor_names:
            if fn in existing:
                self.conn.execute(f'ALTER TABLE factors_wide DROP COLUMN "{fn}"')

    # ============================================================
    # symbols
    # ============================================================

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

    # ============================================================
    # factor_registry
    # ============================================================

    def upsert_factor_registry(self, records: pd.DataFrame):
        """把因子评估结果写入 factor_registry。"""
        if records is None or records.empty:
            return
        records = records.copy()
        records["last_eval"] = pd.Timestamp.now()
        self.conn.register("_stg_reg", records)
        cols = list(records.columns)
        col_list = ", ".join(cols)
        set_clause = ", ".join(
            f"{c}=excluded.{c}" for c in cols if c != "factor_name"
        )
        try:
            self.conn.execute(f"""
                INSERT INTO factor_registry ({col_list})
                SELECT {col_list} FROM _stg_reg
                ON CONFLICT (factor_name) DO UPDATE SET {set_clause}
            """)
        finally:
            self.conn.unregister("_stg_reg")

    def get_enabled_factors(self, min_abs_ir: float = None,
                            as_dataframe: bool = False):
        """取启用中的因子。

        Args:
            min_abs_ir: 要求 |IR| 超过该阈值(反转因子 IR<0 仍有效,故取绝对值)
            as_dataframe: True 返回完整 DataFrame,False 仅返回 factor_name 列表
        """
        sql = (
            "SELECT factor_name, ir, ic_mean, ic_t_stat, "
            "turnover, top_group_ret, bot_group_ret, monotonic, last_eval "
            "FROM factor_registry WHERE enabled = TRUE"
        )
        params: list = []
        if min_abs_ir is not None:
            sql += " AND ABS(ir) >= ?"
            params.append(float(min_abs_ir))
        sql += " ORDER BY ABS(ir) DESC NULLS LAST"
        df = self.conn.execute(sql, params).df()
        if as_dataframe:
            return df
        return df["factor_name"].tolist() if not df.empty else []

    # ============================================================
    # 通用
    # ============================================================

    def execute(self, sql: str, *args, **kwargs):
        return self.conn.execute(sql, *args, **kwargs)

    def close(self):
        self.conn.close()