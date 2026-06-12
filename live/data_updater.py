"""数据更新 — 多源兜底批量补充最新日线数据。

自动按 akshare → baostock → efinance 顺序尝试，
第一个成功返回数据的源即被采用，全部失败则跳过该标的继续下一只。
支持 A 股股票和 LOF/ETF 基金。

用法：
    python3 -m live.data_updater                    # 默认更新(回溯5天)
    python3 -m live.data_updater --days 3            # 回溯3天
    python3 -m live.data_updater --skip-factors      # 跳过因子重算
    python3 -m live.data_updater --funds-only        # 只更新基金
    python3 -m live.data_updater --no-funds          # 不更新基金
    python3 -m live.data_updater --sync-funds       # 同步基金列表到数据库
"""
from __future__ import annotations
import logging
import os
import sys
import time
from datetime import date, datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.database import Database
from core.datasource.fallback_fetcher import FallbackFetcher
from core.datasource import FUND_PREFIXES, is_fund_symbol

logger = logging.getLogger("live.data_updater")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DB_PATH = "data/quant_data.db"
BATCH_COMMIT = 100
REPORT_EVERY = 200


def _fund_like_pattern() -> str:
    """用 REGEXP 替代多 OR LIKE 拼接，避免字符串拼接 SQL。"""
    prefixes = sorted(set(p for p in FUND_PREFIXES if p), key=len, reverse=True)
    pattern = "|".join(f"^{p}" for p in prefixes)
    return pattern


def get_active_symbols(db: Database, min_days: int = 60, include_funds: bool = True) -> list:
    """获取活跃标的列表（包括股票和基金）"""
    if include_funds:
        rows = db.conn.execute(
            "SELECT symbol FROM daily_bars GROUP BY symbol HAVING COUNT(*) >= ? ORDER BY COUNT(*) DESC",
            [min_days]
        ).fetchall()
    else:
        # 排除基金，只获取股票（用 REGEXP 替代 LIKE 拼接）
        pattern = _fund_like_pattern()
        rows = db.conn.execute(
            "SELECT symbol FROM daily_bars WHERE symbol !~ ? GROUP BY symbol HAVING COUNT(*) >= ? ORDER BY COUNT(*) DESC",
            [pattern, min_days]
        ).fetchall()
    return [r[0] for r in rows]


def get_fund_symbols(db: Database) -> list:
    """获取本地数据库中的基金代码列表"""
    pattern = _fund_like_pattern()
    rows = db.conn.execute(
        "SELECT DISTINCT symbol FROM daily_bars WHERE symbol ~ ?",
        [pattern]
    ).fetchall()
    return [r[0] for r in rows]


def sync_fund_list_to_db(db: Database) -> int:
    """从 akshare 同步 LOF/ETF 基金列表到数据库"""
    try:
        from core.datasource.sources.akshare_source import AkshareSource
        source = AkshareSource()
        fund_df = source.fetch_fund_symbols()

        if fund_df is None or fund_df.empty:
            logger.warning("无法获取基金列表")
            return 0

        # 过滤已有数据的基金（避免重复）
        existing = set(get_fund_symbols(db))
        new_funds = fund_df[~fund_df["symbol"].isin(existing)]

        if new_funds.empty:
            logger.info("基金列表已是最新，无需同步")
            return 0

        # 保存到 symbols 表
        new_funds = new_funds.copy()
        new_funds["market"] = new_funds["symbol"].apply(
            lambda x: "SH" if x.startswith(("5", "51", "52", "56", "58")) else "SZ"
        )
        new_funds["list_date"] = None
        new_funds["delist_date"] = None
        new_funds["sector"] = new_funds.get("type", "FUND")

        db.save_symbols(new_funds[["symbol", "name", "market", "list_date", "delist_date", "sector"]])
        logger.info("✅ 基金列表同步完成: 新增 %d 只基金", len(new_funds))
        return len(new_funds)

    except Exception as e:
        logger.warning("基金列表同步失败: %s", e)
        return 0


def get_latest_dates(db: Database, symbols: list) -> dict:
    """批量查询所有股票最新日期（单条 SQL，替代逐只查询）。"""
    if not symbols:
        return {}
    ph = ",".join("?" for _ in symbols)
    rows = db.conn.execute(
        f"SELECT symbol, MAX(date) FROM daily_bars WHERE symbol IN ({ph}) GROUP BY symbol",
        symbols
    ).fetchall()
    return {r[0]: str(r[1]) if r[1] else "2000-01-01" for r in rows}


def _to_date(val: str | date) -> date:
    """将字符串 'YYYY-MM-DD' 转为 date 对象。"""
    return val if isinstance(val, date) else date.fromisoformat(val)


def update_bars(days_back: int = 5, max_stocks: int = 3000,
               include_funds: bool = True, funds_only: bool = False):
    """用多源兜底链批量补充日线数据。

    Args:
        days_back: 回溯天数
        max_stocks: 限制每次更新的标的数量（避免超时）
        include_funds: 是否包含基金更新
        funds_only: 是否只更新基金
    """
    db = Database(DB_PATH)
    today = date.today()
    start_date_str = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    # 获取标的列表
    if funds_only:
        symbols = get_fund_symbols(db)
        logger.info("基金模式: 本地基金 %d 只", len(symbols))
    else:
        symbols = get_active_symbols(db, include_funds=include_funds)
        logger.info("数据库最新: %s  今日: %s  回溯起始: %s", "N/A", today, start_date_str)

    if not symbols:
        logger.warning("无活跃标的")
        db.close()
        return True

    latest = get_latest_dates(db, symbols)
    global_latest = "2000-01-01"
    for v in latest.values():
        if v > global_latest:
            global_latest = v

    if not funds_only:
        logger.info("数据库最新: %s  今日: %s  回溯起始: %s", global_latest, today, start_date_str)

    today_s = today.isoformat()
    need_update = [s for s in symbols if latest.get(s, "2000-01-01") < today_s]
    need_update = need_update[:max_stocks]  # 限制数量

    # 分离股票和基金
    stock_list = [s for s in need_update if not is_fund_symbol(s)]
    fund_list = [s for s in need_update if is_fund_symbol(s)]

    if funds_only:
        logger.info("需要更新的基金: %d 只 (本次限制 %d 只)", len(need_update), max_stocks)
    else:
        logger.info("活跃标的: %d 只(股票 %d, 基金 %d), 需要更新: %d 只(股票 %d, 基金 %d)",
                    len(symbols), len(stock_list), len(fund_list),
                    len(need_update), len(stock_list), len(fund_list))

    if not need_update:
        logger.info("所有标的已是最新")
        db.close()
        return True

    fetcher = FallbackFetcher()
    total_new = 0
    db.conn.execute("BEGIN TRANSACTION")

    for i, sym in enumerate(need_update):
        stock_start = _to_date(latest.get(sym, global_latest))
        if stock_start >= today:
            continue
        df = fetcher.fetch_bars(sym, stock_start.isoformat(), today_s)
        if df is None or df.empty:
            continue
        new_rows = df[df["date"] > stock_start]
        if new_rows.empty:
            continue
        import math
        batch = []
        for _, row in new_rows.iterrows():
            def _v(v, t=float):
                try:
                    return t(v) if not (isinstance(v, float) and math.isnan(v)) else t()
                except (ValueError, TypeError):
                    return t()
            batch.append((
                sym, str(row["date"]),
                _v(row.get("open")), _v(row.get("high")),
                _v(row.get("low")), _v(row.get("close")),
                _v(row.get("volume"), int), _v(row.get("amount")),
                _v(row.get("pct_change"))
            ))
        if batch:
            db.conn.executemany("""
                INSERT INTO daily_bars (symbol,date,open,high,low,close,volume,amount,pct_change)
                VALUES (?,?,?,?,?,?,?,?,?)
                ON CONFLICT (symbol,date) DO UPDATE SET
                    open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low,
                    close=EXCLUDED.close, volume=EXCLUDED.volume, amount=EXCLUDED.amount,
                    pct_change=EXCLUDED.pct_change
            """, batch)
            total_new += len(batch)

        if (i + 1) % REPORT_EVERY == 0:
            db.conn.execute("COMMIT")
            db.conn.execute("BEGIN TRANSACTION")
            logger.info("  进度 %d/%d, 已新增 %d 条", i + 1, len(need_update), total_new)

    db.conn.execute("COMMIT")
    new_latest = db.conn.execute("SELECT MAX(date) FROM daily_bars").fetchone()[0]
    logger.info("更新完成! 新增 %d 条, 数据库最新: %s", total_new, new_latest)
    db.close()
    return True


def recompute_factors():
    """重算因子（失败不中断流程）。"""
    try:
        from core.datasource.daily_updater import DailyUpdater
        updater = DailyUpdater()
        updater.run(fetch_bars=False, compute_factors=True, compute_technical=True)
        logger.info("因子重算完成")
        return True
    except Exception as e:
        logger.warning("因子重算失败(非关键): %s", e)
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="实盘数据更新(多源兜底)")
    parser.add_argument("--days", type=int, default=5, help="回溯天数")
    parser.add_argument("--max-stocks", type=int, default=3000, help="每次最大更新标的数")
    parser.add_argument("--skip-factors", action="store_true", help="跳过因子重算")
    parser.add_argument("--funds-only", action="store_true", help="只更新基金")
    parser.add_argument("--no-funds", action="store_true", help="不更新基金")
    parser.add_argument("--sync-funds", action="store_true", help="同步基金列表到数据库")
    args = parser.parse_args()

    db = Database(DB_PATH)

    # 同步基金列表
    if args.sync_funds:
        sync_fund_list_to_db(db)
        db.close()
        return

    # 确定基金更新策略
    if args.funds_only:
        include_funds = False
        funds_only = True
    elif args.no_funds:
        include_funds = False
        funds_only = False
    else:
        include_funds = True
        funds_only = False

    ok = update_bars(
        days_back=args.days,
        max_stocks=args.max_stocks,
        include_funds=include_funds,
        funds_only=funds_only
    )

    if ok and not args.skip_factors:
        recompute_factors()

    db.close()


if __name__ == "__main__":
    main()
