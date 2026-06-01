"""数据更新 — 多源兜底批量补充最新日线数据。

自动按 akshare → baostock → efinance 顺序尝试，
第一个成功返回数据的源即被采用，全部失败则跳过该股票继续下一只。

用法：
    python3 -m live.data_updater                    # 默认更新(回溯5天)
    python3 -m live.data_updater --days 3            # 回溯3天
    python3 -m live.data_updater --skip-factors      # 跳过因子重算
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

logger = logging.getLogger("live.data_updater")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DB_PATH = "data/quant_data.db"
BATCH_COMMIT = 100
REPORT_EVERY = 200


def get_active_symbols(db: Database, min_days: int = 60) -> list:
    rows = db.conn.execute(
        "SELECT symbol FROM daily_bars GROUP BY symbol HAVING COUNT(*) >= ? ORDER BY COUNT(*) DESC",
        [min_days]
    ).fetchall()
    return [r[0] for r in rows]


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


def update_bars(days_back: int = 5, max_stocks: int = 200):
    """用多源兜底链批量补充日线数据。

    Args:
        days_back: 回溯天数
        max_stocks: 限制每次更新的股票数量（避免超时）
    """
    db = Database(DB_PATH)
    today = date.today()
    start_date_str = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    symbols = get_active_symbols(db)
    latest = get_latest_dates(db, symbols)
    global_latest = "2000-01-01"
    for v in latest.values():
        if v > global_latest:
            global_latest = v
    logger.info("数据库最新: %s  今日: %s  回溯起始: %s", global_latest, today, start_date_str)

    today_s = today.isoformat()
    need_update = [s for s in symbols if latest.get(s, "2000-01-01") < today_s]
    need_update = need_update[:max_stocks]  # 限制数量
    logger.info("活跃标的: %d 只, 需要更新: %d 只 (本次限制 %d 只)", len(symbols), len(need_update), max_stocks)

    if not need_update:
        logger.info("所有标的已是最新")
        db.close()
        return True

    portfolio = ['000429','002973','002700','002298','000880',
                 '002951','001289','002068','002077','002127',
                 '002392','002283','003001','002301','002396']
    ordered = sorted(need_update, key=lambda s: (s not in portfolio, s))

    fetcher = FallbackFetcher()
    total_new = 0
    db.conn.execute("BEGIN TRANSACTION")

    for i, sym in enumerate(ordered):
        stock_start = _to_date(latest.get(sym, global_latest))
        if stock_start >= today:
            continue
        df = fetcher.fetch_bars(sym, stock_start.isoformat(), today_s)
        if df is None or df.empty:
            continue
        new_rows = df[df["date"] > stock_start]
        if new_rows.empty:
            continue
        for _, row in new_rows.iterrows():
            try:
                db.conn.execute("""
                    INSERT INTO daily_bars (symbol,date,open,high,low,close,volume,amount,pct_change)
                    VALUES (?,?,?,?,?,?,?,?,?)
                    ON CONFLICT (symbol,date) DO UPDATE SET
                        open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low,
                        close=EXCLUDED.close, volume=EXCLUDED.volume, amount=EXCLUDED.amount,
                        pct_change=EXCLUDED.pct_change
                """, [
                    sym, str(row["date"]),
                    float(row.get("open", 0)), float(row.get("high", 0)),
                    float(row.get("low", 0)), float(row.get("close", 0)),
                    int(row.get("volume", 0)), float(row.get("amount", 0)),
                    float(row.get("pct_change", 0))
                ])
                total_new += 1
            except Exception:
                pass

        if (i + 1) % REPORT_EVERY == 0:
            db.conn.execute("COMMIT")
            db.conn.execute("BEGIN TRANSACTION")
            logger.info("  进度 %d/%d, 已新增 %d 条", i + 1, len(ordered), total_new)

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
    parser.add_argument("--max-stocks", type=int, default=200, help="每次最大更新股票数")
    parser.add_argument("--skip-factors", action="store_true", help="跳过因子重算")
    args = parser.parse_args()

    ok = update_bars(days_back=args.days, max_stocks=args.max_stocks)
    if ok and not args.skip_factors:
        recompute_factors()


if __name__ == "__main__":
    main()
