"""
快速增量更新：只拉取最近N天最新日线数据 + 补算因子。

用法：
    python3 -m core.datasource.quick_update            # 默认行为
    python3 -m core.datasource.quick_update --days 5   # 补最近5天
"""
from __future__ import annotations
import logging, sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from core.database import Database
from core.datasource.daily_updater import DailyUpdater

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("quick_update")

# 已激活的股票池（从daily_bars获取有数据的股票，避免拉取全量5515只）
# 典型值：2000只活跃股票、持有较多交易日数据的标的
ACTIVE_THRESHOLD = 60  # 至少有60个交易日数据才算活跃


def get_recent_start(days_back: int = 5) -> str:
    from datetime import datetime, timedelta
    return (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")


def get_active_symbols(db: Database, min_days: int = ACTIVE_THRESHOLD) -> list:
    """获取活跃标的（持有足够多交易日数据的股票）。"""
    try:
        df = db.conn.execute(f"""
            SELECT symbol, COUNT(*) AS days
            FROM daily_bars
            GROUP BY symbol
            HAVING days >= {min_days}
            ORDER BY days DESC
        """).df()
        symbols = df["symbol"].tolist()
        logger.info("活跃标的: %d 只 (至少 %d 个交易日)", len(symbols), min_days)
        return symbols
    except Exception as e:
        logger.warning("获取活跃标的失败: %s", e)
        return []


def get_outdated_symbols(db: Database, days_back: int = 5) -> list:
    """获取最新数据早于数据库最新日期的股票（需要增量更新）。"""
    try:
        # 获取数据库中所有股票的最新日期
        df = db.conn.execute("""
            SELECT symbol, MAX(date) AS last_date
            FROM daily_bars
            GROUP BY symbol
            ORDER BY last_date
        """).df()
        if df.empty:
            return []
        # 找数据库全局最新日期
        latest_db_date = df["last_date"].max()
        # 选出最新数据 < 全局最新日期的股票（落后于大部队）
        outdated = df[df["last_date"] < latest_db_date]["symbol"].tolist()
        logger.info("需要更新的标的: %d 只 (最新数据 < 全局最新 %s)", len(outdated), latest_db_date)
        return outdated
    except Exception as e:
        logger.warning("获取过期标的失败: %s", e)
        return []


def quick_update(days_back: int = 5):
    """快速增量更新。

    策略：仅更新活跃的2000只股票中最新的几天数据。
    """
    db = Database()
    start = get_recent_start(days_back)
    logger.info("快速更新: 回溯 %d 天 (起始 %s)", days_back, start)

    # 获取活跃标的
    active = get_active_symbols(db)

    # 从活跃标的中筛选需要更新的
    outdated = get_outdated_symbols(db, days_back=5)
    need_update = [s for s in outdated if s in active]

    if not need_update:
        logger.info("所有活跃标的已是最新数据")
    else:
        logger.info("开始增量拉取 %d 只股票的日线...", len(need_update))

        from core.datasource.fallback_fetcher import FallbackFetcher
        fetcher = FallbackFetcher()

        count = 0
        for i, sym in enumerate(need_update):
            if (i + 1) % 100 == 0:
                logger.info("  [%d/%d] %s ...", i + 1, len(need_update), sym)
            df = fetcher.fetch_bars(sym, start, end_date=None)
            if len(df) > 0:
                # 写入数据库
                from .fetcher import IncrementalFetcher
                inc = IncrementalFetcher(db)
                inc.db.upsert_daily_bars(df)
                count += 1

        logger.info("日线拉取完成: %d / %d 只有新数据", count, len(need_update))

    # 计算因子
    logger.info("补算因子...")
    updater = DailyUpdater(db=db, start_date=start)
    updater.run(fetch_bars=False, compute_factors=True, compute_technical=True, check_quality=False)

    # 验证
    latest_b = db.conn.execute("SELECT MAX(date) FROM daily_bars").fetchone()[0]
    latest_f = db.conn.execute("SELECT MAX(date) FROM factors_wide").fetchone()[0]
    logger.info("更新后: 日线最新=%s, 因子最新=%s", latest_b, latest_f)

    db.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=5)
    args = parser.parse_args()
    quick_update(days_back=args.days)
