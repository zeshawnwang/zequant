"""从 daily_snapshots（真实持仓） + trades（现金）重建实盘绩效。"""
from __future__ import annotations
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.database import Database

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rebuild_perf")

LIVE_DB_PATH = "./data_live/live_data.db"
QUANT_DB_PATH = "./data/quant_data.db"


def rebuild():
    live_db = Database(LIVE_DB_PATH)
    quant_db = Database(QUANT_DB_PATH)

    # 初始资金
    first_cash = float(live_db.conn.execute("SELECT cash FROM daily_snapshots ORDER BY date ASC LIMIT 1").fetchone()[0])
    buys_before = float(live_db.conn.execute("SELECT COALESCE(SUM(amount),0) FROM trades WHERE direction='B' AND date<='2026-05-19'").fetchone()[0])
    sells_before = float(live_db.conn.execute("SELECT COALESCE(SUM(amount),0) FROM trades WHERE direction='S' AND date<='2026-05-19'").fetchone()[0])
    INIT = first_cash + buys_before - sells_before
    logger.info(f"初始资金: {INIT:.0f}")

    # 所有成交（用于现金计算）
    all_trades_raw = live_db.conn.execute("SELECT date, direction, amount FROM trades ORDER BY date").fetchall()
    all_trades = [(str(r[0]), r[1], float(r[2])) for r in all_trades_raw]

    # 所有快照日期
    snap_dates_raw = live_db.conn.execute("SELECT DISTINCT date FROM daily_snapshots ORDER BY date").fetchall()
    snap_dates = [str(r[0]) for r in snap_dates_raw]

    # 对快照缺失的日期补充（06-15, 06-16）
    extended_dates = []
    for d in snap_dates:
        extended_dates.append(d)
    for extra in ['2026-06-15', '2026-06-16']:
        if extra not in extended_dates:
            extended_dates.append(extra)

    # 逐日计算
    prev_positions = {}
    prev_total = INIT

    print()
    header = f"{'日期':<12} {'总资产':>8} {'现金':>8} {'市值':>8} {'只数':>3} {'日收益':>8} {'累计收益':>10}"
    print(header)
    print("-" * len(header))

    for ds in sorted(extended_dates):
        # 持仓：看该日有无快照，有则用快照，无则沿用上日
        snap_row = live_db.conn.execute("SELECT positions FROM daily_snapshots WHERE date=?", [ds]).fetchone()
        if snap_row:
            raw_pos = json.loads(snap_row[0]) if snap_row[0] else {}
            positions = {}
            for sym, info in raw_pos.items():
                shares = info["shares"] if isinstance(info, dict) else info
                positions[sym] = shares
            prev_positions = positions
        else:
            positions = dict(prev_positions)

        # 现金：初始资金 - 累计到该日的净买入
        buy_to_date = sum(r[2] for r in all_trades if r[1] == "B" and r[0] <= ds)
        sell_to_date = sum(r[2] for r in all_trades if r[1] == "S" and r[0] <= ds)
        cash = INIT - buy_to_date + sell_to_date

        # 持仓市值
        pv = 0.0
        for sym, shares in positions.items():
            row = quant_db.conn.execute(
                "SELECT close FROM daily_bars WHERE symbol=? AND date<=? ORDER BY date DESC LIMIT 1",
                [sym, ds],
            ).fetchone()
            if row:
                pv += float(row[0]) * shares

        tv = cash + pv
        daily_ret = (tv - prev_total) / prev_total if prev_total > 0 else 0
        cum_ret = (tv - INIT) / INIT

        print(f"{ds:<12} {tv:>8,.0f} {cash:>8,.0f} {pv:>8,.0f} {len(positions):>3} {daily_ret:>+7.2%} {cum_ret:>+9.2%}")
        prev_total = tv

    print()
    logger.info(f"期初: {INIT:,.0f}")
    logger.info(f"期末: {prev_total:,.0f}")
    logger.info(f"绝对盈亏: {prev_total - INIT:+,.0f}")
    logger.info(f"累计收益率: {(prev_total - INIT) / INIT * 100:+.2f}%")

    live_db.close()
    quant_db.close()


if __name__ == "__main__":
    rebuild()
