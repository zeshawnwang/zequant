"""绩效追踪 — 计算实盘持仓盈亏、信号对比、累计收益。

依赖 live_data.db 中的快照和成交记录，汇总每日绩效。

用法：
    python3 -m live.performance.tracker                             # 今日绩效
    python3 -m live.performance.tracker --date 2026-05-19           # 指定日期
    python3 -m live.performance.tracker --verify                     # 含信号成交对比
    python3 -m live.performance.tracker --json                       # JSON 格式输出
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from core.database import Database

logger = logging.getLogger('live.performance')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

LIVE_DB_PATH = "./data_live/live_data.db"
QUANT_DB_PATH = "./data/quant_data.db"
SIGNAL_DIR = "data_live/mss_dynamic"


def get_prices(quant_db: Database, symbols: List[str], date_str: str) -> Dict[str, float]:
    if not symbols:
        return {}
    ph = ",".join("?" for _ in symbols)
    rows = quant_db.conn.execute(
        f"SELECT symbol, close FROM daily_bars WHERE date=? AND symbol IN ({ph})",
        [date_str] + symbols
    ).fetchall()
    prices = {r[0]: float(r[1]) for r in rows}
    missing = [s for s in symbols if s not in prices]
    for sym in missing:
        r = quant_db.conn.execute(
            "SELECT close FROM daily_bars WHERE symbol=? AND date<=? ORDER BY date DESC LIMIT 1",
            [sym, date_str]
        ).fetchone()
        if r:
            prices[sym] = float(r[0])
    return prices


def get_names(quant_db: Database) -> Dict[str, str]:
    rows = quant_db.conn.execute("SELECT symbol, name FROM symbols").fetchall()
    return {r[0]: r[1] for r in rows}


def load_latest_signal(before_date: str = None) -> Optional[Dict]:
    if not os.path.exists(SIGNAL_DIR):
        return None
    dirs = sorted([d for d in os.listdir(SIGNAL_DIR)
                   if d.isdigit() and os.path.isdir(os.path.join(SIGNAL_DIR, d))],
                  reverse=True)
    for d in dirs:
        if before_date and d >= before_date.replace('-', ''):
            continue
        p = os.path.join(SIGNAL_DIR, d, "build.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                return json.load(f)
    return None


def calc_position_pnl(live_db: Database, quant_db: Database, date_str: str) -> dict:
    row = live_db.conn.execute("""
        SELECT date, total_value, cash, positions, orders
        FROM daily_snapshots ORDER BY date DESC LIMIT 1
    """).fetchone()
    if not row:
        return {"total_value": 0, "cash": 0, "positions": {}, "pnl": {}, "total_pnl": 0}

    snapshot_date = row[0]
    total_value = float(row[1])
    cash = float(row[2])
    positions = json.loads(row[3]) if row[3] else {}
    orders = json.loads(row[4]) if row[4] else []

    symbols = list(positions.keys())
    prices = get_prices(quant_db, symbols, date_str)
    names = get_names(quant_db)

    pnl_by_pos = {}
    total_cost = 0.0
    total_market = 0.0
    for sym, shares in positions.items():
        market_price = prices.get(sym, 0)
        cost_price = prices.get(sym, 0)
        cost_row = live_db.conn.execute("""
            SELECT price, shares FROM trades
            WHERE symbol=? AND strategy='mss_dynamic' AND direction='B'
            ORDER BY date DESC LIMIT 1
        """, [sym]).fetchone()
        if cost_row:
            cost_price = float(cost_row[0])
        else:
            cost_price = prices.get(sym, 0)

        cost_value = cost_price * shares
        market_value = market_price * shares
        pnl = market_value - cost_value
        pnl_pct = pnl / cost_value * 100 if cost_value > 0 else 0
        total_cost += cost_value
        total_market += market_value

        pnl_by_pos[sym] = {
            "shares": shares, "name": names.get(sym, ""),
            "cost_price": round(cost_price, 3),
            "market_price": round(market_price, 2),
            "cost_value": round(cost_value, 2),
            "market_value": round(market_value, 2),
            "pnl": round(pnl, 2), "pnl_pct": round(pnl_pct, 2),
        }

    total_pnl = total_market - total_cost
    total_pnl_pct = total_pnl / total_cost * 100 if total_cost > 0 else 0

    return {
        "date": str(snapshot_date), "query_date": date_str,
        "total_value": round(total_value, 2),
        "cash": round(cash, 2),
        "position_value": round(total_market, 2),
        "total_cost": round(total_cost, 2),
        "total_pnl": round(total_pnl, 2),
        "total_pnl_pct": round(total_pnl_pct, 2),
        "n_positions": len(positions),
        "positions": pnl_by_pos,
    }


def get_signal_comparison(live_db: Database, trade_date: str = None) -> Optional[dict]:
    trade_date = trade_date or str(date.today())
    sig = load_latest_signal(before_date=trade_date)
    if not sig:
        return None

    m = sig.get("meta", {})
    expected = {o["symbol"]: o for o in sig.get("buy_orders", [])}

    actual_rows = live_db.conn.execute("""
        SELECT symbol, direction, shares, price, amount
        FROM trades WHERE strategy='mss_dynamic' AND date=?
        ORDER BY symbol
    """, [trade_date]).fetchall()

    actual = {}
    for r in actual_rows:
        if r[1] in ("B", "BUY"):
            actual[r[0]] = {"shares": r[2], "price": r[3], "amount": r[4]}

    matched, missed, extra = [], [], []
    for sym, e in expected.items():
        if sym in actual:
            a = actual[sym]
            diff = abs(a["shares"] - e.get("shares", 0))
            matched.append({"symbol": sym, "signal_shares": e["shares"],
                            "actual_shares": a["shares"], "diff_shares": diff,
                            "signal_price": e.get("price", 0), "actual_price": a["price"]})
        else:
            missed.append(sym)

    for sym in actual:
        if sym not in expected:
            extra.append(sym)

    return {
        "signal_date": m.get("signal_date"),
        "n_signal_buy": len(expected),
        "n_actual_buy": len(actual),
        "n_matched": len(matched),
        "n_missed": len(missed),
        "n_extra": len(extra),
        "matched": matched, "missed": missed, "extra": extra,
    }


def print_report(pnl: dict, comparison: dict = None):
    print(f'\n╔{"═"*58}╗')
    print(f'║  ZEquant 实盘绩效 — {pnl["query_date"]}')
    print(f'╠{"═"*58}╣')
    print(f'║  总资产: {pnl["total_value"]:>10,.0f}')
    print(f'║  持仓:   {pnl["position_value"]:>10,.0f} ({pnl["n_positions"]}只)')
    print(f'║  现金:   {pnl["cash"]:>10,.0f}')
    print(f'╠{"═"*58}╣')
    if pnl["total_pnl"] >= 0:
        print(f'║  📈 浮动盈亏: +{pnl["total_pnl"]:>7,.0f} ({pnl["total_pnl_pct"]:+.2f}%)')
    else:
        print(f'║  📉 浮动盈亏: {pnl["total_pnl"]:>8,.0f} ({pnl["total_pnl_pct"]:+.2f}%)')
    print(f'╠{"═"*58}╣')
    p_list = sorted(pnl["positions"].items(), key=lambda x: x[1]["pnl"], reverse=True)
    print(f'║  {"":<2}{"代码":<7} {"名称":<10} {"股数":<5} {"成本价":<8} {"现价":<8} {"盈亏":<8} {"涨跌%":<6}')
    for sym, pos in p_list:
        tag = "📈" if pos["pnl"] >= 0 else "📉"
        print(f'║  {tag} {sym:<5} {pos["name"]:<10} {pos["shares"]:<5} '
              f'{pos["cost_price"]:<8.2f} {pos["market_price"]:<8.2f} '
              f'{pos["pnl"]:<+8,.0f} {pos["pnl_pct"]:<+6.2f}')
    if comparison:
        print(f'╠{"═"*58}╣')
        print(f'║  成交确认:')
        print(f'║    信号买入: {comparison["n_signal_buy"]}只 │ 实际买入: {comparison["n_actual_buy"]}只 │ 匹配: {comparison["n_matched"]}只')
        if comparison["missed"]:
            print(f'║    ❌ 信号未成交: {", ".join(comparison["missed"])}')
        if comparison["extra"]:
            print(f'║    ❓ 额外成交: {", ".join(comparison["extra"])}')
    print(f'╚{"═"*58}╝')


def main():
    parser = argparse.ArgumentParser(description="实盘绩效追踪")
    parser.add_argument("--date", default=str(date.today()), help="查询日期")
    parser.add_argument("--verify", action="store_true", help="信号成交对比")
    parser.add_argument("--json", action="store_true", help="JSON 格式输出")
    args = parser.parse_args()

    live_db = Database(LIVE_DB_PATH)
    quant_db = Database(QUANT_DB_PATH)

    pnl = calc_position_pnl(live_db, quant_db, args.date)
    comparison = get_signal_comparison(live_db, trade_date=args.date) if args.verify else None

    if args.json:
        out = {"pnl": pnl, "comparison": comparison}
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        print_report(pnl, comparison)

    live_db.close()
    quant_db.close()


if __name__ == "__main__":
    main()
