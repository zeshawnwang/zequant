"""实盘成交记录 — 只需说成交，其余自动算。

用法：
    python3 -m live.signals.record
    然后输入: 000001 B 100 10.52

也可一条命令:
    python3 -m live.signals.record --trades "000001 B 100 10.52,600519 S 50 1350"
"""
from __future__ import annotations
import sys
import os
import json
import logging
from datetime import datetime
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from live.storage.record_base import (
    Trade, _init_live_db, get_last_snapshot, get_closing_prices,
    get_latest_close, get_benchmark_return, calc_position_value,
    save_trades, save_snapshot, save_performance, parse_trade,
    LIVE_DB_PATH, QUANT_DB_PATH,
)
from core.database import Database

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("record_live")


def _verify_execution(trade_date: str, strategy: str):
    """与信号文件交叉验证成交情况。"""
    sig_base = f"data_live/{strategy}"
    if not os.path.exists(sig_base):
        logger.warning(f"信号目录不存在: {sig_base}，跳过验证")
        return

    dirs = sorted([d for d in os.listdir(sig_base)
                   if d.isdigit() and os.path.isdir(os.path.join(sig_base, d))],
                  reverse=True)
    sig_dir = None
    for d in dirs:
        if d <= trade_date.replace('-', ''):
            sig_dir = os.path.join(sig_base, d)
            break

    if not sig_dir:
        logger.warning(f"未找到 {trade_date} 之前的信号目录，跳过验证")
        return

    sig_path = os.path.join(sig_dir, "build.json")
    if not os.path.exists(sig_path):
        logger.warning(f"信号文件不存在: {sig_path}，跳过验证")
        return

    with open(sig_path) as f:
        sig = json.load(f)

    expected = {o["symbol"]: o for o in sig.get("buy_orders", [])}
    if not expected:
        logger.info("信号无买入，跳过验证")
        return

    live_db = Database(LIVE_DB_PATH)
    actual_rows = live_db.conn.execute("""
        SELECT symbol, direction, shares, price, amount
        FROM trades WHERE strategy=? AND date=?
    """, [strategy, trade_date]).fetchall()
    live_db.close()

    actual = {}
    for r in actual_rows:
        if r[1] in ("B", "BUY", "买入"):
            actual[r[0]] = {"shares": r[2], "price": r[3]}

    matched = 0
    missed = []
    for sym, e in expected.items():
        if sym in actual:
            matched += 1
        else:
            missed.append(sym)

    extra = [sym for sym in actual if sym not in expected]

    logger.info(f"\n📋 成交验证 ({trade_date}):")
    logger.info(f"  信号买入: {len(expected)}只")
    logger.info(f"  实际买入: {len(actual)}只")
    logger.info(f"  已匹配: {matched}只")
    if missed:
        logger.warning(f"  ❌ 未执行买入: {', '.join(missed)}")
    if extra:
        logger.warning(f"  ❓ 额外买入: {', '.join(extra)}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="实盘成交记录")
    parser.add_argument("--trades", help='成交列表, 逗号分隔, 格式: "000001 B 100 10.52,600519 S 50 1350"')
    parser.add_argument("--strategy", default="mss_dynamic", help="策略名")
    parser.add_argument("--date", default=None, help="日期(默认今天)")
    parser.add_argument("--init-cash", type=float, default=None, help="首次建仓初始资金")
    parser.add_argument("--deposit", type=float, default=0, help="追加资金")
    parser.add_argument("--withdraw", type=float, default=0, help="提取资金")
    parser.add_argument("--verify", action="store_true", help="与信号文件交叉验证")
    args = parser.parse_args()

    trade_date = args.date or datetime.now().strftime("%Y-%m-%d")

    # 成交确认模式：读取信号，比对实际成交
    if args.verify:
        _verify_execution(trade_date, args.strategy)

    trades: List[Trade] = []
    if args.trades:
        for t in args.trades.split(","):
            t = t.strip()
            parsed = parse_trade(t)
            if parsed:
                trades.append(parsed)
            else:
                logger.warning("跳过无法解析的成交: %s", t)
    else:
        logger.info("输入今日成交 (一行一条, 空行结束):")
        logger.info("格式: 代码 方向(B/S) 股数 价格  (例如: 000001 B 100 10.52)")
        while True:
            line = input("> ").strip()
            if not line:
                break
            parsed = parse_trade(line)
            if parsed:
                trades.append(parsed)
            else:
                logger.warning("格式错误, 请输: 代码 B/S 股数 价格")

    if not trades:
        logger.warning("无成交记录, 退出")
        return

    total_buy = sum(t.price * t.shares for t in trades if t.direction == "B")
    total_sell = sum(t.price * t.shares for t in trades if t.direction == "S")
    logger.info(f"\n📋 成交摘要:")
    logger.info(f"   买入: {sum(1 for t in trades if t.direction == 'B')} 笔, {total_buy:,.0f} 元")
    logger.info(f"   卖出: {sum(1 for t in trades if t.direction == 'S')} 笔, {total_sell:,.0f} 元")

    live_db = Database(LIVE_DB_PATH)
    quant_db = Database(QUANT_DB_PATH)
    _init_live_db(live_db)

    last = get_last_snapshot(live_db, args.strategy)
    is_first = last["date"] is None
    if is_first:
        init_cash = args.init_cash
        if init_cash is None:
            try:
                s = input("   初始资金 (默认 100000): ").strip()
                init_cash = float(s) if s else 100000.0
            except (EOFError, KeyboardInterrupt):
                init_cash = 100000.0
        last["cash"] = init_cash
        last["total_value"] = init_cash
        logger.info(f"   初始现金: {init_cash:,.0f} 元")
    else:
        logger.info(f"上次快照: {last['date']}, 总资产={last['total_value']:,.0f}, 现金={last['cash']:,.0f}")

    positions = {}
    for k, v in last["positions"].items():
        if isinstance(v, dict):
            positions[k] = dict(v)
        else:
            positions[k] = {"shares": int(v), "cost": 0.0}
    orders = []
    for t in trades:
        orders.append({"symbol": t.symbol, "direction": "买入" if t.direction == "B" else "卖出",
                        "shares": t.shares, "price": t.price, "reason": "实盘录入"})
        if t.direction == "B":
            if t.symbol in positions:
                old = positions[t.symbol]
                total_shares = old["shares"] + t.shares
                total_cost = old["shares"] * old["cost"] + t.shares * t.price
                positions[t.symbol] = {"shares": total_shares, "cost": round(total_cost / total_shares, 3)}
            else:
                positions[t.symbol] = {"shares": t.shares, "cost": t.price}
        else:
            if t.symbol in positions:
                positions[t.symbol]["shares"] -= t.shares
                if positions[t.symbol]["shares"] <= 0:
                    del positions[t.symbol]

    symbols = list(positions.keys()) + [t.symbol for t in trades]
    prices = get_closing_prices(quant_db, list(set(symbols)), trade_date)
    if not prices:
        logger.warning("⚠️ %s 无收盘价, 使用最近价", trade_date)
        for sym in set(symbols):
            p = get_latest_close(quant_db, sym, trade_date)
            if p:
                prices[sym] = p
    for t in trades:
        if t.symbol not in prices or prices[t.symbol] == 0:
            prices[t.symbol] = t.price

    position_value = calc_position_value(positions, prices, quant_db, trade_date)
    cash = last["cash"] - total_buy + total_sell + args.deposit - args.withdraw
    net_cashflow = args.deposit - args.withdraw
    total_value = position_value + cash

    if args.deposit > 0:
        orders.append({"symbol": "💰", "direction": "入金", "shares": 0,
                        "price": 0, "reason": f"追加 {args.deposit:,.0f} 元"})
    if args.withdraw > 0:
        orders.append({"symbol": "💰", "direction": "出金", "shares": 0,
                        "price": 0, "reason": f"提取 {args.withdraw:,.0f} 元"})

    logger.info(f"\n📊 {trade_date} 实盘快照:")
    logger.info(f"   持仓市值: {position_value:,.0f} 元 ({len(positions)} 只)")
    logger.info(f"   现金余额: {cash:,.0f} 元")
    logger.info(f"   总资产:   {total_value:,.0f} 元")

    turnover = (total_buy + total_sell) / total_value if total_value > 0 else 0
    benchmark_ret = get_benchmark_return(quant_db, trade_date)

    # 事务保护：三张表同时写入，避免中间崩溃导致状态不一致
    live_db.conn.execute("BEGIN TRANSACTION")
    save_trades(live_db, trades, args.strategy, trade_date)
    save_snapshot(live_db, trade_date, args.strategy, total_value, cash, positions, orders)
    save_performance(live_db, trade_date, total_value, last["total_value"],
                     len(positions), turnover, benchmark_ret, net_cashflow=net_cashflow)

    # 同步 sub_strategy_state：移除已卖出的 + 添加新买入的
    bought_symbols = [t for t in trades if t.direction == "B"]
    sold_symbols = [t.symbol for t in trades if t.direction == "S"]

    # 从最新信号文件读取 symbol → sub_strategy 归属（处理新买入归属）
    try:
        from live.signals.mss_state import _build_signal_symbol_map
        sig_map = _build_signal_symbol_map()
    except Exception:
        sig_map = {}

    for name_row in live_db.conn.execute(
        "SELECT name, holdings FROM sub_strategy_state"
    ).fetchall():
        name = name_row[0]
        h = json.loads(name_row[1]) if name_row[1] and name_row[1] != "{}" else {}
        changed = False

        # 移除已卖出的
        for sym in sold_symbols:
            if sym in h:
                del h[sym]
                changed = True

        # 添加新买入的（仅限属于本子策略的）
        for t in bought_symbols:
            if sig_map.get(t.symbol) == name and t.symbol not in h:
                h[t.symbol] = {"shares": t.shares, "price": t.price, "peak": t.price}
                changed = True

        if changed:
            used = sum(h[s].get("shares", 0) * h[s].get("price", 0) for s in h)
            live_db.conn.execute(
                "UPDATE sub_strategy_state SET holdings=?, used_capital=?, last_date=? WHERE name=?",
                [json.dumps(h), used, trade_date, name]
            )
            if sold_symbols:
                logger.info(f"  [sync] 从 {name} 移除已卖出的 {len(sold_symbols)} 只")
            if [t for t in bought_symbols if sig_map.get(t.symbol) == name]:
                logger.info(f"  [sync] {name} 已记录新买入成交")

    live_db.conn.execute("COMMIT")
    live_db.close()
    quant_db.close()

    logger.info(f"\n✅ 已保存到 {LIVE_DB_PATH}")
    if positions:
        logger.info("\n📦 当前持仓:")
        logger.info(f"   {'代码':<8} {'股数':<6} {'成本价':<8} {'收盘价':<8} {'市值':<10}")
        for sym, info in sorted(positions.items()):
            shares = info["shares"] if isinstance(info, dict) else info
            cost = info.get("cost", 0) if isinstance(info, dict) else 0
            p = prices.get(sym, 0)
            logger.info(f"   {sym:<8} {shares:<6} {cost:<8.3f} {p:<8.2f} {p*shares:<10,.0f}")


if __name__ == "__main__":
    main()
