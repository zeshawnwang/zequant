"""
实盘成交记录工具 — 你只需说成交，其余自动算。

用法：
    python3 scripts/record_live.py
    然后输入: 000001 B 100 10.52

也可以直接一条命令:
    python3 scripts/record_live.py --trades "000001 B 100 10.52,600519 S 50 1350"
"""
from __future__ import annotations
import sys
import os
import logging
from datetime import datetime
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from live.storage.record_base import (
    Trade, _init_live_db, get_last_snapshot, get_closing_prices,
    get_latest_close, get_benchmark_return, calc_position_value,
    save_trades, save_snapshot, save_performance, parse_trade,
    LIVE_DB_PATH, QUANT_DB_PATH,
)
from core.database import Database

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("record_live")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="实盘成交记录")
    parser.add_argument("--trades", help='成交列表, 用逗号分隔, 格式: "000001 B 100 10.52,600519 S 50 1350"')
    parser.add_argument("--strategy", default="mf_vol_d10_rp", help="策略名")
    parser.add_argument("--date", default=None, help="日期(默认今天)")
    parser.add_argument("--init-cash", type=float, default=None, help="首次建仓初始资金")
    parser.add_argument("--deposit", type=float, default=0, help="追加资金 (加仓)")
    parser.add_argument("--withdraw", type=float, default=0, help="提取资金 (减仓)")
    args = parser.parse_args()

    trade_date = args.date or datetime.now().strftime("%Y-%m-%d")

    # 解析成交
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

    # 打印成交摘要
    total_buy = sum(t.price * t.shares for t in trades if t.direction == "B")
    total_sell = sum(t.price * t.shares for t in trades if t.direction == "S")
    logger.info("\n📋 成交摘要:")
    logger.info(f"   买入: {sum(1 for t in trades if t.direction == 'B')} 笔, {total_buy:,.0f} 元")
    logger.info(f"   卖出: {sum(1 for t in trades if t.direction == 'S')} 笔, {total_sell:,.0f} 元")

    # 打开数据库
    live_db = Database(LIVE_DB_PATH)
    quant_db = Database(QUANT_DB_PATH)
    _init_live_db(live_db)

    # 上次快照为空 = 首次建仓
    last = get_last_snapshot(live_db)
    is_first = last["date"] is None
    if is_first:
        init_cash = args.init_cash
        if init_cash is None:
            logger.info("首次建仓, 请确认初始资金:")
            try:
                init_cash_input = input("   初始资金 (默认 100000): ").strip()
                init_cash = float(init_cash_input) if init_cash_input else 100000.0
            except (EOFError, KeyboardInterrupt):
                init_cash = 100000.0
        last["cash"] = init_cash
        last["total_value"] = init_cash
        logger.info(f"   初始现金: {init_cash:,.0f} 元")
    else:
        logger.info(f"上次快照: {last['date']}, 总资产={last['total_value']:,.0f}, 现金={last['cash']:,.0f}")

    # 计算持仓变化
    positions = dict(last["positions"])  # {symbol: shares}

    orders = []
    for t in trades:
        orders.append({
            "symbol": t.symbol,
            "direction": "买入" if t.direction == "B" else "卖出",
            "shares": t.shares,
            "price": t.price,
            "reason": "实盘录入",
        })
        if t.direction == "B":
            positions[t.symbol] = positions.get(t.symbol, 0) + t.shares
        else:
            positions[t.symbol] = positions.get(t.symbol, 0) - t.shares
            if positions[t.symbol] <= 0:
                del positions[t.symbol]

    # 获取收盘价
    symbols = list(positions.keys()) + [t.symbol for t in trades]
    prices = get_closing_prices(quant_db, list(set(symbols)), trade_date)
    if not prices:
        logger.warning("⚠️ %s 无收盘价, 使用最近交易日价格", trade_date)
        for sym in set(symbols):
            p = get_latest_close(quant_db, sym, trade_date)
            if p:
                prices[sym] = p

    # 用成交价兜底（新股等missing收盘价的情况）
    for t in trades:
        if t.symbol not in prices or prices[t.symbol] == 0:
            prices[t.symbol] = t.price

    # 计算总市值和现金
    position_value = calc_position_value(positions, prices, quant_db, trade_date)
    cash = last["cash"] - total_buy + total_sell + args.deposit - args.withdraw
    net_cashflow = args.deposit - args.withdraw  # 纯资金进出，不影响收益率计算
    total_value = position_value + cash

    # 如果本日有资金进出，记录为订单
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
    if args.deposit > 0:
        logger.info(f"   💰 入金:   {args.deposit:,.0f} 元")
    if args.withdraw > 0:
        logger.info(f"   💳 出金:   {args.withdraw:,.0f} 元")

    # 计算换手率
    turnover = (total_buy + total_sell) / total_value if total_value > 0 else 0

    # 基准收益
    benchmark_ret = get_benchmark_return(quant_db, trade_date)

    # 保存
    save_trades(live_db, trades, args.strategy, trade_date)
    save_snapshot(live_db, trade_date, args.strategy, total_value, cash, positions, orders)
    save_performance(live_db, trade_date, total_value, last["total_value"],
                     len(positions), turnover, benchmark_ret, net_cashflow=net_cashflow)

    live_db.conn.commit()
    live_db.close()
    quant_db.close()

    logger.info(f"\n✅ 已保存到 {LIVE_DB_PATH}")

    # 持仓明细
    if positions:
        logger.info("\n📦 当前持仓:")
        logger.info(f"   {'代码':<8} {'股数':<6} {'收盘价':<8} {'市值':<10}")
        for sym, shr in sorted(positions.items()):
            p = prices.get(sym, 0)
            logger.info(f"   {sym:<8} {shr:<6} {p:<8.2f} {p*shr:<10,.0f}")


if __name__ == "__main__":
    main()
