"""
实盘成交记录工具 — 你只需说成交，其余自动算。

用法：
    python3 scripts/record_live.py
    然后输入: 000001 B 100 10.52

也可以直接一条命令:
    python3 scripts/record_live.py --trades "000001 B 100 10.52,600519 S 50 1350"
"""
from __future__ import annotations
import sys, os, json, logging
from datetime import datetime, date
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.database import Database

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("record_live")

LIVE_DB_PATH = "./data_live/live_data.db"
QUANT_DB_PATH = "./data/quant_data.db"


@dataclass
class Trade:
    symbol: str
    direction: str          # B=买入, S=卖出
    shares: int
    price: float


def _init_live_db(db: Database):
    """初始化实盘数据库表。"""
    db.conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_snapshots (
            date         DATE,
            strategy     VARCHAR,
            total_value  DOUBLE,
            cash         DOUBLE,
            positions    JSON,
            orders       JSON,
            created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # 创建唯一索引供 ON CONFLICT 使用
    db.conn.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_snapshot_date_strategy
        ON daily_snapshots (date, strategy)
    """)
    db.conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            trade_id    VARCHAR PRIMARY KEY,
            date        DATE,
            symbol      VARCHAR,
            direction   VARCHAR,
            price       DOUBLE,
            shares      INT,
            amount      DOUBLE,
            fee         DOUBLE DEFAULT 0,
            strategy    VARCHAR DEFAULT '',
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    db.conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_performance (
            date            DATE PRIMARY KEY,
            total_value     DOUBLE,
            daily_return    DOUBLE,
            cumulative      DOUBLE DEFAULT 0,
            max_drawdown    DOUBLE DEFAULT 0,
            positions_count INT DEFAULT 0,
            turnover        DOUBLE DEFAULT 0,
            benchmark_ret   DOUBLE DEFAULT 0,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    db.conn.commit()


def get_last_snapshot(live_db: Database) -> dict:
    """获取最近一次持仓快照。"""
    row = live_db.conn.execute("""
        SELECT date, total_value, cash, positions, orders
        FROM daily_snapshots
        ORDER BY date DESC LIMIT 1
    """).fetchone()
    if row:
        return {
            "date": row[0],
            "total_value": row[1],
            "cash": row[2],
            "positions": json.loads(row[3]) if row[3] else {},
            "orders": json.loads(row[4]) if row[4] else [],
        }
    return {"date": None, "total_value": 0, "cash": 0, "positions": {}, "orders": []}


def get_closing_prices(quant_db: Database, symbols: List[str], trade_date: str) -> Dict[str, float]:
    """从 quant_data.db 获取指定日期收盘价。"""
    if not symbols:
        return {}
    placeholders = ",".join("?" for _ in symbols)
    rows = quant_db.conn.execute(f"""
        SELECT symbol, close FROM daily_bars
        WHERE date = ? AND symbol IN ({placeholders})
    """, [trade_date] + symbols).fetchall()
    return {r[0]: float(r[1]) for r in rows}


def get_latest_close(quant_db: Database, symbol: str, before_date: str) -> Optional[float]:
    """获取某个日期前的最新收盘价（兜底）。"""
    row = quant_db.conn.execute("""
        SELECT close FROM daily_bars
        WHERE symbol = ? AND date <= ?
        ORDER BY date DESC LIMIT 1
    """, [symbol, before_date]).fetchone()
    return float(row[0]) if row else None


def get_benchmark_return(quant_db: Database, trade_date: str) -> float:
    """获取当日沪深300收益率（近似基准）。"""
    row = quant_db.conn.execute("""
        SELECT pct_change FROM daily_bars
        WHERE symbol = '000300' AND date = ?
    """, [trade_date]).fetchone()
    return float(row[0]) / 100.0 if row else 0.0


def calc_position_value(positions: Dict[str, int], prices: Dict[str, float],
                        quant_db: Database, trade_date: str) -> float:
    """计算持仓总市值。"""
    total = 0.0
    for sym, shares in positions.items():
        price = prices.get(sym)
        if price is None:
            price = get_latest_close(quant_db, sym, trade_date) or 0
        total += price * shares
    return total


def save_trades(live_db: Database, trades: List[Trade], strategy: str, trade_date: str):
    """写入成交记录。"""
    for t in trades:
        tid = f"{trade_date}_{t.symbol}_{t.direction}_{abs(t.shares)}"
        amount = t.price * t.shares
        live_db.conn.execute("""
            INSERT INTO trades
            (trade_id, date, symbol, direction, price, shares, amount, strategy)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (trade_id) DO NOTHING
        """, [tid, trade_date, t.symbol, t.direction, t.price, t.shares, amount, strategy])


def save_snapshot(live_db: Database, trade_date: str, strategy: str,
                  total_value: float, cash: float,
                  positions: dict, orders: list):
    """写入持仓快照。"""
    live_db.conn.execute("""
        INSERT INTO daily_snapshots (date, strategy, total_value, cash, positions, orders)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT (date, strategy) DO UPDATE SET
            total_value = EXCLUDED.total_value,
            cash = EXCLUDED.cash,
            positions = EXCLUDED.positions,
            orders = EXCLUDED.orders
    """, [trade_date, strategy, total_value, cash,
          json.dumps(positions), json.dumps(orders)])


def save_performance(live_db: Database, trade_date: str,
                     total_value: float, last_value: float,
                     position_count: int, turnover: float,
                     benchmark_ret: float, net_cashflow: float = 0):
    """写入当日绩效（净 cashflow 已从收益率中剔除）。"""
    net_asset_change = total_value - net_cashflow
    daily_ret = (net_asset_change - last_value) / last_value if last_value > 0 else 0.0

    # 累计收益率：从第一条记录累乘
    prev = live_db.conn.execute("""
        SELECT cumulative FROM daily_performance
        ORDER BY date DESC LIMIT 1
    """).fetchone()
    prev_cum = float(prev[0]) if prev else 1.0
    cumulative = prev_cum * (1 + daily_ret)

    # 最大回撤（简单版）
    max_cum = live_db.conn.execute("""
        SELECT MAX(cumulative) FROM daily_performance
    """).fetchone()
    peak = max(float(max_cum[0]), cumulative) if max_cum and max_cum[0] else cumulative
    max_dd = (cumulative - peak) / peak if peak > 0 else 0.0

    live_db.conn.execute("""
        INSERT INTO daily_performance
        (date, total_value, daily_return, cumulative, max_drawdown,
         positions_count, turnover, benchmark_ret)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (date) DO UPDATE SET
            total_value = EXCLUDED.total_value,
            daily_return = EXCLUDED.daily_return,
            cumulative = EXCLUDED.cumulative,
            max_drawdown = EXCLUDED.max_drawdown,
            positions_count = EXCLUDED.positions_count,
            turnover = EXCLUDED.turnover,
            benchmark_ret = EXCLUDED.benchmark_ret
    """, [trade_date, total_value, daily_ret, cumulative, max_dd,
          position_count, turnover, benchmark_ret])


def parse_trade(text: str) -> Optional[Trade]:
    """解析一条成交文本。"""
    parts = text.strip().split()
    if len(parts) < 4:
        return None
    sym = parts[0].zfill(6)
    dir_raw = parts[1].upper()
    direction = "B" if dir_raw in ("B", "BUY", "买入") else "S" if dir_raw in ("S", "SELL", "卖出", "卖") else ""
    if not direction:
        return None
    try:
        shares = int(parts[2])
        price = float(parts[3])
    except ValueError:
        return None
    return Trade(symbol=sym, direction=direction, shares=shares, price=price)


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
