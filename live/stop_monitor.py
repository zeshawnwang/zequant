"""盘中止损巡检 — 用实时行情做止损/止盈检查。

独立于收盘后的信号生成流程，盘中任意时刻可跑。

流程：
  1. 从 sub_strategy_state 读取持仓和成本价
  2. 用实时行情获取最新价（多数据源兜底）
  3. 逐只检查固定止损 + 移动止盈
  4. 触发时输出卖出指令到终端并写入记录文件

用法：
    python3 -m live.stop_monitor                     # 盘中巡检（自动检测交易时间）
    python3 -m live.stop_monitor --force              # 强制巡检（跳过交易时间判断）
    python3 -m live.stop_monitor --output sell.json   # 输出到文件

自动化（cron）：
    */30 9-14 * * 1-5  cd /path/to/zequant && python3 -m live.stop_monitor --output data_live/stop_signal.json
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# 共享止损参数（与 mss_state.py 保持一致，避免依赖循环）
STOP_LOSS_CONFIG = {
    "mf_d10_rp": 0.06,
    "mf_vol_d10_rp": 0.06,
    "chip_covrp": 0.08,
    "chip_equal_d3": 0.08,
    "c01_layered_d5": 0.06,
    "mf50_chip50": 0.06,
    "mf60_chip40": 0.06,
    "osr_d10": 0.06,
}
TRAILING_STOP_PCT = 0.03

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.database import Database
from live.realtime_price import get_realtime_prices, check_is_trading_time

logger = logging.getLogger("stop_monitor")

LIVE_DB_PATH = "./data_live/live_data.db"
SIGNAL_DIR = "data_live/stop_signals"
os.makedirs(SIGNAL_DIR, exist_ok=True)


def check_stop_conditions(
    name: str, symbol: str, shares: int, entry_price: float,
    current_price: float, peak: float,
) -> Optional[str]:
    """检查单只股票的止损/止盈条件。

    Returns:
        None 或 触发原因字符串（如 "mf_d10_rp: 止损6% 亏损-7.5%"）
    """
    if current_price <= 0 or entry_price <= 0:
        return None

    loss = (current_price - entry_price) / entry_price
    sl_pct = STOP_LOSS_CONFIG.get(name, 0.08)

    # 1. 固定止损
    if loss < -sl_pct:
        return f"{name}: 止损{sl_pct*100:.0f}% (亏损{loss*100:.1f}%)"

    # 2. 移动止盈：从峰值回撤超过 trail_pct（仅当前仍有利润时）
    if peak > entry_price and current_price > entry_price and current_price < peak * (1.0 - TRAILING_STOP_PCT):
        drawdown = (current_price - peak) / peak
        return f"{name}: 移动止盈{TRAILING_STOP_PCT*100:.0f}% (从峰值回撤{drawdown*100:.1f}%)"

    return None


def run(output_path: Optional[str] = None, force: bool = False) -> List[dict]:
    """执行止损巡检主流程。

    Returns:
        触发止损的卖出指令列表，每项: {symbol, shares, reason, current_price}
    """
    if not check_is_trading_time() and not force:
        logger.info("非交易时间，跳过巡检（使用 --force 强制）")
        return []

    live_db = Database(LIVE_DB_PATH)

    # 读取子策略持仓
    rows = live_db.conn.execute(
        "SELECT name, holdings FROM sub_strategy_state"
    ).fetchall()

    all_holdings: List[Tuple[str, str, int, float, float]] = []
    all_symbols: List[str] = []
    for name, h_json in rows:
        h = json.loads(h_json) if h_json else {}
        for sym, info in h.items():
            shares = info.get("shares", 0)
            price = info.get("price", 0)
            peak = info.get("peak", price)
            all_holdings.append((name, sym, shares, price, peak))
            all_symbols.append(sym)

    if not all_holdings:
        logger.info("无持仓，无需巡检")
        live_db.close()
        return []

    logger.info(f"巡检 {len(all_holdings)} 只持仓: {', '.join(sorted(set(all_symbols)))}")
    logger.info(f"  买入均价: {dict((s, format(p, '.3f')) for _, s, _, p, _ in all_holdings)}")

    # 获取实时价
    prices = get_realtime_prices(all_symbols)
    if not prices:
        logger.warning("无法获取实时行情，跳过巡检")
        live_db.close()
        return []

    # 按子策略分组，批量读、批量写
    states = {}  # name -> {holdings: {symbol: {...}}}
    for name, h_json in rows:
        h = json.loads(h_json) if h_json else {}
        states[name] = {"holdings": h}

    sell_orders = []
    any_change = False

    for name, state in states.items():
        h = state["holdings"]
        changed = False
        for sym, info in list(h.items()):
            cp = prices.get(sym)
            if cp is None:
                continue
            shares = info["shares"]
            price = info["price"]
            peak = info.get("peak", price)

            # 更新 peak（盘中新高）
            if cp > peak:
                h[sym]["peak"] = cp
                changed = True

            reason = check_stop_conditions(name, sym, shares, price, cp, max(peak, cp))
            if reason:
                sell_orders.append({
                    "symbol": sym, "shares": shares,
                    "current_price": round(cp, 3),
                    "entry_price": round(price, 3),
                    "reason": reason,
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                })
                logger.info(f"  ⛔ {sym}: {reason} (实时价{cp:.3f}, 成本{price:.3f})")

        if changed:
            used = sum(h[s].get("shares", 0) * h[s].get("price", 0) for s in h)
            live_db.conn.execute(
                "UPDATE sub_strategy_state SET holdings=?, used_capital=? WHERE name=?",
                [json.dumps(h), used, name],
            )
            any_change = True

    if any_change:
        live_db.commit()
    live_db.close()

    if not sell_orders:
        logger.info("✅ 无触发，一切正常")
    else:
        logger.warning(f"⚠️  触发 {len(sell_orders)} 只卖出信号:")
        for s in sell_orders:
            logger.warning(f"  S {s['symbol']} {s['shares']}股"
                           f" — {s['reason']}")

    # 保存到文件
    if sell_orders and output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({
                "time": datetime.now().isoformat(),
                "sell_orders": sell_orders,
            }, f, indent=2, ensure_ascii=False)
        logger.info(f"已保存到 {output_path}")

    return sell_orders


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    parser = argparse.ArgumentParser(description="盘中止损巡检")
    parser.add_argument("--force", action="store_true", help="强制巡检（跳过交易时间判断）")
    parser.add_argument("--output", default=None, help="输出文件路径")
    args = parser.parse_args()

    run(output_path=args.output, force=args.force)


if __name__ == "__main__":
    main()
