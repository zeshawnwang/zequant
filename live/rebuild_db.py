"""用券商真实成交数据重建 trades 表。
只包含截图中的交易 + 06-17 步步高卖出。
"""
from __future__ import annotations
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.database import Database

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rebuild_db")

LIVE_DB_PATH = "./data_live/live_data.db"
QUANT_DB_PATH = "./data/quant_data.db"
INITIAL_CAPITAL = 50000.0

# 券商截图中的全部成交 + 06-17 步步高
TRADES = [
    ("2026-05-19", "002396", "星网锐捷", "B", 30.71, 100, 3076.0),
    ("2026-05-19", "002973", "侨银股份", "B", 14.01, 200, 2807.0),
    ("2026-05-19", "002283", "天润工业", "B", 11.83, 200, 2371.0),
    ("2026-05-19", "002068", "黑猫股份", "B", 10.36, 300, 3113.0),
    ("2026-05-19", "002392", "北京利尔", "B", 10.28, 300, 3089.0),
    ("2026-05-19", "002298", "中电鑫龙", "B", 11.88, 200, 2381.0),
    ("2026-05-19", "002951", "金时科技", "B", 16.94, 100, 1699.0),
    ("2026-05-19", "000429", "粤高速A", "B", 13.16, 200, 2637.0),
    ("2026-05-19", "002700", "万憬能源", "B", 7.15, 400, 2865.0),
    ("2026-05-19", "000880", "潍柴重机", "B", 32.16, 100, 3221.0),
    ("2026-05-19", "003001", "中岩大地", "B", 21.10, 100, 2115.0),
    ("2026-05-19", "001289", "龙源电力", "B", 17.61, 100, 1766.0),
    ("2026-05-19", "002301", "齐心集团", "B", 8.22, 400, 3293.0),
    ("2026-05-19", "002127", "南极电商", "B", 3.25, 1000, 3255.0),
    ("2026-05-19", "002077", "大港股份", "B", 17.10, 100, 1715.0),
    ("2026-05-25", "002301", "齐心集团", "S", 8.35, 400, 3333.0),
    ("2026-05-25", "002396", "星网锐捷", "S", 23.87, 100, 2381.0),
    ("2026-05-25", "003001", "中岩大地", "S", 21.76, 100, 2170.0),
    ("2026-05-25", "002283", "天润工业", "S", 12.60, 200, 2514.0),
    ("2026-05-25", "002298", "中电鑫龙", "S", 10.49, 200, 2092.0),
    ("2026-05-25", "002745", "木林森", "B", 9.91, 100, 996.0),
    ("2026-05-25", "002732", "燕塘乳业", "B", 15.15, 100, 1520.0),
    ("2026-05-25", "002679", "福建金森", "B", 9.90, 100, 995.0),
    ("2026-05-25", "002651", "利君股份", "B", 8.63, 100, 868.0),
    ("2026-05-25", "002599", "盛通股份", "B", 6.69, 200, 1343.0),
    ("2026-05-25", "002392", "北京利尔", "S", 9.07, 300, 2715.0),
    ("2026-05-28", "000429", "粤高速A", "S", 13.70, 200, 2734.0),
    ("2026-05-28", "002973", "侨银股份", "S", 12.67, 200, 2528.0),
    ("2026-05-28", "002700", "万憬能源", "S", 7.01, 400, 2798.0),
    ("2026-05-28", "000880", "潍柴重机", "S", 32.11, 100, 3204.0),
    ("2026-05-28", "002951", "金时科技", "S", 17.49, 100, 1743.0),
    ("2026-05-28", "001289", "龙源电力", "S", 17.98, 100, 1792.0),
    ("2026-05-28", "002068", "黑猫股份", "S", 9.57, 300, 2865.0),
    ("2026-05-28", "002077", "大港股份", "S", 18.49, 100, 1843.0),
    ("2026-05-28", "002127", "南极电商", "S", 2.94, 1000, 2934.0),
    ("2026-05-28", "002745", "木林森", "S", 10.97, 100, 1091.0),
    ("2026-05-28", "000021", "深科技", "B", 42.98, 200, 8601.0),
    ("2026-05-28", "002180", "奔图科技", "B", 16.64, 400, 6661.0),
    ("2026-05-28", "002917", "金奥博", "B", 14.40, 100, 1445.0),
    ("2026-05-28", "002138", "顺络电子", "B", 48.60, 100, 4865.0),
    ("2026-05-28", "002380", "科远智慧", "B", 42.63, 100, 4268.0),
    ("2026-05-28", "002169", "智光电气", "B", 18.29, 100, 1834.0),
    ("2026-05-28", "000716", "黑芝麻", "B", 5.62, 100, 567.0),
    ("2026-06-03", "002380", "科远智慧", "S", 39.30, 100, 3923.0),
    ("2026-06-03", "000021", "深科技", "S", 37.35, 200, 7461.0),
    ("2026-06-03", "000990", "诚志股份", "B", 9.59, 600, 5759.0),
    ("2026-06-03", "002679", "福建金森", "S", 9.35, 100, 930.0),
    ("2026-06-03", "002651", "利君股份", "S", 8.24, 100, 819.0),
    ("2026-06-03", "002732", "燕塘乳业", "S", 14.56, 100, 1450.0),
    ("2026-06-03", "002138", "顺络电子", "S", 48.85, 100, 4878.0),
    ("2026-06-03", "002169", "智光电气", "S", 15.69, 100, 1563.0),
    ("2026-06-03", "002180", "奔图科技", "S", 15.94, 400, 6368.0),
    ("2026-06-03", "002416", "爱施德", "B", 10.34, 600, 6209.0),
    ("2026-06-03", "000938", "紫光股份", "B", 29.51, 300, 8858.0),
    ("2026-06-03", "002917", "金奥博", "S", 13.44, 100, 1338.0),
    ("2026-06-03", "002599", "盛通股份", "S", 6.28, 100, 623.0),
    ("2026-06-03", "000716", "黑芝麻", "S", 5.57, 100, 552.0),
    ("2026-06-03", "002599", "盛通股份", "S", 6.23, 100, 618.0),
    ("2026-06-04", "001269", "欧晶科技", "B", 27.84, 100, 2789.0),
    ("2026-06-04", "000403", "派林生物", "B", 12.81, 100, 1286.0),
    ("2026-06-04", "002552", "宝鼎科技", "B", 59.72, 100, 5977.0),
    ("2026-06-04", "002251", "步步高", "B", 4.80, 100, 485.0),
    ("2026-06-04", "000990", "诚志股份", "S", 9.56, 600, 5728.0),
    ("2026-06-04", "001301", "尚太科技", "B", 95.80, 100, 9585.0),
    ("2026-06-04", "001301", "尚太科技", "B", 94.49, 100, 9454.0),
    ("2026-06-04", "000598", "兴蓉环境", "B", 7.08, 100, 713.0),
    ("2026-06-09", "000938", "紫光股份", "S", 25.99, 300, 7788.0),
    ("2026-06-09", "001301", "尚太科技", "S", 86.88, 200, 17362.0),
    ("2026-06-09", "003036", "泰坦股份", "B", 79.47, 200, 15899.0),
    ("2026-06-09", "000636", "风华高科", "B", 58.62, 100, 5867.0),
    ("2026-06-09", "000429", "粤高速A", "B", 14.03, 100, 1408.0),
    ("2026-06-10", "000403", "派林生物", "S", 11.84, 100, 1178.0),
    ("2026-06-12", "000429", "粤高速A", "S", 12.79, 100, 1273.0),
    ("2026-06-17", "002251", "步步高", "S", 3.92, 100, 392.0),
]


def rebuild():
    live_db = Database(LIVE_DB_PATH)
    quant_db = Database(QUANT_DB_PATH)

    # trades 重建
    live_db.conn.execute("DELETE FROM trades")
    for t in TRADES:
        date_str, sym, name, direction, price, shares, amount = t
        tid = f"{date_str}_{sym}_{direction}_{abs(shares)}"
        live_db.conn.execute(
            "INSERT INTO trades (trade_id, date, symbol, direction, price, shares, amount, strategy) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 'mss_dynamic') ON CONFLICT (trade_id) DO NOTHING",
            [tid, date_str, sym, direction, price, abs(shares), round(amount, 2)],
        )
    logger.info(f"trades 已重建: {len(TRADES)} 笔")

    # 计算买入总金额（用于算现金）
    all_buys = [t[6] for t in TRADES if t[3] == "B"]
    all_sells = [t[6] for t in TRADES if t[3] == "S"]
    logger.info(f"总买入: {sum(all_buys):,.0f}  总卖出: {sum(all_sells):,.0f}")

    # 重建 daily_snapshots 成本
    snaps = live_db.conn.execute("SELECT date, strategy, positions FROM daily_snapshots ORDER BY date").fetchall()
    for snap_date, strategy, p_json in snaps:
        positions = json.loads(p_json) if p_json else {}
        changed = False
        for sym, info in positions.items():
            # 从 TRADES 找该股票的加权平均买入价
            buys_of_sym = [t for t in TRADES if t[1] == sym and t[3] == "B"]
            if buys_of_sym:
                total_cost = sum(t[4] * t[5] for t in buys_of_sym)
                total_shares = sum(t[5] for t in buys_of_sym)
                avg_cost = round(total_cost / total_shares, 3)
                if isinstance(info, dict):
                    if abs(info.get("cost", 0) - avg_cost) > 0.001:
                        info["cost"] = avg_cost
                        changed = True
                else:
                    positions[sym] = {"shares": int(info), "cost": avg_cost}
                    changed = True
        if changed:
            live_db.conn.execute(
                "UPDATE daily_snapshots SET positions=? WHERE date=? AND strategy=?",
                [json.dumps(positions), str(snap_date), strategy],
            )
    logger.info("daily_snapshots 成本已更新")

    # 重建 sub_strategy_state
    rows = live_db.conn.execute("SELECT name, holdings FROM sub_strategy_state").fetchall()
    for name, h_json in rows:
        h = json.loads(h_json) if h_json else {}
        changed = False
        for sym, info in list(h.items()):
            buys_of_sym = [t for t in TRADES if t[1] == sym and t[3] == "B"]
            if buys_of_sym:
                total_cost = sum(t[4] * t[5] for t in buys_of_sym)
                total_shares = sum(t[5] for t in buys_of_sym)
                avg_cost = round(total_cost / total_shares, 3)
                old_price = info.get("price", 0)
                if abs(old_price - avg_cost) > 0.001:
                    old_peak = info.get("peak", avg_cost)
                    h[sym] = {"shares": info["shares"], "price": avg_cost, "peak": max(avg_cost, old_peak)}
                    changed = True
        if changed:
            used = sum(h[s].get("shares", 0) * h[s].get("price", 0) for s in h)
            live_db.conn.execute(
                "UPDATE sub_strategy_state SET holdings=?, used_capital=? WHERE name=?",
                [json.dumps(h), used, name],
            )
    logger.info("sub_strategy_state 已更新")

    # 重建 daily_performance
    live_db.conn.execute("DELETE FROM daily_performance")
    all_trades = [(str(r[0]), r[1], r[2], r[3], int(r[4]), float(r[5]))
                  for r in live_db.conn.execute("SELECT date, symbol, direction, price, shares, amount FROM trades ORDER BY date").fetchall()]
    trade_dates = sorted(set(r[0] for r in all_trades))
    snap_dates = [str(r[0]) for r in live_db.conn.execute("SELECT DISTINCT date FROM daily_snapshots ORDER BY date").fetchall()]
    ex_dates = sorted(set(snap_dates + trade_dates + ["2026-06-15", "2026-06-16"]))

    prev_pos = {}
    prev_total = INITIAL_CAPITAL
    peaks = [1.0]
    cumulative = 1.0

    print()
    print(f"{'日期':<12} {'总资产':>8} {'现金':>8} {'市值':>8} {'只数':>3}  日收益    累计收益    回撤")
    print("-" * 70)

    for ds in ex_dates:
        snap_row = live_db.conn.execute("SELECT positions FROM daily_snapshots WHERE date=?", [ds]).fetchone()
        if snap_row:
            raw_pos = json.loads(snap_row[0]) if snap_row[0] else {}
            pos_dict = {}
            for sym, info in raw_pos.items():
                pos_dict[sym] = info["shares"] if isinstance(info, dict) else info
            prev_pos = pos_dict
        else:
            pos_dict = dict(prev_pos)

        buy_total = sum(r[5] for r in all_trades if r[2] == "B" and r[0] <= ds)
        sell_total = sum(r[5] for r in all_trades if r[2] == "S" and r[0] <= ds)
        cash = INITIAL_CAPITAL - buy_total + sell_total

        pv = 0.0
        for sym, shares in pos_dict.items():
            row = quant_db.conn.execute(
                "SELECT close FROM daily_bars WHERE symbol=? AND date<=? ORDER BY date DESC LIMIT 1",
                [sym, ds],
            ).fetchone()
            if row:
                pv += float(row[0]) * shares

        tv = cash + pv
        daily_ret = (tv - prev_total) / prev_total if prev_total > 0 else 0
        cumulative *= 1 + daily_ret
        peaks.append(cumulative)
        max_dd = (cumulative - max(peaks)) / max(peaks)

        print(f"{ds:<12} {tv:>8,.0f} {cash:>8,.0f} {pv:>8,.0f} {len(pos_dict):>3}  {daily_ret:>+6.2%}  {cumulative-1:>+7.2%}  {max_dd:>6.2%}")

        live_db.conn.execute(
            "INSERT INTO daily_performance (date, total_value, daily_return, cumulative, max_drawdown, positions_count) "
            "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT (date) DO UPDATE SET "
            "total_value=EXCLUDED.total_value, daily_return=EXCLUDED.daily_return, "
            "cumulative=EXCLUDED.cumulative, max_drawdown=EXCLUDED.max_drawdown, "
            "positions_count=EXCLUDED.positions_count",
            [ds, round(tv, 2), round(daily_ret, 6), round(cumulative, 6), round(max_dd, 6), len(pos_dict)],
        )
        prev_total = tv

    live_db.conn.commit()
    print()
    logger.info(f"初始资金: {INITIAL_CAPITAL:,.0f}")
    logger.info(f"期末: {prev_total:,.0f}")
    logger.info(f"盈亏: {prev_total - INITIAL_CAPITAL:+,.0f}")
    logger.info(f"收益率: {(prev_total - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100:+.2f}%")
    live_db.close()
    quant_db.close()


if __name__ == "__main__":
    rebuild()
