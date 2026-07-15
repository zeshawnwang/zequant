"""risk_off_mf — 实盘信号生成入口。

低频多因子 + m5 熔断策略。
每 10 个交易日调仓，T 日收盘出信号，T+1 日开盘成交。

使用方式:
    python3 -m live.signals.risk_off_mf --capital 50000
    python3 -m live.signals.risk_off_mf --capital 50000 --email
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date, datetime

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logger = logging.getLogger("risk_off_mf")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(PROJECT_DIR, "data", "quant_data.db")
LIVE_DB_PATH = os.path.join(PROJECT_DIR, "data_live", "live_data.db")
SIGNAL_DIR = os.path.join(PROJECT_DIR, "data_live", "risk_off_mf")

from core.screening.impl.risk_off_mf import RiskOffMultiFactorSelector

# === 策略参数 ===
FACTOR_WEIGHTS = {
    "ff_mkt": 0.0413, "gtja142": 0.3005, "gtja144": 0.2045, "gtja171": -0.0443,
    "gtja103": -0.0147, "gtja85": -0.0258, "a88": -0.035, "a31": -0.0249,
    "rsi_14": 0.0253, "gtja139": -0.0112, "gtja123": 0.1666, "a42": 0.1999,
    "a41": 0.2152, "a97": -0.0734, "gtja148": -0.0048, "gtja99": -0.059,
    "gtja117": 0.2324, "gtja76": 0.0032, "gtja90": 0.0437, "volatility_20": -0.1127,
    "gtja113": -0.0874, "gtja141": 0.2104, "a99": -0.072, "gtja12": -0.1859,
    "gtja83": 0.1429, "gtja164": 0.0235, "a98": 0.0657, "gtja49": -0.2478,
    "gtja121": -0.0095, "a85": 0.1419, "gtja104": -0.1303, "gtja185": -0.0565,
    "gtja176": -0.075, "a80": 0.1689, "gtja62": 0.1181, "a8": 0.0657,
    "gtja34": -0.0816, "returns": -0.0508, "gtja168": 0.3003, "gtja108": -0.0791,
    "gtja105": 0.0686, "gtja127": -0.0506, "a27": -0.0627, "a64": 0.0874,
    "gtja91": -0.0399, "a30": -0.0666, "a69": -0.0961, "a91": -0.0582,
    "gtja13": 0.0903, "gtja120": 0.055,
}
FACTOR_NAMES = list(FACTOR_WEIGHTS.keys())
TOP_N = 10
RISK_OFF_Z = -2.5
RISK_OFF_SCALE = 0.55
REBAL_FREQ = 10
VOL_LOOKBACK = 20
WINSOZIZE = 0.01
MAX_POSITION = 0.95
CASH_BUFFER = 0.05
SLIPPAGE = 0.001
TX_COST = 0.0012


def compute_rp_weights(scores, bars_df, date_str):
    if scores.empty:
        return {}
    all_dates = sorted(bars_df["date"].unique())
    if date_str not in all_dates:
        return {}
    idx = all_dates.index(date_str)
    wdates = all_dates[max(0, idx - VOL_LOOKBACK): idx + 1]
    wb = bars_df[bars_df["date"].isin(wdates)]
    vol_dict = {}
    for sym in scores.index:
        s = wb[wb["symbol"] == sym].sort_values("date")
        if len(s) < 5:
            continue
        rets = s["close"].pct_change().dropna()
        if len(rets) < 3:
            continue
        vol_dict[sym] = rets.std()
    if not vol_dict:
        return {}
    inv_vol = {s: 1.0 / v for s, v in vol_dict.items()}
    total = sum(inv_vol.values())
    if total < 1e-10:
        return {}
    return {s: iv / total for s, iv in inv_vol.items()}


def _init_live_db():
    os.makedirs(os.path.dirname(LIVE_DB_PATH), exist_ok=True)
    con = duckdb.connect(LIVE_DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS rebal_state (
            id INTEGER PRIMARY KEY,
            last_rebal_date TEXT,
            holdings_json TEXT,
            capital REAL,
            cash REAL
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS daily_snapshots (
            date TEXT PRIMARY KEY,
            total_value REAL,
            cash REAL,
            positions_json TEXT,
            orders_json TEXT,
            note TEXT
        )
    """)
    con.commit()
    return con


def _load_state(con):
    row = con.execute("SELECT last_rebal_date, holdings_json, capital, cash FROM rebal_state WHERE id=1").fetchone()
    if row and row[0]:
        return {
            "last_rebal_date": row[0],
            "holdings": json.loads(row[1]) if row[1] else {},
            "capital": row[2] or 50000.0,
            "cash": row[3] or 0.0,
        }
    return None


def _save_state(con, state):
    holdings_json = json.dumps(state["holdings"], ensure_ascii=False)
    con.execute("""
        INSERT INTO rebal_state (id, last_rebal_date, holdings_json, capital, cash)
        VALUES (1, ?, ?, ?, ?)
        ON CONFLICT (id) DO UPDATE SET
            last_rebal_date=EXCLUDED.last_rebal_date,
            holdings_json=EXCLUDED.holdings_json,
            capital=EXCLUDED.capital,
            cash=EXCLUDED.cash
    """, [state["last_rebal_date"], holdings_json, state["capital"], state["cash"]])
    con.commit()


def _trading_days_between(qconn, d1, d2):
    rows = qconn.execute(
        "SELECT COUNT(DISTINCT date) FROM daily_bars WHERE date > ? AND date <= ? AND close > 0",
        [d1, d2]
    ).fetchone()
    return rows[0] if rows else 0


def _get_stock_names(qconn):
    try:
        rows = qconn.execute("SELECT symbol, name FROM symbols").fetchall()
        return {r[0]: r[1] for r in rows}
    except Exception:
        return {}


def run(capital: float = 50000, signal_date: str | None = None, force: bool = False) -> dict:
    signal_date = signal_date or str(date.today())

    os.makedirs(os.path.dirname(LIVE_DB_PATH), exist_ok=True)
    live_con = _init_live_db()
    state = _load_state(live_con)

    qconn = duckdb.connect(DB_PATH, read_only=True)

    ds = signal_date
    lf = str(qconn.execute("SELECT MAX(date) FROM factors_wide").fetchone()[0])
    lb = str(qconn.execute("SELECT MAX(date) FROM daily_bars WHERE close>0").fetchone()[0])
    latest_data = min(lf, lb)
    if latest_data and latest_data < ds:
        ds = latest_data
        logger.info(f"{signal_date} 无数据，使用 {latest_data}")

    logger.info(f"信号日期: {signal_date}  数据日期: {ds}  资金: {capital:,.0f}")

    rows = qconn.execute(
        "SELECT DISTINCT date FROM factors_wide WHERE date <= ? ORDER BY date DESC LIMIT 65",
        [ds]
    ).fetchdf()["date"].tolist()
    warmup = str(min(rows))

    fc = ", ".join([f'"{f}"' for f in FACTOR_NAMES + ["momentum_5", "volatility_20"]])
    factors = qconn.execute(
        f"SELECT date, symbol, {fc} FROM factors_wide WHERE date BETWEEN ? AND ? ORDER BY date, symbol",
        [warmup, ds]
    ).fetchdf()
    factors["date"] = factors["date"].astype(str)

    bars = qconn.execute(
        "SELECT date, symbol, open, high, low, close FROM daily_bars WHERE date BETWEEN ? AND ? AND close > 0 ORDER BY date, symbol",
        [warmup, ds]
    ).fetchdf()
    bars["date"] = bars["date"].astype(str)

    close_df = bars[bars["date"] == ds][["symbol", "close"]].copy()

    latest_day = factors[factors["date"] == ds]
    result = RiskOffMultiFactorSelector.compute_signal(
        latest=latest_day,
        weights=FACTOR_WEIGHTS,
        top_n=TOP_N,
        winsorize=WINSOZIZE,
        risk_off_z=RISK_OFF_Z,
        risk_off_scale=RISK_OFF_SCALE,
        trigger_factor="momentum_5",
    )

    if result.score.empty:
        logger.error("信号为空")
        live_con.close()
        qconn.close()
        return {"error": "empty signal", "meta": {"signal_date": signal_date}}

    score = result.score
    risk_off_triggered = result.risk_off_triggered

    top = score.nlargest(TOP_N)
    weights = compute_rp_weights(top, bars, ds)
    if not weights:
        weights = {s: 1.0 / TOP_N for s in top.index}

    is_rebal = force
    if state and state["last_rebal_date"] and not force:
        ndays = _trading_days_between(qconn, state["last_rebal_date"], ds)
        is_rebal = ndays >= REBAL_FREQ

    buy_orders = []
    sell_orders = []
    new_holdings = {}

    if is_rebal:
        target_syms = set(top.index)

        if state:
            for sym, h in state.get("holdings", {}).items():
                if sym not in target_syms:
                    sell_orders.append({
                        "symbol": sym, "direction": "卖出",
                        "shares": h.get("shares", 0),
                        "price": h.get("price", 0),
                        "reason": "调仓卖出"
                    })

        dd = factors[factors["date"] == ds].set_index("symbol")
        old_syms = set(state.get("holdings", {}).keys()) if state else set()

        available_cash = state["cash"] * (1 - CASH_BUFFER) if state and state.get("cash", 0) > 0 else capital * (1 - CASH_BUFFER)
        buy_budget = available_cash * MAX_POSITION
        total_cost = 0.0

        for sym in top.index:
            if sym in old_syms:
                if state:
                    new_holdings[sym] = state["holdings"][sym]
                continue

            w = weights.get(sym, 1.0 / TOP_N)
            alloc = buy_budget * w
            c = close_df.loc[close_df["symbol"] == sym, "close"]
            if c.empty:
                continue
            close_price = float(c.iloc[0])
            actual_price = close_price * (1 + SLIPPAGE)
            shares = int(alloc / (actual_price * 100)) * 100
            if shares < 100:
                shares = 100
            cost = shares * actual_price
            if total_cost + cost > available_cash * 1.05:
                continue

            total_cost += cost
            buy_orders.append({
                "symbol": sym, "direction": "买入",
                "shares": shares, "price": round(close_price, 2),
                "cost": round(cost, 2),
                "weight": round(w, 4),
            })
            new_holdings[sym] = {"shares": shares, "price": round(close_price, 2)}

        if state:
            for sym, h in state.get("holdings", {}).items():
                if sym in target_syms and sym not in new_holdings:
                    new_holdings[sym] = h

        new_state = {
            "last_rebal_date": ds,
            "holdings": new_holdings,
            "capital": capital,
            "cash": round(max(0, available_cash - total_cost), 2),
        }
        _save_state(live_con, new_state)
        state = new_state

    live_con.close()

    names = {}
    relevant_syms = set(top.index)
    if state:
        relevant_syms.update(state.get("holdings", {}).keys())
    all_names = _get_stock_names(qconn)
    for sym in relevant_syms:
        if sym in all_names:
            names[sym] = all_names[sym]

    qconn.close()

    total_occupy = sum(h.get("shares", 0) * h.get("price", 0) for h in (state["holdings"] if state else {}).values())
    remain = state["cash"] if state else capital

    is_hold = len(buy_orders) == 0 and len(sell_orders) == 0

    result = {
        "meta": {
            "strategy": "risk_off_mf",
            "config": "scale=0.55",
            "signal_date": signal_date,
            "data_date": ds,
            "market_state": "risk_off_mf",
            "state_changed": False,
            "last_state": "risk_off_mf",
            "confidence": 1.0 if not risk_off_triggered else 0.8,
            "capital": capital,
            "total_cost": round(total_occupy, 2),
            "total_capital_used": round(total_occupy, 2),
            "remain": round(remain, 2),
            "n_buy": len(buy_orders),
            "n_sell": len(sell_orders),
            "is_hold": is_hold,
            "risk_off_triggered": risk_off_triggered,
        },
        "top_n": top.index.tolist(),
        "scores": {s: round(float(v), 4) for s, v in top.items()},
        "weights": {s: round(v, 4) for s, v in weights.items()},
        "buy_orders": buy_orders,
        "sell_orders": sell_orders,
        "holdings": state.get("holdings", {}) if state else {},
        "names": names,
        "risk_off_triggered": risk_off_triggered,
    }

    date_key = signal_date.replace("-", "")
    sig_dir = os.path.join(SIGNAL_DIR, date_key)
    os.makedirs(sig_dir, exist_ok=True)

    json_path = os.path.join(sig_dir, "build.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info(f"信号已写入: {json_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="risk_off_mf 实盘信号")
    parser.add_argument("--capital", type=float, default=50000, help="资金(默认50000)")
    parser.add_argument("--date", help="信号日期(yyyy-mm-dd)")
    parser.add_argument("--force", action="store_true", help="强制调仓")
    parser.add_argument("--email", action="store_true", help="信号生成后发送邮件通知")
    args = parser.parse_args()

    r = run(capital=args.capital, signal_date=args.date, force=args.force)

    if args.email:
        sys.path.insert(0, os.path.join(PROJECT_DIR, "live"))
        from notification import Mailer

        date_key = r["meta"]["signal_date"].replace("-", "")
        sig_dir = os.path.join(SIGNAL_DIR, date_key)
        sig_path = os.path.join(sig_dir, "build.json")
        if os.path.exists(sig_path):
            mailer = Mailer()
            mailer.send_signal_from_file(sig_path)
            logger.info("邮件已发送")


if __name__ == "__main__":
    main()
