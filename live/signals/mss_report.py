"""mss_report — split from mss_dynamic.py."""
from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd

logger = logging.getLogger('mss_live')

from .mss_state import SIGNAL_DIR, FACTOR_NAMES, SUB_STRATEGY_META, LIVE_DB_PATH, COOLDOWN_DAYS
from .mss_signal import _get_scores


def _version_label(r: dict) -> str:
    """生成版本标签，如 V7.1 v2b"""
    mode = r.get('mode', 'v2b')
    mode_label = {'baseline': 'baseline', 'v2b': 'v2b', 'v2c': 'v2c'}.get(mode, mode)
    if COOLDOWN_DAYS > 0:
        return f"V7.1 {mode_label}"
    return f"V7 {mode_label}"



def _write_html(path: str, r: dict):
    rows = ""
    for i, o in enumerate(r['orders'], 1):
        n = r['names'].get(o['symbol'], '')
        rows += f"""<tr>
          <td>{i}</td>
          <td>{o['symbol']}</td>
          <td>{n}</td>
          <td class="num">{o['shares']}</td>
          <td class="num">{o['price']:.2f}</td>
          <td class="num">{o['cost']:,.0f}</td>
        </tr>"""

    sell_rows = ""
    for o in r.get('sell_orders', []):
        n = r['names'].get(o['symbol'], '')
        sell_rows += f"""<tr>
          <td>{o['symbol']}</td>
          <td>{n}</td>
          <td class="num">{o['shares']}</td>
          <td class="num">{o['price']:.2f}</td>
        </tr>"""

    sub_rows = ""
    status_icon_map = {"rebalanced": "🔄", "hold": "✅", "liquidated": "❌", "skip_empty": "⚠️"}
    for sd in r.get('sub_details', []):
        icon = status_icon_map.get(sd['status'], '❓')
        detail = ""
        if sd['status'] == 'rebalanced':
            detail = f"买入{sd.get('n_buy',0)}只 卖出{sd.get('n_sell',0)}只 占用{sd.get('cost',0):,.0f}/{sd.get('capital',0):,.0f}"
        elif sd['status'] == 'hold':
            detail = f"持仓{sd.get('n_hold',0)}只"
            nr = sd.get('next_rebalance')
            if nr: detail += f" (下次调仓≈{nr})"
        elif sd['status'] == 'liquidated':
            detail = f"清仓{sd.get('n_sell',0)}只"
        sub_rows += f"<li>{icon} {sd['name']} — {detail}</li>"

    alloc_rows = "".join(f"<li>{name} <strong>{w*100:.0f}%</strong></li>" for name, w in r['allocations'].items())

    state_icon = {"bull": "🐂", "bear": "🐻", "oscillate": "〰️", "recovery": "📈"}
    icon = state_icon.get(r['state'], "〰️")
    n_sell = len(r.get('sell_orders', []))

    sell_section = ""
    if n_sell > 0:
        sell_section = f"""<div class="card">
    <h3 style="margin-bottom:12px;">🔴 卖出清单 ({n_sell}只)</h3>
    <table>
      <thead><tr><th>代码</th><th>名称</th><th class="num">股数</th><th class="num">参考价</th></tr></thead>
      <tbody>{sell_rows}</tbody>
    </table>
  </div>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ZEquant 实盘信号 — mss_dynamic</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; background:#f5f7fa; color:#1a1a2e; padding:24px; }}
  .container {{ max-width:800px; margin:0 auto; }}
  .card {{ background:#fff; border-radius:12px; box-shadow:0 2px 12px rgba(0,0,0,.08); padding:24px; margin-bottom:20px; }}
  h1 {{ font-size:22px; margin-bottom:4px; }}
  .badge {{ display:inline-block; padding:4px 12px; border-radius:6px; font-size:14px; font-weight:600; }}
  .badge.bull {{ background:#e8f5e9; color:#2e7d32; }}
  .badge.bear {{ background:#ffebee; color:#c62828; }}
  .badge.oscillate {{ background:#fff3e0; color:#e65100; }}
  .badge.recovery {{ background:#e3f2fd; color:#1565c0; }}
  .meta {{ display:flex; gap:20px; flex-wrap:wrap; margin:12px 0; font-size:14px; color:#666; }}
  .meta span {{ background:#f5f5f5; padding:4px 10px; border-radius:4px; }}
  .alloc {{ display:flex; gap:12px; flex-wrap:wrap; list-style:none; }}
  .alloc li {{ background:#f0f4ff; padding:6px 14px; border-radius:6px; font-size:14px; }}
  .sub-list {{ list-style:none; }}
  .sub-list li {{ padding:6px 0; font-size:14px; }}
  table {{ width:100%; border-collapse:collapse; font-size:14px; }}
  th {{ text-align:left; padding:10px 8px; border-bottom:2px solid #e0e0e0; color:#666; font-weight:600; }}
  td {{ padding:8px; border-bottom:1px solid #f0f0f0; }}
  tr:last-child td {{ border-bottom:none; }}
  .num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  .total {{ font-weight:600; }}
  .remain {{ margin-top:12px; text-align:right; font-size:15px; color:#666; }}
  .footer {{ text-align:center; font-size:12px; color:#999; margin-top:20px; }}
</style>
</head>
<body>
<div class="container">
  <div class="card">
    <h1>ZEquant 实盘信号</h1>
    <div style="font-size:13px;color:#999;margin-bottom:8px;">mss_dynamic — {_version_label(r)} 动态策略切换</div>
    <div class="badge {r['state']}">{icon} {r['state'].upper()}</div>
    <div class="meta">
      <span>📅 {r['date']}</span>
      <span>🎯 置信度 {(r['confidence']*100):.0f}%</span>
      <span>💰 资金 {r['capital']:,.0f}</span>
      <span>🔄 状态变化 {r.get('state_changed', False)}</span>
    </div>
  </div>
  <div class="card">
    <h3 style="margin-bottom:10px;">📊 子策略状态</h3>
    <ul class="sub-list">{sub_rows}</ul>
  </div>
  <div class="card">
    <h3 style="margin-bottom:10px;">📊 策略分配</h3>
    <ul class="alloc">{alloc_rows}</ul>
  </div>
  {sell_section}
  <div class="card">
    <h3 style="margin-bottom:12px;">🟢 买入清单 ({len(r['orders'])}只)</h3>
    <table>
      <thead><tr><th>#</th><th>代码</th><th>名称</th><th class="num">股数</th><th class="num">价格</th><th class="num">金额</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
    <div class="remain">
      新占用: <span class="total">{r['total_cost']:,.0f}</span> &nbsp;|&nbsp; 总占用: <span class="total">{r['total_capital_used']:,.0f}</span> &nbsp;|&nbsp; 剩余: <span class="total">{r['remain']:,.0f}</span>
    </div>
  </div>
  <div class="footer">ZEquant — Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
</div>
</body>
</html>"""
    with open(path, 'w') as f:
        f.write(html)

    print(f'\n╔{"═"*58}╗')
    print(f'║  ZEquant 实盘信号 — mss_dynamic ({_version_label(r)})')
    print(f'╠{"═"*58}╣')
    print(f'║  日期: {r["date"]}')
    print(f'║  市场状态: {r["state"]} ║ 置信度: {r["confidence"]:.2f}')
    if r.get('state_changed'):
        print(f'║  🔄 状态变化: {r.get("last_state")} → {r["state"]}')
    print(f'║  资金: {r["capital"]:>8,.0f}')
    print(f'╠{"═"*58}╣')
    print(f'║  子策略状态:')
    for sd in r.get('sub_details', []):
        st = sd['status']
        if st == 'rebalanced':
            print(f'║    🔄 {sd["name"]}: 买入{sd["n_buy"]}只 卖出{sd["n_sell"]}只')
        elif st == 'hold':
            print(f'║    ✅ {sd["name"]}: 持仓{sd["n_hold"]}只 (未到期)')
        elif st == 'liquidated':
            print(f'║    ❌ {sd["name"]}: 清仓{sd.get("n_sell",0)}只 (退出分配)')
        else:
            print(f'║    ⚠️ {sd["name"]}: 跳过')
    print(f'╠{"═"*58}╣')
    if r.get('sell_orders'):
        print(f'║  🔴 卖出 ({len(r["sell_orders"])}只):')
        for o in r['sell_orders']:
            n = r['names'].get(o['symbol'], '')
            print(f'║    {o["symbol"]} {n:10s} {o["shares"]}股')
    if r['orders']:
        print(f'║  🟢 买入 ({len(r["orders"])}只):')
        print(f'║    {"代码":<8} {"名称":<10} {"股数":<6} {"价格":<7} {"金额":<8}')
        for o in r['orders']:
            n = r['names'].get(o['symbol'], '')
            print(f'║    {o["symbol"]:<8} {n:<10} {o["shares"]:<6} {o["price"]:<7.2f} {o["cost"]:<8,.0f}')
    print(f'╠{"═"*58}╣')
    print(f'║  新占用: {r["total_cost"]:>8,.0f}  总占用: {r["total_capital_used"]:>8,.0f}  剩余: {r["remain"]:>8,.0f}')
    print(f'╚{"═"*58}╝')



def _send_email(signal_path: str):
    """发送信号邮件。"""
    try:
        from live.notification import Mailer
        mailer = Mailer()
        mailer.send_signal_from_file(signal_path)
    except Exception as e:
        logger.error("邮件发送失败: %s", e)



def _sync_hold_snapshot(today: str, used_capital: float, remain: float):
    """非调仓日：将上日的持仓快照同步到今日，用最新市价更新总资产。"""
    try:
        from core.database import Database
        live_db = Database(LIVE_DB_PATH)
        quant_db = Database("./data/quant_data.db")

        # 查上次快照
        last = live_db.conn.execute("""
            SELECT date, total_value, cash, positions, orders
            FROM daily_snapshots ORDER BY date DESC LIMIT 1
        """).fetchone()
        if not last:
            logger.warning("无上次快照，跳过同步")
            live_db.close()
            quant_db.close()
            return

        last_date = last[0]
        positions = json.loads(last[3]) if last[3] else {}
        cash = float(last[2]) if last[2] else 0

        # 用最新收盘价算持仓市值（兼容 dict 和 int 格式）
        position_value = 0.0
        if positions:
            symbols = list(positions.keys())
            ph = ",".join("?" for _ in symbols)
            rows = quant_db.conn.execute(
                f"SELECT symbol, close FROM daily_bars WHERE date=? AND symbol IN ({ph})",
                [today] + symbols
            ).fetchall()
            prices = {r[0]: float(r[1]) for r in rows}
            # 补漏：取最近价
            for sym in symbols:
                if sym not in prices:
                    r = quant_db.conn.execute(
                        "SELECT close FROM daily_bars WHERE symbol=? AND date<=? ORDER BY date DESC LIMIT 1",
                        [sym, today]
                    ).fetchone()
                    if r:
                        prices[sym] = float(r[0])
            # 从 trades 表更新成本价
            for sym in symbols:
                buys = live_db.conn.execute(
                    "SELECT price, shares FROM trades WHERE symbol=? AND direction=? AND date<=?",
                    [sym, 'B', today]
                ).fetchall()
                if buys:
                    total_amount = sum(float(b[0]) * b[1] for b in buys)
                    total_shares = sum(b[1] for b in buys)
                    avg_cost = total_amount / total_shares if total_shares > 0 else 0
                    info = positions[sym]
                    if isinstance(info, dict):
                        info["cost"] = round(avg_cost, 3)
                    else:
                        positions[sym] = {"shares": int(info), "cost": round(avg_cost, 3)}
            for sym, info in positions.items():
                shares = info["shares"] if isinstance(info, dict) else info
                position_value += prices.get(sym, 0) * shares

        total_value = position_value + cash

        # 写入新快照
        live_db.conn.execute("""
            INSERT INTO daily_snapshots (date, strategy, total_value, cash, positions, orders)
            VALUES (?, 'mss_dynamic', ?, ?, ?, '[]')
            ON CONFLICT (date, strategy) DO UPDATE SET
                total_value=EXCLUDED.total_value, cash=EXCLUDED.cash,
                positions=EXCLUDED.positions, orders=EXCLUDED.orders
        """, [today, round(total_value, 2), round(cash, 2),
              json.dumps(positions)])

        # 更新 daily_performance
        prev_perf = live_db.conn.execute(
            "SELECT cumulative FROM daily_performance ORDER BY date DESC LIMIT 1"
        ).fetchone()
        prev_cum = float(prev_perf[0]) if prev_perf else 1.0
        prev_total = float(last[1]) if last[1] else total_value
        daily_ret = (total_value - prev_total) / prev_total if prev_total > 0 else 0.0
        cumulative = prev_cum * (1 + daily_ret)
        # 算回撤
        all_cum = [r[0] for r in live_db.conn.execute(
            "SELECT cumulative FROM daily_performance ORDER BY date").fetchall()]
        all_cum.append(cumulative)
        peak = max(all_cum) if all_cum else cumulative
        max_dd = (cumulative - peak) / peak if peak > 0 else 0.0

        # 基准收益
        bench_ret = 0.0
        br = quant_db.conn.execute(
            "SELECT pct_change FROM daily_bars WHERE symbol='000300' AND date=?",
            [today]
        ).fetchone()
        if br:
            bench_ret = float(br[0]) / 100.0

        live_db.conn.execute("""
            INSERT INTO daily_performance (date, total_value, daily_return, cumulative,
                max_drawdown, positions_count, turnover, benchmark_ret)
            VALUES (?, ?, ?, ?, ?, ?, 0, ?)
            ON CONFLICT (date) DO UPDATE SET
                total_value=EXCLUDED.total_value, daily_return=EXCLUDED.daily_return,
                cumulative=EXCLUDED.cumulative, max_drawdown=EXCLUDED.max_drawdown,
                positions_count=EXCLUDED.positions_count, benchmark_ret=EXCLUDED.benchmark_ret
        """, [today, round(total_value, 2), round(daily_ret, 6),
              round(cumulative, 6), round(max_dd, 6),
              len(positions), round(bench_ret, 6)])

        live_db.conn.commit()
        logger.info(f"快照同步: {today} 持仓{len(positions)}只 总资产{total_value:,.0f} "
                    f"日收益{daily_ret:+.4f} 累计{cumulative:.4f}")
        live_db.close()
        quant_db.close()
    except Exception as e:
        logger.warning("快照同步失败(非关键): %s", e)
