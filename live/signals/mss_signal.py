"""mss_signal — split from mss_dynamic.py."""
from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd

logger = logging.getLogger('mss_live')

from .mss_state import (
    FACTOR_NAMES, SUB_STRATEGY_META, BOARD_PREFIXES,
    DEFAULT_EXCLUDE, STOP_LOSS_CONFIG, TRAILING_STOP_PCT,
    LIVE_DB_PATH, SIGNAL_DIR, V6A_ALLOCATION, COOLDOWN_DAYS
)
from .mss_state import (
    _signal_dir, _init_live_db, _live_db, _load_all_sub_states,
    _load_sub_state, _save_sub_state, _delete_sub_state,
    _qdb,
    _get_last_known_state, _set_last_known_state,
    _get_pending_state, _set_pending_state, _clear_pending_state,
    _trading_days_between
)
from .mss_factors import (
    _factors, _bars, _weights, _zscore, mf_score, chip_score,
    trend_factor, vol_factor, composite_factor, _market_breadth,
    market_state, combo_score, filter_buyable
)


def _write_html(path: str, r: dict):
    from .mss_report import _write_html as _wh
    _wh(path, r)



def generate_orders(df: pd.DataFrame, capital: float, top_n: int,
                     old_holdings: dict = None,
                     exclude_symbols: set = None) -> Tuple[list, list, float, dict]:
    df = df.sort_values('score', ascending=False).dropna(subset=['close'])
    df = df[df['close'] > 0]

    kept = {}; kept_cost = 0.0
    if old_holdings:
        old_syms = set(old_holdings.keys())
        for _, r in df.iterrows():
            sym = r['symbol']
            if sym in old_syms and sym not in kept:
                kept[sym] = old_holdings[sym]
                kept_cost += kept[sym]['shares'] * kept[sym]['price']
    available = max(capital - kept_cost, capital * 0.3)
    new_slots = max(1, top_n - len(kept))

    # 从非保留股票中取 new_slots*3 候选，不限制 head 数量
    candidates = []
    for _, r in df.iterrows():
        if r['symbol'] not in kept:
            if exclude_symbols and r['symbol'] in exclude_symbols:
                continue
            candidates.append(r)
            if len(candidates) >= new_slots * 3:
                break

    # 贪心填充：直到预算用完或填满 new_slots
    new_budget = available
    buy_orders, new_cost = [], 0.0
    new_syms = set()
    equal_share = new_budget / new_slots if new_slots > 0 else new_budget

    for r in candidates:
        p = float(r['close'])
        # 每只最多花 equal_share，最少1手
        budget_per = min(equal_share * 1.2, new_budget)
        s = int(budget_per // (p * 100)) * 100
        if s < 100:
            s = 100
        c = s * p
        if c > new_budget * 1.1:  # 允许轻微超支
            continue
        buy_orders.append({'symbol': r['symbol'], 'direction': '买入',
                           'shares': s, 'price': round(p, 2), 'cost': round(c, 2)})
        new_cost += c; new_budget -= c; new_syms.add(r['symbol'])
        if len(buy_orders) >= new_slots or new_budget < 100:
            break

    sell_orders = []
    if old_holdings:
        for sym, h in old_holdings.items():
            if sym not in new_syms and sym not in kept:
                sell_orders.append({'symbol': sym, 'direction': '卖出',
                                    'shares': h['shares'], 'price': h['price'],
                                    'reason': '子策略调仓'})
    return buy_orders, sell_orders, round(new_cost + kept_cost, 2), kept


def run(capital: float = 50000, signal_date: str = None,
        exclude_boards: list = None, force: bool = False,
        mode: str = "v2b") -> dict:
    """生成信号。signal_date 为信号请求日期（用于目录），ds 为实际数据日期。

    Args:
        mode: "baseline"=原始版, "v2b"=增强ST+止损(默认), "v2c"=增强ST+止损+置信度联动
    """
    exclude_boards = exclude_boards or DEFAULT_EXCLUDE
    signal_date = signal_date or str(date.today())

    enhanced_st = mode in ("v2b", "v2c")
    stop_loss_enabled = mode in ("v2b", "v2c")
    use_confidence_weights = (mode == "v2c")

    _init_live_db()

    # capital <= 0 时自动从 DB 读取总资产和实际现金
    _cash_from_db = None
    if capital <= 0:
        try:
            _ldb = _live_db()
            r = _ldb.conn.execute(
                "SELECT cash, total_value FROM daily_snapshots ORDER BY date DESC LIMIT 1"
            ).fetchone()
            if r and r[1]:
                capital = float(r[1])
                _cash_from_db = float(r[0])
                logger.info(f"  从快照读取: 总资产={capital:.0f}  现金={_cash_from_db:.0f}")
            else:
                r = _ldb.conn.execute(
                    "SELECT total_value FROM daily_performance ORDER BY date DESC LIMIT 1"
                ).fetchone()
                if r:
                    capital = float(r[0])
                    logger.info(f"  从绩效表读取总资产: {capital:.0f}")
            _ldb.close()
        except Exception as e:
            logger.warning(f"  自动读取数据失败: {e}，使用默认 {capital}")
    logger.info(f"  资金: {capital:.0f}")

    qconn = _qdb()
    live_db = None
    ds = signal_date
    latest_data = None
    try:
        lf = qconn.execute("SELECT MAX(date) FROM factors_wide").fetchone()[0]
        lb = qconn.execute("SELECT MAX(date) FROM daily_bars WHERE close>0").fetchone()[0]
        latest_data = min(lf, lb) if lf and lb else (lf or lb)
        if latest_data and str(latest_data) != ds:
            logger.info(f"{ds} 无数据，使用 {latest_data}")
            ds = str(latest_data)
    except Exception:
        pass

    # 止损/止盈始终使用 daily_bars 最新可用数据（以信号日期为准）
    bars_ds = ds
    try:
        lb_max = qconn.execute(
            "SELECT MAX(date) FROM daily_bars WHERE close>0 AND date<=?",
            [signal_date]
        ).fetchone()[0]
        if lb_max and str(lb_max) > bars_ds:
            bars_ds = str(lb_max)
    except Exception:
        pass
    if bars_ds != ds:
        logger.info(f"  止损数据日期: {bars_ds} (因子数据: {ds})")

    state, conf = market_state(qconn, ds)

    # V6 市场广度二次确认: breadth < 0.35 降级为 oscillate
    br = _market_breadth(qconn, ds)
    if br is not None and br < 0.35 and state != "oscillate":
        logger.info(f"  市场广度={br:.3f}<0.35, {state}→oscillate")
        state = "oscillate"

    current_alloc = V6A_ALLOCATION.get(state, V6A_ALLOCATION["oscillate"])

    # V2c: 置信度联动权重调整
    if use_confidence_weights and conf < 0.5:
        adjusted = []
        for name, weight in current_alloc:
            if name == "mf_d10_rp":
                weight = weight * 0.6
            elif name == "c01_layered_d5":
                weight = weight * 0.7
            adjusted.append((name, max(weight, 0.0)))
        total_w = sum(w for _, w in adjusted)
        if total_w > 0:
            adjusted = [(n, w / total_w) for n, w in adjusted]
        current_alloc = adjusted
        logger.info(f"  V2c: 置信度={conf:.2f}<0.5, 已调整分配权重")

    live_db = _live_db()

    # 读取上次状态
    last_state = _get_last_known_state(live_db)
    all_old_states = _load_all_sub_states(live_db)

    # V7.1 状态切换冷却期: 新状态需连续出现 COOLDOWN_DAYS 个交易日才切换
    if last_state is not None and last_state != state:
        pending_state, pending_days = _get_pending_state(live_db)
        if pending_state == state:
            pending_days += 1
            _set_pending_state(live_db, state, pending_days)
        else:
            pending_days = 1
            _set_pending_state(live_db, state, pending_days)
            logger.info(f"  冷却期: {state} 第1天 (需连续{COOLDOWN_DAYS}天)")
        if pending_days >= COOLDOWN_DAYS:
            _clear_pending_state(live_db)
            logger.info(f"  冷却期满: {state} 连续{pending_days}天 → 切换")
        else:
            logger.info(f"  冷却期: 保持 {last_state} ({state} 第{pending_days}/{COOLDOWN_DAYS}天)")
            state = last_state
            current_alloc = V6A_ALLOCATION.get(state, V6A_ALLOCATION["oscillate"])
    elif last_state is not None:
        _clear_pending_state(live_db)

    state_changed = (last_state is not None and last_state != state)

    # 当前分配的子策略名列表
    current_names = {name for name, w in current_alloc}
    # 旧持仓中的所有子策略名
    old_names = set(all_old_states.keys())

    # ===== 收集所有卖出 =====
    all_buy = []
    all_sell = []
    sub_details = []
    total_new_cost = 0.0
    total_capital_used = 0.0
    sold_off_symbols = set()  # 已卖出的，避免重复
    global_bought = set()  # 跨策略已买入的去重追踪

    # 1. 处理状态切换 → 退出分配的子策略，全部清仓卖出
    orphan_names = old_names - current_names
    if orphan_names:
        logger.info(f"状态切换 {last_state}→{state}，以下子策略退出分配:")
    for name in orphan_names:
        old = all_old_states[name]
        holdings = old.get("holdings", {})
        if holdings:
            logger.info(f"  ❌ {name}: 清仓卖出 {len(holdings)} 只")
            for sym, h in holdings.items():
                all_sell.append({'symbol': sym, 'direction': '卖出',
                                 'shares': h['shares'], 'price': h.get('price', 0),
                                 'reason': f'{name}退出分配'})
                sold_off_symbols.add(sym)
        _delete_sub_state(live_db, name)
        # 清仓的资金会释放，不计入占用
        sub_details.append({"name": name, "status": "liquidated",
                            "n_sell": len(holdings)})

    # 2. 处理当前分配中的每个子策略
    for name, weight in current_alloc:
        sub_capital = capital * weight
        meta = SUB_STRATEGY_META.get(name, {"rebal_freq": 10, "top_n": 8})
        old_state = all_old_states.get(name, {})
        last_date_s = old_state.get("last_date")

        due = False
        is_new = (last_date_s is None)

        if is_new:
            due = True
            logger.info(f"  {name}: 首次激活 (权重{weight:.0%})")
        elif force:
            due = True
            logger.info(f"  {name}: 强制调仓")
        elif state_changed:
            due = True
            logger.info(f"  {name}: 状态切换→调仓")
        else:
            ndays = _trading_days_between(qconn, last_date_s, ds)
            due = ndays >= meta["rebal_freq"]

        if due:
            logger.info(f"  {name}: 到期调仓 (距上次 {last_date_s or '首次'} "
                        f"{_trading_days_between(qconn, last_date_s, ds) if last_date_s else 0}天 >= {meta['rebal_freq']}天)")
            scores = _get_scores(qconn, name, ds)
            if scores.empty or 'score' not in scores.columns:
                logger.warning(f"  {name}: 信号为空，跳过")
                sub_details.append({"name": name, "status": "skip_empty"})
                continue

            scores = scores[['symbol', 'close', 'score']].dropna()
            scores['symbol'] = scores['symbol'].astype(str)
            scores = filter_buyable(scores, qconn, ds, exclude_prefixes=exclude_boards, enhanced_st=enhanced_st)

            old_h = old_state.get("holdings", {})
            buys, sells, cost, kept = generate_orders(
                scores, sub_capital, meta["top_n"],
                old_holdings=old_h, exclude_symbols=global_bought
            )

            # 更新状态：新买入 + 保留持仓
            new_holdings = {}
            for sym, h in kept.items():
                new_holdings[sym] = h
            for o in buys:
                new_holdings[o['symbol']] = {"shares": o['shares'], "price": o['price'], "peak": o['price']}
            _save_sub_state(live_db, name, {
                "last_date": ds, "holdings": new_holdings, "used_capital": cost,
            }, state, weight)

            all_buy.extend(buys)
            for o in buys:
                global_bought.add(o['symbol'])
            for sym in kept:
                global_bought.add(sym)
            # 过滤掉已被清仓的卖出
            for o in sells:
                if o['symbol'] not in sold_off_symbols:
                    all_sell.append(o)
            total_new_cost += cost
            total_capital_used += cost
            sub_details.append({
                "name": name, "status": "rebalanced",
                "weight": weight, "capital": round(sub_capital, 2),
                "n_buy": len(buys), "n_sell": len(sells), "cost": cost,
            })
        else:
            old_h = old_state.get("holdings", {})
            kept_cap = old_state.get("used_capital", 0)

            # V2b/V2c: 个股止损 + V6 trailing stop 检查
            stop_sold = set()
            if old_h:
                try:
                    bars_today = _bars(qconn, bars_ds, bars_ds)
                    if not bars_today.empty:
                        pm = dict(zip(bars_today['symbol'].astype(str), bars_today['close'].astype(float)))
                        sl_pct = STOP_LOSS_CONFIG.get(name, 0.08)
                        trail_pct = TRAILING_STOP_PCT
                        for sym, h in list(old_h.items()):
                            cp = pm.get(sym)
                            if cp and h.get('price', 0) > 0:
                                entry = h['price']
                                loss = (cp - entry) / entry
                                # 固定止损
                                if loss < -sl_pct:
                                    all_sell.append({'symbol': sym, 'direction': '卖出',
                                                     'shares': h['shares'], 'price': cp,
                                                     'reason': f'{name}: 止损{sl_pct*100:.0f}%'})
                                    logger.info(f"  ⛔ {name}: {sym} 止损触发 (亏损{loss*100:.1f}%)")
                                    stop_sold.add(sym)
                                    continue
                                # V6 移动止盈: 从峰值回撤 > trail_pct 时卖出（仅当前仍有利润时）
                                peak = h.get('peak', entry)
                                if cp > peak:
                                    peak = cp
                                    old_h[sym]['peak'] = peak
                                if peak > entry and cp > entry and cp < peak * (1.0 - trail_pct):
                                    drawdown = (cp - peak) / peak
                                    all_sell.append({'symbol': sym, 'direction': '卖出',
                                                     'shares': h['shares'], 'price': cp,
                                                     'reason': f'{name}: 移动止盈{trail_pct*100:.0f}%'})
                                    logger.info(f"  🎯 {name}: {sym} 移动止盈触发 (从峰值回撤{drawdown*100:.1f}%)")
                                    stop_sold.add(sym)
                                    continue
                                # 更新 peak
                                if cp > peak:
                                    old_h[sym]['peak'] = cp
                except Exception as e:
                    logger.warning(f"  {name}: 止损检查失败: {e}")

            # 将持有中的股票加入全局去重列表，避免其他策略重复买入
            for sym in old_h:
                if sym not in stop_sold:
                    global_bought.add(sym)

            # 止损后调整占用：从 kept_cap 中扣除已止损股票的原始买入成本
            actual_kept_cap = kept_cap
            if stop_sold:
                for sym in stop_sold:
                    h = old_h.get(sym)
                    if h:
                        actual_kept_cap -= h['shares'] * h['price']
                actual_kept_cap = max(0, actual_kept_cap)
            total_capital_used += actual_kept_cap
            next_rebal = _next_rebal_date(qconn, last_date_s, ds, meta["rebal_freq"])
            logger.info(f"  {name}: 未到期，维持 {len(old_h)} 只持仓"
                        + (f" (止损{len(stop_sold)}只)" if stop_sold else "")
                        + (f", 下次调仓≈{next_rebal}" if next_rebal else ""))
            sub_details.append({
                "name": name, "status": "hold",
                "weight": weight, "n_hold": len(old_h),
                "last_date": last_date_s,
                "rebal_freq": meta["rebal_freq"],
                "next_rebalance": next_rebal,
            })

    # 写入 last_known_state
    _set_last_known_state(live_db, state)
    live_db.conn.commit()

    # 股票名称
    names = {}
    try:
        ndf = qconn.execute("SELECT symbol, name FROM symbols").fetchdf()
        names = dict(zip(ndf['symbol'], ndf['name']))
    except Exception:
        pass
    finally:
        qconn.close()
        if live_db is not None:
            live_db.close()

    # 计算真实占用 = 既有持仓 + 新增买入 - 已卖出持仓中用到的部分
    # total_capital_used 已包含: 退出分配的old capital + 调仓的new cost + hold的kept capital
    occupied = round(total_capital_used, 2)
    if occupied == 0:
        occupied = round(total_new_cost, 2)  # 极端情况：只有新买没有旧持仓

    result = {
        'is_rebalance': len(all_buy) > 0,
        'state_changed': state_changed,
        'last_state': last_state,
        'date': signal_date, 'data_date': ds,
        'state': state, 'confidence': round(float(conf), 3),
        'breadth': round(float(br), 3) if br is not None else None,
        'capital': capital, 'total_cost': occupied,
        'total_capital_used': occupied,
        'remain': round(_cash_from_db, 2) if _cash_from_db is not None else round(capital - occupied, 2),
        'orders': all_buy,
        'sell_orders': all_sell,
        'allocations': {n: w for n, w in current_alloc},
        'names': names,
        'sub_details': sub_details,
        'mode': mode,
    }
    return result



def _next_rebal_date(qconn, last_date: Optional[str], ds: str, rebal_freq: int) -> Optional[str]:
    """计算下次调仓日。"""
    if last_date is None:
        return None
    ndays = _trading_days_between(qconn, last_date, ds)
    remaining = max(0, rebal_freq - ndays)
    if remaining == 0:
        return None  # 今天就该调仓了
    # 获取 future trading days
    rows = qconn.execute(
        "SELECT DISTINCT date FROM daily_bars WHERE date > ? AND close>0 ORDER BY date",
        [ds]
    ).fetchall()
    if len(rows) < remaining:
        return str(rows[-1][0]) + "+" if rows else None  # 数据不够，用最后一天+表示至少
    return str(rows[remaining - 1][0])



def _get_scores(qconn, name: str, date_str: str) -> pd.DataFrame:
    meta = SUB_STRATEGY_META.get(name, {})
    sig = meta.get("signal", "mf")
    mf = mf_score(qconn, date_str)
    chip = chip_score(qconn, date_str)
    vr = vol_factor(qconn, date_str)
    tr = trend_factor(qconn, date_str)
    # V6 composite择时: trend×0.6 + vol×0.4
    cp = composite_factor(qconn, date_str)
    mfv = c01 = None
    if not mf.empty:
        mfv = mf.copy()
        mfv['score'] = mfv['score'] * cp
        c01 = mf.copy()
        c01['score'] = c01['score'] * tr

    if sig == "mf":
        return mf
    elif sig == "mf_vol":
        return mfv if mfv is not None else mf
    elif sig == "chip":
        return chip
    elif sig == "mf_trend":
        return c01 if c01 is not None else mf
    elif sig == "osr":
        # 超跌反��: chip(低波低动量) × trend(趋势确认)，适合recovery状态
        if not chip.empty:
            osr = chip.copy()
            osr['score'] = osr['score'] * tr
            return osr
        return mf
    elif sig == "combo_50" and not mf.empty and not chip.empty:
        return combo_score(mf, chip, 0.5)
    elif sig == "combo_60" and not mf.empty and not chip.empty:
        return combo_score(mf, chip, 0.6)
    return mf


def main():
    parser = argparse.ArgumentParser(description='mss_dynamic 实盘信号')
    parser.add_argument('--capital', type=float, default=50000)
    parser.add_argument('--date', help='信号日期(yyyy-mm-dd)')
    parser.add_argument('--exclude-boards', nargs='*',
                        default=DEFAULT_EXCLUDE,
                        help='排除板块代码前缀，默认排除创业板 300 301')
    parser.add_argument('--force', action='store_true', help='强制所有子策略调仓')
    parser.add_argument('--email', action='store_true', help='信号生成后发送邮件通知')
    parser.add_argument('--mode', choices=['baseline', 'v2b', 'v2c'], default='v2b',
                        help='基线版 / V2b增强ST+止损(默认) / V2c+置信度联动')
    args = parser.parse_args()

    r = run(capital=args.capital, signal_date=args.date,
            exclude_boards=args.exclude_boards, force=args.force,
            mode=args.mode)

    date_key = r["date"].replace("-", "")
    sig_dir = _signal_dir(date_key)

    # 总是写入信号文件（即使是持有不动）
    meta = {
        'strategy': 'mss_dynamic', 'config': r.get('mode', 'v2b'),
        'signal_date': r['date'], 'data_date': r.get('data_date', r['date']), 'market_state': r['state'],
        'state_changed': r.get('state_changed', False),
        'last_state': r.get('last_state'),
        'confidence': r['confidence'],
        'capital': r['capital'],
        'total_cost': r['total_cost'],
        'total_capital_used': r['total_capital_used'],
        'remain': r['remain'],
        'breadth': r.get('breadth'),
        'n_buy': len(r['orders']),
        'n_sell': len(r.get('sell_orders', [])),
        'is_hold': len(r['orders']) == 0 and len(r.get('sell_orders', [])) == 0,
    }
    sf = os.path.join(sig_dir, 'build.json')
    with open(sf, 'w') as f:
        json.dump({'meta': meta, 'allocation': r['allocations'],
                    'sell_orders': r.get('sell_orders', []),
                    'buy_orders': r['orders'],
                    'sub_details': r.get('sub_details', []),
                    'names': r.get('names', {})},
                  f, indent=2, ensure_ascii=False)
    logger.info(f'信号已写入 {sf}')

    hf = os.path.join(sig_dir, 'build.html')
    _write_html(hf, r)
    logger.info(f'HTML已写入 {hf}')

    # 自动同步持仓快照
    if meta['is_hold']:
        from .mss_report import _sync_hold_snapshot as _shs
        _shs(r['date'], meta['total_capital_used'], meta['remain'])
    else:
        logger.info(f'调仓日，等待 record.py 录入成交')

    if args.email:
        from .mss_report import _send_email as _se
        _se(sf)
