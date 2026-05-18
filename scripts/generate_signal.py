"""每日信号生成脚本。

每天运行一次，自动判断是否当前策略的调仓日。
- 调仓日: 生成买入/卖出订单
- 非调仓日: 输出"持有"信号

Usage:
    python3 scripts/generate_signal.py --strategy mf_d10_rp --capital 50000
    python3 scripts/generate_signal.py --strategy mf_d10_rp --capital 50000 --build  # 强制建仓
"""
from __future__ import annotations
import argparse, json, os, sys, logging
from datetime import date, datetime
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.database import Database
from core.screening.universe import SymbolUniverse

logger = logging.getLogger('signal_gen')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

STRATEGY_CONFIGS = {
    'mf_d10_rp': {
        'top_n': 20,
        'rebal_freq': 10,
        'cfg_path': 'core/strategies/impl/mf_d10_rp/config.json',
        'name': 'MF_D10_RP',
    },
    'mf_d10_emergency_v1': {
        'top_n': 20,
        'rebal_freq': 10,
        'cfg_path': 'core/strategies/impl/mf_d10_emergency_v1/config.json',
        'name': 'MF_D10_EMERGENCY_V1',
    },
}

def load_weights(cfg_path):
    import json as j
    with open(cfg_path) as f:
        cfg = j.load(f)
    weights = cfg['selector']['weights']
    top_n = cfg['composer']['top_n']
    return weights, top_n

def load_holdings(strategy_dir):
    """读取当前持仓（从上一期信号）。"""
    holdings = {}
    if not os.path.exists(strategy_dir):
        return holdings
    builds = sorted([f for f in os.listdir(strategy_dir) if f.startswith('build_') and f.endswith('.json')])
    if not builds:
        return holdings
    latest = os.path.join(strategy_dir, builds[-1])
    with open(latest) as f:
        data = json.load(f)
    for o in data.get('orders', []):
        if o['direction'] == '买入':
            holdings[o['symbol']] = {'shares': o['shares'], 'price': o['price']}
    return holdings

def is_rebalance_day(strategy_dir, rebal_freq, signal_date):
    """判断今天是否是调仓日。

    从上一期 build 文件的日期算起，如果间隔 >= rebal_freq 个交易日，
    且上一期建仓后已经过了足够时间，则是调仓日。
    如果没有历史信号（首日），则返回 True（需要建仓）。
    """
    db = Database()
    builds = sorted([f for f in os.listdir(strategy_dir) if f.startswith('build_') and f.endswith('.json')])
    if not builds:
        db.close()
        return True  # 首日建仓

    latest = builds[-1]
    date_str = latest.replace('build_', '').replace('.json', '')
    try:
        last_date = datetime.strptime(date_str, '%Y%m%d').date()
    except:
        db.close()
        return True

    if last_date >= signal_date:
        db.close()
        return False  # 今天已经出过信号

    dates = db.conn.execute('''
        SELECT DISTINCT date FROM daily_bars
        WHERE date > ? AND date <= ?
        ORDER BY date
    ''', [last_date.isoformat(), signal_date.isoformat()]).df()
    db.close()
    trading_days = len(dates)
    return trading_days >= rebal_freq

def compute_scores(db, weights, top_n, signal_date):
    """计算因子综合得分，返回候选排名。"""
    factor_cols = [c for c in weights.keys() if c in db.list_factor_columns()]
    df = db.get_factors(factor_names=factor_cols, start_date=str(signal_date), end_date=str(signal_date), with_close=True)

    available = [c for c in factor_cols if c in df.columns]
    scores = np.zeros(len(df))
    for fn in available:
        w = weights.get(fn, 0)
        vals = df[fn].fillna(0).values
        lo, hi = np.percentile(vals[~np.isnan(vals)], [1, 99]) if np.sum(~np.isnan(vals)) > 1 else (-1, 1)
        vals = np.clip(vals, lo, hi)
        mu, sd = np.mean(vals), np.std(vals)
        if sd > 1e-10:
            vals = (vals - mu) / sd
        scores += vals * w
    df['score'] = scores

    universe = SymbolUniverse(db)
    bars = db.get_daily_bars(start_date=str(signal_date), end_date=str(signal_date))
    buyable = universe.filter_buyable(signal_date.isoformat(), bars)
    df = df[df['symbol'].isin(buyable)].copy()
    df = df.sort_values('score', ascending=False)
    return df, factor_cols

def generate_orders(df, top_n, capital, holdings):
    """生成订单列表。"""
    top_candidates = df.head(40)
    selected = []
    n = top_n
    while n >= 5 and not selected:
        alloc = capital / n
        for _, r in top_candidates.iterrows():
            price = float(r.get('close', 0))
            if price > 0 and 100 * price <= alloc:
                selected.append(r)
                if len(selected) == n:
                    break
        if not selected:
            n -= 1

    if not selected:
        alloc = capital / min(len(top_candidates), 10)
        for _, r in top_candidates.iterrows():
            price = float(r.get('close', 0))
            if price > 0 and 100 * price <= alloc:
                selected.append(r)
                if len(selected) == 10:
                    break

    n_actual = len(selected)
    alloc_per = capital / n_actual
    orders = []
    total_cost = 0
    for r in selected:
        sym = r['symbol']
        price = float(r.get('close', 0))
        shares = int(alloc_per // (price * 100)) * 100
        if shares < 100:
            continue
        total_cost += shares * price
        orders.append({
            'symbol': sym, 'direction': '买入',
            'shares': shares, 'price': round(price, 2),
            'reason': f'{STRATEGY_CONFIGS[args.strategy]["name"]}调仓',
        })

    # 卖出信号：持仓中有但不在新买入列表的
    new_symbols = {o['symbol'] for o in orders}
    sell_orders = []
    sell_revenue = 0
    for sym, holding in holdings.items():
        if sym not in new_symbols:
            price = float(df[df['symbol'] == sym]['close'].iloc[0]) if sym in df['symbol'].values else 0
            sell_price = round(price * 1.002, 2) if price > 0 else 0  # 模拟卖一价
            sell_orders.append({
                'symbol': sym, 'direction': '卖出',
                'shares': holding['shares'],
                'price': sell_price,
                'reason': '调仓卖出',
            })
            sell_revenue += sell_price * holding['shares']

    return orders, sell_orders, round(total_cost, 2), round(capital - total_cost + sell_revenue, 2)

def main():
    parser = argparse.ArgumentParser(description='每日信号生成')
    parser.add_argument('--strategy', default='mf_d10_rp', help='策略名')
    parser.add_argument('--capital', type=float, default=50000, help='总资金')
    parser.add_argument('--date', help='信号日期(默认今天)，格式YYYY-MM-DD')
    parser.add_argument('--build', action='store_true', help='强制建仓(无视调仓日判定)')
    global args
    args = parser.parse_args()

    if args.strategy not in STRATEGY_CONFIGS:
        logger.error(f'未知策略: {args.strategy}')
        sys.exit(1)

    config = STRATEGY_CONFIGS[args.strategy]
    signal_date = date.fromisoformat(args.date) if args.date else date.today()
    strategy_dir = f'data_live/{args.strategy}'
    os.makedirs(strategy_dir, exist_ok=True)

    weights, top_n = load_weights(config['cfg_path'])
    db = Database()

    # 判断是否调仓日
    is_rebal = args.build or is_rebalance_day(strategy_dir, config['rebal_freq'], signal_date)

    if not is_rebal:
        holdings = load_holdings(strategy_dir)
        holding_symbols = list(holdings.keys())
        sym_df = db.get_symbols()
        name_map = dict(zip(sym_df['symbol'], sym_df['name'])) if not sym_df.empty else {}
        logger.info(f'{signal_date} 非调仓日，持有 {len(holding_symbols)} 只')
        print()
        print(f'系统: ZEquant 每日信号')
        print(f'策略: {args.strategy}（{config["rebal_freq"]}个交易日调仓）')
        print(f'日期: {signal_date}')
        print('─' * 50)
        print(f'🟢 操作: 持有不动（今日非调仓日）')
        print(f'📦 当前持仓: {len(holding_symbols)} 只')
        for sym in holding_symbols:
            name = name_map.get(sym, '')
            h = holdings[sym]
            print(f'   {sym} {name:10s} {h["shares"]}股 @ {h["price"]:.2f}')
        print('─' * 50)
        print(f'下次调仓: 当距上次建仓满{config["rebal_freq"]}个交易日时')
        db.close()
        return

    # 调仓日：生成信号
    logger.info(f'{signal_date} 调仓日，生成信号...')
    df, factor_cols = compute_scores(db, weights, top_n, signal_date)
    holdings = load_holdings(strategy_dir)
    orders, sell_orders, total_cost, remain = generate_orders(df, top_n, args.capital, holdings)

    sym_df = db.get_symbols()
    name_map = dict(zip(sym_df['symbol'], sym_df['name'])) if not sym_df.empty else {}

    signal_file = f'{strategy_dir}/build_{signal_date.strftime("%Y%m%d")}.json'
    with open(signal_file, 'w') as f:
        json.dump({
            'meta': {
                'strategy': args.strategy,
                'signal_date': str(signal_date),
                'execution_date': str(signal_date),
                'capital': args.capital,
                'n_buy': len(orders),
                'n_sell': len(sell_orders),
                'total_cost': total_cost,
                'remain': remain,
                'is_rebalance': True,
            },
            'sell_orders': sell_orders,
            'buy_orders': orders,
        }, f, indent=2, ensure_ascii=False)
    logger.info(f'信号已写入 {signal_file}')

    print()
    print(f'系统: ZEquant 每日信号')
    print(f'策略: {args.strategy}（{config["rebal_freq"]}个交易日调仓）')
    print(f'日期: {signal_date}')
    print('─' * 70)
    if sell_orders:
        print(f'🔴 卖出 ({len(sell_orders)}只):')
        for o in sell_orders:
            name = name_map.get(o['symbol'], '')
            print(f'   {o["symbol"]} {name:10s} {o["shares"]}股')
    print(f'🟢 买入 ({len(orders)}只):')
    print(f'   {"代码":<8} {"名称":<10} {"股数":<6} {"价格":<7} {"金额":<8}')
    for o in orders:
        name = name_map.get(o['symbol'], '')
        cost = o['shares'] * o['price']
        print(f'   {o["symbol"]:<8} {name:<10} {o["shares"]:<6} {o["price"]:<7.2f} {cost:<8,.0f}')
    print('─' * 70)
    print(f'占用: {total_cost:,.0f}  剩余: {remain:,.0f}')

    db.close()

if __name__ == '__main__':
    main()
