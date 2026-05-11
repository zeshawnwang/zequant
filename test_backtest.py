#!/usr/bin/env python3
"""测试回测流程。"""
import sys
sys.path.insert(0, '.')

from core.database import Database
from core.backtest import BacktestEngine
from core.screening import FactorRankSelector
from core.timings import TrendVolatilityTiming
from core.positioners import EqualWeightBuilder
from core.strategy import QuantStrategy

print('测试完整回测流程...')

db = Database()

required_factors = ['momentum_5', 'momentum_20', 'rsi_14', 'macd', 'macd_signal', 'volatility_20', 'volume_ratio', 'boll_position']

factor_data = db.get_factors(
    factor_names=required_factors,
    start_date='2020-01-01',
    end_date='2020-06-30',
    with_close=True,
)

print(f'因子数据行数: {len(factor_data)}')

selector = FactorRankSelector(factor_name='momentum_20', ascending=False)
timing = TrendVolatilityTiming()
portfolio = EqualWeightBuilder()

strategy = QuantStrategy(
    name='测试策略',
    selector=selector,
    timing=timing,
    portfolio=portfolio,
    top_n=30,
)

engine = BacktestEngine()
report = engine.run(
    strategy=strategy,
    factor_data=factor_data,
    start_date='2020-01-01',
    end_date='2020-06-30',
)

print(f'\n回测结果:')
print(f'总收益率: {report.total_return:.2%}')
print(f'年化收益: {report.annualized_return:.2%}')
print(f'最大回撤: {report.max_drawdown:.2%}')
print(f'夏普比率: {report.sharpe_ratio:.2f}')
print(f'胜率: {report.win_rate:.2%}')
print(f'交易次数: {report.total_trades}')

if report.selection_log:
    print(f'\n选股记录(前3条):')
    for rec in report.selection_log[:3]:
        print(f"  {rec['date']}: {rec['n']} 只股票")

print('\n测试完成!')
