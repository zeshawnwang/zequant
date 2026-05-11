#!/usr/bin/env python3
"""测试优化流程。"""
import sys
sys.path.insert(0, '.')

from datetime import datetime
from core.database import Database
from core.optimization import (
    RiskConstraints,
    FactorSelector,
    GeneticWeightOptimizer,
    ConfigManager,
)

print('测试策略优化流程...')
print(f'开始时间: {datetime.now()}')

db = Database()

risk_constraints = RiskConstraints(
    max_drawdown=0.20,
    single_stock_weight=0.15,
    single_sector_weight=0.25,
    max_volatility=0.30,
    max_turnover=1.00,
    min_calmar_ratio=0.5,
    min_win_rate=0.50,
)

# 阶段1: 因子筛选
print('\n=== 阶段1: 因子筛选 ===')
factor_selector = FactorSelector(
    db=db,
    risk_constraints=risk_constraints,
    top_n=30,
    target_factor_count=5,
)

test_factors = ['momentum_20', 'momentum_5', 'volatility_20', 'rsi_14', 'macd']
print(f'测试因子: {test_factors}')

factor_results = factor_selector.run(
    factor_names=test_factors,
    start_date='2020-01-01',
    end_date='2020-12-31',
    parallel=False,
)

print(f'筛选完成,共 {len(factor_results)} 个结果')

if factor_results:
    df = factor_selector.get_results_df()
    print('\n因子筛选结果:')
    print(df[['factor_name', 'total_return', 'annual_return', 'max_drawdown', 'sharpe_ratio', 'risk_passed']])

# 阶段2: 权重优化
print('\n=== 阶段2: 权重优化 ===')
top_factors = [fr.factor_name for fr in factor_results if fr.risk_check_passed]
if not top_factors:
    top_factors = test_factors[:3]

print(f'用于优化的因子: {top_factors}')

optimizer = GeneticWeightOptimizer(
    db=db,
    risk_constraints=risk_constraints,
    factor_names=top_factors,
    top_n=30,
    population_size=5,
    generations=2,
)

best_weights = optimizer.run(
    start_date='2021-01-01',
    end_date='2021-12-31',
    target_config_count=2,
)

print(f'优化完成,获得 {len(best_weights)} 个配置')

if best_weights:
    print('\n权重优化结果:')
    for i, config in enumerate(best_weights, 1):
        print(f'\n配置{i}: {config.name}')
        print(f'  得分: {config.score:.4f}')
        print(f'  年化收益: {config.report.annualized_return:.2%}')
        print(f'  最大回撤: {config.report.max_drawdown:.2%}')
        print(f'  夏普比率: {config.report.sharpe_ratio:.2f}')
        print(f'  权重: {dict(sorted(config.weights.items(), key=lambda x: -abs(x[1])))}')

# 阶段3: 保存配置
print('\n=== 阶段3: 保存配置 ===')
config_manager = ConfigManager()

if factor_results and best_weights:
    config = config_manager.build_config(
        factor_results=factor_results,
        weight_configs=best_weights,
        risk_constraints=risk_constraints,
        created_at=str(datetime.now()),
    )
    config_manager.save(config)
    print('配置已保存')

print(f'\n结束时间: {datetime.now()}')
print('测试完成!')
