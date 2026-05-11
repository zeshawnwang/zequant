#!/usr/bin/env python3
"""完整策略优化 - 对所有因子进行筛选和权重优化。"""
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

print('=' * 80)
print('策略优化完整流程 - 所有因子')
print('=' * 80)
print(f'开始时间: {datetime.now()}')

db = Database()

all_factors = db.list_factor_columns()
print(f'数据库中共有 {len(all_factors)} 个因子')

# 获取最新数据日期
latest_date = db.get_max_date('daily_bars')
print(f'最新数据日期: {latest_date}')

risk_constraints = RiskConstraints(
    max_drawdown=0.60,
    single_stock_weight=0.15,
    single_sector_weight=0.25,
    max_volatility=0.30,
    max_turnover=1.00,
    min_calmar_ratio=0.5,
    min_win_rate=0.50,
)

print('\n【风控配置】')
print(f'  最大回撤: {risk_constraints.max_drawdown:.0%}')
print(f'  单只股票仓位: {risk_constraints.single_stock_weight:.0%}')
print(f'  单行业仓位: {risk_constraints.single_sector_weight:.0%}')

# 阶段1: 因子筛选
print('\n' + '=' * 80)
print('阶段1: 因子筛选')
print('=' * 80)
print(f'回测区间: 2019-01-01 ~ {latest_date}')
print(f'筛选目标: 选出30个最佳因子')

factor_selector = FactorSelector(
    db=db,
    risk_constraints=risk_constraints,
    top_n=30,
    target_factor_count=30,
)

factor_results = factor_selector.run(
    factor_names=all_factors,
    start_date='2019-01-01',
    end_date=latest_date,
    parallel=False,
)

print(f'\n筛选完成,共 {len(factor_results)} 个结果')

if factor_results:
    df = factor_selector.get_results_df()
    print('\n因子筛选结果:')
    print(df[['factor_name', 'total_return', 'annual_return', 'max_drawdown', 'sharpe_ratio', 'win_rate', 'risk_passed', 'score']].head(20).to_string())

    top_factors = factor_selector.get_top_factors()
    print(f'\n通过风控的因子 ({len(top_factors)} 个):')
    for i, fr in enumerate(top_factors[:30], 1):
        print(f'  {i:2d}. {fr.factor_name:<20s} 得分={fr.score:.4f} 年化={fr.report.annualized_return:+.2%} 回撤={fr.report.max_drawdown:.2%}')
else:
    top_factors = []
    print('没有因子通过筛选')

# 阶段2: 权重优化
if top_factors:
    print('\n' + '=' * 80)
    print('阶段2: 权重优化')
    print('=' * 80)

    factor_names = [fr.factor_name for fr in top_factors]
    print(f'优化因子数: {len(factor_names)}')
    print(f'回测区间: 2021-01-01 ~ 2023-12-31')
    print(f'遗传算法: 种群=50, 代数=100')

    optimizer = GeneticWeightOptimizer(
        db=db,
        risk_constraints=risk_constraints,
        factor_names=factor_names,
        top_n=30,
        population_size=50,
        generations=100,
    )

    best_weights = optimizer.run(
        start_date='2021-01-01',
        end_date='2023-12-31',
        target_config_count=5,
    )

    print(f'\n优化完成,获得 {len(best_weights)} 个配置')

    if best_weights:
        print('\n权重优化结果:')
        for i, config in enumerate(best_weights, 1):
            print(f'\n配置{i}: {config.name}')
            print(f'  得分: {config.score:.4f}')
            print(f'  年化收益: {config.report.annualized_return:.2%}')
            print(f'  最大回撤: {config.report.max_drawdown:.2%}')
            print(f'  夏普比率: {config.report.sharpe_ratio:.2f}')
            print(f'  胜率: {config.report.win_rate:.2%}')

            non_zero_weights = {k: v for k, v in config.weights.items() if abs(v) > 0.001}
            sorted_weights = sorted(non_zero_weights.items(), key=lambda x: -abs(x[1]))[:10]
            print(f'  主要权重: {dict(sorted_weights)}')

        # 阶段3: 保存配置
        print('\n' + '=' * 80)
        print('阶段3: 保存配置')
        print('=' * 80)

        config_manager = ConfigManager()
        config = config_manager.build_config(
            factor_results=factor_results,
            weight_configs=best_weights,
            risk_constraints=risk_constraints,
            created_at=str(datetime.now()),
        )
        config_manager.save(config)

        print(f'配置已保存到: config/optimized_strategy_config.yaml')
    else:
        print('\n没有有效的权重配置')
else:
    print('\n没有通过风控的因子,跳过权重优化')

print('\n' + '=' * 80)
print(f'结束时间: {datetime.now()}')
print('=' * 80)
