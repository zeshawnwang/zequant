#!/usr/bin/env python3
"""分类因子优化启动脚本 - 支持断点续传。"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
from daily_optimizer import DailyOptimizer

print('=' * 60)
print('分类因子优化 - 按10类因子分组')
print('=' * 60)
print(f'开始时间: {datetime.now()}')

optimizer = DailyOptimizer(
    db_path="./data/quant_data.db",
    output_dir="./daily",
    top_n_per_category=5,
    max_drawdown=0.60,
)

print('\n【风控配置】')
print(f'  最大回撤: {optimizer.risk_constraints.max_drawdown:.0%}')
print(f'  单只股票仓位: {optimizer.risk_constraints.single_stock_weight:.0%}')
print(f'  单行业仓位: {optimizer.risk_constraints.single_sector_weight:.0%}')

optimizer.run_full(
    start_date='2019-01-01',
    end_date=None,
)

print('\n' + '=' * 60)
print(f'结束时间: {datetime.now()}')
print('=' * 60)
