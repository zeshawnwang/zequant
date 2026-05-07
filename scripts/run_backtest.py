#!/usr/bin/env python3
"""
回测脚本
运行策略回测并打印报告。
"""
import sys
sys.path.insert(0, '.')

import yaml
from core.database import Database
from core.backtest import BacktestEngine
from strategies.momentum_strategy import create_momentum_strategy, create_low_vol_strategy

def main():
    # 加载配置
    with open('./config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    db = Database()

    # 获取因子数据
    print("加载因子数据...")
    factors = db.get_factors()

    if factors.empty:
        print("无因子数据，请先运行 compute_factors.py")
        db.close()
        return

    print(f"因子数据: {len(factors)} 条, {factors['date'].min()} ~ {factors['date'].max()}")

    # 创建策略
    strategy = create_momentum_strategy(top_n=50)
    print(f"\n策略: {strategy.name}")
    print(strategy.get_description())

    # 回测
    engine = BacktestEngine(
        initial_capital=config['backtest']['initial_capital'],
        fee_config=config['fees'],
        risk_config={'max_position_pct': config['risk']['max_position_pct'],
                     'max_total_position': config['risk']['max_total_position'],
                     'stop_loss': config['risk']['stop_loss'],
                     'take_profit': config['risk']['take_profit']}
    )

    print(f"\n运行回测: {config['backtest']['start_date']} ~ {config['backtest']['end_date']}...")
    report = engine.run(
        strategy=strategy,
        factor_data=factors,
        start_date=config['backtest']['start_date'],
        end_date=config['backtest']['end_date'],
        rebalance_freq=config['backtest']['rebalance_freq']
    )

    # 打印报告
    print("\n" + "="*50)
    print("回测报告")
    print("="*50)
    print(f"总收益:       {report.total_return:.2%}")
    print(f"年化收益:     {report.annualized_return:.2%}")
    print(f"最大回撤:     {report.max_drawdown:.2%}")
    print(f"夏普比率:     {report.sharpe_ratio:.2f}")
    print(f"胜率:         {report.win_rate:.2%}")
    print(f"profit_factor: {report.profit_factor:.2f}")
    print(f"总交易次数:   {report.total_trades}")
    print("="*50)

    db.close()

if __name__ == "__main__":
    main()
