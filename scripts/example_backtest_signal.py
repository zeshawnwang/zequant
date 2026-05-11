"""新架构策略回测示例

展示如何使用新架构策略进行回测。
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from core.execution.backtest import BacktestEngine, BacktestReport
from core.monitor.performance import PerformanceMonitor
from strategies.example_signal_strategy import STRATEGIES
from strategies.config_signal_strategy import get_config, list_configs
from core.data_fetcher import fetch_market_data
from core.config import get_config as get_global_config


def run_backtest(strategy_name: str, config_name: str = None,
                start_date: str = "2020-01-01",
                end_date: str = "2024-12-31",
                initial_capital: float = 1_000_000) -> BacktestReport:
    """
    运行新架构策略回测

    Args:
        strategy_name: 策略名称 (momentum_v2, low_vol_v2, trend_vol_v2)
        config_name: 配置名称 (可选)
        start_date: 回测开始日期
        end_date: 回测结束日期
        initial_capital: 初始资金

    Returns:
        BacktestReport: 回测报告
    """
    print("=" * 80)
    print(f"新架构策略回测: {strategy_name}")
    if config_name:
        print(f"使用配置: {config_name}")
    print("=" * 80)

    # 获取策略配置
    strategy_config = {}
    if config_name:
        strategy_config = get_config(config_name)

    # 构建策略
    if strategy_name not in STRATEGIES:
        raise ValueError(f"未知策略: {strategy_name}, 可用: {list(STRATEGIES.keys())}")

    top_n = strategy_config.get("top_n", 30)
    strategy = STRATEGIES[strategy_name](
        top_n=top_n,
        strategy_config=strategy_config,
    )
    print(f"\n✓ 策略已构建: {strategy.name}")

    # 加载数据
    print("\n正在加载市场数据...")
    market_data = fetch_market_data(start_date, end_date)
    if market_data is None or len(market_data) == 0:
        raise ValueError("未能加载市场数据")
    print(f"✓ 数据加载完成: {len(market_data)} 条记录")

    # 创建回测引擎
    print("\n初始化回测引擎...")
    backtest_engine = BacktestEngine(
        initial_capital=initial_capital,
        fee_config={"commission": 0.0003, "min_fee": 5},
    )
    print("✓ 回测引擎已初始化")

    # 运行回测
    print("\n开始回测...")
    report = backtest_engine.run(
        strategy=strategy,
        factor_data=market_data,
        start_date=start_date,
        end_date=end_date,
    )
    print("✓ 回测完成")

    # 分析绩效
    print("\n" + "=" * 80)
    print("绩效分析")
    print("=" * 80)

    monitor = PerformanceMonitor()
    perf_report = monitor.analyze(
        equity_curve=report.equity_curve,
        trades=report.trades,
    )

    # 打印报告
    print_report(report, perf_report)

    return report


def print_report(report: BacktestReport, perf_report: PerformanceMonitor):
    """打印回测报告"""
    print(f"\n{'指标':<20}{'结果':>15}")
    print("-" * 35)
    print(f"{'总收益率':<20}{report.total_return * 100:>15.2f}%")
    print(f"{'年化收益率':<20}{report.annualized_return * 100:>15.2f}%")
    print(f"{'最大回撤':<20}{report.max_drawdown * 100:>15.2f}%")
    print(f"{'夏普比率':<20}{report.sharpe_ratio:>15.2f}")
    print(f"{'胜率':<20}{report.win_rate * 100:>15.2f}%")
    print(f"{'盈亏比':<20}{report.profit_factor:>15.2f}")
    print(f"{'交易次数':<20}{report.total_trades:>15}")
    print(f"{'期末总净值':<20}{report.final_value:>15,.2f}")

    print("\n" + "=" * 80)
    print(f"策略配置")
    print("=" * 80)
    print(f"策略名称: {report.strategy_name}")
    print(f"回测区间: {report.start_date} ~ {report.end_date}")


def main():
    """主函数"""
    print("新架构策略回测系统")
    print("=" * 80)

    # 列出可用策略
    print("\n可用策略:")
    for name in STRATEGIES.keys():
        print(f"  - {name}")

    print("\n可用配置:")
    for name in list_configs():
        print(f"  - {name}")

    # 选择一个策略运行（示例）
    print("\n" + "=" * 80)
    print("运行示例回测")
    print("=" * 80)

    try:
        # 运行示例回测
        report = run_backtest(
            strategy_name="momentum_v2",
            config_name="momentum_aggressive",
            start_date="2020-01-01",
            end_date="2024-12-31",
        )
    except Exception as e:
        print(f"\n✗ 回测失败: {e}")
        print("\n提示: 请确保数据库中有足够的因子数据")


if __name__ == "__main__":
    main()
