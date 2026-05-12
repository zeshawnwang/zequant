#!/usr/bin/env python3
"""
新架构快速验证脚本

测试新架构的 SignalStrategy 是否正常工作
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 80)
print("测试新架构核心模块导入")
print("=" * 80)

try:
    # 测试 1: 导入核心模块
    print("\n1. 导入核心模块...")
    from core import Database, SignalStrategy
    print("   ✓ 核心模块导入成功")

    from core.screening import FactorRankSelector, MultiFactorSelector
    print("   ✓ 选股器导入成功")

    from core.timings import TrendTiming
    from core.signals.position import TrendPositionSizer, VolatilityPositionSizer
    print("   ✓ 择时器/仓位管理导入成功")

    from core.signals import LayeredComposer, DirectComposer
    from core.risk import RiskManager
    print("   ✓ 信号组合和风控导入成功")

    from core.execution import BacktestEngine
    print("   ✓ 回测引擎导入成功")

    print("\n2. 导入策略库...")
    from strategies import build_momentum_strategy_v2, build_low_vol_strategy_v2
    print("   ✓ 策略库导入成功")

except Exception as e:
    print(f"   ✗ 导入失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("创建测试策略实例")
print("=" * 80)

try:
    # 测试 2: 创建基础策略
    print("\n3. 创建动量策略...")
    momentum = build_momentum_strategy_v2(top_n=30)
    print(f"   ✓ 动量策略创建成功: {momentum.name}")

    print("\n4. 创建低波策略...")
    low_vol = build_low_vol_strategy_v2(top_n=30)
    print(f"   ✓ 低波策略创建成功: {low_vol.name}")

    # 测试 3: 策略信息
    print("\n5. 获取策略描述...")
    print(f"   动量策略:\n{momentum.get_description()}\n")
    print(f"   低波策略:\n{low_vol.get_description()}\n")

    print("\n" + "=" * 80)
    print("架构测试完成！✅")
    print("=" * 80)
    print("""
新架构特点:
  - 模块化设计: 每个组件独立在 core/ 子目录
  - 基类/实现分离: 每个模块都有 base/ 和 impl/
  - 信号流驱动: Selector → PositionSizer → Composer → RiskManager
  - 积木式拼装: 可以自由组合不同组件

核心模块:
  - core/screening/      选股器
  - core/timings/        择时器（独立基类目录）
  - core/positioners/    仓位分配器（旧 portfolios/）
  - core/signals/        信号组合和仓位管理
  - core/risk/           风控系统
  - core/execution/      回测引擎和实盘执行
  - core/research/       研究工具（因子评估、归因）
  - core/optimization/   参数优化
  - core/monitor/        绩效监控

策略库:
  - strategies/         策略实现（使用新架构）

研究日志:
  - research/            研究记录和模板
""")

except Exception as e:
    print(f"   ✗ 测试失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
